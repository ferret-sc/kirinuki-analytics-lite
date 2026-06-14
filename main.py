# ============================================================
# きりぬき分析ライト - main.py
# ------------------------------------------------------------
# 配信のチャットを分析して「盛り上がり・スパチャ・メンバー」を
# 可視化する軽量版。動画編集・AI(ローカル/外部)機能は持たない。
# どんなPC環境でも動く・配布が軽い・ブラウザだけで完結。
#
# 起動: streamlit run main.py
# ============================================================

from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

import analyzer

# ============================================================
# ページ設定 & デザイン
# ============================================================

st.set_page_config(
    page_title="きりぬき分析ライト 📊",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_BG = "#FAFAFA"
COLOR_CORAL = "#FF6B81"
COLOR_LAVENDER = "#A55EEA"

FETCH_RANGE_CHOICES = {
    "30分": 0.5, "1時間": 1.0, "1時間30分": 1.5, "2時間": 2.0,
    "3時間": 3.0, "4時間": 4.0, "6時間": 6.0, "8時間": 8.0, "12時間": 12.0,
}
SENSITIVITY_MAP = {"低(大きな山のみ)": 2.5, "標準": 2.0, "高(小さな山も拾う)": 1.5}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {COLOR_BG}; color: #333333; }}
        [data-testid="stSidebar"] {{ background: linear-gradient(180deg,#FFF5F6 0%,#F7F0FF 100%); }}
        h1, h2, h3 {{ color: #3A3A3A; font-weight: 700; }}
        .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg,{COLOR_CORAL} 0%,{COLOR_LAVENDER} 100%);
            color: white; border: none; border-radius: 12px; padding: 0.6rem 1.4rem;
            font-weight: 600; transition: all 0.25s ease;
            box-shadow: 0 4px 12px rgba(255,107,129,0.25);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-2px); box-shadow: 0 6px 18px rgba(165,94,234,0.35); color: white;
        }}
        .stTextInput input {{ border-radius: 12px !important; border: 1.5px solid #EEE !important; }}
        .content-card {{
            background: white; border-radius: 12px; padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1.5px solid #F0F0F0; margin-bottom: 1rem;
        }}
        .situation-badge {{
            display: inline-block; color: white; border-radius: 12px; padding: 0.3rem 0.9rem;
            font-weight: 700; font-size: 0.95rem; margin-right: 0.5rem;
        }}
        .preview-frame {{
            width: 100%; aspect-ratio: 16/9; border: none; border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        }}
        [data-testid="stExpander"] {{
            background: white; border-radius: 12px !important; border: 1.5px solid #F0F0F0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# セッション管理
# ============================================================

def init_state() -> None:
    defaults = {
        "page": "home", "video_url": "", "chat_df": None,
        "agg_df": None, "spikes_df": None, "points": None, "demo_mode": False,
        "preview_minute": 0, "preview_slider_min": 0, "last_chart_click": None,
        "video_library": {}, "active_video_id": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def goto(page: str) -> None:
    st.session_state["page"] = page


def _reset_analysis() -> None:
    for k in ("agg_df", "spikes_df", "points"):
        st.session_state[k] = None
    for k in ("preview_minute", "preview_slider_min", "last_chart_click"):
        st.session_state.pop(k, None)


def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"約{seconds}秒"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"約{m}分{s:02d}秒"
    h, m = divmod(m, 60)
    return f"約{h}時間{m:02d}分"


def _filename(prefix: str, ext: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.{ext}"


def _csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# 複数動画ライブラリ
# ============================================================

def _video_key() -> str:
    if st.session_state.get("demo_mode"):
        return "demo"
    vid = analyzer.extract_video_id(st.session_state.get("video_url", "") or "")
    return vid or "unknown"


def _video_title() -> str:
    df = st.session_state.get("chat_df")
    if df is not None and df.attrs.get("video_title"):
        return df.attrs["video_title"]
    if st.session_state.get("demo_mode"):
        return "🧪 デモ配信"
    return (st.session_state.get("video_url", "") or "")[:45] or "(不明)"


def _snapshot() -> None:
    if st.session_state.get("points") is None:
        return
    st.session_state["video_library"][_video_key()] = {
        "video_url": st.session_state["video_url"], "demo_mode": st.session_state["demo_mode"],
        "chat_df": st.session_state["chat_df"], "agg_df": st.session_state["agg_df"],
        "spikes_df": st.session_state["spikes_df"], "points": st.session_state["points"],
        "title": _video_title(),
    }
    st.session_state["active_video_id"] = _video_key()


def _restore(key: str) -> None:
    snap = st.session_state["video_library"].get(key)
    if not snap:
        return
    for k in ("video_url", "demo_mode", "chat_df", "agg_df", "spikes_df", "points"):
        st.session_state[k] = snap[k]
    st.session_state["active_video_id"] = key
    for k in ("preview_minute", "preview_slider_min", "last_chart_click"):
        st.session_state.pop(k, None)


# ============================================================
# 分析(スパイク検出 + ルールベース分類。AI・構成案は持たない)
# ============================================================

def run_analysis(chat_df, z_threshold, max_spikes, per_hour, context=1):
    agg = analyzer.aggregate_per_minute(chat_df)
    spikes = analyzer.detect_spikes(agg, z_threshold=z_threshold,
                                    max_spikes=max_spikes, per_hour=per_hour)
    points = []
    for _, sp in spikes.iterrows():
        minute = int(sp["minute"])
        window = chat_df[(chat_df["minute"] >= minute - context)
                         & (chat_df["minute"] <= minute + context)]
        messages = window["message"].astype(str).tolist()
        situation, reason = analyzer.classify_situation_rule_based(messages)
        info = analyzer.SITUATIONS[situation]
        points.append({
            "minute": minute,
            "timestamp": analyzer.minute_to_label(minute),
            "count": int(sp["count"]),
            "situation": situation,
            "emoji": info["emoji"],
            "color": info["color"],
            "reason": reason,
            "samples": messages[:10],
        })
    return agg, spikes, points


# ============================================================
# 共通UI
# ============================================================

def header() -> None:
    st.markdown(
        f"""
        <div style="text-align:center; padding:0.5rem 0 1rem;">
            <h1 style="margin-bottom:0.2rem;">📊 きりぬき分析ライト</h1>
            <p style="color:#888;">URLを貼るだけ。配信の<span style="color:{COLOR_CORAL};font-weight:700;">
            盛り上がり・スパチャ・メンバー</span>をまるっと分析(軽量版)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📊 メニュー")
        if st.button("🏠 ① URL入力", use_container_width=True):
            goto("home")
        if st.button("📈 ② 分析結果", use_container_width=True):
            goto("analyze")

        lib = st.session_state.get("video_library", {})
        if lib:
            st.divider()
            st.markdown("### 📚 分析した動画")
            keys = list(lib)
            active = st.session_state.get("active_video_id")
            idx = keys.index(active) if active in keys else 0

            def _switch():
                sel = st.session_state.get("lib_select")
                if sel and sel != st.session_state.get("active_video_id"):
                    _restore(sel)
                    goto("analyze")

            st.selectbox("切り替え", options=keys, index=idx,
                         format_func=lambda k: f"🎬 {lib[k].get('title', k)[:26]}",
                         key="lib_select", on_change=_switch, label_visibility="collapsed")
            if st.button("➕ 別の動画を分析", use_container_width=True):
                goto("home")

        st.divider()
        st.caption("🎁 軽量版 — 動画編集・AI機能は含みません")
        st.caption(f"{'🟢' if analyzer.YTDLP_CHAT_AVAILABLE else '🔴'} チャット取得 (yt-dlp)")
        st.caption("ℹ️ 高機能な切り抜き作成は『きりぬきスタジオ(フル版)』をご利用ください")


# ============================================================
# 画面1: URL入力
# ============================================================

def page_home() -> None:
    st.markdown(
        """
        <div class="content-card">
            <h3>🔗 ステップ① 配信アーカイブのURLを入力</h3>
            <p style="color:#888;">チャットリプレイが有効なYouTube配信アーカイブのURLを貼り付けてください。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    url = st.text_input("YouTube URL", value=st.session_state["video_url"],
                        placeholder="https://www.youtube.com/watch?v=XXXXXXXXXXX",
                        label_visibility="collapsed")

    with st.expander("⚙️ 詳細設定", expanded=True):
        range_label = st.select_slider("チャットを取得する範囲(配信開始からどこまで)",
                                       options=list(FETCH_RANGE_CHOICES), value="3時間",
                                       help="動画がこれより短い場合は自動で全編が対象になります。")
        max_hours = FETCH_RANGE_CHOICES[range_label]
        use_cache = st.checkbox("♻️ キャッシュを使う(同じ配信の2回目以降は瞬時)", value=True)
        if st.button("🗑️ キャッシュをすべて削除"):
            st.success(f"キャッシュを{analyzer.clear_cache()}件削除しました。")

    col1, col2 = st.columns(2)
    fetch_clicked = col1.button("📥 チャットを取得して分析", use_container_width=True)
    demo_clicked = col2.button("🧪 デモデータで試す", use_container_width=True)

    if fetch_clicked:
        if not analyzer.extract_video_id(url):
            st.error("URLが正しくありません。YouTubeの動画URLを確認してください。")
            return
        st.session_state["video_url"] = url
        bar = st.progress(0.0, text="📡 動画情報を確認中…")
        t0 = time.time()

        def on_prog(count, pos, target):
            frac = min(pos / target, 1.0) if target else 0.0
            txt = f"📥 配信 {analyzer.minute_to_label(int(pos // 60))} 地点まで取得 ({frac*100:.0f}%)"
            if count:
                txt += f" / {count:,}コメント"
            if frac > 0.03:
                txt += f" — 残り目安 {_fmt_eta((time.time()-t0)*(1-frac)/frac)}"
            bar.progress(frac, text=txt)

        try:
            df = analyzer.fetch_chat_log(url, max_seconds=int(max_hours * 3600),
                                         use_cache=use_cache, progress_callback=on_prog)
            st.session_state["chat_df"] = df
            st.session_state["demo_mode"] = False
            st.toast(f"✅ {len(df):,}件取得 ({_fmt_eta(time.time()-t0)})", icon="✅")
            _reset_analysis()
            bar.empty()
            goto("analyze")
            st.rerun()
        except Exception as e:
            bar.empty()
            st.error(f"取得に失敗しました: {e}")
            st.info("💡 「デモデータで試す」で動作確認ができます。")

    if demo_clicked:
        st.session_state["chat_df"] = analyzer.generate_mock_chat_log()
        st.session_state["demo_mode"] = True
        _reset_analysis()
        goto("analyze")
        st.rerun()


# ============================================================
# 画面2: 分析結果
# ============================================================

def _set_preview(minute: int) -> None:
    st.session_state["preview_minute"] = int(minute)
    st.session_state["preview_slider_min"] = int(minute)


def _jump_notify(minute: int, label: str) -> None:
    _set_preview(minute)
    st.toast(f"▶️ {label} をプレビューにセットしました", icon="🎬")


def _scene_preview(max_minute: int) -> None:
    with st.expander("🎬 シーンプレビュー(クリックした場面を確認)", expanded=False):
        if st.session_state["demo_mode"] or not st.session_state["video_url"]:
            st.caption("🧪 デモモードではプレビューできません。実際の配信URLで分析するとここに動画が表示されます。")
            return
        st.session_state.setdefault("preview_minute", 0)
        st.session_state.setdefault("preview_slider_min", 0)

        def _slider_jump():
            st.session_state["preview_minute"] = int(st.session_state["preview_slider_min"])

        col1, col2 = st.columns([3, 1])
        with col1:
            autoplay = st.toggle("自動再生", value=True,
                                 help="ジャンプ時にミュート再生します。音はプレーヤーで解除できます。")
        st.slider("⏩ 位置(分)", 0, max(1, max_minute), step=1,
                  key="preview_slider_min", on_change=_slider_jump)
        minute = int(st.session_state.get("preview_minute", 0))
        vid = analyzer.extract_video_id(st.session_state["video_url"])
        params = f"start={minute*60}" + ("&autoplay=1&mute=1" if autoplay else "")
        # st.markdown で埋め込むことで .preview-frame の16:9比率が効く(高さ0にならない)
        st.markdown(
            f'<iframe class="preview-frame" src="https://www.youtube.com/embed/{vid}?{params}" '
            f'allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>',
            unsafe_allow_html=True,
        )
        st.caption(f"▶️ {analyzer.minute_to_label(minute)} 地点(プレーヤーで自由にシーク・ミュート解除できます)")


def _chart_clicked_minute(event):
    try:
        pts = event.selection.points
        if pts:
            return int(round(float(pts[0]["x"])))
    except Exception:
        pass
    return None


def page_analyze() -> None:
    chat_df = st.session_state["chat_df"]
    if chat_df is None:
        st.info("まずはステップ①でURLを入力するか、デモデータを読み込んでください。")
        if st.button("← URL入力へ戻る"):
            goto("home")
            st.rerun()
        return

    if st.session_state["demo_mode"]:
        st.markdown(
            f"<span style='background:{COLOR_LAVENDER};color:white;border-radius:12px;"
            f"padding:0.2rem 0.8rem;font-size:0.85rem;'>🧪 デモモードで動作中</span>",
            unsafe_allow_html=True,
        )

    data_end = float(chat_df["elapsed_sec"].max())
    st.caption(f"📡 分析対象: 0:00:00 〜 {analyzer.minute_to_label(int(data_end//60))} "
               f"/ {len(chat_df):,}コメント")

    with st.expander("🔧 検出設定(任意)"):
        st.slider("検出する最大ポイント数", 4, 20, value=12, key="cfg_max")
        st.select_slider("検出感度", options=list(SENSITIVITY_MAP), value="標準", key="cfg_sens")
        st.checkbox("⏳ 時間帯ごとに評価(長時間配信向け)", value=data_end > 3 * 3600, key="cfg_ph")
        if st.button("🔁 この設定で再分析", use_container_width=True):
            _reset_analysis()
            st.rerun()

    if st.session_state["points"] is None:
        with st.spinner("📊 盛り上がりを分析中…"):
            agg, spikes, points = run_analysis(
                chat_df,
                z_threshold=SENSITIVITY_MAP[st.session_state.get("cfg_sens", "標準")],
                max_spikes=st.session_state.get("cfg_max", 12),
                per_hour=st.session_state.get("cfg_ph", False),
            )
            st.session_state.update({"agg_df": agg, "spikes_df": spikes, "points": points})

    agg_df = st.session_state["agg_df"]
    spikes_df = st.session_state["spikes_df"]
    points = st.session_state["points"]

    # グラフ
    with st.expander("📈 チャット盛り上がりグラフ", expanded=True):
        event = st.plotly_chart(analyzer.build_spike_chart(agg_df, spikes_df),
                                use_container_width=True, on_select="rerun",
                                selection_mode=("points",), key="chart")
        clicked = _chart_clicked_minute(event)
        if clicked is not None and clicked != st.session_state.get("last_chart_click"):
            st.session_state["last_chart_click"] = clicked
            _set_preview(clicked)

    _scene_preview(int(data_end // 60))

    # スパチャ
    sc = analyzer.superchat_summary(chat_df)
    with st.expander("💰 スーパーチャット分析", expanded=False):
        if not sc:
            st.info("この配信にはスーパーチャットがありませんでした。")
        else:
            cur = sc["main_currency"]
            c1, c2, c3 = st.columns(3)
            c1.metric("スパチャ件数", f"{sc['total_count']}件")
            c2.metric(f"合計金額 ({cur})", f"{sc['by_currency'].iloc[0]['total']:,.0f}")
            c3.metric("最高額の瞬間", analyzer.minute_to_label(int(sc["top_moments"].iloc[0]["minute"])))
            st.plotly_chart(analyzer.build_superchat_chart(sc["per_minute"], cur),
                            use_container_width=True)
            t1, t2, t3 = st.tabs(["💎 高額TOP5", "🧾 全スパチャ一覧", "👥 投稿者ごと"])
            with t1:
                for _, r in sc["top_moments"].iterrows():
                    st.markdown(f"- ⏰ **{analyzer.minute_to_label(int(r['minute']))}** — "
                                f"{r['sc_amount']:,.0f} {r['sc_currency']} "
                                f"({r['author']})「{r['message']}」")
            with t2:
                st.dataframe(sc["all"], use_container_width=True, hide_index=True)
                st.download_button("⬇️ 全スパチャをCSVで保存", _csv_bytes(sc["all"]),
                                   _filename("スーパーチャット一覧", "csv"), "text/csv",
                                   use_container_width=True, key="sc_all")
            with t3:
                st.dataframe(sc["by_author"], use_container_width=True, hide_index=True)
                st.download_button("⬇️ 投稿者別集計をCSVで保存", _csv_bytes(sc["by_author"]),
                                   _filename("スパチャ投稿者別", "csv"), "text/csv",
                                   use_container_width=True, key="sc_au")

    # メンバーシップ
    mem = analyzer.membership_summary(chat_df)
    with st.expander("🎫 メンバーシップ分析", expanded=False):
        if not mem:
            st.info("この配信にはメンバーシップ加入(新規・継続・ギフト)がありませんでした。")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("加入イベント", f"{mem['total_events']}件")
            m2.metric("新規", f"{mem['new_count']}人")
            m3.metric("継続", f"{mem['milestone_count']}人")
            m4.metric("ギフト", f"{mem['gifted_total']}口")
            st.plotly_chart(analyzer.build_membership_chart(mem["per_minute"]),
                            use_container_width=True)
            t1, t2 = st.tabs(["🧾 全加入一覧", "👥 投稿者ごと"])
            with t1:
                st.dataframe(mem["all"], use_container_width=True, hide_index=True)
                st.download_button("⬇️ 全メンバー加入をCSVで保存", _csv_bytes(mem["all"]),
                                   _filename("メンバーシップ一覧", "csv"), "text/csv",
                                   use_container_width=True, key="mem_all")
            with t2:
                st.dataframe(mem["by_author"], use_container_width=True, hide_index=True)
                st.download_button("⬇️ 投稿者別集計をCSVで保存", _csv_bytes(mem["by_author"]),
                                   _filename("メンバー投稿者別", "csv"), "text/csv",
                                   use_container_width=True, key="mem_au")

    # キーワード検索
    with st.expander("🔍 キーワードで深掘り検索(任意)"):
        kw = st.text_input("検索ワード", placeholder="例: 草 / てぇてぇ / 名前など")
        if kw:
            hits = analyzer.keyword_search(chat_df, kw)
            if hits.empty:
                st.warning("そのキーワードは見つかりませんでした。")
            else:
                st.plotly_chart(analyzer.build_spike_chart(hits, hits.nlargest(3, "count")),
                                use_container_width=True)

    # 盛り上がりポイント一覧
    st.markdown("### 🎯 盛り上がりポイント一覧")
    if not points:
        st.warning("盛り上がりが検出されませんでした。チャット数が少ない可能性があります。")
        _snapshot()
        return

    # CSVエクスポート
    import pandas as pd
    pts_df = pd.DataFrame([{
        "時刻": p["timestamp"], "経過分": p["minute"],
        "シチュエーション": f"{p['emoji']} {p['situation']}",
        "コメント数/分": p["count"], "判定根拠": p["reason"],
        "チャット抜粋": " / ".join(p["samples"][:8]),
    } for p in points])
    st.download_button("⬇️ 盛り上がりポイント一覧をCSVで保存(Excel対応)",
                       _csv_bytes(pts_df), _filename("盛り上がりポイント", "csv"), "text/csv",
                       use_container_width=True, key="pts_dl")

    for p in points:
        col_card, col_btn = st.columns([4, 1], vertical_alignment="center")
        with col_card:
            st.markdown(
                f"""
                <div class="content-card" style="margin-bottom:0.6rem;">
                    <span class="situation-badge" style="background:{p['color']};">
                        {p['emoji']} {p['situation']}</span>
                    <b>⏰ {p['timestamp']}</b>
                    <span style="color:#999;"> | 💬 {p['count']}コメント/分</span>
                    <div style="color:#888;font-size:0.85rem;margin-top:0.4rem;">🤖 {p['reason']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_btn:
            st.button(f"▶️ {p['timestamp']}", key=f"jump_{p['minute']}",
                      on_click=_jump_notify, args=(p["minute"], p["timestamp"]),
                      use_container_width=True, disabled=st.session_state["demo_mode"],
                      help="この場面を上部のシーンプレビューで再生")
        if p["samples"]:
            with st.expander(f"💬 {p['timestamp']} のチャット抜粋"):
                st.caption(" / ".join(p["samples"][:10]))

    _snapshot()


# ============================================================
# エントリポイント
# ============================================================

def main() -> None:
    inject_css()
    init_state()
    header()
    sidebar()
    page = st.session_state["page"]
    if page == "analyze":
        page_analyze()
    else:
        page_home()


if __name__ == "__main__":
    main()
