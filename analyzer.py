# ============================================================
# analyzer.py - チャットログ分析・AIシチュエーション分類・構成案生成
# ------------------------------------------------------------
# 1. pytchat でアーカイブ配信のチャットログを取得 (不可ならモック生成)
# 2. pandas で1分単位に集計し、スパイク(盛り上がり)を統計的に検出
# 3. スパイク前後のチャット群を Gemini API に渡し、
#    シチュエーション(爆笑/感動/てぇてぇ/絶叫)を分類
# 4. 各スパイクに対する「タイトル案・構成指示・サムネ指示書」を生成
# ============================================================

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import time
import types
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---- 任意依存: yt-dlp (高速チャット取得バックエンド・優先) ----
# 注: chat-downloader はYouTubeの2026年仕様に未対応(ParsingError)のため、
#     活発にメンテされている yt-dlp の live_chat 取得機能を採用している。
try:
    import yt_dlp  # type: ignore
    from yt_dlp.utils import DownloadCancelled  # type: ignore

    YTDLP_CHAT_AVAILABLE = True
except Exception:
    yt_dlp = None
    DownloadCancelled = KeyboardInterrupt
    YTDLP_CHAT_AVAILABLE = False

# ---- 任意依存: pytchat (フォールバック) ----
try:
    import pytchat  # type: ignore

    PYTCHAT_AVAILABLE = True
except Exception:
    pytchat = None
    PYTCHAT_AVAILABLE = False

# チャットログのローカルキャッシュ保存先
CACHE_DIR = Path(__file__).parent / "chat_cache"

# ---- 任意依存: Gemini API (新SDK google-genai を優先、旧SDKにフォールバック) ----
GEMINI_SDK: str | None = None  # "google-genai" / "google-generativeai" / None
try:
    from google import genai as _google_genai  # type: ignore

    GEMINI_SDK = "google-genai"
except Exception:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as _legacy_genai  # type: ignore

        GEMINI_SDK = "google-generativeai"
    except Exception:
        pass

GEMINI_SDK_AVAILABLE = GEMINI_SDK is not None
GEMINI_MODEL_NAME = "gemini-2.0-flash"


class _GeminiClient:
    """新旧SDKのAPI差を吸収する薄いラッパー。

    呼び出し側は client.generate_content(prompt).text だけを使う。
    """

    def __init__(self, api_key: str):
        if GEMINI_SDK == "google-genai":
            self._client = _google_genai.Client(api_key=api_key)
        else:
            _legacy_genai.configure(api_key=api_key)
            self._client = _legacy_genai.GenerativeModel(GEMINI_MODEL_NAME)

    def generate_content(self, prompt: str):
        if GEMINI_SDK == "google-genai":
            return self._client.models.generate_content(
                model=GEMINI_MODEL_NAME, contents=prompt
            )
        return self._client.generate_content(prompt)


# ---- ローカルLLM: Ollama (Gemma等) 連携。APIキー不要・完全無料・課金リスクなし ----
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_ollama_cache: dict = {"ts": 0.0, "models": []}


def _ollama_list_models() -> list[str]:
    """Ollamaにインストール済みのモデル名一覧を返す(30秒キャッシュ)。

    Ollamaが起動していなければ即座に空リストを返す(接続拒否で高速失敗)。
    """
    now = time.time()
    if now - _ollama_cache["ts"] < 30:
        return _ollama_cache["models"]
    models: list[str] = []
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
        models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        models = []
    _ollama_cache["ts"] = now
    _ollama_cache["models"] = models
    return models


def _pick_ollama_model() -> str | None:
    """使用するOllamaモデルを選ぶ。明示設定 > Gemma系の新しいもの > 先頭。"""
    models = _ollama_list_models()
    if not models:
        return None
    configured = None
    try:
        configured = st.secrets.get("OLLAMA_MODEL")
    except Exception:
        pass
    configured = configured or os.environ.get("OLLAMA_MODEL")
    if configured:
        for m in models:
            if m == configured or m.startswith(configured):
                return m
    # gemma系を優先(タグ降順で新しい/大きいものを優先)、無ければ先頭
    gemmas = sorted((m for m in models if "gemma" in m.lower()), reverse=True)
    return gemmas[0] if gemmas else models[0]


class _OllamaClient:
    """ローカルOllamaのHTTP APIラッパー。client.generate_content(prompt).text で使える。"""

    def __init__(self, model: str):
        self.model = model

    def generate_content(self, prompt: str):
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        }
        # このアプリのプロンプトは全てJSON出力を要求するため、format=json で
        # 必ず妥当なJSONを出させる(ローカルモデルの出力ブレによる解析失敗を防ぐ)。
        if "JSON" in prompt or "json" in prompt:
            body["format"] = "json"
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        # ローカルCPU推論は時間がかかるため長めのタイムアウト
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
        return types.SimpleNamespace(text=data.get("response", ""))


# ============================================================
# シチュエーション定義 (分類カテゴリの単一情報源)
# ============================================================

SITUATIONS: dict[str, dict] = {
    "爆笑": {
        "emoji": "🤣",
        "color": "#FF6B81",
        "keywords": ["草", "w", "ｗ", "笑", "くさ", "ワロタ", "腹筋", "lol"],
        "description": "視聴者が大爆笑しているシーン",
    },
    "感動": {
        "emoji": "😭",
        "color": "#A55EEA",
        "keywords": ["涙", "泣", "エモ", "感動", "うるっ", "じーん", "良い話"],
        "description": "感動・エモーショナルなシーン",
    },
    "てぇてぇ": {
        "emoji": "🙏",
        "color": "#FFA94D",
        "keywords": ["てぇてぇ", "尊い", "とうとい", "好き", "ガチ恋", "かわいい", "可愛い"],
        "description": "尊さ・てぇてぇが溢れるシーン",
    },
    "絶叫・ハプニング": {
        "emoji": "😱",
        "color": "#4DABF7",
        "keywords": ["！？", "!?", "やばい", "ヤバい", "やば", "えぇ", "ええ…", "は？", "悲報", "事故"],
        "description": "絶叫・予想外のハプニングシーン",
    },
}


@dataclass
class SpikeAnalysis:
    """1つのスパイク(盛り上がり地点)に対する分析結果。"""

    minute: int                      # 配信開始からの経過分
    timestamp_label: str             # "1:23:45" 形式
    message_count: int               # その1分間のチャット数
    situation: str                   # SITUATIONS のキー
    situation_emoji: str
    situation_color: str
    reason: str                      # 分類根拠 (AIの説明)
    sample_messages: list[str] = field(default_factory=list)
    # 構成案 (generate_clip_plan で充填)
    title_ideas: list[str] = field(default_factory=list)
    structure_guide: str = ""
    thumbnail_guide: str = ""


# ============================================================
# 1. チャットログ取得
# ============================================================

def extract_video_id(url: str) -> str | None:
    """YouTubeのURLから動画IDを抽出する。"""
    patterns = [
        r"(?:v=|youtu\.be/|/live/|/shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url.strip())
        if m:
            return m.group(1)
    return None


CHAT_COLUMNS = [
    "elapsed_sec", "minute", "author", "message",
    "is_superchat", "sc_amount", "sc_currency",
    "is_member", "member_type", "member_count",
]


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=CHAT_COLUMNS)
    df["is_superchat"] = df["is_superchat"].fillna(False).astype(bool)
    df["sc_amount"] = pd.to_numeric(df["sc_amount"], errors="coerce").fillna(0.0)
    df["is_member"] = df["is_member"].fillna(False).astype(bool)
    df["member_type"] = df["member_type"].fillna("").astype(str)
    df["member_count"] = pd.to_numeric(df["member_count"], errors="coerce").fillna(0).astype(int)
    return df


def _runs_to_text(message_obj: dict | None) -> str:
    """YouTubeのrich textランをプレーンテキストに変換する。"""
    if not message_obj:
        return ""
    parts: list[str] = []
    for run in message_obj.get("runs", []):
        if "text" in run:
            parts.append(run["text"])
        elif "emoji" in run:
            shortcuts = run["emoji"].get("shortcuts") or []
            parts.append(shortcuts[0] if shortcuts else "")
    return "".join(parts)


def _parse_amount_text(amount_text: str) -> tuple[float, str]:
    """スパチャ金額表記 '¥1,000' / 'US$5.00' を (金額, 通貨表記) に分解する。"""
    m = re.match(r"\s*([^\d\-]*)([\d,]+(?:\.\d+)?)", amount_text)
    if not m:
        return 0.0, amount_text.strip()
    try:
        amount = float(m.group(2).replace(",", ""))
    except ValueError:
        amount = 0.0
    return amount, m.group(1).strip() or amount_text.strip()


def _parse_live_chat_line(obj: dict) -> dict | None:
    """yt-dlpのlive_chat JSON 1行をチャット行dictに変換する。対象外はNone。"""
    repl = obj.get("replayChatItemAction") or {}
    try:
        elapsed = max(0.0, float(repl.get("videoOffsetTimeMsec", 0)) / 1000.0)
    except (TypeError, ValueError):
        return None
    for action in repl.get("actions", []):
        item = (action.get("addChatItemAction") or {}).get("item") or {}
        r = item.get("liveChatTextMessageRenderer")
        if r:
            return {
                "elapsed_sec": elapsed,
                "minute": int(elapsed // 60),
                "author": (r.get("authorName") or {}).get("simpleText", ""),
                "message": _runs_to_text(r.get("message")),
                "is_superchat": False,
                "sc_amount": 0.0,
                "sc_currency": "",
            }
        r = item.get("liveChatPaidMessageRenderer") or item.get("liveChatPaidStickerRenderer")
        if r:
            amount, currency = _parse_amount_text(
                (r.get("purchaseAmountText") or {}).get("simpleText", "")
            )
            return {
                "elapsed_sec": elapsed,
                "minute": int(elapsed // 60),
                "author": (r.get("authorName") or {}).get("simpleText", ""),
                "message": _runs_to_text(r.get("message")) or "(スーパーステッカー)",
                "is_superchat": True,
                "sc_amount": amount,
                "sc_currency": currency,
            }
        # メンバーシップ加入・継続(マイルストーン)
        r = item.get("liveChatMembershipItemRenderer")
        if r:
            primary = _runs_to_text(r.get("headerPrimaryText"))  # 継続Nヶ月などのテキスト
            sub = (r.get("headerSubtext") or {}).get("simpleText", "") \
                or _runs_to_text(r.get("headerSubtext"))
            if primary:
                mtype, detail = "継続", primary
            else:
                mtype, detail = "新規", (sub or "新規メンバー")
            msg = _runs_to_text(r.get("message"))
            return {
                "elapsed_sec": elapsed,
                "minute": int(elapsed // 60),
                "author": (r.get("authorName") or {}).get("simpleText", ""),
                "message": (f"{detail} {msg}".strip()),
                "is_member": True,
                "member_type": mtype,
                "member_count": 1,
            }
        # メンバーシップ ギフト(まとめ贈与)
        r = item.get("liveChatSponsorshipsGiftPurchaseAnnouncementRenderer")
        if r:
            hdr = (r.get("header") or {}).get("liveChatSponsorshipsHeaderRenderer") or {}
            primary = _runs_to_text(hdr.get("primaryText"))
            m = re.search(r"(\d+)", primary)
            count = int(m.group(1)) if m else 1
            return {
                "elapsed_sec": elapsed,
                "minute": int(elapsed // 60),
                "author": (hdr.get("authorName") or {}).get("simpleText", ""),
                "message": primary or f"メンバーシップギフト {count}件",
                "is_member": True,
                "member_type": "ギフト",
                "member_count": count,
            }
        # メンバーシップ ギフト受領(贈られた側)
        r = item.get("liveChatSponsorshipsGiftRedemptionAnnouncementRenderer")
        if r:
            return {
                "elapsed_sec": elapsed,
                "minute": int(elapsed // 60),
                "author": (r.get("authorName") or {}).get("simpleText", ""),
                "message": _runs_to_text(r.get("message")) or "メンバーギフトを受け取りました",
                "is_member": True,
                "member_type": "ギフト受領",
                "member_count": 1,
            }
    return None


def _peek_last_offset(path: str) -> float | None:
    """ダウンロード途中のlive_chatファイル末尾から現在の動画位置(秒)を読む。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read()
        for line in reversed(tail.split(b"\n")):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            off = (obj.get("replayChatItemAction") or {}).get("videoOffsetTimeMsec")
            if off is not None:
                return max(0.0, float(off) / 1000.0)
    except Exception:
        pass
    return None


def _fetch_via_ytdlp_livechat(video_url: str, max_seconds: int,
                              progress_callback=None) -> pd.DataFrame:
    """yt-dlp の live_chat 字幕ダウンロード機能による高速取得。

    ダウンロード中はファイル末尾を覗いて現在の動画位置を取得し、
    progress_callback(None, 現在位置秒, 目標秒) で進捗を通知する。
    目標位置を超えたら DownloadCancelled で早期終了し、部分ファイルを解析する。
    """
    tmpdir = tempfile.mkdtemp(prefix="livechat_")
    base = os.path.join(tmpdir, "chat")

    # 1) 動画情報を先に取得して実効的な目標秒数を確定 (進捗%とETAの分母になる)
    probe_opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    duration = float(info.get("duration") or 0)
    target_sec = min(max_seconds, duration) if duration > 0 else max_seconds
    if not any(c.get("ext") == "json" for c in
               (info.get("subtitles") or {}).get("live_chat", [])):
        raise RuntimeError("この動画にはチャットリプレイがありません。")

    # 2) live_chat をダウンロード (進捗フックで位置監視・早期終了)
    state = {"hooks": 0}

    def _hook(d: dict) -> None:
        if d.get("status") != "downloading":
            return
        state["hooks"] += 1
        if state["hooks"] % 3 != 0:
            return
        path = d.get("tmpfilename") or d.get("filename") or ""
        off = _peek_last_offset(path)
        if off is None:
            return
        if progress_callback:
            progress_callback(None, min(off, target_sec), target_sec)
        if off > max_seconds + 60:  # 目標+1分のバッファで打ち切り
            raise DownloadCancelled("指定範囲まで取得完了")

    dl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "outtmpl": {"default": base},
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "noplaylist": True,
        "progress_hooks": [_hook],
    }
    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([video_url])
    except (DownloadCancelled, KeyboardInterrupt):
        pass  # 早期終了は正常系 (部分ファイルを解析する)

    # 3) 出力ファイル (完了形 or .part) を解析
    candidates = list(Path(tmpdir).glob("*.live_chat.json")) + \
        list(Path(tmpdir).glob("*.live_chat.json.part"))
    if not candidates:
        raise RuntimeError("チャットデータのダウンロードに失敗しました。")
    chat_file = max(candidates, key=lambda p: p.stat().st_size)

    rows: list[dict] = []
    with open(chat_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # .part末尾の不完全な行はスキップ
            row = _parse_live_chat_line(obj)
            if row is None:
                continue
            if row["elapsed_sec"] > max_seconds:
                break
            rows.append(row)

    df = _rows_to_df(rows)
    df.attrs["video_duration"] = duration
    df.attrs["video_title"] = info.get("title", "") or ""
    return df


def _fetch_via_pytchat(video_id: str, max_seconds: int,
                       progress_callback=None) -> pd.DataFrame:
    """pytchat によるフォールバック取得 (chat-downloader より低速)。"""
    rows: list[dict] = []
    # interruptable=False: pytchatはデフォルトでCtrl+C用のシグナルハンドラを登録するが、
    # Streamlitはスクリプトをメインスレッド以外で実行するため登録に失敗する。
    # ("シグナルはメインインタプリタのメインスレッドでのみ機能します" エラーの回避)
    chat = pytchat.create(video_id=video_id, interruptable=False)
    base_ts: float | None = None
    while chat.is_alive():
        for c in chat.get().sync_items():
            ts = c.timestamp / 1000.0  # ms -> sec (エポック秒)
            if base_ts is None:
                base_ts = ts
            elapsed = max(0.0, ts - base_ts)
            if elapsed > max_seconds:
                chat.terminate()
                break
            is_sc = getattr(c, "type", "") in ("superChat", "superSticker")
            rows.append(
                {
                    "elapsed_sec": elapsed,
                    "minute": int(elapsed // 60),
                    "author": c.author.name,
                    "message": c.message,
                    "is_superchat": is_sc,
                    "sc_amount": float(getattr(c, "amountValue", 0) or 0) if is_sc else 0.0,
                    "sc_currency": str(getattr(c, "currency", "")) if is_sc else "",
                }
            )
            if progress_callback and len(rows) % 500 == 0:
                progress_callback(len(rows), elapsed, float(max_seconds))
    chat.terminate()
    return _rows_to_df(rows)


# ---- ローカルキャッシュ ----

def _cache_path(video_id: str, max_seconds: int) -> Path:
    return CACHE_DIR / f"{video_id}__{max_seconds}.pkl"


def _cache_lookup(video_id: str, max_seconds: int) -> pd.DataFrame | None:
    """要求範囲をカバーするキャッシュがあれば読み込み、範囲を切り出して返す。"""
    if not CACHE_DIR.exists():
        return None
    best: tuple[int, Path] | None = None
    for f in CACHE_DIR.glob(f"{video_id}__*.pkl"):
        try:
            covered = int(f.stem.split("__")[1])
        except (IndexError, ValueError):
            continue
        if covered >= max_seconds and (best is None or covered < best[0]):
            best = (covered, f)
    if best is None:
        return None
    try:
        df = pd.read_pickle(best[1])
        # 古い形式のキャッシュ(新機能の列が無い)は破棄して取り直させる
        if not set(CHAT_COLUMNS).issubset(df.columns):
            try:
                best[1].unlink()
            except OSError:
                pass
            return None
        sliced = df[df["elapsed_sec"] <= max_seconds].reset_index(drop=True)
        # スライスで失われる動画タイトル/長さのattrsを引き継ぐ
        for k in ("video_title", "video_duration"):
            if k in df.attrs:
                sliced.attrs[k] = df.attrs[k]
        return sliced
    except Exception:
        return None


def _cache_store(video_id: str, max_seconds: int, df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        # 狭い範囲の古いキャッシュは上位互換となるため削除
        for f in CACHE_DIR.glob(f"{video_id}__*.pkl"):
            try:
                if int(f.stem.split("__")[1]) <= max_seconds:
                    f.unlink()
            except (IndexError, ValueError, OSError):
                pass
        df.to_pickle(_cache_path(video_id, max_seconds))
    except Exception:
        pass  # キャッシュ保存失敗は致命的でないため握りつぶす


def clear_cache(video_id: str | None = None) -> int:
    """キャッシュを削除する。video_id指定時はその動画のみ。削除件数を返す。"""
    if not CACHE_DIR.exists():
        return 0
    pattern = f"{video_id}__*.pkl" if video_id else "*.pkl"
    n = 0
    for f in CACHE_DIR.glob(pattern):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


def fetch_chat_log(video_url: str, max_seconds: int = 7200,
                   progress_callback=None, use_cache: bool = True) -> pd.DataFrame:
    """チャットログを取得する (キャッシュ → chat-downloader → pytchat の優先順)。

    progress_callback の引数: (取得済み件数 or None, 現在の動画位置秒, 目標秒数)

    Returns:
        DataFrame [elapsed_sec, minute, author, message, is_superchat, sc_amount, sc_currency]
        attrs["from_cache"] にキャッシュ利用有無、attrs["backend"] に使用バックエンド名を格納。
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("YouTubeのURLから動画IDを抽出できませんでした。")

    if use_cache:
        cached = _cache_lookup(video_id, max_seconds)
        if cached is not None and not cached.empty:
            cached.attrs["from_cache"] = True
            cached.attrs["backend"] = "cache"
            return cached

    df = pd.DataFrame()
    backend = ""
    errors: list[str] = []
    if YTDLP_CHAT_AVAILABLE:
        try:
            df = _fetch_via_ytdlp_livechat(video_url, max_seconds, progress_callback)
            backend = "yt-dlp"
        except Exception as e:
            errors.append(f"yt-dlp: {e}")
    if df.empty and PYTCHAT_AVAILABLE:
        try:
            df = _fetch_via_pytchat(video_id, max_seconds, progress_callback)
            backend = "pytchat"
        except Exception as e:
            errors.append(f"pytchat: {e}")

    if df.empty:
        detail = f" (内部エラー: {' / '.join(errors)})" if errors else ""
        raise RuntimeError(
            "チャットログを取得できませんでした。チャットリプレイが有効なアーカイブか確認してください。" + detail
        )

    _cache_store(video_id, max_seconds, df)
    df.attrs["from_cache"] = False
    df.attrs["backend"] = backend
    return df


def generate_mock_chat_log(duration_minutes: int = 90, seed: int = 42) -> pd.DataFrame:
    """デモ・開発用のリアルなモックチャットログを生成する。

    通常時は静かな流量、ランダムな数地点でシチュエーション付きスパイクを起こす。
    """
    rng = random.Random(seed)
    normal_pool = ["こんちゃ", "初見です", "がんばれ～", "8888", "おつかれ", "それな", "うんうん"]
    rows: list[dict] = []

    # スパイク地点をシチュエーションごとに配置
    spike_plan = rng.sample(range(5, duration_minutes - 5), k=min(6, duration_minutes // 12))
    situations = list(SITUATIONS.keys())
    spike_map = {m: situations[i % len(situations)] for i, m in enumerate(spike_plan)}

    sc_amounts = [200, 500, 1000, 2000, 5000, 10000]
    for minute in range(duration_minutes):
        is_spike = minute in spike_map
        if is_spike:
            situation = spike_map[minute]
            pool = SITUATIONS[situation]["keywords"]
            n_msgs = rng.randint(80, 150)
        else:
            pool = normal_pool
            n_msgs = rng.randint(5, 20)
        for _ in range(n_msgs):
            sec = minute * 60 + rng.uniform(0, 60)
            msg = rng.choice(pool)
            if pool is not normal_pool and rng.random() < 0.4:
                msg = msg * rng.randint(1, 3)  # "草草草" のような連打を再現
            # スパイク中は3%、通常時は0.2%の確率でスーパーチャットを発生させる
            is_sc = rng.random() < (0.03 if is_spike else 0.002)
            # メンバーシップ加入(スパイク中1.5%、通常0.1%)
            is_mem = (not is_sc) and rng.random() < (0.015 if is_spike else 0.001)
            mtype = rng.choice(["新規", "新規", "継続", "ギフト"]) if is_mem else ""
            rows.append(
                {
                    "elapsed_sec": sec,
                    "minute": minute,
                    "author": f"listener_{rng.randint(1, 500)}",
                    "message": ("ナイス切り抜きポイント！" if is_sc else
                                (f"メンバーシップ({mtype})" if is_mem else msg)),
                    "is_superchat": is_sc,
                    "sc_amount": float(rng.choice(sc_amounts)) if is_sc else 0.0,
                    "sc_currency": "JPY" if is_sc else "",
                    "is_member": is_mem,
                    "member_type": mtype,
                    "member_count": (rng.choice([5, 10, 20]) if mtype == "ギフト" else 1) if is_mem else 0,
                }
            )
    df = _rows_to_df(sorted(rows, key=lambda r: r["elapsed_sec"]))
    df.attrs["from_cache"] = False
    df.attrs["backend"] = "mock"
    df.attrs["video_title"] = "🧪 デモ配信"
    return df


# ============================================================
# 2. 集計・スパイク検出
# ============================================================

def aggregate_per_minute(chat_df: pd.DataFrame) -> pd.DataFrame:
    """1分単位のチャット数集計。欠損分は0で補完する。"""
    counts = chat_df.groupby("minute").size()
    full_index = range(0, int(chat_df["minute"].max()) + 1)
    counts = counts.reindex(full_index, fill_value=0)
    return pd.DataFrame({"minute": counts.index, "count": counts.values})


def detect_spikes(agg_df: pd.DataFrame, z_threshold: float = 2.0,
                  max_spikes: int = 12, per_hour: bool = False) -> pd.DataFrame:
    """盛り上がり地点を検出する(隣接分はピークのみ採用)。

    Args:
        z_threshold: 検出感度。平均+zσ を超えた分を候補とする(小さいほど敏感)。
        max_spikes: 返す最大件数。
        per_hour: True なら1時間ブロックごとの相対評価で検出する。
            長時間配信では序盤に視聴者が集中して全体平均が歪み、
            中盤以降の山を取りこぼすため、マラソン配信はこちらが有効。
    """
    if agg_df.empty:
        return pd.DataFrame(columns=["minute", "count"])

    candidates: list[dict] = []
    if per_hour:
        for block_start in range(0, int(agg_df["minute"].max()) + 1, 60):
            block = agg_df[(agg_df["minute"] >= block_start) & (agg_df["minute"] < block_start + 60)]
            if len(block) < 5 or block["count"].sum() == 0:
                continue
            mean, std = block["count"].mean(), block["count"].std()
            threshold = mean + z_threshold * (std if std > 0 else 1.0)
            for _, row in block[block["count"] >= threshold].iterrows():
                candidates.append(
                    {
                        "minute": int(row["minute"]),
                        "count": int(row["count"]),
                        # ブロック平均に対する倍率でランク付け (時間帯間で公平に比較)
                        "score": row["count"] / max(mean, 1.0),
                    }
                )
    else:
        mean, std = agg_df["count"].mean(), agg_df["count"].std()
        threshold = mean + z_threshold * (std if std > 0 else 1.0)
        for _, row in agg_df[agg_df["count"] >= threshold].iterrows():
            candidates.append(
                {"minute": int(row["minute"]), "count": int(row["count"]),
                 "score": float(row["count"])}
            )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected: list[dict] = []
    for c in candidates:
        if any(abs(c["minute"] - s["minute"]) <= 1 for s in selected):
            continue  # 隣接スパイクは最大値のみ残す
        selected.append({"minute": c["minute"], "count": c["count"]})
        if len(selected) >= max_spikes:
            break
    return pd.DataFrame(selected).sort_values("minute").reset_index(drop=True) if selected \
        else pd.DataFrame(columns=["minute", "count"])


def minute_to_label(minute: int) -> str:
    h, m = divmod(minute, 60)
    return f"{h}:{m:02d}:00"


def build_spike_chart(agg_df: pd.DataFrame, spikes_df: pd.DataFrame):
    """テーマ配色に沿った折れ線グラフ(plotly Figure)を構築する。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg_df["minute"], y=agg_df["count"],
            mode="lines+markers", name="チャット数/分",
            line=dict(color="#A55EEA", width=2.5, shape="spline"),
            # 全点にマーカーを置き、どこをクリックしてもプレビューにジャンプできるようにする
            marker=dict(size=5, color="#A55EEA", opacity=0.55),
            fill="tozeroy", fillcolor="rgba(165, 94, 234, 0.12)",
        )
    )
    if not spikes_df.empty:
        fig.add_trace(
            go.Scatter(
                x=spikes_df["minute"], y=spikes_df["count"],
                mode="markers", name="盛り上がり🔥",
                marker=dict(color="#FF6B81", size=13, symbol="star",
                            line=dict(color="white", width=1.5)),
            )
        )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FAFAFA", plot_bgcolor="#FFFFFF",
        font=dict(family="sans-serif", color="#444"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="経過時間(分)", yaxis_title="チャット数",
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified",
    )
    return fig


# ============================================================
# 3. LLM 連携レイヤー (Gemini API / ローカルOllama)
# ============================================================

def _get_gemini_client() -> _GeminiClient | None:
    """Gemini APIキーがあればクライアントを返す。

    キーの探索順: st.secrets["GEMINI_API_KEY"] -> 環境変数 GEMINI_API_KEY
    """
    if not GEMINI_SDK_AVAILABLE:
        return None
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return _GeminiClient(api_key)
    except Exception:
        return None


def get_llm():
    """利用可能なLLMクライアントを返す。優先順: Gemini(キーあれば) → ローカルOllama → None。

    どれも無ければ None(各機能はモック/フォールバック動作)。
    返り値は client.generate_content(prompt).text の形で使える。
    """
    client = _get_gemini_client()
    if client is not None:
        return client
    model = _pick_ollama_model()
    if model:
        try:
            return _OllamaClient(model)
        except Exception:
            return None
    return None


# 後方互換: 既存コードは get_gemini_model() を呼ぶ(中身は統合LLM取得)
get_gemini_model = get_llm


def llm_backend_label() -> str:
    """現在有効なLLMバックエンドの表示名(サイドバー用)。無ければ空文字。"""
    if _get_gemini_client() is not None:
        return "Gemini AI"
    model = _pick_ollama_model()
    if model:
        return f"ローカルAI ({model})"
    return ""


def _extract_json(text: str) -> dict | None:
    """LLM応答からJSONオブジェクトを頑健に抽出する。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ============================================================
# 3a. シチュエーション分類
# ============================================================

def classify_situation_rule_based(messages: list[str]) -> tuple[str, str]:
    """キーワード出現頻度によるフォールバック分類。(situation, reason) を返す。"""
    scores = {name: 0 for name in SITUATIONS}
    for msg in messages:
        for name, info in SITUATIONS.items():
            if any(kw in msg for kw in info["keywords"]):
                scores[name] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "絶叫・ハプニング"
        return best, "特徴的なキーワードが少ないため、文脈不明の盛り上がりとして分類しました。"
    return best, f"チャット{len(messages)}件中{scores[best]}件に「{best}」系の反応が含まれていました。"


def classify_situation(messages: list[str], model=None) -> tuple[str, str]:
    """スパイク前後のチャット群から感情・文脈を判定する。

    Gemini が利用可能ならAI分類、不可ならルールベースにフォールバック。
    Returns: (situation名, 判定理由)
    """
    if model is None:
        model = get_gemini_model()
    if model is None:
        return classify_situation_rule_based(messages)

    sample = "\n".join(messages[:80])
    categories = "\n".join(
        f"- {name}: {info['description']} (例: {', '.join(info['keywords'][:4])})"
        for name, info in SITUATIONS.items()
    )
    prompt = f"""あなたはVtuber切り抜き動画の編集者です。
以下はライブ配信の盛り上がった瞬間のチャットログです。
視聴者の感情・文脈を読み取り、最も当てはまるシチュエーションを1つ選んでください。

【分類カテゴリ】
{categories}

【チャットログ】
{sample}

【出力形式】次のJSONのみを出力:
{{"situation": "カテゴリ名", "reason": "判定理由を日本語で1文"}}
"""
    try:
        response = model.generate_content(prompt)
        data = _extract_json(response.text)
        if data and data.get("situation") in SITUATIONS:
            return data["situation"], data.get("reason", "AIによる文脈判定")
    except Exception:
        pass
    return classify_situation_rule_based(messages)


# ============================================================
# 3b. 構成案・サムネイル指示書の生成
# ============================================================

def _mock_clip_plan(situation: str, timestamp_label: str) -> dict:
    """Gemini未接続時のデモ用構成案。"""
    info = SITUATIONS[situation]
    plans = {
        "爆笑": {
            "titles": [
                f"【腹筋崩壊】{timestamp_label}からの伝説のシーンがヤバすぎたwww",
                "配信中にまさかの大事故！笑いが止まらない神回まとめ",
                "ガチで咳き込むレベルで笑ったシーン【切り抜き】",
            ],
            "structure": (
                "1. スパイク30秒前から導入(フリ部分を必ず含める)\n"
                "2. 爆笑の瞬間はノーカット・チャット欄を画面内に表示\n"
                "3. 笑い終わりの一言コメントで締め(15秒)\n"
                "4. 効果音: ドンッ→笑い声SE / テロップは大きく揺らす"
            ),
            "thumbnail": (
                "・背景: 黄色の集中線で勢いを演出\n"
                "・キャラの大笑い表情のスクショを右側に大きく配置\n"
                "・キャッチコピー: 「腹筋崩壊www」を極太ゴシック+赤縁取りで左上に\n"
                "・「草」の文字を背景に薄く散りばめる"
            ),
        },
        "感動": {
            "titles": [
                f"【涙腺崩壊】ファンへの想いを語る{timestamp_label}のシーンが尊すぎた",
                "ガチ泣き不可避…リスナー全員が涙したあの瞬間",
                "この話を聞いて泣かない人はいない【感動切り抜き】",
            ],
            "structure": (
                "1. 静かなBGM(ピアノ系)で雰囲気を作る\n"
                "2. 語りのシーンはテロップを丁寧に全文字起こし\n"
                "3. チャットの「泣いた」コメントをハイライト表示\n"
                "4. 余韻を残すため最後に3秒の無音+ロゴ"
            ),
            "thumbnail": (
                "・背景: 淡い紫→白のグラデーションで切ない雰囲気\n"
                "・キャラの優しい表情のスクショを中央に\n"
                "・キャッチコピー: 「全員が泣いた」を明朝体+白文字で下部に\n"
                "・涙の雫エフェクトをワンポイントで追加"
            ),
        },
        "てぇてぇ": {
            "titles": [
                f"【てぇてぇ注意】{timestamp_label}の尊さで天に召されるリスナー続出",
                "ガチ恋勢が死んだ瞬間がこちらです【尊い】",
                "今日も推しが可愛すぎて生きるのが辛い【切り抜き】",
            ],
            "structure": (
                "1. 可愛い瞬間の直前3秒からスタート(テンポ重視)\n"
                "2. 尊さポイントでズームイン+ハートエフェクト\n"
                "3. チャットの「てぇてぇ」「尊い」の流れを画面端に表示\n"
                "4. ループしたくなる10〜30秒のショート向け構成"
            ),
            "thumbnail": (
                "・背景: 桜ピンクのふんわりグラデーション+ハート柄\n"
                "・キャラの最高に可愛い表情を中央ドアップで\n"
                "・キャッチコピー: 「尊すぎ注意」を丸ゴシック+ピンク縁取りで\n"
                "・キラキラエフェクトを全体に散らす"
            ),
        },
        "絶叫・ハプニング": {
            "titles": [
                f"【放送事故】{timestamp_label}に起きたまさかのハプニングに本人も絶叫",
                "「は！？」配信史上最大のアクシデントまとめ",
                "絶叫注意⚠️ 心臓に悪すぎる神リアクション集",
            ],
            "structure": (
                "1. ハプニング5秒前から「この後とんでもないことが…」のテロップ\n"
                "2. 絶叫の瞬間は画面シェイク+ビックリマークエフェクト\n"
                "3. リプレイをスロー+ズームでもう一度見せる\n"
                "4. 本人の冷静になった後のコメントで落として締め"
            ),
            "thumbnail": (
                "・背景: 赤と黒の警告色+ヒビ割れエフェクト\n"
                "・キャラの驚愕表情スクショを左に、「！？」を右に巨大配置\n"
                "・キャッチコピー: 「まさかの放送事故」を黒帯+黄色文字で\n"
                "・⚠️マークをアクセントに追加"
            ),
        },
    }
    p = plans[situation]
    return {
        "titles": p["titles"],
        "structure": p["structure"],
        "thumbnail": p["thumbnail"],
    }


def generate_clip_plan(situation: str, messages: list[str],
                       timestamp_label: str, model=None) -> dict:
    """切り抜き動画の「タイトル案・構成指示・サムネ指示書」を生成する。

    Returns: {"titles": [str, ...], "structure": str, "thumbnail": str}
    """
    if model is None:
        model = get_gemini_model()
    if model is None:
        return _mock_clip_plan(situation, timestamp_label)

    sample = "\n".join(messages[:60])
    prompt = f"""あなたは登録者100万人クラスのVtuber切り抜きチャンネルの敏腕編集者です。
配信の {timestamp_label} 地点で「{situation}」系の盛り上がりが発生しました。

【その時のチャットログ】
{sample}

このシーンを切り抜き動画にするための企画書を作成してください。
クリック率(CTR)と視聴維持率を最大化する、思わず見たくなる案にしてください。

【出力形式】次のJSONのみを出力:
{{
  "titles": ["タイトル案1", "タイトル案2", "タイトル案3"],
  "structure": "動画の構成指示(導入→本編→締めの流れ、テロップ・効果音・BGMの指示を含む)",
  "thumbnail": "サムネイルのデザイン指示書(背景・キャラ配置・キャッチコピー・配色を箇条書き)"
}}
"""
    try:
        response = model.generate_content(prompt)
        data = _extract_json(response.text)
        if data and all(k in data for k in ("titles", "structure", "thumbnail")):
            return data
    except Exception:
        pass
    return _mock_clip_plan(situation, timestamp_label)


# ============================================================
# 4. 分析パイプライン (main.py から呼ぶ統合関数)
# ============================================================

def analyze_stream(chat_df: pd.DataFrame, context_window: int = 1,
                   progress_callback=None, z_threshold: float = 2.0,
                   max_spikes: int = 12, per_hour: bool = False,
                   use_ai: bool = True,
                   ) -> tuple[pd.DataFrame, pd.DataFrame, list[SpikeAnalysis]]:
    """チャットDataFrameを受け取り、集計→スパイク検出→AI分析まで一括実行する。

    use_ai=False のときはLLMを一切呼ばず、ルールベース/モックで即座に返す
    (デモ用・高速処理用)。

    Returns:
        (1分集計DataFrame, スパイクDataFrame, SpikeAnalysisのリスト)
    """
    agg_df = aggregate_per_minute(chat_df)
    spikes_df = detect_spikes(agg_df, z_threshold=z_threshold,
                              max_spikes=max_spikes, per_hour=per_hour)
    model = get_gemini_model() if use_ai else None

    analyses: list[SpikeAnalysis] = []
    for i, (_, spike) in enumerate(spikes_df.iterrows()):
        minute = int(spike["minute"])
        # スパイク前後 context_window 分のチャットを文脈として収集
        window = chat_df[
            (chat_df["minute"] >= minute - context_window)
            & (chat_df["minute"] <= minute + context_window)
        ]
        messages = window["message"].astype(str).tolist()
        label = minute_to_label(minute)

        situation, reason = classify_situation(messages, model=model)
        plan = generate_clip_plan(situation, messages, label, model=model)
        info = SITUATIONS[situation]

        analyses.append(
            SpikeAnalysis(
                minute=minute,
                timestamp_label=label,
                message_count=int(spike["count"]),
                situation=situation,
                situation_emoji=info["emoji"],
                situation_color=info["color"],
                reason=reason,
                sample_messages=messages[:10],
                title_ideas=list(plan["titles"]),
                structure_guide=str(plan["structure"]),
                thumbnail_guide=str(plan["thumbnail"]),
            )
        )
        if progress_callback:
            progress_callback(i + 1, len(spikes_df))

    return agg_df, spikes_df, analyses


# ============================================================
# 5. スーパーチャット分析
# ============================================================

def superchat_summary(chat_df: pd.DataFrame) -> dict | None:
    """スーパーチャットの集計結果を返す。スパチャが無ければ None。

    Returns:
        {
          "total_count": int,
          "by_currency": DataFrame [sc_currency, total, count],
          "per_minute": DataFrame [minute, amount]   # 通貨混在のため主要通貨のみ
          "main_currency": str,
          "top_moments": DataFrame [...],
          "all": DataFrame [時刻, 投稿者, 金額, 通貨, メッセージ]      # 全スパチャ(時系列)
          "by_author": DataFrame [投稿者, 回数, 合計金額, 通貨]       # 投稿者ごと集計
        }
    """
    if "is_superchat" not in chat_df.columns:
        return None
    sc = chat_df[chat_df["is_superchat"]]
    if sc.empty:
        return None

    by_currency = (
        sc.groupby("sc_currency")["sc_amount"]
        .agg(total="sum", count="count")
        .reset_index()
        .sort_values("total", ascending=False)
    )
    main_currency = str(by_currency.iloc[0]["sc_currency"])
    main_sc = sc[sc["sc_currency"] == main_currency]
    per_minute = (
        main_sc.groupby("minute")["sc_amount"].sum().reset_index(name="amount")
    )
    top_moments = sc.sort_values("sc_amount", ascending=False).head(5)[
        ["minute", "elapsed_sec", "author", "message", "sc_amount", "sc_currency"]
    ]

    # 全スパチャ一覧(いつ・誰が・金額・通貨・内容)を時系列で
    all_sc = sc.sort_values("elapsed_sec").copy()
    all_df = pd.DataFrame({
        "時刻": all_sc["elapsed_sec"].apply(
            lambda s: f"{int(s) // 3600}:{(int(s) % 3600) // 60:02d}:{int(s) % 60:02d}"
        ).values,
        "投稿者": all_sc["author"].astype(str).values,
        "金額": all_sc["sc_amount"].values,
        "通貨": all_sc["sc_currency"].astype(str).values,
        "メッセージ": all_sc["message"].astype(str).values,
    })

    # 投稿者ごと: 何回・合計金額(主要通貨ベース、通貨混在は通貨も表示)
    by_author = (
        sc.groupby(["author", "sc_currency"])["sc_amount"]
        .agg(回数="count", 合計金額="sum")
        .reset_index()
        .rename(columns={"author": "投稿者", "sc_currency": "通貨"})
        .sort_values("合計金額", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "total_count": int(len(sc)),
        "by_currency": by_currency,
        "per_minute": per_minute,
        "main_currency": main_currency,
        "top_moments": top_moments,
        "all": all_df,
        "by_author": by_author,
    }


def build_superchat_chart(per_minute: pd.DataFrame, currency: str):
    """1分ごとのスパチャ金額バーチャート (ゴールド配色)。"""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=per_minute["minute"], y=per_minute["amount"],
            name=f"スパチャ金額 ({currency})",
            marker=dict(color="#FFC107", line=dict(color="#FFA000", width=1)),
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FAFAFA", plot_bgcolor="#FFFFFF",
        font=dict(family="sans-serif", color="#444"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="経過時間(分)", yaxis_title=f"金額 ({currency})",
        hovermode="x unified",
    )
    return fig


# ============================================================
# 5b. メンバーシップ分析
# ============================================================

def membership_summary(chat_df: pd.DataFrame) -> dict | None:
    """メンバーシップ加入(新規/継続/ギフト)の集計結果を返す。無ければ None。

    Returns:
        {
          "total_events": int,                # イベント数(行数)
          "new_count": int, "milestone_count": int,
          "gift_count": int, "gifted_total": int,   # ギフト贈与イベント数 / 贈られた総数
          "all": DataFrame [時刻, 投稿者, 種別, 件数, 内容],
          "by_author": DataFrame [投稿者, 種別, 回数, 件数合計],
          "per_minute": DataFrame [minute, count],
        }
    """
    if "is_member" not in chat_df.columns:
        return None
    mem = chat_df[chat_df["is_member"]]
    if mem.empty:
        return None

    new_count = int((mem["member_type"] == "新規").sum())
    milestone_count = int((mem["member_type"] == "継続").sum())
    gift_rows = mem[mem["member_type"] == "ギフト"]
    gift_count = int(len(gift_rows))
    gifted_total = int(gift_rows["member_count"].sum())

    mem_sorted = mem.sort_values("elapsed_sec")
    all_df = pd.DataFrame({
        "時刻": mem_sorted["elapsed_sec"].apply(
            lambda s: f"{int(s) // 3600}:{(int(s) % 3600) // 60:02d}:{int(s) % 60:02d}"
        ).values,
        "投稿者": mem_sorted["author"].astype(str).values,
        "種別": mem_sorted["member_type"].astype(str).values,
        "件数": mem_sorted["member_count"].astype(int).values,
        "内容": mem_sorted["message"].astype(str).values,
    })

    by_author = (
        mem.groupby(["author", "member_type"])["member_count"]
        .agg(回数="count", 件数合計="sum")
        .reset_index()
        .rename(columns={"author": "投稿者", "member_type": "種別"})
        .sort_values(["回数", "件数合計"], ascending=False)
        .reset_index(drop=True)
    )

    per_minute = mem.groupby("minute").size().reset_index(name="count")

    return {
        "total_events": int(len(mem)),
        "new_count": new_count,
        "milestone_count": milestone_count,
        "gift_count": gift_count,
        "gifted_total": gifted_total,
        "all": all_df,
        "by_author": by_author,
        "per_minute": per_minute,
    }


def build_membership_chart(per_minute: pd.DataFrame):
    """1分ごとのメンバーシップ加入数バーチャート (グリーン配色)。"""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=per_minute["minute"], y=per_minute["count"],
            name="メンバー加入数",
            marker=dict(color="#2ED573", line=dict(color="#26AE60", width=1)),
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FAFAFA", plot_bgcolor="#FFFFFF",
        font=dict(family="sans-serif", color="#444"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="経過時間(分)", yaxis_title="加入数",
        hovermode="x unified",
    )
    return fig


# ============================================================
# 6. YouTubeチャプター名の生成
# ============================================================

CHAPTER_FALLBACK = {
    "爆笑": "爆笑シーン",
    "感動": "感動の場面",
    "てぇてぇ": "尊いシーン",
    "絶叫・ハプニング": "ハプニング発生",
}


def generate_chapter_titles(analyses: list, model=None) -> list[str]:
    """各スパイクのYouTubeチャプター名(簡潔・20字以内)を返す。

    Gemini接続時は目次専用の見出しを一括生成。未接続時は状況ラベルにフォールバック。
    切り抜き用のクリックベイトなタイトルとは別物として生成する。
    """
    fallback = [CHAPTER_FALLBACK.get(getattr(a, "situation", ""), "ハイライト") for a in analyses]
    if not analyses:
        return fallback
    if model is None:
        model = get_gemini_model()
    if model is None:
        return fallback

    blocks = []
    for i, a in enumerate(analyses):
        msgs = " / ".join(a.sample_messages[:10])
        blocks.append(f"{i}. 状況:{a.situation} / チャット例:{msgs}")
    prompt = f"""あなたはVtuber配信アーカイブに目次(チャプター)を付ける編集者です。
以下の各盛り上がり地点に、YouTubeチャプター名を付けてください。

【条件】
- 20文字以内。内容が一目で分かる簡潔で魅力的な見出し
- 絵文字・記号・時刻は含めない
- 「雑談」「神回避シーン」「感動の告知」のような自然な見出しにする

【盛り上がり地点】
{chr(10).join(blocks)}

【出力】次のJSONのみ:
{{"titles": {{"0": "見出し", "1": "見出し"}}}}
"""
    try:
        text = model.generate_content(prompt).text
        # まずJSONとして解釈
        data = _extract_json(text)
        if data and isinstance(data.get("titles"), dict):
            out = []
            for i in range(len(analyses)):
                name = str(data["titles"].get(str(i), "")).strip()
                out.append(name or fallback[i])
            return out
        # ローカルモデル等でJSONにならない場合は "0: 見出し" 形式を行解析
        parsed: dict[int, str] = {}
        for line in text.splitlines():
            m = re.match(r"\s*\"?(\d+)\"?\s*[:.、]\s*(.+)", line)
            if m:
                parsed[int(m.group(1))] = m.group(2).strip().strip('",。').strip()
        if parsed:
            return [parsed.get(i) or fallback[i] for i in range(len(analyses))]
    except Exception:
        pass
    return fallback


# ============================================================
# 6b. 配信全体の目次(チャプター)生成 — 切り抜きポイントとは別物
# ============================================================

def _segment_boundaries(agg_df: pd.DataFrame, target_count: int) -> list[int]:
    """配信を target_count 個の区間に分ける境界(分)を返す。先頭は必ず0。

    均等割りした各境界を、近傍のチャット最小(=話題の切れ目/休憩)にスナップして
    自然な区切りにする。
    """
    if agg_df.empty:
        return [0]
    max_min = int(agg_df["minute"].max())
    if max_min <= 1 or target_count <= 1:
        return [0]
    counts = dict(zip(agg_df["minute"].astype(int), agg_df["count"]))
    step = max_min / target_count
    boundaries = [0]
    for i in range(1, target_count):
        center = i * step
        win = max(1, int(step * 0.35))
        lo = max(boundaries[-1] + max(2, int(step * 0.4)), int(center - win))
        hi = min(max_min - 1, int(center + win))
        cand = int(center) if lo >= hi else min(range(lo, hi + 1), key=lambda m: counts.get(m, 0))
        if cand > boundaries[-1] + 1:
            boundaries.append(cand)
    return boundaries


def _segment_sample_messages(chat_df: pd.DataFrame, start_min: int, end_min: int,
                             limit: int = 40) -> list[str]:
    """区間内のチャットを、話題が分かりやすいよう抜粋する。

    一定間隔のサンプル + 長め(=内容を含みやすい)のメッセージを混ぜる。
    """
    seg = chat_df[(chat_df["minute"] >= start_min) & (chat_df["minute"] < end_min)]
    msgs = seg["message"].astype(str).tolist()
    if not msgs:
        return []
    spread = msgs[:: max(1, len(msgs) // 25)][:25]
    longest = sorted(set(msgs), key=len, reverse=True)[:15]
    out: list[str] = []
    for m in spread + longest:
        m = m.strip()
        if m and m not in out:
            out.append(m)
        if len(out) >= limit:
            break
    return out


def _clean_chat_text(m: str) -> str:
    """チャットからノイズ(絵文字スタンプ・連打)を除去して内容を読みやすくする。"""
    m = re.sub(r":[A-Za-z0-9_\-]+:", "", m)        # :emote: 形式のスタンプ除去
    m = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]", "", m)
    m = re.sub(r"(.)\1{3,}", r"\1\1", m)            # 同一文字の連打を圧縮(かわいいい→かわいい)
    return m.strip()


def _digest_window_messages(chat_df: pd.DataFrame, start_min: int, end_min: int,
                            limit: int = 8) -> list[str]:
    """時系列ダイジェスト用に、その区間の「内容を含むコメント」を抜粋する。"""
    seg = chat_df[(chat_df["minute"] >= start_min) & (chat_df["minute"] < end_min)]
    cleaned = []
    for raw in seg["message"].astype(str):
        c = _clean_chat_text(raw)
        if len(c) >= 5:  # 短い相槌・スタンプのみは除外
            cleaned.append(c[:50])
    # 長め(=情報量が多い)を優先しつつ重複排除
    out: list[str] = []
    for m in sorted(dict.fromkeys(cleaned), key=len, reverse=True):
        out.append(m)
        if len(out) >= limit:
            break
    return out


def _parse_time_to_sec(value) -> int | None:
    """"H:MM:SS" / "MM:SS" / 秒数 を秒(int)に変換する。"""
    s = str(value).strip()
    if not s:
        return None
    if ":" in s:
        try:
            nums = [int(p) for p in s.split(":")]
        except ValueError:
            return None
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _llm_toc_core(digest_lines: list[str], duration_min: float, model,
                  source_hint: str) -> list[tuple[int, str]]:
    """時系列ダイジェスト(時刻|内容 の行)からAIに目次を区切らせる共通処理。

    source_hint: "チャット" / "音声の書き起こし" など、入力の種類の説明。
    Returns: [(秒, 見出し)] または [](失敗・品質不足)。
    """
    if not digest_lines:
        return []
    lo, hi = max(4, int(duration_min / 18)), max(6, int(duration_min / 8))
    last_label = digest_lines[-1].split(" | ")[0] if digest_lines else "終わり"
    prompt = f"""以下はVtuber配信の{source_hint}を時間帯順に並べたものです(各行: 時刻 | 内容)。
配信内容の流れを読み取り、YouTubeのチャプター(目次)を作成してください。

【厳守事項】
- 必ず {lo} 個以上 {hi} 個以内のチャプターを作成する(1〜2個だけにしない)
- 配信の最初(0:00:00)から最後({last_label}付近)まで、全体をまんべんなくカバーする
- 話題が変わる所で区切る(雑談→ゲーム→告知 等)。等間隔でなく内容に従う
- 最初のチャプターは必ず time="0:00:00"
- time は実際の配信時刻(その内容が話されている時刻)にする
- title は20文字以内、絵文字や記号や時刻は含めない、内容が一目で分かる見出し

【時系列の内容】
{chr(10).join(digest_lines)}

【出力】次のJSONのみ(チャプターは{lo}個以上):
{{"chapters": [
  {{"time": "0:00:00", "title": "オープニング"}},
  {{"time": "0:08:30", "title": "今日の本題トーク"}},
  {{"time": "0:25:00", "title": "ゲーム開始"}},
  {{"time": "0:52:00", "title": "視聴者と雑談"}},
  {{"time": "1:10:00", "title": "告知・エンディング"}}
]}}
"""
    try:
        data = _extract_json(model.generate_content(prompt).text)
        chapters = data.get("chapters") if isinstance(data, dict) else None
        if not isinstance(chapters, list):
            return []
        total_sec = int(duration_min * 60)
        out: list[tuple[int, str]] = []
        for ch in chapters:
            if not isinstance(ch, dict):
                continue
            sec = _parse_time_to_sec(ch.get("time"))
            title = str(ch.get("title", "")).strip()
            if sec is None or sec < 0 or sec > total_sec + 120 or not title:
                continue
            out.append((sec, title))
        out = sorted(dict(out).items())  # 重複秒を排除して昇順
        if len(out) < max(3, lo - 1):
            return []
        gap_limit = max(25 * 60, total_sec * 0.4)
        bounds_sec = [s for s, _ in out] + [total_sec]
        if any(bounds_sec[i + 1] - bounds_sec[i] > gap_limit for i in range(len(bounds_sec) - 1)):
            return []
        return out
    except Exception:
        return []


def _build_chat_digest(chat_df: pd.DataFrame, max_min: int, max_lines: int = 30) -> list[str]:
    """チャットを時系列ダイジェスト化。長時間でも行数を上限以内に抑える。"""
    win = max(3, -(-max_min // max_lines))  # 行数が max_lines を超えないよう窓幅を調整
    digest = []
    for start in range(0, max_min + 1, win):
        msgs = _digest_window_messages(chat_df, start, start + win)
        if msgs:
            digest.append(f"{minute_to_label(start)} | {' / '.join(msgs)}")
    return digest


def _llm_toc_from_digest(chat_df: pd.DataFrame, max_min: int, duration_min: float,
                         model) -> list[tuple[int, str]]:
    """チャットの時系列ダイジェストから目次を生成する。"""
    digest = _build_chat_digest(chat_df, max_min)
    return _llm_toc_core(digest, duration_min, model, "チャット")


def _llm_toc_per_sample(samples: list[tuple[int, str]], model) -> list[str]:
    """各サンプル地点に見出しを1つずつ付ける(自由区切りより確実に全体をカバー)。"""
    fallback = ["配信開始"] + ["トーク"] * (len(samples) - 1)
    if model is None or not samples:
        return fallback
    blocks = []
    for i, (sec, text) in enumerate(samples):
        snippet = text[:90] if text.strip() else "(発話なし)"
        blocks.append(f"{i}. {minute_to_label(sec // 60)} 発言: {snippet}")
    prompt = f"""Vtuber配信の音声を一定間隔で書き起こしました(各行: 番号 時刻 発言)。
各時間帯の発言内容から、その時間帯に何をしていたかを表す見出しを付けてください。

【厳守】
- 全ての番号(0〜{len(samples) - 1})に必ず見出しを付ける
- 各見出しは15文字以内、絵文字や記号や時刻は含めない
- 0番は「配信開始」「オープニング」など冒頭らしい見出し
- 同じ話題が続く区間は同じ見出しにしてよい(後でまとめます)
- 書き起こしが不明瞭でも前後から推測する

【各時間帯の発言】
{chr(10).join(blocks)}

【出力】次のJSONのみ:
{{"titles": {{"0": "配信開始", "1": "見出し"}}}}
"""
    try:
        data = _extract_json(model.generate_content(prompt).text)
        if data and isinstance(data.get("titles"), dict):
            parsed = {}
            for k, v in data["titles"].items():
                try:
                    parsed[int(k)] = str(v).strip()
                except (ValueError, TypeError):
                    continue
            if parsed:
                return [parsed.get(i) or fallback[i] for i in range(len(samples))]
    except Exception:
        pass
    return fallback


def generate_toc_from_samples(samples: list[tuple[int, str]], duration_min: float,
                              model=None) -> list[tuple[int, str]]:
    """音声サンプル文字起こし [(秒, テキスト)] から目次を生成する(高精度)。

    各サンプルに見出しを付け、連続する同じ見出しをまとめてチャプターにする。
    自由区切り方式が時々1章に collapse するのを防ぎ、全体を確実にカバーする。
    """
    if model is None:
        model = get_llm()
    if model is None or not samples:
        return [(s, t) for s, t in [(0, "配信開始")]]
    titles = _llm_toc_per_sample(samples, model)
    merged: list[tuple[int, str]] = []
    for (sec, _), title in zip(samples, titles):
        if merged and merged[-1][1] == title:
            continue  # 直前と同じ見出しはまとめる
        merged.append((sec, title))
    return merged


def generate_toc_chapters(chat_df: pd.DataFrame, video_duration: float | None = None,
                          model=None) -> list[tuple[int, str]]:
    """配信全体を時間帯で区切った目次(チャプター)を生成する。

    切り抜きポイント(盛り上がり)ではなく、配信全体をカバーする目次。
    第一候補: 時系列ダイジェストをAIに見せ、話題の変わり目で区切らせる(時間と内容が一致)。
    フォールバック: 均等区間に分けて各区間の見出しを付ける。

    Returns: [(開始秒, チャプター名), ...] (先頭は0秒付近)
    """
    agg = aggregate_per_minute(chat_df)
    if agg.empty:
        return [(0, "配信開始")]
    max_min = int(agg["minute"].max())
    duration_min = (video_duration / 60) if video_duration else max_min
    if model is None:
        model = get_llm()

    # 第一候補: AIに時系列から区切らせる
    if model is not None:
        toc = _llm_toc_from_digest(chat_df, max_min, duration_min, model)
        if toc:
            return toc

    # フォールバック: 均等区間 + 見出し
    target = int(min(12, max(4, round(duration_min / 12)))) if duration_min > 0 else 4
    bounds = _segment_boundaries(agg, target)
    segments: list[tuple[int, list[str]]] = []
    for i, start_min in enumerate(bounds):
        end_min = bounds[i + 1] if i + 1 < len(bounds) else max_min + 1
        segments.append((start_min * 60, _segment_sample_messages(chat_df, start_min, end_min)))
    titles = _llm_segment_titles(segments, model)
    return [(sec, title) for (sec, _), title in zip(segments, titles)]


def _llm_segment_titles(segments: list[tuple[int, list[str]]], model) -> list[str]:
    """各区間のチャットから目次見出しをまとめて生成する。"""
    fallback = ["配信開始"] + [f"パート{i}" for i in range(1, len(segments))]
    if model is None or not segments:
        return fallback

    blocks = []
    for i, (sec, msgs) in enumerate(segments):
        sample = " / ".join(msgs[:40]) if msgs else "(コメントなし)"
        blocks.append(f"{i}. 開始{minute_to_label(sec // 60)} / チャット: {sample}")
    prompt = f"""あなたはVtuber配信アーカイブに目次(チャプター)を付ける編集者です。
配信を時間帯で区切りました。各区間のチャットの様子から、その時間帯に配信者が
何をしていたか(雑談・ゲーム・歌・告知など)を推測し、目次の見出しを付けてください。

【条件】
- 各区間20文字以内。内容が一目で分かる簡潔な見出し
- 0番は「配信開始」「オープニング」など冒頭らしい見出しにする
- 絵文字・記号・時刻は含めない
- 例: 雑談、ゲーム開始、ボス戦、休憩・マシュマロ読み、エンディング・告知

【各区間のチャット】
{chr(10).join(blocks)}

【出力】次のJSONのみ:
{{"titles": {{"0": "見出し", "1": "見出し"}}}}
"""
    try:
        text = model.generate_content(prompt).text
        data = _extract_json(text)
        parsed: dict[int, str] = {}
        if data and isinstance(data.get("titles"), dict):
            for k, v in data["titles"].items():
                try:
                    parsed[int(k)] = str(v).strip()
                except (ValueError, TypeError):
                    continue
        if not parsed:  # JSONにならない場合は "0: 見出し" 形式を行解析
            for line in text.splitlines():
                m = re.match(r"\s*\"?(\d+)\"?\s*[:.、]\s*(.+)", line)
                if m:
                    parsed[int(m.group(1))] = m.group(2).strip().strip('",。').strip()
        if parsed:
            return [parsed.get(i) or fallback[i] for i in range(len(segments))]
    except Exception:
        pass
    return fallback


def keyword_search(chat_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """任意キーワードでチャットを横断検索し、出現分布を返す。"""
    if not keyword.strip():
        return pd.DataFrame(columns=["minute", "count"])
    hits = chat_df[chat_df["message"].astype(str).str.contains(re.escape(keyword), case=False)]
    if hits.empty:
        return pd.DataFrame(columns=["minute", "count"])
    counts = hits.groupby("minute").size()
    return pd.DataFrame({"minute": counts.index, "count": counts.values})
