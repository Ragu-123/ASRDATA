import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Optional

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from secret_resolver import resolve_secret

# Robust secret retrieval: check env, then Kaggle secrets with exact pattern, then .env
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("HF_TOKEN")
    secret_value_1 = user_secrets.get_secret("TELEGRAM_ADMIN_ID")
    secret_value_2 = user_secrets.get_secret("TELEGRAM_BOT_TOKEN")
    if secret_value_0 and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = str(secret_value_0).strip()
    if secret_value_1 and not os.getenv("TELEGRAM_ADMIN_ID"):
        os.environ["TELEGRAM_ADMIN_ID"] = str(secret_value_1).strip()
    if secret_value_2 and not os.getenv("TELEGRAM_BOT_TOKEN"):
        os.environ["TELEGRAM_BOT_TOKEN"] = str(secret_value_2).strip()
except Exception:
    pass

# Runtime environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else "./data").resolve()
STAGING_DIR = WORKING_DIR / "segment_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR = WORKING_DIR / "shards"
SHARDS_DIR.mkdir(parents=True, exist_ok=True)

# Storage & Target Repositories
# Tier 1: Tracking Bucket strictly for JSON metadata (catalogs, leases, manifest < 5 MB, ZERO audio)
TRACKING_BUCKET = os.getenv("TRACKING_BUCKET", "King758/asr-tracking-01")

# Dedicated Separate Raw Audio Storage (from Stage 1)
RAW_DATASET_REPO = os.getenv("RAW_DATASET_REPO", "King758/tamil-raw-audio")
RAW_BUCKET = os.getenv("RAW_BUCKET", "King758/media-archive-01")
SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", RAW_BUCKET)

# Target Segmented Dataset (stores Parquet shards with 10s audio clips & metadata)
SEGMENTED_DATASET_REPO = os.getenv("SEGMENTED_DATASET_REPO", "King758/tamil-asr-segmented")

# Shard & Disk Controls
# ~300-500 clips (~70 MB per Parquet shard)
SHARD_CLIP_THRESHOLD = int(os.getenv("SHARD_CLIP_THRESHOLD", "400"))
# Strictly maintain local disk usage under 1.0 GB (within 19GB total quota)
MAX_LOCAL_DISK_GB = float(os.getenv("MAX_LOCAL_DISK_GB", "1.0"))
DISK_LIMIT_GB = 19.0

# Audio Slicing & Silence Detection Thresholds
TARGET_CHUNK_SEC = 10.0
VAD_MIN_CHUNK_SEC = float(os.getenv("VAD_MIN_CHUNK_SEC", "4.0"))
VAD_MAX_CHUNK_SEC = float(os.getenv("VAD_MAX_CHUNK_SEC", "15.0"))
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "350"))

# Distributed Lease Controls
BATCH_LEASE_SIZE = int(os.getenv("BATCH_LEASE_SIZE", "5"))
LEASE_TTL_MINUTES = int(os.getenv("LEASE_TTL_MINUTES", "45"))
WORKER_ID = os.getenv("SEGMENTER_ID", f"segmenter-{uuid.uuid4().hex[:6]}")

# Telegram Bot Settings
TELEGRAM_BOT_TOKEN = resolve_secret("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_ID = resolve_secret("TELEGRAM_ADMIN_ID", resolve_secret("ADMIN_USER_ID", resolve_secret("TELEGRAM_CHAT_ID", "5959376217")))
TELEGRAM_ADMIN_PASSCODE = resolve_secret("TELEGRAM_ADMIN_PASSCODE", resolve_secret("ADMIN_PASSCODE", ""))

# Cookie File Detection (Kaggle secrets, inputs, .env, or local cookies.txt)
COOKIE_FILE = None

def set_cookie_file(path_str: str) -> Optional[str]:
    global COOKIE_FILE
    if not path_str:
        return COOKIE_FILE
    p = Path(str(path_str).strip().strip("'\""))
    if p.exists() and p.is_file() and p.stat().st_size > 10:
        resolved_str = str(p.resolve())
        if "/kaggle/input" in resolved_str or not os.access(resolved_str, os.W_OK):
            target = Path("/kaggle/working/cookies.txt" if IS_KAGGLE else WORKING_DIR / "cookies.txt")
            try:
                if target.resolve() != p.resolve():
                    shutil.copy2(resolved_str, str(target))
                    p = target
            except Exception:
                pass
        COOKIE_FILE = str(p.resolve())
        os.environ["COOKIE_FILE"] = COOKIE_FILE
        return COOKIE_FILE
    return None

_cookie_candidates = [
    os.getenv("COOKIE_FILE", ""),
    os.getenv("COOKIES_FILE", ""),
    "/kaggle/input/datasets/dhanalakshmikavitha/cookies/cookies (1).txt",
    "/kaggle/input/datasets/dhanalakshmikavitha/cookies/cookies.txt",
    str(WORKING_DIR / "cookies.txt"),
    str(ROOT_DIR / "cookies.txt")
]
for _cp in _cookie_candidates:
    if _cp and set_cookie_file(_cp):
        break

if not COOKIE_FILE and os.path.exists("/kaggle/input"):
    try:
        for _f in Path("/kaggle/input").rglob("*cookie*.txt"):
            if _f.is_file() and _f.stat().st_size > 10:
                set_cookie_file(str(_f))
                break
    except Exception:
        pass

if not COOKIE_FILE:
    _env_cookies = os.getenv("YOUTUBE_COOKIES", os.getenv("COOKIES", ""))
    if _env_cookies.strip():
        try:
            _target = WORKING_DIR / "cookies.txt"
            _target.write_text(_env_cookies.strip(), encoding="utf-8")
            set_cookie_file(str(_target))
        except Exception:
            pass

# yt-dlp Bot Detection Bypass Extractor Args
YTDLP_EXTRACTOR_ARGS = {
    "youtube": {
        "player_client": ["android", "ios", "android_vr"],
        "player_skip": ["web", "tv", "web_embedded"]
    }
}
YTDLP_EXTRACTOR_ARGS_CLI = '--extractor-args "youtube:player_client=android,ios,android_vr;player_skip=web,tv,web_embedded"'

# Default Seed Channels if catalog is empty
DEFAULT_CHANNELS = [
    {
        "id": "madangowri",
        "name": "Madan Gowri",
        "url": "https://www.youtube.com/@madangowri/videos",
        "active": True
    },
    {
        "id": "TheBookShowbyrjananthi",
        "name": "The Book Show by Ananthi",
        "url": "https://www.youtube.com/@TheBookShowbyrjananthi/videos",
        "active": True
    },
    {
        "id": "savukkumedianetwork",
        "name": "Savukku Media Network",
        "url": "https://www.youtube.com/@savukkumedianetwork/videos",
        "active": True
    }
]

def get_hf_token() -> str:
    """Retrieve Hugging Face token safely from environment variables, Kaggle Secrets, or local .env."""
    return resolve_secret("HF_TOKEN", "")
