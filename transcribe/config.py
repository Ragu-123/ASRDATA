import os
import sys
import uuid
from pathlib import Path

# Runtime environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else "./data").resolve()
# Use /tmp on Kaggle when available (over 30GB+ available space, preserving /kaggle/working 20GB limit)
TMP_ROOT = Path("/tmp/asr_cache") if (IS_KAGGLE and Path("/tmp").exists()) else WORKING_DIR
STAGING_DIR = Path(os.getenv("STAGING_DIR", str(TMP_ROOT / "transcribe_staging"))).resolve()
STAGING_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR = Path(os.getenv("SHARDS_DIR", str(TMP_ROOT / "transcribe_shards"))).resolve()
SHARDS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from secret_resolver import (
    resolve_secret,
    add_secret_cli_args,
    apply_cli_secrets,
    set_secret,
    HARDCODED_SECRETS
)

# --- Storage & Target Settings ---
TRACKING_BUCKET = os.getenv("TRACKING_BUCKET", "King758/asr-tracking-01")
SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", "King758/media-archive-01")
TARGET_BUCKET = TRACKING_BUCKET  # Alias for backward compatibility

# Segmented dataset (input for transcribe) and audited corpus (output of transcribe)
SEGMENTED_DATASET_REPO = os.getenv("SEGMENTED_DATASET_REPO", "King758/tamil-asr-segmented")
DATASET_REPO = os.getenv("DATASET_REPO", "King758/tamil-asr-corpus")
AUDITED_DATASET_REPO = DATASET_REPO

# Quota & Shard Controls (Target ~1 GB Parquet Shards)
BUFFER_GB = float(os.getenv("BUFFER_GB", os.getenv("SHARD_GB", "1.0")))
BUFFER_BYTES = int(BUFFER_GB * (1024**3))
SHARD_CLIP_THRESHOLD = int(os.getenv("SHARD_CLIP_THRESHOLD", "5000"))
MAX_LOCAL_DISK_GB = float(os.getenv("MAX_LOCAL_DISK_GB", "5.0"))

# Telegram Bot Configuration
def sync_runtime_secrets():
    global TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID, TELEGRAM_ADMIN_PASSCODE, PROXY_TOKEN
    TELEGRAM_BOT_TOKEN = resolve_secret("TELEGRAM_BOT_TOKEN", "8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ")
    TELEGRAM_ADMIN_ID = resolve_secret("TELEGRAM_ADMIN_ID", "5959376217")
    TELEGRAM_ADMIN_PASSCODE = resolve_secret("TELEGRAM_ADMIN_PASSCODE", "")
    PROXY_TOKEN = resolve_secret("PROXY_TOKEN", "")
    resolve_secret("HF_TOKEN", "")
    return TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID

TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID = sync_runtime_secrets()
TELEGRAM_ADMIN_PASSCODE = resolve_secret("TELEGRAM_ADMIN_PASSCODE", "")

def get_ytdlp_extractor_args() -> dict:
    """Return optimal yt-dlp extractor args based on whether cookies are active.
    - With authenticated cookies: prioritize clients that support Netscape cookies
      ('tv', 'mweb', 'web_safari', 'default') so YouTube receives the authenticated session.
    - Without cookies: use mobile/tv clients ('android', 'ios', 'tv') and skip desktop web
      to bypass datacenter IP bot detection challenges.
    """
    if os.getenv("COOKIE_FILE"):
        return {
            "youtube": {
                "player_client": ["tv", "mweb", "web_safari", "default", "android"]
            }
        }
    return {
        "youtube": {
            "player_client": ["android", "ios", "tv"],
            "player_skip": ["web", "web_embedded", "android_vr"]
        }
    }

# yt-dlp Bot Detection Bypass Extractor Args
YTDLP_EXTRACTOR_ARGS = get_ytdlp_extractor_args()
YTDLP_EXTRACTOR_ARGS_CLI = (
    '--extractor-args "youtube:player_client=tv,mweb,web_safari,default,android"'
    if os.getenv("COOKIE_FILE") else
    '--extractor-args "youtube:player_client=android,ios,tv;player_skip=web,web_embedded,android_vr"'
)

# Default Seed Channels for Upfront Scraping
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

# Gemini Canvas Proxy Configuration
DEFAULT_PROXY_URL = os.getenv("GEMINI_PROXY_URL", "")
PROXY_TOKEN = resolve_secret("PROXY_TOKEN", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# Parallel Request Concurrency (Sweet spot: 3 to 5 parallel requests)
CONCURRENCY = int(os.getenv("CONCURRENCY", "4"))
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "2048"))  # 2k reasoning tokens (fast & avoids Cloudflare 100s timeout)

# Distributed Lease Controls
BATCH_LEASE_SIZE = int(os.getenv("BATCH_LEASE_SIZE", "3"))  # 3 videos per lease
LEASE_TTL_MINUTES = int(os.getenv("LEASE_TTL_MINUTES", "60"))
WORKER_ID = os.getenv("TRANSCRIBER_ID", f"transcriber-{uuid.uuid4().hex[:6]}")

# Audio VAD & Chunking Thresholds
VAD_MIN_CHUNK_SEC = float(os.getenv("VAD_MIN_CHUNK_SEC", "4.0"))
VAD_MAX_CHUNK_SEC = float(os.getenv("VAD_MAX_CHUNK_SEC", "15.0"))
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "400"))


def get_hf_token() -> str:
    """Retrieve Hugging Face token from environment variables, Kaggle Secrets, or local .env."""
    return resolve_secret("HF_TOKEN", "")
