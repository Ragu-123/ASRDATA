import os
import sys
import uuid
from pathlib import Path

# --- Two-Tier Storage & Target Settings ---
# Tier 1: Lightweight tracking bucket (< 5 MB total). Zero audio files.
TRACKING_BUCKET = os.getenv("TRACKING_BUCKET", "King758/asr-tracking-01")
SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", "King758/media-archive-01")
TARGET_BUCKET = TRACKING_BUCKET  # Alias for backward compatibility

# Tier 2: Hugging Face Dataset storing Apache Parquet shards with PyArrow audio struct
DATASET_REPO = os.getenv("DATASET_REPO", "King758/tamil-asr-corpus")

# Quota & Shard Controls
SHARD_CLIP_THRESHOLD = int(os.getenv("SHARD_CLIP_THRESHOLD", "400"))  # ~300-500 clips (~70 MB)
MAX_LOCAL_DISK_GB = float(os.getenv("MAX_LOCAL_DISK_GB", "1.0"))       # Strictly under 1.0 GB local disk

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ")
TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID", os.getenv("ADMIN_USER_ID", os.getenv("TELEGRAM_CHAT_ID", "5959376217")))
TELEGRAM_ADMIN_PASSCODE = os.getenv("TELEGRAM_ADMIN_PASSCODE", os.getenv("ADMIN_PASSCODE", ""))

# Runtime environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else "./data").resolve()
STAGING_DIR = WORKING_DIR / "transcribe_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR = WORKING_DIR / "shards"
SHARDS_DIR.mkdir(parents=True, exist_ok=True)

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
PROXY_TOKEN = os.getenv("PROXY_TOKEN", "51dfb957-7347-449b-b153-df797a72c9d5")
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
    """Retrieve Hugging Face token from environment variables or Kaggle Secrets."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    if IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            token = user_secrets.get_secret("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
                return token
        except Exception as e:
            print(f"[CONFIG] Kaggle secrets retrieval notice: {e}", file=sys.stderr)
    return ""
