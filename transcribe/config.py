import os
import sys
import uuid
from pathlib import Path

# Runtime environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else "./data").resolve()
STAGING_DIR = WORKING_DIR / "transcribe_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR = WORKING_DIR / "shards"
SHARDS_DIR.mkdir(parents=True, exist_ok=True)

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

# --- Storage & Target Settings ---
TRACKING_BUCKET = os.getenv("TRACKING_BUCKET", "King758/asr-tracking-01")
SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", "King758/media-archive-01")
TARGET_BUCKET = TRACKING_BUCKET  # Alias for backward compatibility

# Segmented dataset (input for transcribe) and audited corpus (output of transcribe)
SEGMENTED_DATASET_REPO = os.getenv("SEGMENTED_DATASET_REPO", "King758/tamil-asr-segmented")
DATASET_REPO = os.getenv("DATASET_REPO", "King758/tamil-asr-corpus")
AUDITED_DATASET_REPO = DATASET_REPO

# Quota & Shard Controls
SHARD_CLIP_THRESHOLD = int(os.getenv("SHARD_CLIP_THRESHOLD", "400"))  # ~300-500 clips (~70 MB)
MAX_LOCAL_DISK_GB = float(os.getenv("MAX_LOCAL_DISK_GB", "1.0"))       # Strictly under 1.0 GB local disk

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = resolve_secret("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_ID = resolve_secret("TELEGRAM_ADMIN_ID", resolve_secret("ADMIN_USER_ID", resolve_secret("TELEGRAM_CHAT_ID", "")))
TELEGRAM_ADMIN_PASSCODE = resolve_secret("TELEGRAM_ADMIN_PASSCODE", resolve_secret("ADMIN_PASSCODE", ""))

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
