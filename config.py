import os
import sys
import uuid
import time
from pathlib import Path

# --- Storage & Target Settings (Two-Tier Architecture) ---
TRACKING_BUCKET = os.getenv("TRACKING_BUCKET", "King758/asr-tracking-01")
DATASET_REPO = os.getenv("DATASET_REPO", "King758/tamil-asr-corpus")
BUCKET_ID = os.getenv("BUCKET_ID", TRACKING_BUCKET)

# Determine runtime environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else "./data").resolve()
STAGING_DIR = WORKING_DIR / "staging_audios"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Cloudflare Binary Path
CLOUDFLARED_BIN = WORKING_DIR / "cloudflared"

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
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

# Telegram Bot Settings
TELEGRAM_BOT_TOKEN = resolve_secret("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_ID = resolve_secret("TELEGRAM_ADMIN_ID", resolve_secret("ADMIN_USER_ID", resolve_secret("TELEGRAM_CHAT_ID", "")))

# Quota & Batch Controls
DISK_PER_NODE_GB = 19.5
DISK_THRESHOLD_GB = float(os.getenv("DISK_THRESHOLD_GB", "14.0"))
BATCH_LEASE_SIZE = int(os.getenv("BATCH_LEASE_SIZE", "30"))
LEASE_TTL_MINUTES = int(os.getenv("LEASE_TTL_MINUTES", "45"))

# Heartbeat & Mesh Timing
HEARTBEAT_INTERVAL_SEC = 10              # Heartbeat frequency (daemon thread)
WORKER_OFFLINE_THRESHOLD_SEC = 90        # Mark offline after 90s without heartbeat
STALE_PRUNE_THRESHOLD_SEC = 600          # Prune from bucket if > 10 mins inactive

# Stream Concurrency per Node
CONCURRENT_DOWNLOADS_PER_NODE = 2        # 2 concurrent streams per node to stay within YouTube limits

# Web Dashboard Port
PORT = int(os.getenv("PORT", "8000"))

# Guaranteed Unique Node ID per instance
NODE_ID = os.getenv("WORKER_ID", f"node-{uuid.uuid4().hex[:6]}")

# Metadata Catalog Fetch Chunk Size
CATALOG_FETCH_CHUNK = 1000

# Default Seed Channels if channels.json is blank
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
    """Retrieve Hugging Face token from environment, Kaggle Secrets, or local .env."""
    return resolve_secret("HF_TOKEN", "")
