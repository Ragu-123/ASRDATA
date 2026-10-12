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
# Use /tmp on Kaggle when available (over 30GB+ available space, preserving /kaggle/working 20GB limit)
TMP_ROOT = Path("/tmp/asr_cache") if (IS_KAGGLE and Path("/tmp").exists()) else WORKING_DIR
STAGING_DIR = Path(os.getenv("STAGING_DIR", str(TMP_ROOT / "segment_staging"))).resolve()
STAGING_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR = Path(os.getenv("SHARDS_DIR", str(TMP_ROOT / "segment_shards"))).resolve()
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

# Shard & Buffer Controls (Target ~1 GB Parquet Shards)
BUFFER_GB = float(os.getenv("BUFFER_GB", os.getenv("SHARD_GB", "1.0")))
BUFFER_BYTES = int(BUFFER_GB * (1024**3))
SHARD_CLIP_THRESHOLD = int(os.getenv("SHARD_CLIP_THRESHOLD", "5000"))
MAX_LOCAL_DISK_GB = float(os.getenv("MAX_LOCAL_DISK_GB", "5.0"))
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

def get_ytdlp_extractor_args() -> dict:
    """Return clean mobile yt-dlp extractor args (segmentation does not need cookies)."""
    return {
        "youtube": {
            "player_client": ["android", "ios"]
        }
    }

def set_cookie_file(path_str: str, validate: bool = True) -> Optional[str]:
    return None

def ensure_deno_installed() -> Optional[str]:
    """Ensure Deno JS runtime is installed for yt-dlp EJS challenge solving."""
    deno_in_path = shutil.which("deno")
    if deno_in_path:
        return deno_in_path
    user_deno = Path.home() / ".deno" / "bin" / "deno"
    if user_deno.exists():
        bin_dir = str(user_deno.parent)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        return str(user_deno)
    if sys.platform.startswith("linux") or IS_KAGGLE:
        try:
            import subprocess
            subprocess.run("curl -fsSL https://deno.land/install.sh | sh", shell=True, capture_output=True, timeout=30)
            if user_deno.exists():
                bin_dir = str(user_deno.parent)
                os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
                return str(user_deno)
        except Exception:
            pass
    return None

DENO_PATH = ensure_deno_installed()
YTDLP_REMOTE_COMPONENTS = {"ejs": "github"}
YTDLP_REMOTE_COMPONENTS_CLI = "--remote-components ejs:github"

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
