import os
import sys
import uuid
import time
from pathlib import Path

# --- Storage & Target Settings ---
BUCKET_ID = os.getenv("BUCKET_ID", "King758/media-archive-01")

# Determine runtime environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else "./data").resolve()
STAGING_DIR = WORKING_DIR / "staging_audios"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Cloudflare Binary Path
CLOUDFLARED_BIN = WORKING_DIR / "cloudflared"

# Quota & Batch Controls
DISK_PER_NODE_GB = 19.5
DISK_THRESHOLD_GB = float(os.getenv("DISK_THRESHOLD_GB", "14.0"))
BATCH_LEASE_SIZE = int(os.getenv("BATCH_LEASE_SIZE", "30"))
LEASE_TTL_MINUTES = int(os.getenv("LEASE_TTL_MINUTES", "45"))

# Heartbeat & Mesh Timing
HEARTBEAT_INTERVAL_SEC = 8              # Heartbeat frequency (daemon thread)
WORKER_OFFLINE_THRESHOLD_SEC = 90       # Mark offline only after 90s of no heartbeat
STALE_PRUNE_THRESHOLD_SEC = 1200        # Prune from bucket only after 20 minutes offline

# Speed Optimization for Kaggle (4 vCPUs)
CONCURRENT_DOWNLOADS_PER_NODE = 3       # 3 concurrent streams per node to saturate 4 Kaggle cores
HTTP_CHUNK_SIZE = 10485760              # 10MB chunk size (bypasses YouTube CDN throttling)

# Web Dashboard Port
PORT = int(os.getenv("PORT", "8000"))

# Guaranteed Unique Node ID per instance (prevents node collisions)
NODE_ID = os.getenv("WORKER_ID", f"node-{uuid.uuid4().hex[:6]}")

# Metadata Catalog Fetch Chunk Size
CATALOG_FETCH_CHUNK = 500

# Default Seed Channels if channels.json is blank
DEFAULT_CHANNELS = [
    {
        "id": "madangowri",
        "name": "Madan Gowri",
        "url": "https://www.youtube.com/@madangowri/videos",
        "active": True
    }
]

def get_hf_token() -> str:
    """Retrieve Hugging Face token from Kaggle Secrets or environment variables."""
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
            print(f"[CONFIG] Kaggle secrets retrieval warning: {e}", file=sys.stderr)
            
    return ""
