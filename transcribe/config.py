import os
import sys
import uuid
from pathlib import Path

# --- Storage & Bucket Configuration ---
SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", "King758/media-archive-01")
TARGET_BUCKET = os.getenv("TARGET_BUCKET", "King758/asr-transcripts-01")

# Runtime environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else "./data").resolve()
STAGING_DIR = WORKING_DIR / "transcribe_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Gemini Canvas Proxy Configuration
DEFAULT_PROXY_URL = os.getenv("GEMINI_PROXY_URL", "")
PROXY_TOKEN = os.getenv("PROXY_TOKEN", "51dfb957-7347-449b-b153-df797a72c9d5")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# Parallel Request Concurrency (Sweet spot: 3 to 5 parallel requests)
CONCURRENCY = int(os.getenv("CONCURRENCY", "4"))
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "16384"))  # 16k reasoning tokens

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
