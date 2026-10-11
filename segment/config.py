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

# Cookie File Detection & Live Validation
COOKIE_FILE = None

def validate_youtube_cookies(path_or_str: str) -> bool:
    """Validate whether YouTube cookies file exists and has recognizable cookie format."""
    p = Path(str(path_or_str).strip().strip("'\""))
    if not p.exists() or not p.is_file() or p.stat().st_size < 10:
        return False
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        return any(k in content for k in ("youtube.com", "google.com", "LOGIN_INFO", "SID", "VISITOR_INFO", "NETSCAPE")) or len(content.strip()) > 20
    except Exception:
        return True

def get_ytdlp_extractor_args() -> dict:
    """Return optimal yt-dlp extractor args based on whether cookies are active.
    - With authenticated cookies: prioritize clients that support Netscape cookies
      ('tv', 'mweb', 'web_safari', 'default') so YouTube receives the authenticated session.
    - Without cookies: use mobile/tv clients ('android', 'ios', 'tv') and skip desktop web
      to bypass datacenter IP bot detection challenges.
    """
    if COOKIE_FILE or os.getenv("COOKIE_FILE"):
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

def set_cookie_file(path_str: str, validate: bool = True) -> Optional[str]:
    global COOKIE_FILE, YTDLP_EXTRACTOR_ARGS, YTDLP_EXTRACTOR_ARGS_CLI
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
        YTDLP_EXTRACTOR_ARGS = get_ytdlp_extractor_args()
        YTDLP_EXTRACTOR_ARGS_CLI = '--extractor-args "youtube:player_client=tv,mweb,web_safari,default,android"'
        print(f"[CONFIG] 🍪 YouTube Cookies: ACTIVE ({COOKIE_FILE})")
        return COOKIE_FILE
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

# Prioritize scanning /kaggle/input for the freshest cookies first
if os.path.exists("/kaggle/input"):
    try:
        for _f in sorted(Path("/kaggle/input").rglob("*cookie*.txt"), key=lambda x: x.stat().st_mtime, reverse=True):
            if _f.is_file() and _f.stat().st_size > 10:
                set_cookie_file(str(_f))
                break
    except Exception:
        pass

if not COOKIE_FILE:
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
YTDLP_EXTRACTOR_ARGS = get_ytdlp_extractor_args()
YTDLP_EXTRACTOR_ARGS_CLI = (
    '--extractor-args "youtube:player_client=tv,mweb,web_safari,default,android"'
    if COOKIE_FILE else
    '--extractor-args "youtube:player_client=android,ios,tv;player_skip=web,web_embedded,android_vr"'
)

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
