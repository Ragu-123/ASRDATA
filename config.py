import os
import sys
import uuid
import time
import shutil
from pathlib import Path
from typing import Optional

# --- Storage & Target Settings (Two-Tier Architecture) ---
# Tier 1: Tracking Bucket strictly for JSON metadata (manifest, leases, catalogs < 5 MB, ZERO audios)
TRACKING_BUCKET = os.getenv("TRACKING_BUCKET", "King758/asr-tracking-01")
BUCKET_ID = TRACKING_BUCKET

# Dedicated Separate Storage for Stage 1 Raw Audios (Completely separated from tracking bucket)
RAW_DATASET_REPO = os.getenv("RAW_DATASET_REPO", "King758/tamil-raw-audio")
RAW_BUCKET = os.getenv("RAW_BUCKET", "King758/media-archive-01")
RAW_STORAGE_TYPE = os.getenv("RAW_STORAGE_TYPE", "dataset")  # 'dataset' or 'bucket'

# Tier 2: Hugging Face Datasets
SEGMENTED_DATASET_REPO = os.getenv("SEGMENTED_DATASET_REPO", "King758/tamil-asr-segmented")
DATASET_REPO = os.getenv("DATASET_REPO", "King758/tamil-asr-corpus")

# Determine runtime environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists("/kaggle/working")
WORKING_DIR = Path("/kaggle/working" if IS_KAGGLE else "./data").resolve()
# Use /tmp on Kaggle when available (over 30GB+ available space, preserving /kaggle/working 20GB limit)
TMP_ROOT = Path("/tmp/asr_cache") if (IS_KAGGLE and Path("/tmp").exists()) else WORKING_DIR
STAGING_DIR = Path(os.getenv("STAGING_DIR", str(TMP_ROOT / "staging_audios"))).resolve()
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Buffer & GB Flushing Controls
BUFFER_GB = float(os.getenv("BUFFER_GB", os.getenv("BUFFER_FLUSH_GB", "1.0")))
BUFFER_BYTES = int(BUFFER_GB * (1024**3))

# Cloudflare Binary Path
CLOUDFLARED_BIN = WORKING_DIR / "cloudflared"

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from secret_resolver import (
    resolve_secret,
    add_secret_cli_args,
    apply_cli_secrets,
    set_secret,
    HARDCODED_SECRETS
)

# Cookie File Detection & Live Validation
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

def ensure_js_runtime() -> Optional[str]:
    """Ensure Deno or Node JS runtime is available and on PATH for yt-dlp EJS challenge solving."""
    # 1. Check existing Deno on PATH
    deno_in_path = shutil.which("deno")
    if deno_in_path:
        return deno_in_path

    # 2. Check ~/.deno/bin/deno
    user_deno = Path.home() / ".deno" / "bin" / "deno"
    if user_deno.exists():
        bin_dir = str(user_deno.parent)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        return str(user_deno)

    # 3. Check common Node.js paths (e.g. Kaggle /tools/node/bin/node)
    for node_candidate in ["/tools/node/bin", "/usr/bin", "/usr/local/bin"]:
        node_file = Path(node_candidate) / "node"
        if node_file.exists() and os.access(str(node_file), os.X_OK):
            if node_candidate not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{node_candidate}:{os.environ.get('PATH', '')}"

    # 4. If neither Deno nor Node is present, install Deno on Linux/Kaggle
    if not shutil.which("deno") and not shutil.which("node"):
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

    return shutil.which("deno") or shutil.which("node")

def ensure_deno_installed() -> Optional[str]:
    return ensure_js_runtime()

DENO_PATH = ensure_js_runtime()
YTDLP_REMOTE_COMPONENTS = ["ejs:github"]
YTDLP_REMOTE_COMPONENTS_CLI = "--remote-components ejs:github"

def get_impersonate_target():
    """Return ImpersonateTarget for curl_cffi if explicitly enabled via USE_IMPERSONATE=1.
    Defaults to None in multi-threaded ingestion to prevent curl_cffi thread pool contention.
    """
    if os.getenv("USE_IMPERSONATE", "false").lower() in ("true", "1"):
        try:
            from yt_dlp.networking.impersonate import ImpersonateTarget
            return ImpersonateTarget.from_str("chrome")
        except Exception:
            return None
    return None

def get_ytdlp_extractor_args(force_mobile: bool = False) -> dict:
    """Return optimal yt-dlp extractor args for robust high-volume bulk downloads.
    - If background PO Token provider (bgutil-pot) is active on 127.0.0.1:4416:
      Supplies authentic Proof-of-Origin (PO) tokens and visitor data, unlocking
      'web' & 'web_creator' high-fidelity direct streaming without bot blocks.
    - Multi-tier client waterfall:
      With cookies: ['web_creator', 'web', 'tv_samsung', 'visionos', 'android', 'ios']
      Clean guest mode: ['web', 'tv_samsung', 'visionos', 'android', 'ios']
    - fetch_pot: ['always'] forces proactive PO token generation to eliminate 403 & bot challenges.
    - Skips broken embedded/vr players and redundant webpage calls (youtubetab:skip=webpage).
    """
    pot_active = False
    pot_url = "http://127.0.0.1:4416"
    try:
        from pot_service import is_pot_service_alive, POT_URL
        pot_url = POT_URL
        pot_active = is_pot_service_alive(timeout=0.3)
    except Exception:
        pass

    has_valid_cookies = bool(COOKIE_FILE and Path(COOKIE_FILE).exists() and Path(COOKIE_FILE).stat().st_size > 10)

    if pot_active:
        clients = ["web", "web_creator", "tv_samsung", "visionos", "android", "ios"]
        return {
            "youtubepot-bgutilhttp": {
                "base_url": [pot_url]
            },
            "youtube": {
                "player_client": clients,
                "player_skip": ["web_embedded", "android_vr"],
                "fetch_pot": ["always"]
            },
            "youtubetab": {
                "skip": ["webpage"]
            }
        }

    return {
        "youtube": {
            "player_client": ["tv_samsung", "visionos", "android", "ios"],
            "player_skip": ["web_embedded", "android_vr"]
        },
        "youtubetab": {
            "skip": ["webpage"]
        }
    }

def get_ytdlp_js_runtimes() -> dict:
    """Return available JavaScript runtimes for yt-dlp (Node.js or Deno)."""
    if shutil.which("node"):
        return {"node": {}}
    if shutil.which("deno"):
        return {"deno": {}}
    return {}

def set_cookie_file(path_str: str, validate: bool = True) -> Optional[str]:
    """Set global COOKIE_FILE and environment variable if file exists.
    If the file is in a read-only filesystem (e.g. /kaggle/input), automatically
    copies it to a writable location (/kaggle/working/cookies.txt) so yt-dlp's
    cookiejar does not crash with OSError: [Errno 30] Read-only file system.
    """
    global COOKIE_FILE, YTDLP_EXTRACTOR_ARGS, YTDLP_EXTRACTOR_ARGS_CLI
    if not path_str:
        return COOKIE_FILE
    p = Path(str(path_str).strip().strip("'\""))

    # If explicit path doesn't exist, search /kaggle/input and working directory
    if not (p.exists() and p.is_file()):
        candidate = None
        search_roots = [Path("/kaggle/input"), Path("/kaggle/working")] if IS_KAGGLE else [Path("/kaggle/input"), Path(".")]
        for root in search_roots:
            if root.exists():
                # 1. Try finding by matching filename
                try:
                    for found in root.rglob(p.name):
                        if found.is_file() and found.stat().st_size > 10 and not any(ign in str(found).lower() for ign in ["testdata", "yt-dlp", ".git"]):
                            candidate = found
                            break
                except Exception:
                    pass
                if candidate:
                    break
                # 2. Try finding any *cookie*.txt
                try:
                    for found in root.rglob("*cookie*.txt"):
                        if found.is_file() and found.stat().st_size > 10 and not any(ign in str(found).lower() for ign in ["testdata", "yt-dlp", ".git"]):
                            candidate = found
                            break
                except Exception:
                    pass
                if candidate:
                    break
        if candidate:
            print(f"[CONFIG] 🔍 Auto-discovered cookie file at: {candidate}")
            p = candidate

    if p.exists() and p.is_file() and p.stat().st_size > 10:
        resolved_str = str(p.resolve())
        # /kaggle/input is read-only. Copy to writable /kaggle/working/cookies.txt
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
        YTDLP_EXTRACTOR_ARGS_CLI = '--extractor-args "youtube:player_client=tv_samsung,visionos,android,ios;player_skip=web_embedded,android_vr;youtubetab:skip=webpage"'
        print(f"[CONFIG] 🍪 YouTube Cookies: ACTIVE ({COOKIE_FILE})")
        return COOKIE_FILE
    return None

# Use aria2c toggle (Default False: native yt-dlp downloader is 10x faster and avoids YouTube CDN throttling)
USE_ARIA2C = os.getenv("USE_ARIA2C", "false").lower() in ("true", "1")

# Scan /kaggle/input for cookies if explicitly present, but treat as fallback asset
if os.path.exists("/kaggle/input"):
    try:
        for _f in sorted(Path("/kaggle/input").rglob("*cookie*.txt"), key=lambda x: x.stat().st_mtime, reverse=True):
            if _f.is_file() and _f.stat().st_size > 10:
                set_cookie_file(str(_f))
                break
    except Exception:
        pass

# Check env or common Kaggle paths if not found
if not COOKIE_FILE:
    _candidate_cookie_paths = [
        os.getenv("COOKIE_FILE", ""),
        os.getenv("COOKIES_FILE", ""),
        "/kaggle/input/cookies/cookies (1).txt",
        "/kaggle/input/cookies/cookies.txt",
        "/kaggle/input/datasets/dhanalakshmikavitha/cookies/cookies (1).txt",
        "/kaggle/input/datasets/dhanalakshmikavitha/cookies/cookies.txt",
        str(WORKING_DIR / "cookies.txt"),
        str(ROOT_DIR / "cookies.txt"),
    ]

    for _cp in _candidate_cookie_paths:
        if _cp and set_cookie_file(_cp):
            break

# Kaggle Secrets / ENV fallback
if not COOKIE_FILE:
    _env_cookies = os.getenv("YOUTUBE_COOKIES", os.getenv("COOKIES", ""))
    if _env_cookies.strip():
        try:
            _target = WORKING_DIR / "cookies.txt"
            _target.write_text(_env_cookies.strip(), encoding="utf-8")
            set_cookie_file(str(_target))
        except Exception:
            pass

# Telegram Bot Settings
def sync_runtime_secrets():
    global TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID
    TELEGRAM_BOT_TOKEN = resolve_secret("TELEGRAM_BOT_TOKEN", "8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ")
    TELEGRAM_ADMIN_ID = resolve_secret("TELEGRAM_ADMIN_ID", "5959376217")
    resolve_secret("HF_TOKEN", "")
    return TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID

TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID = sync_runtime_secrets()

# yt-dlp Bot Detection Bypass Extractor Args
YTDLP_EXTRACTOR_ARGS = get_ytdlp_extractor_args()
YTDLP_EXTRACTOR_ARGS_CLI = '--extractor-args "youtube:player_skip=web_embedded,android_vr"'

# Quota & Batch Controls
DISK_PER_NODE_GB = 19.5
DISK_THRESHOLD_GB = float(os.getenv("DISK_THRESHOLD_GB", "14.0"))
BATCH_LEASE_SIZE = int(os.getenv("BATCH_LEASE_SIZE", "30"))
LEASE_TTL_MINUTES = int(os.getenv("LEASE_TTL_MINUTES", "45"))

# Heartbeat & Mesh Timing
HEARTBEAT_INTERVAL_SEC = 10              # Heartbeat frequency (daemon thread)
WORKER_OFFLINE_THRESHOLD_SEC = 90        # Mark offline after 90s without heartbeat
STALE_PRUNE_THRESHOLD_SEC = 600          # Prune from bucket if > 10 mins inactive

# Stream Concurrency per Node:
# Kaggle notebook sessions provide 4 vCPUs. Default to 4 on Kaggle (utilizing all 4 cores), 2 locally.
CONCURRENT_DOWNLOADS_PER_NODE = int(os.getenv("CONCURRENT_DOWNLOADS", os.getenv("CONCURRENCY", "4" if IS_KAGGLE else "2")))

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
