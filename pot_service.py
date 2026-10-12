import os
import sys
import time
import shutil
import urllib.request
import subprocess
from pathlib import Path
from typing import Optional

POT_PORT = int(os.getenv("POT_PORT", "4416"))
POT_URL = f"http://127.0.0.1:{POT_PORT}"
POT_GITHUB_RELEASE = "https://github.com/jim60105/bgutil-ytdlp-pot-provider-rs/releases/latest/download/bgutil-pot-linux-x86_64"

_pot_process: Optional[subprocess.Popen] = None

def is_pot_service_alive(timeout: float = 1.0) -> bool:
    """Check if the local bgutil PO Token provider HTTP service is responding on 127.0.0.1:4416."""
    try:
        req = urllib.request.Request(f"{POT_URL}/ping", headers={"User-Agent": "ASR-Worker"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 204, 404)
    except urllib.error.HTTPError as he:
        # If server responds with HTTP code (even 404), the server is alive
        return True
    except Exception:
        # Try root endpoint as fallback
        try:
            req = urllib.request.Request(POT_URL, headers={"User-Agent": "ASR-Worker"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status in (200, 204, 404)
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

def find_pot_binary() -> Optional[Path]:
    """Search for bgutil-pot binary in common system locations and PATH."""
    bin_in_path = shutil.which("bgutil-pot")
    if bin_in_path:
        return Path(bin_in_path)

    candidates = [
        Path("/usr/local/bin/bgutil-pot"),
        Path("/usr/bin/bgutil-pot"),
        Path.home() / ".local" / "bin" / "bgutil-pot",
        Path("/tmp/bgutil-pot"),
        Path("/kaggle/working/bgutil-pot"),
        Path("./bgutil-pot")
    ]
    for c in candidates:
        if c.exists() and os.access(str(c), os.X_OK):
            return c
    return None

def download_pot_binary() -> Optional[Path]:
    """Download pre-compiled bgutil-pot binary for Linux x86_64 from GitHub Releases."""
    if not sys.platform.startswith("linux"):
        return None

    # Choose writable destination
    target_dir = Path("/usr/local/bin") if os.access("/usr/local/bin", os.W_OK) else Path("/tmp")
    dest_path = target_dir / "bgutil-pot"

    print(f"[POT SERVICE] ⬇️ Downloading standalone bgutil-pot binary from GitHub Releases to {dest_path}...")
    try:
        req = urllib.request.Request(
            POT_GITHUB_RELEASE,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ASR-Harvester"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)

        dest_path.chmod(0o755)
        print(f"[POT SERVICE] ✅ Successfully downloaded and installed bgutil-pot binary ({dest_path.stat().st_size // 1024} KB).")
        return dest_path
    except Exception as e:
        print(f"[POT SERVICE] ⚠️ Notice downloading bgutil-pot binary: {e}")
        return None

def ensure_pot_plugin_installed() -> bool:
    """Ensure bgutil-ytdlp-pot-provider Python plugin is installed and registered with yt-dlp."""
    try:
        import yt_dlp.extractor.youtube.pot._registry as reg
        if "BgUtilHTTP" in reg._pot_providers.value:
            return True
    except Exception:
        pass

    try:
        _load_yt_dlp_plugins()
        import yt_dlp.extractor.youtube.pot._registry as reg
        if "BgUtilHTTP" in reg._pot_providers.value:
            return True
    except Exception:
        pass

    print("[POT SERVICE] 📦 Installing bgutil-ytdlp-pot-provider plugin via pip...")
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "bgutil-ytdlp-pot-provider"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if res.returncode == 0:
            import importlib
            importlib.invalidate_caches()
            _load_yt_dlp_plugins()
            return True
        return False
    except Exception as e:
        print(f"[POT SERVICE] Notice installing plugin: {e}")
        return False

def _load_yt_dlp_plugins():
    """Dynamically register yt-dlp plugin extractors if yt_dlp is already in memory."""
    try:
        import yt_dlp.plugins
        yt_dlp.plugins.load_all_plugins()
    except Exception:
        pass

def verify_pot_provider() -> Optional[str]:
    """Verify bgutil-pot server and return its reported version."""
    try:
        req = urllib.request.Request(f"{POT_URL}/ping", headers={"User-Agent": "ASR-Worker"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            import json
            data = json.loads(resp.read().decode())
            return data.get("version", "active")
    except Exception:
        return None

def start_pot_server(pot_binary: Path) -> bool:
    """Launch bgutil-pot server daemon in background with verbose logging."""
    global _pot_process
    print(f"[POT SERVICE] 🚀 Spawning PO Token daemon: {pot_binary} server (Port {POT_PORT})...")
    log_file = Path("/tmp/bgutil_pot_server.log") if Path("/tmp").exists() else Path("bgutil_pot_server.log")
    
    try:
        f = open(log_file, "a", encoding="utf-8")
        env = os.environ.copy()
        env["RUST_LOG"] = "info"
        _pot_process = subprocess.Popen(
            [str(pot_binary), "server", "--verbose"],
            stdout=f,
            stderr=f,
            env=env,
            start_new_session=True
        )
        
        # Wait up to 5 seconds for daemon to accept connections
        for _ in range(10):
            time.sleep(0.5)
            if is_pot_service_alive(timeout=0.5):
                version = verify_pot_provider() or "v1.x"
                print(f"[POT SERVICE] ✅ Background PO Token daemon is ACTIVE and listening on {POT_URL}! (Version: {version})")
                return True
        return is_pot_service_alive()
    except Exception as e:
        print(f"[POT SERVICE] ⚠️ Failed to launch PO Token daemon: {e}")
        return False

def ensure_pot_service() -> bool:
    """
    Ensure the Proof of Origin (PO) token provider is running and healthy.
    - If already alive on 127.0.0.1:4416: returns True.
    - If not alive on Linux: locates or downloads the binary, starts the daemon, and ensures pip plugin.
    - Returns True if service is running, False otherwise.
    """
    if is_pot_service_alive():
        version = verify_pot_provider() or "active"
        print(f"[POT SERVICE] ✅ PO Token daemon is already running on {POT_URL} (Version: {version}).")
        ensure_pot_plugin_installed()
        return True

    # Check or download binary on Linux
    pot_bin = find_pot_binary()
    if not pot_bin and (sys.platform.startswith("linux") or os.path.exists("/kaggle/working")):
        pot_bin = download_pot_binary()

    if pot_bin:
        started = start_pot_server(pot_bin)
        ensure_pot_plugin_installed()
        return started

    return False

def stop_pot_service():
    """Terminate daemon process if managed by this process."""
    global _pot_process
    if _pot_process and _pot_process.poll() is None:
        try:
            _pot_process.terminate()
            _pot_process.wait(timeout=2)
            print("[POT SERVICE] Stopped background PO Token daemon.")
        except Exception:
            try:
                _pot_process.kill()
            except Exception:
                pass
        _pot_process = None
