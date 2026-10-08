import os
import sys
import re
import time
import shutil
import urllib.request
import subprocess
import threading
from typing import Optional
import config

class CloudflareTunnel:
    """
    Manages cloudflared binary download, execution, and public HTTPS URL capture.
    """
    def __init__(self, port: int = config.PORT):
        self.port = port
        self.url: str = ""
        self.process: Optional[subprocess.Popen] = None
        self._bin_path = config.CLOUDFLARED_BIN
        self._ensure_binary()

    def _ensure_binary(self):
        if self._bin_path.exists():
            return

        # Check if already in PATH
        which_path = shutil.which("cloudflared")
        if which_path:
            self._bin_path = config.WORKING_DIR / "cloudflared"
            shutil.copy(which_path, self._bin_path)
            return

        print("[TUNNEL] Downloading cloudflared binary...")
        is_windows = sys.platform.startswith("win")
        if is_windows:
            download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            self._bin_path = config.WORKING_DIR / "cloudflared.exe"
        else:
            download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

        try:
            urllib.request.urlretrieve(download_url, str(self._bin_path))
            if not is_windows:
                os.chmod(str(self._bin_path), 0o755)
            print("[TUNNEL] cloudflared downloaded successfully.")
        except Exception as e:
            print(f"[TUNNEL] Failed to download cloudflared: {e}", file=sys.stderr)

    def start(self) -> str:
        """Start the tunnel and wait for the public URL."""
        if not self._bin_path.exists():
            print("[TUNNEL] Error: cloudflared binary not available.", file=sys.stderr)
            return ""

        # Terminate any stray cloudflared processes on this machine/container
        try:
            if not sys.platform.startswith("win"):
                subprocess.run(["pkill", "-9", "-f", "cloudflared"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        cmd = [str(self._bin_path), "tunnel", "--url", f"http://127.0.0.1:{self.port}", "--no-autoupdate"]
        print(f"[TUNNEL] Launching Cloudflare Tunnel for port {self.port} on {config.NODE_ID}...")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        url_event = threading.Event()

        def scan_output():
            pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            for line in iter(self.process.stderr.readline, ''):
                match = pattern.search(line)
                if match and not self.url:
                    self.url = match.group(0)
                    url_event.set()
                    print(f"\n========================================================")
                    print(f"🚀 CLOUDFLARE PUBLIC TUNNEL LIVE FOR [{config.NODE_ID}]:")
                    print(f"👉 {self.url}")
                    print(f"========================================================\n")
                    try:
                        with open(config.WORKING_DIR / "tunnel_url.txt", "w") as f:
                            f.write(self.url)
                    except Exception:
                        pass
                if not line and self.process.poll() is not None:
                    break

        t = threading.Thread(target=scan_output, daemon=True)
        t.start()

        # Wait up to 35 seconds for URL
        url_event.wait(timeout=35)
        return self.url

    def stop(self):
        """Terminate the tunnel process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("[TUNNEL] Cloudflare Tunnel stopped.")
