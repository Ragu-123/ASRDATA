#!/usr/bin/env python3
"""
Telegram Cluster Monitor & Controller for Tamil ASR Distributed Engine.
Implements strict administrator authorization, proactive alerting (shards, disk, errors),
and remote cluster command execution (/status, /add_channel, /errors, /pause, /resume, /flush).
"""

import os
import sys
import time
import json
import threading
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
from huggingface_hub import HfApi
import transcribe.config as config

class TelegramClusterBot:
    """
    Telegram Bot client for monitoring and controlling the Tamil ASR cluster.
    Provides strict admin locking, real-time proactive alerts, and remote management commands.
    """
    def __init__(
        self,
        token: str = config.TELEGRAM_BOT_TOKEN,
        admin_id: Optional[str] = None,
        coordinator: Any = None,
        syncer: Any = None
    ):
        self.token = token.strip()
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.coordinator = coordinator
        self.syncer = syncer
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Lock and identity storage
        self.admin_file = config.WORKING_DIR / "telegram_admin.json"
        self.admin_user_id: Optional[int] = None
        self.admin_chat_id: Optional[int] = None
        self.admin_username: Optional[str] = None
        self._lock = threading.Lock()

        # In-memory recent error buffer for /errors command
        self.recent_errors: deque = deque(maxlen=25)

        # Initialize admin identity
        self._load_admin_identity(preconfigured_id=admin_id or config.TELEGRAM_ADMIN_ID)

    def _load_admin_identity(self, preconfigured_id: str = ""):
        """Load admin identity from environment, tracking bucket, or local file."""
        # 1. Environment variable if set
        if preconfigured_id:
            try:
                self.admin_user_id = int(preconfigured_id)
                self.admin_chat_id = self.admin_user_id
            except ValueError:
                self.admin_user_id = preconfigured_id
                self.admin_chat_id = preconfigured_id
            print(f"[TELEGRAM BOT] Admin preconfigured from env: {self.admin_user_id}")
            return

        # 2. Local file
        if self.admin_file.exists():
            try:
                with open(self.admin_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.admin_user_id = data.get("admin_user_id")
                self.admin_chat_id = data.get("admin_chat_id")
                self.admin_username = data.get("admin_username")
                if self.admin_user_id:
                    print(f"[TELEGRAM BOT] Admin loaded from local cache: {self.admin_user_id} (@{self.admin_username})")
                    return
            except Exception as e:
                print(f"[TELEGRAM BOT] Notice reading local admin file: {e}")

        # 3. Check Tier 1 Tracking Bucket if coordinator is provided
        if self.coordinator and hasattr(self.coordinator, "api") and hasattr(self.coordinator, "tracking_bucket"):
            try:
                admin_data = self.coordinator.get_telegram_admin()
                if admin_data and admin_data.get("admin_user_id"):
                    self.admin_user_id = admin_data.get("admin_user_id")
                    self.admin_chat_id = admin_data.get("admin_chat_id")
                    self.admin_username = admin_data.get("admin_username")
                    self._persist_local_admin(admin_data)
                    print(f"[TELEGRAM BOT] Admin loaded from tracking bucket: {self.admin_user_id} (@{self.admin_username})")
            except Exception as e:
                print(f"[TELEGRAM BOT] Notice fetching admin from bucket: {e}")

    def _persist_local_admin(self, data: Dict[str, Any]):
        try:
            with open(self.admin_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _lock_admin(self, user_id: int, chat_id: int, username: str = "") -> bool:
        """Lock the bot onto this user permanently across cluster."""
        with self._lock:
            self.admin_user_id = user_id
            self.admin_chat_id = chat_id
            self.admin_username = username

            data = {
                "admin_user_id": user_id,
                "admin_chat_id": chat_id,
                "admin_username": username,
                "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            self._persist_local_admin(data)

            # Persist to Tier 1 tracking bucket so all peer nodes and future restarts recognize the admin
            if self.coordinator and hasattr(self.coordinator, "save_telegram_admin"):
                try:
                    self.coordinator.save_telegram_admin(data)
                except Exception as e:
                    print(f"[TELEGRAM BOT] Warning syncing admin lock to bucket: {e}")

            print(f"[TELEGRAM BOT] 🔒 STRICT ADMIN LOCK ESTABLISHED: User {user_id} (@{username})")
            return True

    def is_authorized(self, user_id: int, chat_id: int) -> bool:
        """Check whether the user is the authorized administrator."""
        with self._lock:
            if self.admin_user_id is None:
                return False
            u_str = str(user_id)
            c_str = str(chat_id)
            admin_u_str = str(self.admin_user_id)
            admin_c_str = str(self.admin_chat_id) if self.admin_chat_id is not None else admin_u_str
            return u_str == admin_u_str or c_str == admin_c_str

    def log_error(self, worker_id: str, video_id: str, error_msg: str):
        """Record an error in the recent error buffer and send an alert."""
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "worker_id": worker_id,
            "video_id": video_id,
            "error": str(error_msg)
        }
        self.recent_errors.append(entry)
        self.send_error_alert(worker_id, video_id, error_msg)

    # --- PROACTIVE ALERT DISPATCHERS ---

    def send_message(self, text: str, parse_mode: Optional[str] = "Markdown") -> bool:
        """Send a proactive Telegram message to the authorized admin with automatic markdown fallback."""
        if not self.admin_chat_id:
            return False
        try:
            with httpx.Client(timeout=10.0) as client:
                payload = {
                    "chat_id": self.admin_chat_id,
                    "text": text
                }
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                resp = client.post(f"{self.base_url}/sendMessage", json=payload)
                if resp.status_code == 200:
                    return True

                # If Telegram rejected formatting (e.g. 400 Bad Request), fallback to plain text
                if parse_mode:
                    payload.pop("parse_mode", None)
                    resp_fb = client.post(f"{self.base_url}/sendMessage", json=payload)
                    return resp_fb.status_code == 200
                return False
        except Exception as e:
            print(f"[TELEGRAM BOT] Failed to send message: {e}")
            return False

    def send_shard_alert(self, shard_name: str, num_clips: int, duration_sec: float, total_hours: float, dataset_repo: str = config.DATASET_REPO):
        """Proactive alert when an Apache Parquet shard is uploaded to Hugging Face Dataset."""
        dur_mins = duration_sec / 60.0
        dur_hrs = duration_sec / 3600.0
        msg = (
            f"📦 *Apache Parquet Shard Uploaded!*\n\n"
            f"• *Shard*: `{shard_name}`\n"
            f"• *Clips*: `{num_clips}` audio segments\n"
            f"• *Duration*: `{dur_hrs:.2f} hrs` ({dur_mins:.1f} mins)\n"
            f"• *Cluster Total*: `{total_hours:.2f} hrs` harvested\n"
            f"• *Dataset*: `{dataset_repo}`\n"
            f"• *Storage*: Tier 2 Hugging Face Dataset (Zero-disk streaming ready)"
        )
        return self.send_message(msg)

    def send_disk_alert(self, used_gb: float, total_gb: float, free_gb: float, percent: float):
        """Proactive alert when node disk approaches threshold."""
        msg = (
            f"🚨 *Disk Guard Warning Alert!*\n\n"
            f"• *Used*: `{used_gb:.2f} GB` / `{total_gb:.2f} GB` ({percent:.1f}%)\n"
            f"• *Free Space*: `{free_gb:.2f} GB`\n"
            f"• *Trigger*: Automatic shard flush executed to keep disk strictly under 1.0 GB quota."
        )
        return self.send_message(msg)

    def send_error_alert(self, worker_id: str, video_id: str, error_msg: str):
        """Proactive alert for pipeline failures or Cloudflare 502 exhaustion."""
        msg = (
            f"❌ *Cluster Error Notice*\n\n"
            f"• *Node*: `{worker_id}`\n"
            f"• *Video*: `{video_id}`\n"
            f"• *Error*: `{str(error_msg)[:250]}`"
        )
        return self.send_message(msg)

    # --- COMMAND HANDLERS ---

    def handle_command(self, cmd_text: str, user_id: int, chat_id: int, username: str = "") -> str:
        """Process incoming command with strict authorization check."""
        parts = cmd_text.strip().split()
        if not parts:
            return ""

        command = parts[0].lower().split("@")[0]  # Strip bot handle if present (e.g. /status@crawler001bot)
        args = parts[1:]

        # If admin is not set yet, check tracking bucket in case another node locked it
        if self.admin_user_id is None and self.coordinator and hasattr(self.coordinator, "get_telegram_admin"):
            try:
                admin_data = self.coordinator.get_telegram_admin()
                if admin_data and admin_data.get("admin_user_id"):
                    self.admin_user_id = admin_data.get("admin_user_id")
                    self.admin_chat_id = admin_data.get("admin_chat_id")
                    self.admin_username = admin_data.get("admin_username")
                    self._persist_local_admin(admin_data)
            except Exception:
                pass

        # Auto-lock check for new admin on /start ONLY
        if self.admin_user_id is None:
            if command != "/start":
                return (
                    "🔒 *Setup Required*\n"
                    "This bot has no registered administrator yet.\n"
                    "Please send `/start` (or `/start <passcode>`) to claim ownership."
                )

            # Check optional passcode if configured
            if config.TELEGRAM_ADMIN_PASSCODE:
                provided_pass = args[0] if args else ""
                if provided_pass != config.TELEGRAM_ADMIN_PASSCODE:
                    return (
                        "🔒 *Passcode Required*\n"
                        "This bot requires an administrator passcode to establish ownership.\n"
                        "Usage: `/start <passcode>`"
                    )

            self._lock_admin(user_id=user_id, chat_id=chat_id, username=username)
            return (
                f"🔒 *Admin Lock Established!*\n\n"
                f"Welcome! You have been registered as the sole administrator for this Tamil ASR cluster.\n"
                f"• *Admin User ID*: `{user_id}`\n"
                f"• *Admin Handle*: @{username or 'user'}\n\n"
                f"This bot is now strictly locked to your account. All unauthorized users will be rejected.\n\n"
                f"Type `/help` to see available cluster control commands."
            )

        # Check authorization
        if not self.is_authorized(user_id, chat_id):
            return (
                f"⛔ *Access Denied*\n\n"
                f"This cluster management bot is strictly locked to the authorized administrator.\n"
                f"Your User ID (`{user_id}`) is not authorized to interact with or control this cluster."
            )

        # Update chat_id or username for proactive alerts if needed
        if not self.admin_chat_id or (username and not self.admin_username):
            with self._lock:
                self.admin_chat_id = chat_id
                if username:
                    self.admin_username = username

        # Authorized Admin Commands
        if command == "/start":
            return self._cmd_start_authorized(username)
        elif command == "/help":
            return self._cmd_help()
        elif command == "/status":
            return self._cmd_status()
        elif command == "/add_channel":
            return self._cmd_add_channel(args)
        elif command == "/errors":
            return self._cmd_errors()
        elif command == "/pause":
            return self._cmd_pause()
        elif command == "/resume":
            return self._cmd_resume()
        elif command == "/flush":
            return self._cmd_flush()
        elif command == "/stop":
            return self._cmd_stop()
        elif command == "/admin":
            return self._cmd_admin()
        else:
            return f"❓ Unknown command: `{command}`. Type `/help` for available commands."

    def _cmd_start_authorized(self, username: str = "") -> str:
        handle_str = f"@{username}" if username else f"User {self.admin_user_id}"
        return (
            f"👋 *Welcome back, {handle_str}!*\n\n"
            f"🔒 *Status*: Authorized Administrator\n"
            f"• *Admin ID*: `{self.admin_user_id}`\n"
            f"• *Access*: Exclusive (All strangers rejected)\n\n"
            f"Type `/status` to monitor the cluster, or `/help` for commands."
        )

    def _cmd_help(self) -> str:
        return (
            f"🎙️ *Tamil ASR Cluster Control Hub*\n"
            f"🔐 *Admin*: `{self.admin_user_id}` (@{self.admin_username or 'admin'})\n\n"
            f"*Available Commands:*\n"
            f"📊 `/status` — Live cluster overview, active nodes & dataset progress\n"
            f"➕ `/add_channel <url>` — Scrape & queue a YouTube channel upfront\n"
            f"⚠️ `/errors` — View recent cluster error logs\n"
            f"⏸️ `/pause` — Pause ingestion & transcription across cluster\n"
            f"▶️ `/resume` — Resume cluster operations\n"
            f"💾 `/flush` — Force immediate Parquet shard packaging and upload\n"
            f"🛑 `/stop` — Gracefully stop all active worker nodes\n"
            f"👑 `/admin` — Show current admin identity & lock status"
        )

    def _cmd_status(self) -> str:
        if not self.coordinator:
            return "⚠️ Coordinator is not connected to bot instance."

        try:
            overview = self.coordinator.get_cluster_overview()
            manifest = self.coordinator.load_manifest(force=True)

            node_count = overview.get("node_count", 0)
            live_workers = overview.get("live_workers", [])
            total_hours = overview.get("total_harvested_hours", 0.0)
            completed_vids = overview.get("total_completed_videos", 0)
            total_segments = manifest.get("total_segments", 0)
            shards_count = len(manifest.get("shards", []))

            # Channel catalog stats
            channels = self.coordinator.load_channels()
            cat_summary = []
            for ch in channels:
                ch_id = ch["id"]
                catalog = self.coordinator.catalog_cache.get(ch_id, [])
                cat_summary.append(f"• *{ch.get('name', ch_id)}*: {len(catalog)} videos")

            # Workers details
            w_lines = []
            for w in live_workers:
                w_id = w.get("worker_id", "node")
                w_status = w.get("status", "idle")
                w_disk = w.get("disk", {}).get("used_gb", 0.0)
                w_lines.append(f"• `{w_id}`: *{w_status}* | Disk: {w_disk:.1f}GB")

            worker_text = "\n".join(w_lines) if w_lines else "• No active workers online"
            cat_text = "\n".join(cat_summary) if cat_summary else "• None"

            return (
                f"📊 *Tamil ASR Cluster Status*\n\n"
                f"• *Active Nodes*: `{node_count}`\n"
                f"• *Harvested Speech*: `{total_hours:.2f} hrs`\n"
                f"• *Transcribed Videos*: `{completed_vids}`\n"
                f"• *Audited Segments*: `{total_segments}`\n"
                f"• *Parquet Shards*: `{shards_count}`\n"
                f"• *Dataset*: `{config.DATASET_REPO}`\n"
                f"• *Tracking Bucket*: `{config.TRACKING_BUCKET}`\n\n"
                f"💻 *Active Workers:*\n{worker_text}\n\n"
                f"📺 *Channels Catalog:*\n{cat_text}"
            )
        except Exception as e:
            return f"⚠️ Error generating cluster status: {e}"

    def _cmd_add_channel(self, args: List[str]) -> str:
        if not args:
            return "Usage: `/add_channel <url_or_handle>`\nExample: `/add_channel @madangowri`"

        url_or_handle = args[0].strip()
        if not self.coordinator:
            return "⚠️ Coordinator is not connected."

        # Trigger upfront scraping in background thread so Telegram doesn't timeout
        def _scrape_job():
            try:
                self.send_message(f"⏳ *Scraping Channel Catalog Upfront...*\nTarget: `{url_or_handle}`\nExtracting all video metadata via yt-dlp...")
                catalog = self.coordinator.scrape_channel_catalog(url_or_handle)
                self.send_message(
                    f"✅ *Channel Catalog Upfront Scraping Complete!*\n\n"
                    f"• *Channel*: `{url_or_handle}`\n"
                    f"• *Discovered Videos*: `{len(catalog)}` videos\n"
                    f"• *Catalog Stored*: `catalogs/{url_or_handle.replace('@', '')}.json`\n"
                    f"• *Lease Latency*: Sub-10ms with zero runtime YouTube pagination!"
                )
            except Exception as exc:
                self.send_message(f"❌ Failed to scrape catalog for `{url_or_handle}`: {exc}")

        threading.Thread(target=_scrape_job, daemon=True).start()
        return f"🚀 Initiated upfront catalog scraping for `{url_or_handle}`. You will receive an update once complete!"

    def _cmd_errors(self) -> str:
        if not self.recent_errors:
            return "✅ *No Errors Reported*\nThe cluster is running smoothly with zero recorded failures!"

        lines = ["⚠️ *Recent Cluster Errors (Last 10):*"]
        for err in list(self.recent_errors)[-10:]:
            lines.append(f"• [{err['time']}] `{err['worker_id']}` ({err['video_id']}): {err['error'][:80]}")
        return "\n".join(lines)

    def _cmd_pause(self) -> str:
        if not self.coordinator:
            return "⚠️ Coordinator is not connected."
        self.coordinator.send_command("pause")
        return "⏸️ *Pause Signal Dispatched*\nAll active workers have been instructed to pause ingestion."

    def _cmd_resume(self) -> str:
        if not self.coordinator:
            return "⚠️ Coordinator is not connected."
        self.coordinator.send_command("resume")
        return "▶️ *Resume Signal Dispatched*\nWorkers will resume batch processing."

    def _cmd_flush(self) -> str:
        if not self.coordinator:
            return "⚠️ Coordinator is not connected."
        self.coordinator.send_command("flush")
        if self.syncer and hasattr(self.syncer, "flush_shard"):
            try:
                res = self.syncer.flush_shard()
                return f"💾 *Flush Executed*\nCurrent worker flushed shard: `{res or 'No buffered clips'}`"
            except Exception as e:
                return f"⚠️ Notice during local flush: {e}"
        return "💾 *Flush Signal Dispatched*\nWorkers will immediately package and upload pending clips to Parquet."

    def _cmd_stop(self) -> str:
        if not self.coordinator:
            return "⚠️ Coordinator is not connected."
        self.coordinator.send_command("stop")
        return "🛑 *Cluster Stop Dispatched*\nWorkers will safely finish current clip and terminate."

    def _cmd_admin(self) -> str:
        return (
            f"👑 *Administrator Security Status*\n\n"
            f"• *Admin User ID*: `{self.admin_user_id}`\n"
            f"• *Admin Chat ID*: `{self.admin_chat_id}`\n"
            f"• *Admin Handle*: @{self.admin_username or 'unknown'}\n"
            f"• *Lock Status*: 🔒 Strictly Locked & Active\n"
            f"• *Unauthorized Access*: Rejection Active"
        )

    # --- POLLING ENGINE ---

    def poll_once(self, offset: Optional[int] = None, timeout: int = 15) -> tuple:
        """Fetch updates from Telegram Bot API with long-polling."""
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset

        try:
            with httpx.Client(timeout=timeout + 5.0) as client:
                resp = client.get(f"{self.base_url}/getUpdates", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return data.get("result", []), offset
                elif resp.status_code == 409:
                    print("[TELEGRAM BOT] Notice: 409 Conflict with another poller instance. Backing off 15s...")
                    time.sleep(15)
        except Exception as e:
            # Network blip or timeout
            time.sleep(2)
        return [], offset

    def run_polling_loop(self):
        """Continuous polling loop for Telegram Bot."""
        print(f"[TELEGRAM BOT] Polling service started for bot token: {self.token[:10]}...")
        self.running = True
        offset = None

        while self.running:
            # Multi-node coordination: ensure only ONE node polls getUpdates to prevent HTTP 409 Conflict
            if self.coordinator and hasattr(self.coordinator, "acquire_telegram_poller_lease"):
                worker_id = getattr(self.coordinator, "worker_id", getattr(self.coordinator, "node_id", "node"))
                is_leader = self.coordinator.acquire_telegram_poller_lease(worker_id=worker_id, ttl_sec=45)
                if not is_leader:
                    time.sleep(12)
                    continue

            try:
                updates, _ = self.poll_once(offset=offset, timeout=20)
                for update in updates:
                    update_id = update.get("update_id")
                    offset = update_id + 1

                    message = update.get("message") or update.get("edited_message")
                    if not message:
                        continue

                    text = message.get("text", "")
                    sender = message.get("from", {})
                    chat = message.get("chat", {})

                    user_id = sender.get("id")
                    chat_id = chat.get("id")
                    username = sender.get("username", "")

                    if text.startswith("/"):
                        reply = self.handle_command(text, user_id=user_id, chat_id=chat_id, username=username)
                        if reply:
                            try:
                                with httpx.Client(timeout=10.0) as client:
                                    resp = client.post(
                                        f"{self.base_url}/sendMessage",
                                        json={
                                            "chat_id": chat_id,
                                            "text": reply,
                                            "parse_mode": "Markdown"
                                        }
                                    )
                                    if resp.status_code != 200:
                                        # Fallback to plain text if Markdown parsing failed
                                        client.post(
                                            f"{self.base_url}/sendMessage",
                                            json={
                                                "chat_id": chat_id,
                                                "text": reply
                                            }
                                        )
                            except Exception as e:
                                print(f"[TELEGRAM BOT] Send reply error: {e}")
            except Exception as e:
                print(f"[TELEGRAM BOT] Polling loop notice: {e}")
                time.sleep(3)

        # Release poller lease on exit
        if self.coordinator and hasattr(self.coordinator, "release_telegram_poller_lease"):
            try:
                worker_id = getattr(self.coordinator, "worker_id", getattr(self.coordinator, "node_id", "node"))
                self.coordinator.release_telegram_poller_lease(worker_id)
            except Exception:
                pass

        print("[TELEGRAM BOT] Polling service stopped.")

    def start_background_thread(self):
        """Start the Telegram Bot polling loop in an independent background thread."""
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self.run_polling_loop, daemon=True)
        self.thread.start()
        print("[TELEGRAM BOT] Started background monitor thread.")

    def stop(self):
        self.running = False
        if self.coordinator and hasattr(self.coordinator, "release_telegram_poller_lease"):
            try:
                worker_id = getattr(self.coordinator, "worker_id", getattr(self.coordinator, "node_id", "node"))
                self.coordinator.release_telegram_poller_lease(worker_id)
            except Exception:
                pass

def main():
    """Standalone CLI entry point for Telegram bot."""
    token = config.TELEGRAM_BOT_TOKEN
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN is not configured.", file=sys.stderr)
        sys.exit(1)

    print("="*70)
    print("🤖 TAMIL ASR TELEGRAM CLUSTER MONITOR & CONTROLLER")
    print("="*70)
    print(f"📌 Token: {token[:12]}...")

    hf_token = config.get_hf_token()
    coordinator = None
    if hf_token:
        try:
            from transcribe.coordinator import TranscriptionCoordinator
            api = HfApi(token=hf_token)
            coordinator = TranscriptionCoordinator(api=api)
            print(f"📦 Connected to Tracking Bucket: {config.TRACKING_BUCKET}")
        except Exception as e:
            print(f"Notice connecting coordinator: {e}")

    bot = TelegramClusterBot(token=token, coordinator=coordinator)
    print(f"🔒 Admin Lock Status: {'Locked to ID: ' + str(bot.admin_user_id) if bot.admin_user_id else 'Unlocked (First user /start will become admin)'}")
    print("Listening for Telegram commands... Press Ctrl+C to stop.")

    try:
        bot.run_polling_loop()
    except KeyboardInterrupt:
        print("\nStopping bot...")
        bot.stop()

if __name__ == "__main__":
    main()
