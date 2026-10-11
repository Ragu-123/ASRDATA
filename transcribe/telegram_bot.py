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
import tempfile
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

DEFAULT_ADMIN_ID = "5959376217"

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
        self.token = token.strip() if token else ""
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""
        self.coordinator = coordinator
        self.syncer = syncer
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.bot_info: Optional[Dict[str, Any]] = None

        # Lock and identity storage
        self.admin_file = config.WORKING_DIR / "telegram_admin.json"
        self.admin_user_id: Optional[int] = None
        self.admin_chat_id: Optional[int] = None
        self.admin_username: Optional[str] = None
        self._lock = threading.Lock()

        # In-memory recent error buffer for /errors command
        self.recent_errors: deque = deque(maxlen=25)
        self._stalled_alerted: set = set()
        self._last_watchdog_check: float = 0.0

        # Initialize admin identity
        self._load_admin_identity(preconfigured_id=admin_id or config.TELEGRAM_ADMIN_ID)

        # Proactively verify connection and register command menu if token is present
        if self.token:
            self.bot_info = self.verify_connection()
            self.setup_bot_commands()

    def verify_connection(self) -> Optional[Dict[str, Any]]:
        """Verify the bot token with Telegram getMe endpoint."""
        if not self.token or not self.base_url:
            return None
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/getMe")
                if resp.status_code == 200:
                    data = resp.json().get("result", {})
                    username = data.get("username", "")
                    bot_id = data.get("id", "")
                    print(f"[TELEGRAM BOT] ✅ Connected successfully as @{username} (ID: {bot_id})")
                    return data
                elif resp.status_code == 401:
                    print(f"[TELEGRAM BOT] ❌ Bot token is unauthorized (401). Verify TELEGRAM_BOT_TOKEN.")
                else:
                    print(f"[TELEGRAM BOT] ⚠️ Telegram getMe returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[TELEGRAM BOT] Notice during getMe check: {e}")
        return None

    def get_reply_keyboard(self) -> Dict[str, Any]:
        """Dismiss custom keyboard so Telegram displays native left-side Menu button."""
        return {"remove_keyboard": True}

    def normalize_command(self, text: str) -> str:
        """Map button text or plain text to slash command."""
        t = text.strip()
        if t.startswith("/"):
            return t

        t_lower = t.lower()
        if "status" in t_lower:
            return "/status"
        if "add_channel" in t_lower or "add channel" in t_lower:
            parts = t.split(maxsplit=1)
            arg = parts[1] if len(parts) > 1 and not parts[1].startswith("/") else ""
            return f"/add_channel {arg}".strip()
        if "help" in t_lower:
            return "/help"
        if "errors" in t_lower or "error" in t_lower:
            return "/errors"
        if "flush" in t_lower:
            return "/flush"
        if "pause" in t_lower:
            return "/pause"
        if "resume" in t_lower:
            return "/resume"
        if "stop" in t_lower:
            return "/stop"
        if "admin" in t_lower:
            return "/admin"
        return t

    def setup_bot_commands(self, target_chat_id: Optional[Any] = None) -> bool:
        """Register the bot command menu and menu button with Telegram so the Menu button appears in chat."""
        if not self.token or not self.base_url:
            return False
        commands = [
            {"command": "status", "description": "📊 Live throughput, progress & worker tabs"},
            {"command": "add_channel", "description": "➕ Upfront scrape & queue YouTube channel"},
            {"command": "pause", "description": "⏸️ Pause cluster operations"},
            {"command": "resume", "description": "▶️ Resume cluster operations"},
            {"command": "flush", "description": "💾 Force immediate Parquet shard sync"},
            {"command": "errors", "description": "⚠️ View recent cluster errors"},
            {"command": "stop", "description": "🛑 Graceful cluster shutdown"},
            {"command": "help", "description": "❓ Available commands guide"},
            {"command": "delete_channel", "description": "🗑️ Delete channel catalog & records"},
            {"command": "delete_manifest", "description": "🗑️ Reset completed tracking manifest"},
            {"command": "admin", "description": "👑 Show admin lock status"}
        ]
        cid = target_chat_id or self.admin_chat_id or self.admin_user_id or DEFAULT_ADMIN_ID
        try:
            with httpx.Client(timeout=10.0) as client:
                # 1. Global command registration
                resp = client.post(f"{self.base_url}/setMyCommands", json={"commands": commands})
                ok = resp.status_code == 200 and resp.json().get("ok")

                # 2. Scope for all private chats
                try:
                    client.post(
                        f"{self.base_url}/setMyCommands",
                        json={"commands": commands, "scope": {"type": "all_private_chats"}}
                    )
                except Exception:
                    pass

                # 3. Global chat menu button
                try:
                    client.post(
                        f"{self.base_url}/setChatMenuButton",
                        json={"menu_button": {"type": "commands"}}
                    )
                except Exception:
                    pass

                # 4. Scope and menu button specifically for target user/admin
                if cid:
                    try:
                        c_int = int(cid)
                        client.post(
                            f"{self.base_url}/setMyCommands",
                            json={"commands": commands, "scope": {"type": "chat", "chat_id": c_int}}
                        )
                        client.post(
                            f"{self.base_url}/setChatMenuButton",
                            json={"chat_id": c_int, "menu_button": {"type": "commands"}}
                        )
                    except Exception:
                        pass

                if ok:
                    print("[TELEGRAM BOT] ✅ Bot command menu & menu button successfully registered with Telegram!")
                else:
                    print(f"[TELEGRAM BOT] Notice setting commands: {resp.status_code} {resp.text}")
                return ok
        except Exception as e:
            print(f"[TELEGRAM BOT] Notice during setup_bot_commands: {e}")
            return False


    def _load_admin_identity(self, preconfigured_id: str = ""):
        """Load admin identity from environment, tracking bucket, local file, or default."""
        target_id = preconfigured_id or os.getenv("TELEGRAM_ADMIN_ID") or os.getenv("ADMIN_USER_ID") or os.getenv("TELEGRAM_CHAT_ID")
        if target_id:
            try:
                self.admin_user_id = int(target_id)
                self.admin_chat_id = self.admin_user_id
            except ValueError:
                self.admin_user_id = target_id
                self.admin_chat_id = target_id
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

    @property
    def chat_id(self) -> Optional[Any]:
        return self.admin_chat_id or self.admin_user_id

    # --- PROACTIVE ALERT DISPATCHERS ---

    def send_message(
        self,
        text: str,
        parse_mode: Optional[str] = "Markdown",
        chat_id: Optional[Any] = None,
        reply_markup: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a proactive Telegram message to the authorized admin with automatic markdown fallback."""
        target_chat_id = chat_id or self.admin_chat_id or self.admin_user_id or DEFAULT_ADMIN_ID
        if not target_chat_id:
            print("[TELEGRAM BOT] Notice: Cannot send message, no valid chat_id or admin_id found.")
            return False
        if not self.token or not self.base_url:
            print("[TELEGRAM BOT] Notice: Cannot send message, TELEGRAM_BOT_TOKEN is empty.")
            return False
        try:
            with httpx.Client(timeout=10.0) as client:
                markup = reply_markup if reply_markup is not None else self.get_reply_keyboard()
                payload = {
                    "chat_id": target_chat_id,
                    "text": text
                }
                if markup:
                    payload["reply_markup"] = markup
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                resp = client.post(f"{self.base_url}/sendMessage", json=payload)
                if resp.status_code == 200:
                    return True

                # If Telegram rejected formatting (e.g. 400 Bad Request with bad Markdown), fallback to plain text
                if parse_mode:
                    payload.pop("parse_mode", None)
                    resp_fb = client.post(f"{self.base_url}/sendMessage", json=payload)
                    if resp_fb.status_code == 200:
                        return True
                    resp = resp_fb

                # Diagnose failed message delivery
                try:
                    err_json = resp.json()
                    err_desc = err_json.get("description", "")
                except Exception:
                    err_desc = resp.text

                if "chat not found" in err_desc.lower() or "bot was blocked" in err_desc.lower():
                    bot_name = f"@{self.bot_info.get('username')}" if self.bot_info and self.bot_info.get("username") else "your bot"
                    print(f"[TELEGRAM BOT] ⚠️ Telegram alert to {target_chat_id} failed ({err_desc}). The admin user must open the Telegram app, search for {bot_name}, and send /start to begin receiving messages!")
                elif "unauthorized" in err_desc.lower() or resp.status_code == 401:
                    print(f"[TELEGRAM BOT] ❌ Telegram bot token is unauthorized or invalid! Please check TELEGRAM_BOT_TOKEN in Kaggle Secrets.")
                else:
                    print(f"[TELEGRAM BOT] Warning sending message: {resp.status_code} {err_desc}")
                return False
        except Exception as e:
            print(f"[TELEGRAM BOT] Failed to send message: {e}")
            return False

    def send_alert(self, text_or_title: str, details: str = "", chat_id: Optional[Any] = None) -> bool:
        """Send a general alert to admin with automatic markdown fallback."""
        if details:
            msg = f"⚠️ *{text_or_title}*\n\n{details}"
        else:
            msg = f"⚠️ {text_or_title}"
        return self.send_message(text=msg, chat_id=chat_id)

    def send_shard_alert(self, shard_name: str, num_clips: int, duration_sec: float, total_hours: float, dataset_repo: str = config.DATASET_REPO, chat_id: Optional[Any] = None):
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
        return self.send_message(text=msg, chat_id=chat_id)

    def send_disk_alert(self, used_gb: float, total_gb: float, free_gb: float, percent: float, chat_id: Optional[Any] = None):
        """Proactive alert when node disk approaches threshold."""
        msg = (
            f"🚨 *Disk Guard Warning Alert!*\n\n"
            f"• *Used*: `{used_gb:.2f} GB` / `{total_gb:.2f} GB` ({percent:.1f}%)\n"
            f"• *Free Space*: `{free_gb:.2f} GB`\n"
            f"• *Trigger*: Automatic shard flush executed to keep disk strictly under 1.0 GB quota."
        )
        return self.send_message(text=msg, chat_id=chat_id)

    def send_error_alert(self, worker_id: str, video_id: str, error_msg: str, traceback_str: str = "", chat_id: Optional[Any] = None):
        """Proactive alert for pipeline failures or Cloudflare 502 exhaustion with traceback."""
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "worker_id": worker_id,
            "video_id": video_id,
            "error": str(error_msg)
        }
        self.recent_errors.append(entry)
        tb_block = f"\n\nTraceback:\n`{traceback_str[-350:]}`" if traceback_str else ""
        msg = (
            f"❌ *Cluster Error Notice*\n\n"
            f"• *Node*: `{worker_id}`\n"
            f"• *Video*: `{video_id}`\n"
            f"• *Error*: `{str(error_msg)[:250]}`"
            f"{tb_block}"
        )
        return self.send_message(text=msg, chat_id=chat_id)

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
        elif command in ["/delete_channel", "/del_channel", "/remove_channel"]:
            return self._cmd_delete_channel(args)
        elif command in ["/delete_manifest", "/reset_manifest"]:
            return self._cmd_delete_manifest(args)
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
            f"🗑️ `/delete_channel <url>` — Delete a channel, catalog & its manifest records\n"
            f"🗑️ `/delete_manifest` — Reset/clear tracking manifest records in bucket\n"
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
            live_workers = overview.get("live_workers", [])
            bucket = getattr(self.coordinator, "tracking_bucket", getattr(self.coordinator, "bucket_id", config.TRACKING_BUCKET))

            # Fetch segment_manifest.json if present
            seg_manifest = {"total_videos": 0, "total_segments": 0, "total_hours": 0.0, "shards": []}
            if hasattr(self.coordinator, "api"):
                try:
                    tree = list(self.coordinator.api.list_bucket_tree(bucket_id=bucket))
                    if any(it.path == "segment_manifest.json" for it in tree):
                        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                            sm_p = Path(tmp.name)
                        try:
                            self.coordinator.api.download_bucket_files(bucket_id=bucket, files=[("segment_manifest.json", sm_p)])
                            with open(sm_p, "r", encoding="utf-8") as f:
                                seg_manifest = json.load(f)
                        finally:
                            if sm_p.exists():
                                sm_p.unlink()
                except Exception:
                    pass

            # Classify workers by stage
            ingest_workers = []
            segment_workers = []
            transcribe_workers = []

            for w in live_workers:
                stage = w.get("stage", "")
                w_id = w.get("worker_id", "")
                if stage == "ingest" or w_id.startswith("node-"):
                    ingest_workers.append(w)
                elif stage == "segment" or w_id.startswith("segmenter-"):
                    segment_workers.append(w)
                elif stage == "transcribe" or w_id.startswith("transcriber-"):
                    transcribe_workers.append(w)
                else:
                    transcribe_workers.append(w)

            # Stage 1: Ingest lines
            raw_vids = len(manifest.get("completed", {})) or manifest.get("total_videos", 0)
            in_lines = []
            for w in ingest_workers:
                cur_v = w.get("current_video", {}).get("id") or w.get("current_video", {}).get("title") or "waiting"
                speed = w.get("speed", "")
                sp_str = f" ({speed})" if speed else ""
                in_lines.append(f"  • `{w['worker_id']}`: *{w.get('status','idle')}* | {cur_v}{sp_str}")
            in_text = "\n".join(in_lines) if in_lines else "  • No active ingest workers online"

            # Stage 2: Segment lines
            seg_vids = seg_manifest.get("total_videos", 0)
            seg_clips = seg_manifest.get("total_segments", 0)
            seg_shards = len(seg_manifest.get("shards", []))
            sg_lines = []
            for w in segment_workers:
                cur_v = w.get("current_video", {}).get("video_id") or "idle"
                sg_lines.append(f"  • `{w['worker_id']}`: *{w.get('status','idle')}* | Video: {cur_v}")
            sg_text = "\n".join(sg_lines) if sg_lines else "  • No active segment workers online"

            # Stage 3: Transcribe lines
            trans_clips = manifest.get("total_segments", 0)
            trans_hours = manifest.get("total_hours", 0.0)
            trans_shards = len(manifest.get("shards", []))
            tr_lines = []
            for w in transcribe_workers:
                cur_v = w.get("current_video", {}).get("video_id") or w.get("current_video", {}).get("segment_id") or "active"
                tr_lines.append(f"  • `{w['worker_id']}`: *{w.get('status','idle')}* | Clip: {cur_v}")
            tr_text = "\n".join(tr_lines) if tr_lines else "  • No active transcribe workers online"

            return (
                f"📊 *Tamil ASR Cluster Status & Overview (3 Stages)*\n\n"
                f"• *Active Nodes*: `{len(live_workers)}`\n"
                f"• *Tracking Bucket*: `{bucket}`\n"
                f"• *Audited Corpus*: `{config.DATASET_REPO}`\n\n"
                f"📥 *Stage 1: Raw Ingest (`main.py`)*\n"
                f"• Active Workers: `{len(ingest_workers)}`\n"
                f"• Raw Audios Ingested: `{raw_vids}` videos\n"
                f"{in_text}\n\n"
                f"✂️ *Stage 2: Silence Segmentation (`segment.py`)*\n"
                f"• Active Workers: `{len(segment_workers)}`\n"
                f"• Videos Segmented: `{seg_vids}`\n"
                f"• 10s Clips Built: `{seg_clips}`\n"
                f"• Parquet Shards Uploaded: `{seg_shards}`\n"
                f"{sg_text}\n\n"
                f"🎙️ *Stage 3: Transcription (`transcribe.py`)*\n"
                f"• Active Workers: `{len(transcribe_workers)}`\n"
                f"• Audited Speech: `{trans_hours:.2f} hrs`\n"
                f"• Audited Clips: `{trans_clips}`\n"
                f"• Final Parquet Shards: `{trans_shards}`\n"
                f"{tr_text}"
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

    def _cmd_delete_channel(self, args: List[str]) -> str:
        if not args:
            return "Usage: `/delete_channel <url_or_handle>`\nExample: `/delete_channel @madangowri`"
        if not self.coordinator:
            return "⚠️ Coordinator is not connected."

        target = args[0].strip()
        try:
            if hasattr(self.coordinator, "delete_channel"):
                res = self.coordinator.delete_channel(target)
                ch_id = res.get("channel_id", target)
                ch_name = res.get("channel_name", ch_id)
                cat_del = res.get("catalog_videos_removed", 0)
                man_del = res.get("manifest_videos_removed", 0)
                rem_vids = res.get("remaining_total_videos", 0)
                rem_hrs = res.get("remaining_total_hours", 0.0)
                bucket = getattr(self.coordinator, "tracking_bucket", getattr(self.coordinator, "bucket_id", config.TRACKING_BUCKET))

                return (
                    f"🗑️ *Channel & Manifest Cleaned!*\n\n"
                    f"• *Channel*: `{ch_name}` (`{ch_id}`)\n"
                    f"• *Catalog*: Removed from bucket (`{cat_del}` videos purged)\n"
                    f"• *Manifest*: Removed `{man_del}` completed video records\n"
                    f"• *Remaining Videos*: `{rem_vids}`\n"
                    f"• *Remaining Speech*: `{rem_hrs:.2f} hrs`\n"
                    f"• *Tracking Bucket*: `{bucket}` (Updated)"
                )
            else:
                return "⚠️ Current coordinator does not support delete_channel."
        except Exception as e:
            return f"❌ Error deleting channel `{target}`: {e}"

    def _cmd_delete_manifest(self, args: List[str]) -> str:
        if not self.coordinator:
            return "⚠️ Coordinator is not connected."
        try:
            if hasattr(self.coordinator, "delete_manifest"):
                self.coordinator.delete_manifest()
                bucket = getattr(self.coordinator, "tracking_bucket", getattr(self.coordinator, "bucket_id", config.TRACKING_BUCKET))
                return (
                    f"🗑️ *Manifest Reset Complete!*\n\n"
                    f"• *Manifests Reset*: `manifest.json` & `segment_manifest.json`\n"
                    f"• *Tracking Bucket*: `{bucket}`\n"
                    f"• *Status*: All completed video records cleared. Ready for fresh harvesting!"
                )
            else:
                return "⚠️ Current coordinator does not support delete_manifest."
        except Exception as e:
            return f"❌ Error resetting manifest: {e}"

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

    def check_watchdog_stalls(self, force: bool = False):
        """
        Watchdog check: verifies if any active worker has had no heartbeat for > 10 minutes (600s).
        Proactively alerts Telegram:
        🚨 [STALL ALERT] Worker {worker_id} ({stage}) has been inactive for >10 mins on {video_id}! Check Kaggle session.
        """
        now = time.time()
        if not force and (now - self._last_watchdog_check < 30.0):
            return
        self._last_watchdog_check = now

        if not self.coordinator or not hasattr(self.coordinator, "api"):
            return

        bucket = getattr(self.coordinator, "tracking_bucket", getattr(self.coordinator, "bucket_id", config.TRACKING_BUCKET))
        try:
            tree = list(self.coordinator.api.list_bucket_tree(bucket_id=bucket))
            w_paths = [it.path for it in tree if it.path.startswith("workers/") and it.path.endswith(".json")]
            for w_path in w_paths:
                tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                t_path = Path(tmp.name)
                tmp.close()
                try:
                    self.coordinator.api.download_bucket_files(bucket_id=bucket, files=[(w_path, t_path)])
                    with open(t_path, "r", encoding="utf-8") as f:
                        w_data = json.load(f)
                    w_id = w_data.get("worker_id", "unknown")
                    last_hb = w_data.get("last_heartbeat", 0)
                    status = w_data.get("status", "idle")
                    stage = w_data.get("stage", "general")
                    cur_vid = w_data.get("current_video", {})
                    v_id = cur_vid.get("id") or cur_vid.get("video_id") or "unspecified"

                    diff_sec = now - last_hb
                    if 600 < diff_sec < 2400 and status not in ["idle", "paused", "offline"]:
                        if w_id not in self._stalled_alerted:
                            self._stalled_alerted.add(w_id)
                            mins = int(diff_sec // 60)
                            alert_msg = (
                                f"🚨 *[STALL ALERT]* Worker `{w_id}` ({stage}) has been inactive for >{mins} mins on `{v_id}`!\n"
                                f"• *Status*: `{status}`\n"
                                f"• *Stage*: `{stage}`\n"
                                f"• *Action*: Check Kaggle session or restart node."
                            )
                            self.send_message(alert_msg)
                    elif diff_sec <= 600:
                        self._stalled_alerted.discard(w_id)
                except Exception:
                    pass
                finally:
                    if t_path.exists():
                        t_path.unlink()
        except Exception:
            pass

    def run_polling_loop(self):
        """Continuous polling loop for Telegram Bot with watchdog monitoring."""
        print(f"[TELEGRAM BOT] Polling service started for bot token: {self.token[:10]}...")
        self.running = True
        offset = None

        while self.running:
            # 1. Run watchdog stall check
            self.check_watchdog_stalls()

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

                    if text:
                        norm_cmd = self.normalize_command(text)
                        if norm_cmd.startswith("/") or text.startswith("/"):
                            # Proactively register commands and chat menu button specifically for this chat
                            if self.is_authorized(user_id, chat_id) or self.admin_user_id is None:
                                self.setup_bot_commands(target_chat_id=chat_id)

                            reply = self.handle_command(norm_cmd, user_id=user_id, chat_id=chat_id, username=username)
                            if reply:
                                self.send_message(text=reply, chat_id=chat_id)
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
        if not self.token or not self.base_url:
            print("[TELEGRAM BOT] Notice: Skipping background monitor thread because TELEGRAM_BOT_TOKEN is not configured.")
            return
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
