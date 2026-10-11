#!/usr/bin/env python3
"""
Stage 1: Raw YouTube Audio Ingestion Engine (main.py)

Task:
  - Scrapes channel catalog into tracking bucket (catalogs/{channel}.json).
  - Leases batches of videos.
  - Downloads full raw audio (.m4a) from YouTube using aria2c + yt-dlp (Android client spoofing).
  - Uploads audio files to Hugging Face storage/dataset.
  - Updates manifest in tracking bucket.
  - NO SEGMENTATION. NO SILENCE DETECTION.
    (Run 'python segment.py' for Stage 2 silence segmentation).
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi
import config
from coordinator import ClusterCoordinator
from bucket_sync import BucketSync
from worker import IngestionWorker

def print_banner(channel: str = "", batch_size: int = config.BATCH_LEASE_SIZE, concurrency: int = config.CONCURRENT_DOWNLOADS_PER_NODE, buffer_gb: float = config.BUFFER_GB):
    print("\n" + "="*70)
    print("📥  TAMIL ASR STAGE 1: RAW AUDIO INGESTION ENGINE (main.py)")
    print("="*70)
    print(f"📌 Node ID:         {config.NODE_ID}")
    print(f"📦 Target Bucket:   {config.BUCKET_ID}")
    print(f"💾 Staging Dir:     {config.STAGING_DIR}")
    print(f"⚡ Batch Lease:     {batch_size} videos per lease")
    print(f"📦 Buffer Flush:    {buffer_gb:.1f} GB threshold (Single commit upload)")
    print(f"🔥 Concurrency:     {concurrency} parallel streams (aria2c 8x sockets each)")
    if getattr(config, "COOKIE_FILE", None):
        print(f"🍪 YouTube Cookies: ACTIVE ({config.COOKIE_FILE})")
    else:
        print("🍪 YouTube Cookies: Not configured (Using Android/VR player_skip guard)")
    if channel:
        print(f"📺 Target Channel:  {channel}")
    print("🚀 Task:            Download raw audio (.m4a) from YouTube via yt-dlp + aria2c")
    print("✂️  Segmentation:    DISABLED (Run 'python segment.py' for Stage 2)")
    print("🤖 Telegram Bot:    Interactive Menu Bar & Real-time Alerts Active")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Tamil ASR Stage 1: Download raw YouTube audio files (main.py)"
    )
    parser.add_argument(
        "--channel", "-c",
        type=str,
        default=os.getenv("CHANNEL", os.getenv("CHANNEL_URL", "")),
        help="YouTube channel URL, handle, or ID to download from (e.g. @madangowri)"
    )
    parser.add_argument(
        "--batch-size", "--batch_size",
        dest="batch_size",
        type=int,
        default=config.BATCH_LEASE_SIZE,
        help=f"Videos per lease (default: {config.BATCH_LEASE_SIZE})"
    )
    parser.add_argument(
        "--concurrency", "-w", "--workers",
        dest="concurrency",
        type=int,
        default=config.CONCURRENT_DOWNLOADS_PER_NODE,
        help=f"Number of concurrent video download streams (default: {config.CONCURRENT_DOWNLOADS_PER_NODE})"
    )
    parser.add_argument(
        "--cookies", "--cookie", "--cookie-file",
        dest="cookies",
        nargs="*",
        default=[],
        help="Path to YouTube cookies.txt file (e.g. /kaggle/input/.../cookies.txt)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single batch and exit"
    )
    parser.add_argument(
        "--buffer-gb", "--buffer_gb",
        dest="buffer_gb",
        type=float,
        default=config.BUFFER_GB,
        help=f"Buffer size in GB before uploading raw audios in single commit (default: {config.BUFFER_GB} GB)"
    )
    # Backward compatibility flags (no-op)
    parser.add_argument("--stage", "-s", type=str, default="ingest", help=argparse.SUPPRESS)
    parser.add_argument("--no-web", "--cli", dest="no_web", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8000, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--url", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--video-id", "-v", dest="video_id", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("extra_args", nargs="*", default=[], help=argparse.SUPPRESS)

    args, unknown = parser.parse_known_args()

    # Smart fallback for positional or unrecognized arguments (e.g. unquoted paths with spaces/parentheses)
    leftovers = list(args.extra_args) + list(unknown)
    if leftovers:
        # Check if any leftover is channel (@channel or http)
        for token in list(leftovers):
            if token.startswith("@") or "youtube.com" in token or "youtu.be" in token:
                if not args.channel:
                    args.channel = token
                leftovers.remove(token)
            elif token.isdigit() and args.concurrency == config.CONCURRENT_DOWNLOADS_PER_NODE:
                args.concurrency = int(token)
                leftovers.remove(token)

        # Remaining tokens could form a cookie file path (e.g. /kaggle/input/.../cookies (1).txt)
        if leftovers and not args.cookies:
            joined_candidate = " ".join(leftovers).strip().strip("'\"")
            if Path(joined_candidate).exists() or ".txt" in joined_candidate or "cookie" in joined_candidate.lower():
                args.cookies = [joined_candidate]

    # Process cookies flag
    if args.cookies:
        raw_cookie_str = " ".join(args.cookies) if isinstance(args.cookies, list) else str(args.cookies)
        cookie_res = config.set_cookie_file(raw_cookie_str)
        if cookie_res:
            print(f"[BOOT] 🍪 Successfully loaded YouTube cookies from: {cookie_res}")
        else:
            print(f"[BOOT] ⚠️ Cookie file not found at: {raw_cookie_str}", file=sys.stderr)
    elif getattr(config, "COOKIE_FILE", None):
        print(f"[BOOT] 🍪 Auto-detected YouTube cookies from: {config.COOKIE_FILE}")

    # 1. Retrieve Hugging Face Token
    hf_token = config.get_hf_token()
    if not hf_token:
        print("\n[ERROR] HF_TOKEN is not set!", file=sys.stderr)
        print("On Kaggle: Add 'HF_TOKEN' to your Kaggle Notebook Secrets (Add-ons -> Secrets).", file=sys.stderr)
        print("Locally: Run 'export HF_TOKEN=your_token' or set HF_TOKEN=your_token in .env.\n", file=sys.stderr)
        sys.exit(1)

    print(f"[BOOT] Initializing Stage 1 Ingestion Node '{config.NODE_ID}'...")
    api = HfApi(token=hf_token)

    # 2. Setup Coordinator and Bucket Sync (Stage 1)
    coordinator = ClusterCoordinator(api=api, bucket_id=config.BUCKET_ID, node_id=config.NODE_ID)
    bucket_sync = BucketSync(api=api, coordinator=coordinator)

    # 3. Add or prioritize channel if given
    channel_input = args.channel.strip()
    if not channel_input and sys.stdin.isatty():
        try:
            print("\n" + "="*70)
            print("📺 YOUTUBE CHANNEL INGESTION")
            print("="*70)
            val = input("👉 Enter YouTube Channel URL/handle (press Enter for default seed channels): ").strip()
            if val:
                channel_input = val
        except (KeyboardInterrupt, EOFError):
            pass

    if channel_input:
        print(f"[BOOT] Adding and prioritizing target channel: {channel_input}")
        coordinator.add_channel(channel_input)

    # 4. Initialize Telegram Bot
    telegram_bot = None
    if config.TELEGRAM_BOT_TOKEN:
        try:
            from transcribe.telegram_bot import TelegramClusterBot
            telegram_bot = TelegramClusterBot(
                token=config.TELEGRAM_BOT_TOKEN,
                admin_id=config.TELEGRAM_ADMIN_ID or "5959376217",
                coordinator=coordinator
            )
            telegram_bot.start_background_thread()
            env_str = "Kaggle" if config.IS_KAGGLE else "Local Cluster"
            cookie_label = f"Active (`{Path(config.COOKIE_FILE).name}`)" if getattr(config, "COOKIE_FILE", None) else "None"
            telegram_bot.send_message(
                f"🚀 *Worker `{config.NODE_ID}` started (Stage 1: Raw Ingest - `main.py`) on {env_str}!*\n\n"
                f"• *Bucket*: `{config.BUCKET_ID}`\n"
                f"• *Batch Size*: `{args.batch_size}` videos\n"
                f"• *Concurrency*: `{args.concurrency}` parallel streams (Kaggle 4-vCPU optimized)\n"
                f"• *Cookies*: {cookie_label}\n"
                f"• *Stage*: Ingestion only (No segmentation)\n"
                f"• *Status*: Active and streaming downloads"
            )
            print("[BOOT] Telegram cluster monitor bot connected and running.")
        except Exception as e:
            print(f"[BOOT] Notice starting Telegram bot: {e}")
    else:
        print("[BOOT] ℹ️ TELEGRAM_BOT_TOKEN is not configured. Telegram alerts disabled.")

    # 5. Initialize Ingestion Worker
    worker = IngestionWorker(
        coordinator=coordinator,
        bucket_sync=bucket_sync,
        telegram_bot=telegram_bot,
        concurrency=args.concurrency,
        buffer_gb=args.buffer_gb
    )
    if telegram_bot:
        worker.startup_notified = True

    # 6. Signal Handlers for Graceful Shutdown
    def shutdown_handler(signum, frame):
        print("\n[SHUTDOWN] Signal received. Gracefully flushing and releasing lease...")
        if telegram_bot:
            try:
                telegram_bot.send_message(f"⏹️ *Worker `{config.NODE_ID}` stopped (Stage 1: Raw Ingest)*")
            except Exception:
                pass
        worker.stop()
        coordinator.deregister_worker()
        if telegram_bot:
            telegram_bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 7. Start Ingestion Worker
    worker.start()
    print_banner(channel=channel_input, batch_size=args.batch_size, concurrency=args.concurrency, buffer_gb=args.buffer_gb)

    print(f"[BOOT] Running Stage 1 ingestion worker. Downloading audio for channel(s)...")
    try:
        while worker.running:
            time.sleep(2)
            if args.once and worker.session_count > 0:
                break
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        worker.stop()
        coordinator.deregister_worker()
        if telegram_bot:
            telegram_bot.stop()

if __name__ == "__main__":
    main()
