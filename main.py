import os
import sys
import time
import signal
import threading
import uvicorn
from huggingface_hub import HfApi

import config
from coordinator import ClusterCoordinator
from bucket_sync import BucketSync
from worker import IngestionWorker
from tunnel import CloudflareTunnel
import app

def print_banner(tunnel_url: str):
    print("\n" + "="*70)
    print("🎙️  TAMIL ASR AUDIO HARVEST HUB - CLUSTER NODE ONLINE")
    print("="*70)
    print(f"📌 Node ID:         {config.NODE_ID}")
    print(f"📦 Target Bucket:   {config.BUCKET_ID}")
    print(f"💾 Working Dir:     {config.WORKING_DIR}")
    print(f"⚠️  Disk Threshold:  {config.DISK_THRESHOLD_GB} GB (Auto-flush trigger)")
    print(f"⚡ Batch Lease:     {config.BATCH_LEASE_SIZE} videos per lease")
    if tunnel_url:
        print("\n" + "-"*70)
        print("🌐 LIVE CLUSTER MESH DASHBOARD AVAILABLE AT:")
        print(f"👉 {tunnel_url}")
        print("-"*70)
    else:
        print(f"🌐 Local Dashboard: http://127.0.0.1:{config.PORT}")
    print("="*70 + "\n")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Tamil ASR Audio Harvest Ingestion Engine")
    parser.add_argument("--channel", "-c", type=str, default=os.getenv("CHANNEL", os.getenv("CHANNEL_URL", "")), help="YouTube channel URL, handle, or ID to download from (e.g. @madangowri or https://www.youtube.com/@madangowri/videos)")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=config.BATCH_LEASE_SIZE, help=f"Videos per lease (default: {config.BATCH_LEASE_SIZE})")
    parser.add_argument("--port", type=int, default=config.PORT, help=f"Web server dashboard port (default: {config.PORT})")
    parser.add_argument("--no-web", "--cli", dest="no_web", action="store_true", help="Run in headless CLI mode without web server")
    parser.add_argument("--once", action="store_true", help="Process a single batch and exit")

    args = parser.parse_args()

    # 1. Retrieve Hugging Face Token
    hf_token = config.get_hf_token()
    if not hf_token:
        print("\n[ERROR] HF_TOKEN is not set!", file=sys.stderr)
        print("On Kaggle: Add 'HF_TOKEN' to your Kaggle Notebook Secrets (Add-ons -> Secrets).", file=sys.stderr)
        print("Locally: Run 'export HF_TOKEN=your_token' or set HF_TOKEN=your_token in .env.\n", file=sys.stderr)
        sys.exit(1)

    print(f"[BOOT] Initializing node '{config.NODE_ID}'...")
    api = HfApi(token=hf_token)

    # 2. Setup Coordinator and Bucket Sync
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
            telegram_bot = TelegramClusterBot(token=config.TELEGRAM_BOT_TOKEN, coordinator=coordinator)
            telegram_bot.start_background_thread()
            print("[BOOT] Telegram cluster monitor bot connected and running.")
        except Exception as e:
            print(f"[BOOT] Notice starting Telegram bot: {e}")

    # 5. Start Cloudflare Tunnel (if web enabled)
    tunnel = None
    tunnel_url = ""
    if not args.no_web:
        tunnel = CloudflareTunnel(port=args.port)
        tunnel_url = tunnel.start()

    # 6. Initialize Worker
    worker = IngestionWorker(coordinator=coordinator, bucket_sync=bucket_sync, tunnel_url=tunnel_url)

    # Connect references to FastAPI app
    app.coordinator_ref = coordinator
    app.worker_ref = worker

    # Signal Handlers for Graceful Shutdown
    def shutdown_handler(signum, frame):
        print("\n[SHUTDOWN] Signal received. Gracefully flushing and releasing lease...")
        worker.stop()
        coordinator.deregister_worker()
        if tunnel:
            tunnel.stop()
        if telegram_bot:
            telegram_bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 7. Start Ingestion Worker
    worker.start()
    print_banner(tunnel_url)

    if args.no_web:
        print(f"[BOOT] Running in CLI ingestion mode. Downloading videos for channel(s)...")
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
        return

    # 8. Start FastAPI Web Server (Runs until interrupted)
    try:
        uvicorn.run(app.app, host="0.0.0.0", port=args.port, log_level="warning")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        worker.stop()
        coordinator.deregister_worker()
        if tunnel:
            tunnel.stop()
        if telegram_bot:
            telegram_bot.stop()

if __name__ == "__main__":
    main()
