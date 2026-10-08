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

def main():
    # 1. Retrieve Hugging Face Token
    hf_token = config.get_hf_token()
    if not hf_token:
        print("\n[ERROR] HF_TOKEN is not set!", file=sys.stderr)
        print("On Kaggle: Add 'HF_TOKEN' to your Kaggle Notebook Secrets (Add-ons -> Secrets).", file=sys.stderr)
        print("Locally: Run 'export HF_TOKEN=your_token' or 'set HF_TOKEN=your_token'.\n", file=sys.stderr)
        sys.exit(1)

    print(f"[BOOT] Initializing node '{config.NODE_ID}'...")
    api = HfApi(token=hf_token)

    # 2. Setup Coordinator and Bucket Sync
    coordinator = ClusterCoordinator(api=api, bucket_id=config.BUCKET_ID, node_id=config.NODE_ID)
    bucket_sync = BucketSync(api=api, coordinator=coordinator)

    # 3. Start Cloudflare Tunnel
    tunnel = CloudflareTunnel(port=config.PORT)
    tunnel_url = tunnel.start()

    # 4. Initialize Worker
    worker = IngestionWorker(coordinator=coordinator, bucket_sync=bucket_sync, tunnel_url=tunnel_url)

    # 5. Connect references to FastAPI app
    app.coordinator_ref = coordinator
    app.worker_ref = worker

    # 6. Signal Handlers for Graceful Shutdown
    def shutdown_handler(signum, frame):
        print("\n[SHUTDOWN] Signal received. Gracefully flushing and releasing lease...")
        worker.stop()
        coordinator.deregister_worker()
        tunnel.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 7. Start Ingestion Worker
    worker.start()

    # Print Banner
    print_banner(tunnel_url)

    # 8. Start FastAPI Web Server (Runs until interrupted)
    try:
        uvicorn.run(app.app, host="0.0.0.0", port=config.PORT, log_level="warning")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        worker.stop()
        coordinator.deregister_worker()
        tunnel.stop()

if __name__ == "__main__":
    main()
