import os
import sys
import time
import signal
import asyncio
import argparse
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi

import transcribe.config as config
from transcribe.coordinator import TranscriptionCoordinator
from transcribe.segmenter import AudioSegmenter
from transcribe.auditor import GeminiAuditor
from transcribe.syncer import ParquetDatasetSyncer, BucketSyncer
from transcribe.disk_guard import DiskGuard
from transcribe.telegram_bot import TelegramClusterBot

def print_banner(proxy_url: str, concurrency: int, bot_active: bool = False):
    print("\n" + "="*70)
    print("🎙️  TAMIL ASR DISTRIBUTED TRANSCRIPTION ENGINE (TWO-TIER ARCHITECTURE)")
    print("="*70)
    print(f"📌 Worker ID:        {config.WORKER_ID}")
    print(f"📦 Tier 1 Tracking:  {config.TRACKING_BUCKET} (< 5MB, zero audio)")
    print(f"🎯 Tier 2 Dataset:   {config.DATASET_REPO} (Apache Parquet Shards)")
    print(f"⚡ Concurrency:      {concurrency} parallel requests (Sweet spot: 3 to 5)")
    print(f"🤖 Gemini Model:     {config.MODEL_NAME}")
    print(f"🔗 Gemini Proxy:     {proxy_url}")
    print(f"🤖 Telegram Bot:     {'ONLINE (@crawler001bot)' if bot_active else 'OFFLINE'}")
    print(f"💾 Local Disk Quota: Strictly under {config.MAX_LOCAL_DISK_GB} GB")
    print("="*70 + "\n")

async def run_transcription_pipeline(proxy_url: str, concurrency: int, batch_size: int):
    # 1. Retrieve Hugging Face Token
    hf_token = config.get_hf_token()
    if not hf_token:
        print("[ERROR] HF_TOKEN is not set!", file=sys.stderr)
        print("On Kaggle: Add 'HF_TOKEN' to Notebook Secrets (Add-ons -> Secrets).", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=hf_token)

    # 2. Initialize Coordinator
    coordinator = TranscriptionCoordinator(api=api)

    # 3. Initialize Telegram Bot and launch background monitor thread
    telegram_bot = None
    if config.TELEGRAM_BOT_TOKEN:
        try:
            telegram_bot = TelegramClusterBot(
                token=config.TELEGRAM_BOT_TOKEN,
                coordinator=coordinator
            )
            telegram_bot.start_background_thread()
            print("[INIT] Telegram cluster monitor bot connected and running.")
        except Exception as e:
            print(f"[INIT] Notice starting Telegram bot: {e}")

    # 4. Initialize Segmenter, Auditor, Parquet Syncer, and Disk Guard
    auditor = GeminiAuditor(proxy_url=proxy_url, concurrency=concurrency)
    segmenter = AudioSegmenter(api=api)
    syncer = ParquetDatasetSyncer(
        api=api,
        coordinator=coordinator,
        dataset_repo=config.DATASET_REPO,
        shard_threshold=config.SHARD_CLIP_THRESHOLD,
        telegram_bot=telegram_bot
    )
    if telegram_bot:
        telegram_bot.syncer = syncer

    disk_guard = DiskGuard(
        max_staging_gb=config.MAX_LOCAL_DISK_GB,
        telegram_bot=telegram_bot
    )

    # 5. Test Proxy Connectivity
    print("[INIT] Testing connection to Gemini Canvas Proxy...")
    connected = await auditor.test_connection()
    if not connected:
        print(f"\n[ERROR] Unable to connect to Gemini Canvas Proxy at '{proxy_url}'.", file=sys.stderr)
        print("Please verify:", file=sys.stderr)
        print("1. Google Chrome has the Gemini Canvas tab open and active.", file=sys.stderr)
        print("2. The canvas-proxy.html shows green 'Proxy Active' status.", file=sys.stderr)
        print("3. The /v1 URL is correct and includes /v1 at the end.\n", file=sys.stderr)
        sys.exit(1)

    print_banner(auditor.proxy_url, concurrency, bot_active=(telegram_bot is not None))

    running = True
    paused = False

    def shutdown(signum, frame):
        nonlocal running
        print("\n[SHUTDOWN] Signal received. Flushing pending shards and releasing lease...")
        running = False
        try:
            syncer.flush_shard()
        except Exception:
            pass
        coordinator.release_lease()
        coordinator.deregister_worker()
        if telegram_bot:
            telegram_bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 6. Main Distributed Ingestion & Auditing Loop
    while running:
        # Check for remote commands from Telegram (/pause, /resume, /flush, /stop)
        cmd = coordinator.check_commands()
        if cmd == "pause":
            paused = True
            print("[COORDINATOR] Remote PAUSE command received.")
        elif cmd == "resume":
            paused = False
            print("[COORDINATOR] Remote RESUME command received.")
        elif cmd == "flush":
            print("[COORDINATOR] Remote FLUSH command received.")
            syncer.flush_shard()
        elif cmd == "stop":
            print("[COORDINATOR] Remote STOP command received.")
            break

        if paused:
            coordinator.update_heartbeat(status="paused")
            await asyncio.sleep(5)
            continue

        # Check and enforce strict local disk quota (< 1.0 GB)
        disk_guard.print_disk_status()
        disk_guard.check_and_clean(emergency_sync_cb=syncer.flush_shard)

        # Update worker heartbeat in Tier 1 tracking bucket
        coordinator.update_heartbeat(status="leasing")

        print(f"[COORDINATOR] Leasing next batch of up to {batch_size} videos (< 10ms from catalog)...")
        batch = coordinator.lease_next_batch(batch_size=batch_size)

        if not batch:
            print("[COORDINATOR] No uncompleted or unleased videos available right now. Waiting 25s...")
            coordinator.update_heartbeat(status="idle")
            await asyncio.sleep(25)
            continue

        print(f"[COORDINATOR] Successfully leased {len(batch)} video(s): {[v['video_id'] for v in batch]}")
        coordinator.update_heartbeat(status="processing", active_streams=batch)

        # --- Process Leased Batch Video-by-Video (Maintains local disk strictly under 1.0 GB) ---
        print("\n" + "="*70)
        print(f"📦 [BATCH PROCESSING] Processing {len(batch)} leased video(s)...")
        print("="*70)

        for vid_meta in batch:
            vid_id = vid_meta["video_id"]
            vid_title = vid_meta.get("title", vid_id)
            source_file = vid_meta.get("file", f"{vid_id}.m4a")
            staging_video_dir = config.STAGING_DIR / vid_id
            staging_video_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n🎬 Slicing Video [{vid_id}]: {vid_title[:55]}")
            local_raw_audio = segmenter.fetch_audio(vid_id, config.STAGING_DIR, source_file)
            if not local_raw_audio:
                print(f"  ⚠️ Failed to fetch audio for {vid_id}. Skipping.")
                if telegram_bot:
                    telegram_bot.log_error(config.WORKER_ID, vid_id, f"Audio fetch failed for {vid_id}")
                continue

            segments = segmenter.segment_audio_by_silence(local_raw_audio, vid_id, staging_video_dir)
            # Instantly purge raw source audio file to conserve disk space!
            local_raw_audio.unlink(missing_ok=True)

            if not segments:
                print(f"  ⚠️ No speech segments detected for {vid_id}. Skipping.")
                DiskGuard.cleanup_video_artifacts(vid_id, config.STAGING_DIR)
                continue

            print(f"\n──────────────────────────────────────────────────────────────────────")
            print(f"🎙️  Transcribing Video [{vid_id}]: {vid_title[:55]} ({len(segments)} clips)")
            print(f"──────────────────────────────────────────────────────────────────────")

            t0 = time.time()
            audited_segments = await auditor.audit_batch(
                segments,
                desc=f"[{vid_id}] {vid_title[:25]}"
            )
            elapsed = time.time() - t0

            success_count = sum(1 for s in audited_segments if s.get("status") == "ok")
            print(f"[STAGE 2] Audited {len(audited_segments)} clips in {elapsed:.1f}s ({success_count} Gemini OK)!")

            # Stage 3: Ingest into Parquet Shard Buffer (auto-flushes when threshold reached)
            syncer.sync_video_transcripts(vid_meta, audited_segments, config.STAGING_DIR)

            # Stage 4: Verify disk usage remains strictly under 1.0 GB
            disk_guard.check_and_clean(emergency_sync_cb=syncer.flush_shard)

        # Flush any remaining clips in the shard buffer at the end of the batch
        if syncer.buffered_rows:
            syncer.flush_shard()

        # Release current lease after batch is complete
        coordinator.release_lease()

    # Clean exit
    try:
        syncer.flush_shard()
    except Exception:
        pass
    coordinator.release_lease()
    coordinator.deregister_worker()
    if telegram_bot:
        telegram_bot.stop()

def main():
    parser = argparse.ArgumentParser(description="Distributed Tamil ASR Transcriber with Two-Tier Architecture and Telegram Monitor")
    parser.add_argument("--proxy-url", "--url", "--proxy_url", dest="proxy_url", type=str, default="", help="Gemini Canvas Proxy /v1 endpoint (e.g. https://...trycloudflare.com/v1)")
    parser.add_argument("--concurrency", type=int, default=config.CONCURRENCY, help="Parallel requests to Gemini (default: 4, recommended: 3-5)")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=config.BATCH_LEASE_SIZE, help="Number of videos to lease per batch (default: 3)")

    args = parser.parse_args()

    proxy_url = args.proxy_url or config.DEFAULT_PROXY_URL
    if not proxy_url:
        print("\n" + "="*70)
        print("🔗 GEMINI CANVAS PROXY URL REQUIRED")
        print("="*70)
        try:
            proxy_url = input("👉 Enter Gemini Canvas /v1 URL (e.g. https://...trycloudflare.com/v1): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not proxy_url:
        print("[ERROR] Proxy URL cannot be empty.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_transcription_pipeline(
        proxy_url=proxy_url,
        concurrency=args.concurrency,
        batch_size=args.batch_size
    ))

if __name__ == "__main__":
    main()
