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

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import shutil
import tempfile
import pyarrow.parquet as pq
from huggingface_hub import HfApi

import transcribe.config as config
from transcribe.coordinator import TranscriptionCoordinator
from transcribe.segmenter import AudioSegmenter
from transcribe.auditor import GeminiAuditor
from transcribe.syncer import ParquetDatasetSyncer, BucketSyncer
from transcribe.disk_guard import DiskGuard
from transcribe.telegram_bot import TelegramClusterBot

def print_banner(proxy_url: str, concurrency: int, segmented_dataset: str = config.SEGMENTED_DATASET_REPO, audited_dataset: str = config.DATASET_REPO, buffer_gb: float = config.BUFFER_GB, bot_active: bool = False):
    print("\n" + "="*70)
    print("🎙️  TAMIL ASR DISTRIBUTED TRANSCRIPTION ENGINE (TWO-TIER ARCHITECTURE)")
    print("="*70)
    print(f"📌 Worker ID:          {config.WORKER_ID}")
    print(f"📦 Tier 1 Tracking:    {config.TRACKING_BUCKET} (< 5MB, zero audio)")
    print(f"✂️  Segmented Dataset:  {segmented_dataset} (Input Shards)")
    print(f"🎯 Audited Dataset:    {audited_dataset} (Final Corpus)")
    print(f"📦 Target Shard Size:  {buffer_gb:.1f} GB per Parquet shard (Single commit per shard)")
    print(f"⚡ Concurrency:        {concurrency} parallel requests (Sweet spot: 3 to 5)")
    print(f"🤖 Gemini Model:       {config.MODEL_NAME}")
    print(f"🔗 Gemini Proxy:       {proxy_url}")
    print(f"🤖 Telegram Bot:       {'ONLINE' if bot_active else 'OFFLINE'}")
    print(f"💾 Staging Directory:  {config.STAGING_DIR}")
    print("="*70 + "\n")

async def run_transcription_pipeline(
    proxy_url: str,
    concurrency: int = config.CONCURRENCY,
    batch_size: int = config.BATCH_LEASE_SIZE,
    segmented_dataset: str = config.SEGMENTED_DATASET_REPO,
    audited_dataset: str = config.DATASET_REPO,
    buffer_gb: float = config.BUFFER_GB,
    run_once: bool = False
):
    config.sync_runtime_secrets()
    # 1. Retrieve Hugging Face Token
    hf_token = config.get_hf_token()
    if not hf_token:
        print("[ERROR] HF_TOKEN is not set!", file=sys.stderr)
        print("1. CLI flag: Run with '--hf-token hf_...'", file=sys.stderr)
        print("2. Hardcoded: Set HARDCODED_SECRETS['HF_TOKEN'] in secret_resolver.py", file=sys.stderr)
        print("3. Kaggle Secrets: Add 'HF_TOKEN' into Notebook Secrets (Add-ons -> Secrets)", file=sys.stderr)
        print("4. Locally: Set HF_TOKEN in .env or run 'export HF_TOKEN=...'\n", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=hf_token)

    # 2. Initialize Coordinator
    coordinator = TranscriptionCoordinator(api=api, dataset_repo=audited_dataset)

    # 3. Initialize Telegram Bot and launch background monitor thread
    telegram_bot = None
    if config.TELEGRAM_BOT_TOKEN:
        try:
            telegram_bot = TelegramClusterBot(
                token=config.TELEGRAM_BOT_TOKEN,
                admin_id=config.TELEGRAM_ADMIN_ID or "5959376217",
                coordinator=coordinator
            )
            telegram_bot.start_background_thread()
            env_str = "Kaggle" if config.IS_KAGGLE else "Local"
            telegram_bot.send_message(
                f"🚀 *Worker `{config.WORKER_ID}` started (Transcription / Stage 3) on {env_str}!*\n\n"
                f"• *Audited Dataset*: `{audited_dataset}`\n"
                f"• *Concurrency*: `{concurrency}`\n"
                f"• *Status*: Active and transcribing audio"
            )
            print("[INIT] Telegram cluster monitor bot connected and running.")
        except Exception as e:
            print(f"[INIT] Notice starting Telegram bot: {e}")
    else:
        print("[INIT] ℹ️ TELEGRAM_BOT_TOKEN is not configured. Telegram alerts disabled.")

    # 4. Initialize Segmenter, Auditor, Parquet Syncer, and Disk Guard
    auditor = GeminiAuditor(proxy_url=proxy_url, token=config.PROXY_TOKEN, concurrency=concurrency)
    segmenter = AudioSegmenter(api=api, telegram_bot=telegram_bot)
    syncer = ParquetDatasetSyncer(
        api=api,
        coordinator=coordinator,
        dataset_repo=audited_dataset,
        shard_threshold=config.SHARD_CLIP_THRESHOLD,
        buffer_gb=buffer_gb,
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

    print_banner(
        proxy_url=auditor.proxy_url,
        concurrency=concurrency,
        segmented_dataset=segmented_dataset,
        audited_dataset=audited_dataset,
        buffer_gb=buffer_gb,
        bot_active=(telegram_bot is not None)
    )

    running = True
    paused = False

    def shutdown(signum, frame):
        nonlocal running
        print("\n[SHUTDOWN] Signal received. Flushing pending shards and releasing lease...")
        if telegram_bot:
            try:
                telegram_bot.send_message(f"⏹️ *Worker `{config.WORKER_ID}` stopped (Transcription stage)*")
            except Exception:
                pass
        running = False
        try:
            syncer.flush_shard()
        except Exception:
            pass
        coordinator.release_shard_lease()
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

        # --- MODE 1: Check for un-transcribed segmented shards in the segmented dataset ---
        shard_lease = coordinator.lease_segmented_shard(segmented_repo=segmented_dataset)
        if shard_lease:
            shard_file = shard_lease["shard_file"]
            repo_id = shard_lease["repo_id"]
            print(f"\n========================================================")
            print(f"📦 [SEGMENTED SHARD] Transcribing leased shard: {shard_file} from {repo_id}")
            print(f"========================================================")

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                local_shard_p = Path(tmp.name)
            try:
                # Download shard
                if hasattr(api, "download_bucket_files"):
                    api.download_bucket_files(bucket_id=repo_id, files=[(shard_file, local_shard_p)])
                elif hasattr(api, "storage") and repo_id in getattr(api, "storage", {}):
                    # In mock tests
                    local_shard_p.write_bytes(api.storage[repo_id][shard_file])
                else:
                    from huggingface_hub import hf_hub_download
                    down = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=shard_file, token=config.get_hf_token())
                    shutil.copyfile(down, local_shard_p)

                table = pq.read_table(str(local_shard_p))
                clips_to_audit = []
                for row_idx in range(table.num_rows):
                    row_dict = table.slice(row_idx, 1).to_pylist()[0]
                    audio_field = row_dict.get("audio", {})
                    audio_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) else None
                    clips_to_audit.append({
                        "segment_id": row_dict.get("segment_id", f"clip_{row_idx:04d}"),
                        "video_id": row_dict.get("video_id", "unknown"),
                        "channel_name": row_dict.get("channel_name", "Unknown"),
                        "duration": float(row_dict.get("duration", 0.0)),
                        "audio_bytes": audio_bytes,
                        "audio_format": "m4a",
                        "draft_text": ""
                    })

                print(f"[STAGE 2] Loaded {len(clips_to_audit)} clips from {shard_file}. Transcribing with concurrency {concurrency}...")
                t0 = time.time()
                audited = await auditor.audit_batch(clips_to_audit, desc=f"Shard {Path(shard_file).name}")
                elapsed = time.time() - t0
                success_count = sum(1 for s in audited if s.get("status") == "ok")
                print(f"[STAGE 2] Audited {len(audited)} clips in {elapsed:.1f}s ({success_count} Gemini OK)!")

                syncer.sync_audited_segments(audited, shard_id=shard_file)
                if syncer.buffered_bytes >= syncer.target_buffer_bytes or len(syncer.buffered_rows) >= syncer.shard_threshold:
                    syncer.flush_shard()
                else:
                    cur_mb = syncer.buffered_bytes / (1024 * 1024)
                    target_mb = syncer.target_buffer_bytes / (1024 * 1024)
                    print(f"[TRANSCRIBER] Buffer at {cur_mb:.1f} MB / {target_mb:.0f} MB ({len(syncer.buffered_rows)} clips). Continuing to next shard before uploading...")

                # Mark shard completed in manifest
                manifest = coordinator.load_manifest(force=True)
                manifest.setdefault("completed_segmented_shards", [])
                if shard_file not in manifest["completed_segmented_shards"]:
                    manifest["completed_segmented_shards"].append(shard_file)
                coordinator.save_manifest(manifest)
                print(f"[COORDINATOR] Successfully transcribed and recorded shard {shard_file}!")

            except Exception as e:
                print(f"[COORDINATOR] Error processing shard {shard_file}: {e}")
            finally:
                if local_shard_p.exists():
                    local_shard_p.unlink(missing_ok=True)
                coordinator.release_shard_lease()

            if run_once:
                break
            continue

        # --- MODE 2: Fallback to video batch leasing from catalog ---
        print(f"[COORDINATOR] No pending segmented shards. Leasing next batch of up to {batch_size} videos...")
        batch = coordinator.lease_next_batch(batch_size=batch_size)

        if not batch:
            print("[COORDINATOR] No uncompleted or unleased items available right now. Waiting 20s...")
            coordinator.update_heartbeat(status="idle")
            if run_once:
                break
            await asyncio.sleep(20)
            continue

        print(f"[COORDINATOR] Successfully leased {len(batch)} video(s): {[v.get('video_id') or v.get('id') for v in batch]}")
        coordinator.update_heartbeat(status="processing", active_streams=batch)

        for vid_meta in batch:
            vid_id = vid_meta.get("video_id") or vid_meta.get("id")
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

            syncer.sync_video_transcripts(vid_meta, audited_segments, config.STAGING_DIR)
            disk_guard.check_and_clean(emergency_sync_cb=syncer.flush_shard)

        # Check if buffer has reached target GB or threshold
        if syncer.buffered_bytes >= syncer.target_buffer_bytes or len(syncer.buffered_rows) >= syncer.shard_threshold:
            syncer.flush_shard()
        else:
            cur_mb = syncer.buffered_bytes / (1024 * 1024)
            target_mb = syncer.target_buffer_bytes / (1024 * 1024)
            print(f"[TRANSCRIBER] Buffer at {cur_mb:.1f} MB / {target_mb:.0f} MB ({len(syncer.buffered_rows)} clips). Continuing to next batch...")

        coordinator.release_lease()

        if run_once:
            break

    # Clean exit
    try:
        syncer.flush_shard()
    except Exception:
        pass
    coordinator.release_shard_lease()
    coordinator.release_lease()
    coordinator.deregister_worker()
    if telegram_bot:
        telegram_bot.stop()

def main():
    parser = argparse.ArgumentParser(description="Distributed Tamil ASR Transcriber with Segmented Dataset Pulling")
    parser.add_argument("--url", "--proxy-url", "--proxy_url", dest="proxy_url", type=str, default="", help="Gemini Canvas Proxy /v1 endpoint (e.g. https://...trycloudflare.com/v1)")
    parser.add_argument("--concurrency", type=int, default=config.CONCURRENCY, help="Parallel requests to Gemini (default: 4, recommended: 3-5)")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=config.BATCH_LEASE_SIZE, help="Number of videos to lease per batch (default: 3)")
    parser.add_argument("--segmented-dataset", dest="segmented_dataset", type=str, default=config.SEGMENTED_DATASET_REPO, help=f"Source segmented HF dataset (default: {config.SEGMENTED_DATASET_REPO})")
    parser.add_argument("--dataset-repo", "--audited-dataset", dest="audited_dataset", type=str, default=config.AUDITED_DATASET_REPO, help=f"Target audited HF dataset (default: {config.AUDITED_DATASET_REPO})")
    parser.add_argument("--buffer-gb", "--buffer_gb", "--shard-gb", dest="buffer_gb", type=float, default=config.BUFFER_GB, help=f"Target Parquet shard size in GB (default: {config.BUFFER_GB} GB)")
    parser.add_argument("--once", action="store_true", help="Process a single batch and exit")
    parser.add_argument("--cookies", "--cookie", "--cookie-file", dest="cookies", nargs="*", default=[], help="Path to YouTube cookies.txt file")
    config.add_secret_cli_args(parser)
    parser.add_argument("extra_args", nargs="*", default=[], help=argparse.SUPPRESS)

    args, unknown = parser.parse_known_args()
    config.apply_cli_secrets(args)
    config.sync_runtime_secrets()

    # Process cookies if provided
    if args.cookies:
        raw_cookie_str = " ".join(args.cookies) if isinstance(args.cookies, list) else str(args.cookies)
        config.set_cookie_file(raw_cookie_str)

    leftovers = list(args.extra_args) + list(unknown)
    if leftovers and not args.proxy_url:
        for token in list(leftovers):
            if token.startswith("http://") or token.startswith("https://"):
                args.proxy_url = token
                break

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
        batch_size=args.batch_size,
        segmented_dataset=args.segmented_dataset,
        audited_dataset=args.audited_dataset,
        buffer_gb=args.buffer_gb,
        run_once=args.once
    ))

if __name__ == "__main__":
    main()

