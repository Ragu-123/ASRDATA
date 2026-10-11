import os
import sys
import time
import signal
import argparse
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi

import segment.config as config
from segment.coordinator import SegmentCoordinator
from segment.segmenter import AudioSegmenter
from segment.syncer import ParquetSegmentSyncer

def print_banner(dataset_repo: str, channel: str = ""):
    print("\n" + "="*70)
    print("✂️  TAMIL ASR DISTRIBUTED AUDIO SEGMENTER & SILENCE DETECTOR")
    print("="*70)
    print(f"📌 Worker ID:         {config.WORKER_ID}")
    print(f"📦 Tracking Bucket:   {config.TRACKING_BUCKET}")
    print(f"🎯 Segmented Dataset: {dataset_repo}")
    print(f"⏱️  Target Clip Size:  {config.TARGET_CHUNK_SEC}s (Window: {config.VAD_MIN_CHUNK_SEC}-{config.VAD_MAX_CHUNK_SEC}s)")
    print(f"💾 Shard Threshold:   {config.SHARD_CLIP_THRESHOLD} clips (~70 MB per Parquet shard)")
    print(f"🛡️  Disk Quota Guard:  Strictly under {config.MAX_LOCAL_DISK_GB} GB (< 19GB limit)")
    if channel:
        print(f"📺 Active Channel:    {channel}")
    print("="*70 + "\n")

def run_segment_pipeline(
    channel_input: str = "",
    video_id_input: str = "",
    dataset_repo: str = config.SEGMENTED_DATASET_REPO,
    batch_size: int = config.BATCH_LEASE_SIZE,
    shard_threshold: int = config.SHARD_CLIP_THRESHOLD,
    run_once: bool = False
):
    hf_token = config.get_hf_token()
    if not hf_token:
        print("[ERROR] HF_TOKEN is not set!", file=sys.stderr)
        print("On Kaggle: Add 'HF_TOKEN' into Notebook Secrets (Add-ons -> Secrets).", file=sys.stderr)
        print("Locally: export HF_TOKEN=your_token or set HF_TOKEN=your_token in .env.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=hf_token)
    coordinator = SegmentCoordinator(api=api, dataset_repo=dataset_repo)

    # Initialize Telegram Bot if token configured
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
            env_str = "Kaggle" if config.IS_KAGGLE else "Local"
            telegram_bot.send_message(
                f"🚀 *Worker `{config.WORKER_ID}` started (Segmentation / Stage 2) on {env_str}!*\n\n"
                f"• *Target Dataset*: `{dataset_repo}`\n"
                f"• *Batch Size*: `{batch_size}` videos\n"
                f"• *Status*: Active and segmenting audio"
            )
            print("[INIT] Telegram cluster monitor bot connected.")
        except Exception as e:
            print(f"[INIT] Notice starting Telegram bot: {e}")
    else:
        print("[INIT] ℹ️ TELEGRAM_BOT_TOKEN is not configured. Telegram alerts disabled.")

    segmenter = AudioSegmenter(api=api, telegram_bot=telegram_bot)
    syncer = ParquetSegmentSyncer(
        api=api,
        coordinator=coordinator,
        dataset_repo=dataset_repo,
        shard_threshold=shard_threshold,
        telegram_bot=telegram_bot
    )

    if channel_input:
        coordinator.add_channel(channel_input)

    print_banner(dataset_repo=dataset_repo, channel=channel_input)

    running = True

    def shutdown(signum, frame):
        nonlocal running
        print("\n[SHUTDOWN] Signal received. Flushing buffered shards and releasing lease...")
        if telegram_bot:
            try:
                telegram_bot.send_message(f"⏹️ *Worker `{config.WORKER_ID}` stopped (Segmentation stage)*")
            except Exception:
                pass
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

    # If single video requested directly:
    if video_id_input:
        print(f"[SEGMENT] Processing single video: {video_id_input}")
        v_meta = {
            "video_id": video_id_input,
            "title": video_id_input,
            "channel_name": channel_input or "Manual",
            "duration_seconds": 0.0
        }
        raw_audio = segmenter.fetch_audio(video_id_input, config.STAGING_DIR)
        if raw_audio:
            staging_video_dir = config.STAGING_DIR / video_id_input
            segments = segmenter.segment_audio_by_silence(raw_audio, video_id_input, v_meta["channel_name"], staging_video_dir)
            raw_audio.unlink(missing_ok=True)
            if segments:
                syncer.sync_video_segments(v_meta, segments, config.STAGING_DIR)
                syncer.flush_shard()
                print(f"[SEGMENT] Successfully segmented video {video_id_input} ({len(segments)} clips)!")
        return

    # Distributed Ingestion Loop
    while running:
        coordinator.update_heartbeat(status="leasing")
        print(f"[COORDINATOR] Leasing next batch of up to {batch_size} videos...")
        batch = coordinator.lease_next_batch(batch_size=batch_size, channel_filter=channel_input)

        if not batch:
            print("[COORDINATOR] No unsegmented or unleased videos available right now. Waiting 20s...")
            coordinator.update_heartbeat(status="idle")
            if run_once:
                break
            time.sleep(20)
            continue

        print(f"[COORDINATOR] Leased {len(batch)} video(s): {[v['video_id'] for v in batch]}")
        coordinator.update_heartbeat(status="segmenting", active_streams=batch)

        for vid_meta in batch:
            if not running:
                break
            vid_id = vid_meta["video_id"]
            vid_title = vid_meta.get("title", vid_id)
            channel_name = vid_meta.get("channel_name", "Unknown")
            staging_video_dir = config.STAGING_DIR / vid_id

            print(f"\n🎬 Slicing Video [{vid_id}]: {vid_title[:55]} ({channel_name})")
            local_raw_audio = segmenter.fetch_audio(vid_id, config.STAGING_DIR)
            if not local_raw_audio:
                print(f"  ⚠️ Failed to fetch audio for {vid_id}. Skipping.")
                continue

            segments = segmenter.segment_audio_by_silence(
                audio_path=local_raw_audio,
                video_id=vid_id,
                channel_name=channel_name,
                output_dir=staging_video_dir
            )
            # Instantly purge raw source audio to strictly maintain disk quota
            local_raw_audio.unlink(missing_ok=True)

            if not segments:
                print(f"  ⚠️ No speech segments detected for {vid_id}. Skipping.")
                continue

            # Ingest into Parquet Shard Buffer (auto-flushes when threshold reached)
            syncer.sync_video_segments(vid_meta, segments, config.STAGING_DIR)

        # Flush any remaining clips at the end of the batch
        if syncer.buffered_rows:
            syncer.flush_shard()

        # Release current lease after batch is complete
        coordinator.release_lease()

        if run_once:
            break

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
    parser = argparse.ArgumentParser(description="Distributed Tamil ASR Silence Segmentation Pipeline")
    parser.add_argument("--channel", "-c", type=str, default="", help="YouTube channel URL, handle, or ID to segment")
    parser.add_argument("--video-id", "-v", dest="video_id", type=str, default="", help="Specific YouTube video ID to segment")
    parser.add_argument("--dataset", type=str, default=config.SEGMENTED_DATASET_REPO, help=f"Target Hugging Face Dataset for segmented shards (default: {config.SEGMENTED_DATASET_REPO})")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=config.BATCH_LEASE_SIZE, help="Number of videos to lease per batch (default: 5)")
    parser.add_argument("--shard-threshold", "--threshold", dest="shard_threshold", type=int, default=config.SHARD_CLIP_THRESHOLD, help="Clips per Parquet shard (default: 400, ~70 MB)")
    parser.add_argument("--once", action="store_true", help="Process a single batch and exit")
    parser.add_argument("--cookies", "--cookie", "--cookie-file", dest="cookies", nargs="*", default=[], help="Path to YouTube cookies.txt file")
    parser.add_argument("extra_args", nargs="*", default=[], help=argparse.SUPPRESS)

    args, unknown = parser.parse_known_args()

    # Process cookies if provided
    if args.cookies:
        raw_cookie_str = " ".join(args.cookies) if isinstance(args.cookies, list) else str(args.cookies)
        config.set_cookie_file(raw_cookie_str)

    leftovers = list(args.extra_args) + list(unknown)
    if leftovers:
        for token in list(leftovers):
            if token.startswith("@") or "youtube.com" in token or "youtu.be" in token:
                if not args.channel:
                    args.channel = token

    run_segment_pipeline(
        channel_input=args.channel,
        video_id_input=args.video_id,
        dataset_repo=args.dataset,
        batch_size=args.batch_size,
        shard_threshold=args.shard_threshold,
        run_once=bool(getattr(args, "once", False))
    )

if __name__ == "__main__":
    main()
