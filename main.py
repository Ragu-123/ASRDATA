#!/usr/bin/env python3
"""
Main CLI Entry Point for Tamil ASR Two-Tier Engine.

Architecture:
  - Tier 1: Tracking Bucket (King758/asr-tracking-01) for tiny JSON metadata only (< 5 MB).
            ZERO raw audio files ever stored in buckets!
  - Tier 2: Hugging Face Dataset (King758/tamil-asr-segmented / King758/tamil-asr-corpus).
            All audio packaged as compressed Apache Parquet shards (~70 MB per shard).
  - Disk Guard: Local Kaggle disk maintained strictly under 1.0 GB (< 20 GB quota).
"""

import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def print_banner(stage: str, channel: str = "", dataset: str = ""):
    print("\n" + "="*70)
    print("🎙️  TAMIL ASR TWO-TIER ENGINE (DATASET SHARDED ARCHITECTURE)")
    print("="*70)
    print(f"🚀 Execution Stage:   {stage.upper()}")
    print(f"📦 Tracking Bucket:   King758/asr-tracking-01 (Metadata JSON only, zero audio)")
    if dataset:
        print(f"🎯 Target Dataset:    {dataset} (Apache Parquet Shards)")
    if channel:
        print(f"📺 Target Channel:    {channel}")
    print(f"🛡️  Disk Protection:   Strictly < 1.0 GB temp usage (Immediate raw purge)")
    print(f"🤖 Telegram Bot:       Interactive Menu Bar & Cluster Alerts Active")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Tamil ASR Two-Tier Pipeline CLI (Parquet Datasets + Lightweight Bucket Tracking)"
    )
    parser.add_argument(
        "--stage", "-s",
        type=str,
        default=os.getenv("STAGE", "segment"),
        choices=["segment", "transcribe", "catalog"],
        help="Pipeline stage to execute: 'segment' (default: download, silence slice & upload Parquet dataset), 'transcribe' (Canvas Gemini proxy transcription), or 'catalog' (upfront scrape only)"
    )
    parser.add_argument(
        "--channel", "-c",
        type=str,
        default=os.getenv("CHANNEL", os.getenv("CHANNEL_URL", "")),
        help="YouTube channel URL, handle, or ID (e.g. @madangowri or https://www.youtube.com/@madangowri/videos)"
    )
    parser.add_argument(
        "--video-id", "-v",
        dest="video_id",
        type=str,
        default="",
        help="Single YouTube video ID to process directly"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Target Hugging Face Dataset repo (default: King758/tamil-asr-segmented for segment, King758/tamil-asr-corpus for transcribe)"
    )
    parser.add_argument(
        "--batch-size", "--batch_size",
        dest="batch_size",
        type=int,
        default=5,
        help="Number of videos to lease per batch (default: 5)"
    )
    parser.add_argument(
        "--shard-threshold", "--threshold",
        dest="shard_threshold",
        type=int,
        default=400,
        help="Number of ~10s clips per Apache Parquet shard (default: 400, ~70 MB)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("PROXY_URL", os.getenv("CLOUDFLARE_URL", "")),
        help="Cloudflare Canvas Gemini Proxy URL (for transcribe stage, e.g. https://.../v1)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent transcription requests (default: 4)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single batch and exit gracefully"
    )
    # Backward compatibility flags (no-op so legacy commands don't crash)
    parser.add_argument("--no-web", "--cli", dest="no_web", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8000, help=argparse.SUPPRESS)

    args = parser.parse_args()

    stage = args.stage.lower().strip()
    channel_input = args.channel.strip()

    # Prompt interactively only if running in an interactive TTY without channel
    if not channel_input and stage == "segment" and not args.video_id and sys.stdin.isatty():
        try:
            print("\n" + "="*70)
            print("📺 YOUTUBE CHANNEL INPUT")
            print("="*70)
            val = input("👉 Enter YouTube Channel handle or URL (press Enter for default queued channels): ").strip()
            if val:
                channel_input = val
        except (KeyboardInterrupt, EOFError):
            pass

    # STAGE 1 & 2: AUDIO SEGMENTATION (Direct to Parquet Dataset)
    if stage in ["segment", "catalog"]:
        import segment.config as seg_config
        target_dataset = args.dataset or seg_config.SEGMENTED_DATASET_REPO

        print_banner(stage=stage, channel=channel_input, dataset=target_dataset)

        if stage == "catalog":
            # Upfront channel metadata scraping only (< 1 MB JSON to tracking bucket)
            from huggingface_hub import HfApi
            from segment.coordinator import SegmentCoordinator
            hf_token = seg_config.get_hf_token()
            if not hf_token:
                print("[ERROR] HF_TOKEN is not set in secrets or environment!", file=sys.stderr)
                sys.exit(1)
            api = HfApi(token=hf_token)
            coordinator = SegmentCoordinator(api=api, dataset_repo=target_dataset)
            if channel_input:
                coordinator.add_channel(channel_input)
                print(f"✅ Channel '{channel_input}' catalog scraped and queued in tracking bucket successfully!")
            else:
                print("⚠️ Please specify --channel <url> to scrape.")
            return

        from segment.main import run_segment_pipeline
        run_segment_pipeline(
            channel_input=channel_input,
            video_id_input=args.video_id,
            dataset_repo=target_dataset,
            batch_size=args.batch_size,
            shard_threshold=args.shard_threshold,
            run_once=args.once
        )

    # STAGE 3: GEMINI CANVAS TRANSCRIPTION (From Segmented Dataset to Audited Corpus)
    elif stage == "transcribe":
        import asyncio
        import transcribe.config as trans_config
        from transcribe.main import run_transcription_pipeline

        target_dataset = args.dataset or trans_config.DATASET_REPO
        segmented_repo = trans_config.SEGMENTED_DATASET_REPO

        print_banner(stage=stage, dataset=target_dataset)

        proxy_url = args.url.strip()
        if not proxy_url:
            print("[ERROR] --url <Cloudflare_Proxy_URL> is required for transcription stage!", file=sys.stderr)
            print("Example: python main.py --stage transcribe --url https://your-worker.workers.dev/v1", file=sys.stderr)
            sys.exit(1)

        asyncio.run(
            run_transcription_pipeline(
                proxy_url=proxy_url,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                segmented_dataset=segmented_repo,
                audited_dataset=target_dataset,
                run_once=args.once
            )
        )

if __name__ == "__main__":
    main()
