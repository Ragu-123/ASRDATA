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
from transcribe.syncer import BucketSyncer

def print_banner(proxy_url: str, concurrency: int):
    print("\n" + "="*70)
    print("🎙️  TAMIL ASR DISTRIBUTED TRANSCRIPTION ENGINE (STAGE 1 & 2)")
    print("="*70)
    print(f"📌 Worker ID:        {config.WORKER_ID}")
    print(f"📦 Source Bucket:    {config.SOURCE_BUCKET} (Raw YouTube Audios)")
    print(f"🎯 Target Bucket:    {config.TARGET_BUCKET} (Audited ASR Transcripts)")
    print(f"⚡ Concurrency:      {concurrency} parallel requests (Sweet spot: 3 to 5)")
    print(f"🤖 Gemini Model:     {config.MODEL_NAME}")
    print(f"🔗 Gemini Proxy:     {proxy_url}")
    print("="*70 + "\n")

async def run_transcription_pipeline(proxy_url: str, concurrency: int, batch_size: int):
    # 1. Retrieve Hugging Face Token
    hf_token = config.get_hf_token()
    if not hf_token:
        print("[ERROR] HF_TOKEN is not set!", file=sys.stderr)
        print("On Kaggle: Add 'HF_TOKEN' to Notebook Secrets (Add-ons -> Secrets).", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=hf_token)

    # 2. Initialize Coordinator, Segmenter, Auditor, and Syncer
    coordinator = TranscriptionCoordinator(api=api)
    auditor = GeminiAuditor(proxy_url=proxy_url, concurrency=concurrency)
    segmenter = AudioSegmenter(api=api)
    syncer = BucketSyncer(api=api, coordinator=coordinator)

    # 3. Test Proxy Connectivity
    print("[INIT] Testing connection to Gemini Canvas Proxy...")
    connected = await auditor.test_connection()
    if not connected:
        print(f"\n[ERROR] Unable to connect to Gemini Canvas Proxy at '{proxy_url}'.", file=sys.stderr)
        print("Please verify:", file=sys.stderr)
        print("1. Google Chrome has the Gemini Canvas tab open and active.", file=sys.stderr)
        print("2. The canvas-proxy.html shows green 'Proxy Active' status.", file=sys.stderr)
        print("3. The /v1 URL is correct and includes /v1 at the end.\n", file=sys.stderr)
        sys.exit(1)

    print_banner(auditor.proxy_url, concurrency)

    running = True

    def shutdown(signum, frame):
        nonlocal running
        print("\n[SHUTDOWN] Signal received. Releasing lease and shutting down cleanly...")
        coordinator.release_lease()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 4. Main Distributed Ingestion & Auditing Loop
    while running:
        print(f"[COORDINATOR] Leasing next batch of up to {batch_size} videos...")
        batch = coordinator.lease_next_batch(batch_size=batch_size)

        if not batch:
            print("[COORDINATOR] No uncompleted or unleased videos available right now. Waiting 30s...")
            await asyncio.sleep(30)
            continue

        print(f"[COORDINATOR] Successfully leased {len(batch)} video(s): {[v['video_id'] for v in batch]}")

        for vid_meta in batch:
            vid_id = vid_meta["video_id"]
            vid_title = vid_meta.get("title", vid_id)
            source_file = vid_meta.get("file", f"{vid_id}.m4a")

            print(f"\n──────────────────────────────────────────────────────────────────────")
            print(f"🎬 Processing Video: [{vid_id}] {vid_title}")
            print(f"──────────────────────────────────────────────────────────────────────")

            # A. Download raw audio from source bucket
            staging_video_dir = config.STAGING_DIR / vid_id
            staging_video_dir.mkdir(parents=True, exist_ok=True)

            print(f"[STAGE 0] Downloading audio from {config.SOURCE_BUCKET}...")
            local_raw_audio = segmenter.download_source_audio(source_file, config.STAGING_DIR)
            if not local_raw_audio:
                print(f"[WARN] Failed to download {source_file}. Skipping.")
                continue

            # B. Stage 1: Segment into 5-15s clips with faster-whisper VAD & Tamil draft
            print(f"[STAGE 1] Segmenting audio into 5-15s speech chunks with VAD...")
            segments = segmenter.segment_with_whisper_and_vad(local_raw_audio, vid_id, staging_video_dir)

            if not segments:
                print(f"[WARN] No speech segments detected for {vid_id}. Skipping.")
                local_raw_audio.unlink(missing_ok=True)
                continue

            print(f"[STAGE 1] Generated {len(segments)} speech segments. Starting Stage 2 Gemini Audit...")

            # C. Stage 2: Audit with Gemini Canvas Proxy in parallel (3-5 requests)
            t0 = time.time()
            audited_segments = await auditor.audit_batch(segments)
            elapsed = time.time() - t0

            success_count = sum(1 for s in audited_segments if s.get("status") == "ok")
            print(f"[STAGE 2] Audited {len(audited_segments)} segments in {elapsed:.1f}s ({success_count} Gemini OK)!")

            # D. Stage 3: Sync to Target Bucket King758/asr-transcripts-01
            syncer.sync_video_transcripts(vid_meta, audited_segments, config.STAGING_DIR)

        # Release current lease after batch is complete
        coordinator.release_lease()

def main():
    parser = argparse.ArgumentParser(description="Distributed Tamil ASR Transcriber with Gemini Canvas Proxy")
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
