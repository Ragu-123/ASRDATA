#!/usr/bin/env python3
"""
Rescue script: Extracts the 369 valid transcribed audio segments from King758/asr-transcripts-01
(from the 4 successful videos: WLUErZVZQkQ, YNEGWXd_Xec, cR8IVdDD_i4, zmeUTrYtQ-c)
and packages them into a clean Apache Parquet shard (train-00000.parquet)
ready for a Hugging Face Dataset.
"""

import os
import sys
import json
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq

# 4 videos with verified non-empty Tamil transcripts
VALID_VIDEO_IDS = [
    "WLUErZVZQkQ",
    "YNEGWXd_Xec",
    "cR8IVdDD_i4",
    "zmeUTrYtQ-c"
]

SOURCE_BUCKET = "King758/asr-transcripts-01"
OUTPUT_DIR = Path("./rescued_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_OUT = OUTPUT_DIR / "train-00000.parquet"

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    # Try reading from Kaggle secrets if running on Kaggle
    if os.path.exists("/kaggle/working"):
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    return ""

def main():
    token = get_hf_token()
    if not token:
        print("Error: HF_TOKEN is not set. Please set export HF_TOKEN=... or pass it.")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Connecting to bucket '{SOURCE_BUCKET}'...")

    valid_rows = []
    total_duration = 0.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for vid_id in VALID_VIDEO_IDS:
            print(f"\nProcessing valid video: {vid_id}...")
            # 1. Download transcript JSON
            remote_json = f"transcripts/{vid_id}.json"
            local_json = tmp_path / f"{vid_id}.json"

            try:
                api.download_bucket_files(
                    bucket_id=SOURCE_BUCKET,
                    files=[(remote_json, local_json)]
                )
            except Exception as e:
                print(f"  Warning: Failed to download {remote_json}: {e}")
                continue

            if not local_json.exists():
                continue

            with open(local_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = data.get("segments", [])
            print(f"  Found {len(segments)} segments in {vid_id}.json")

            # 2. Download and package each valid segment
            for seg in segments:
                text = (seg.get("transcript") or seg.get("clean_text") or "").strip()
                if not text:
                    continue  # Skip any empty blips

                seg_id = seg.get("id") or seg.get("segment_id")
                dur = float(seg.get("duration", 0.0))
                emotion = seg.get("emotion", "None")

                remote_audio = f"segments/{vid_id}/{seg_id}.m4a"
                local_audio = tmp_path / f"{seg_id}.m4a"

                try:
                    api.download_bucket_files(
                        bucket_id=SOURCE_BUCKET,
                        files=[(remote_audio, local_audio)]
                    )
                except Exception as e:
                    print(f"    Failed to download audio {remote_audio}: {e}")
                    continue

                if not local_audio.exists() or local_audio.stat().st_size < 1000:
                    continue

                audio_bytes = local_audio.read_bytes()
                local_audio.unlink(missing_ok=True)  # Clean immediately

                valid_rows.append({
                    "audio": {"bytes": audio_bytes, "path": f"{seg_id}.m4a"},
                    "sentence": text,
                    "duration": dur,
                    "video_id": vid_id,
                    "segment_id": seg_id,
                    "emotion": emotion
                })
                total_duration += dur

    print(f"\n========================================================")
    print(f"✅ Total Rescued Valid Segments: {len(valid_rows)}")
    print(f"⏱️  Total Rescued Speech Duration: {total_duration/60:.2f} minutes ({total_duration/3600:.2f} hours)")
    print(f"========================================================")

    if not valid_rows:
        print("No valid rows rescued. Exiting.")
        return

    # 3. Create PyArrow Table with Struct for Hugging Face Audio Feature
    audio_type = pa.struct([
        ("bytes", pa.binary()),
        ("path", pa.string())
    ])

    schema = pa.schema([
        ("audio", audio_type),
        ("sentence", pa.string()),
        ("duration", pa.float64()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("emotion", pa.string())
    ])

    table = pa.Table.from_pylist(valid_rows, schema=schema)
    pq.write_table(table, str(PARQUET_OUT), compression="snappy")

    file_size_mb = PARQUET_OUT.stat().st_size / (1024 * 1024)
    print(f"📦 Successfully created Parquet shard: {PARQUET_OUT} ({file_size_mb:.2f} MB)")
    print("Now this file can be pushed directly to your new Hugging Face Dataset repository!")

if __name__ == "__main__":
    main()
