import re
import os
import sys
import time
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq

import transcribe.config as config
from transcribe.coordinator import TranscriptionCoordinator
from transcribe.disk_guard import DiskGuard

class ParquetDatasetSyncer:
    """
    Tier 2 Hugging Face Dataset Syncer for Tamil ASR Corpus (King758/tamil-asr-corpus).
    - Buffers audited clips in memory / staging.
    - Packages clips into Apache Parquet shards (data/train-{worker_id}-{shard_index:05d}.parquet)
      using PyArrow audio struct {"bytes": binary, "path": string} for zero-disk streaming.
    - Flushes shards when buffer reaches ~300-500 clips (~70 MB) or batch ends.
    - Strictly maintains local Kaggle disk usage under 1.0 GB by immediately purging sliced audio files.
    - Dispatches proactive Telegram alerts on each shard upload.
    - Updates Tier 1 tracking bucket manifest.json.
    """
    def __init__(
        self,
        api: HfApi,
        coordinator: TranscriptionCoordinator,
        dataset_repo: str = config.DATASET_REPO,
        shard_threshold: int = config.SHARD_CLIP_THRESHOLD,
        telegram_bot: Any = None
    ):
        self.api = api
        self.coordinator = coordinator
        self.dataset_repo = dataset_repo
        self.shard_threshold = shard_threshold
        self.telegram_bot = telegram_bot
        self.worker_id = coordinator.worker_id

        self.buffered_rows: List[Dict[str, Any]] = []
        self.buffered_videos: Dict[str, Dict[str, Any]] = {}
        self.shard_index: int = 0

        # Output directory for temporary parquet shards
        self.shards_dir = config.SHARDS_DIR
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        # PyArrow Audio Struct schema compatible with Hugging Face datasets streaming Audio feature
        self.audio_type = pa.struct([
            ("bytes", pa.binary()),
            ("path", pa.string())
        ])
        hf_meta = {
            "info": {
                "features": {
                    "audio": {"_type": "Audio"},
                    "sentence": {"dtype": "string", "_type": "Value"},
                    "duration": {"dtype": "float64", "_type": "Value"},
                    "video_id": {"dtype": "string", "_type": "Value"},
                    "segment_id": {"dtype": "string", "_type": "Value"},
                    "emotion": {"dtype": "string", "_type": "Value"}
                }
            }
        }
        self.schema = pa.schema([
            ("audio", self.audio_type),
            ("sentence", pa.string()),
            ("duration", pa.float64()),
            ("video_id", pa.string()),
            ("segment_id", pa.string()),
            ("emotion", pa.string())
        ], metadata={b"huggingface": json.dumps(hf_meta).encode("utf-8")})

        # Discover initial shard_index from manifest if any
        self._init_shard_index()

    def _init_shard_index(self):
        """Set shard_index based on maximum existing shard index for this worker."""
        try:
            manifest = self.coordinator.load_manifest()
            max_idx = -1
            pattern = re.compile(rf"train-{re.escape(self.worker_id)}-(\d+)\.parquet")
            for s in manifest.get("shards", []):
                shard_str = s.get("shard", "")
                m = pattern.search(shard_str)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
            if max_idx >= 0:
                self.shard_index = max_idx + 1
        except Exception:
            pass

    def sync_video_transcripts(
        self,
        video_meta: Dict[str, Any],
        audited_segments: List[Dict[str, Any]],
        staging_dir: Path
    ) -> bool:
        """
        Ingests audited segments for a video into the shard buffer.
        Automatically flushes to Parquet shard if threshold is reached.
        Purges local raw audio and slice files to maintain disk under 1.0 GB.
        """
        video_id = video_meta["video_id"]
        total_dur = video_meta.get("duration_seconds") or sum(s.get("duration", 0) for s in audited_segments)

        print(f"\n========================================================")
        print(f"[PARQUET SYNCER] Packaging video '{video_id}' ({len(audited_segments)} clips)...")

        added_clips = 0
        video_duration = 0.0

        for s in audited_segments:
            seg_id = s.get("segment_id")
            clean_text = (s.get("clean_text") or s.get("audited_text") or s.get("transcript") or "").strip()
            chunk_path: Optional[Path] = s.get("audio_path")
            dur = float(s.get("duration", 0.0))
            emotion = s.get("emotion", "None")

            # Validate that transcript is non-empty and audio file exists
            if not clean_text:
                continue

            audio_bytes = s.get("audio_bytes")
            if audio_bytes is None:
                if not chunk_path or not chunk_path.exists() or chunk_path.stat().st_size < 500:
                    continue

                try:
                    audio_bytes = chunk_path.read_bytes()
                except Exception as e:
                    print(f"[PARQUET SYNCER] Notice reading {chunk_path}: {e}")
                    continue

            self.buffered_rows.append({
                "audio": {"bytes": audio_bytes, "path": f"{seg_id}.m4a"},
                "sentence": clean_text,
                "duration": dur,
                "video_id": video_id,
                "segment_id": seg_id,
                "emotion": emotion
            })
            added_clips += 1
            video_duration += dur

        if added_clips > 0:
            self.buffered_videos[video_id] = {
                "title": video_meta.get("title", ""),
                "channel": video_meta.get("channel", "Unknown"),
                "duration_seconds": round(video_duration if video_duration > 0 else total_dur, 2),
                "segments_count": added_clips,
                "transcribed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "worker": self.worker_id
            }

        print(f"[PARQUET SYNCER] Added {added_clips} valid clips (Current buffer: {len(self.buffered_rows)}/{self.shard_threshold}).")

        # Immediately purge local sliced audios for this video
        DiskGuard.cleanup_video_artifacts(video_id, staging_dir)
        print(f"[PARQUET SYNCER] Purged local audio staging for {video_id} (Disk strictly protected).")

        # Flush shard if buffer has reached or exceeded threshold (~300-500 clips)
        if len(self.buffered_rows) >= self.shard_threshold:
            print(f"[PARQUET SYNCER] Buffer reached threshold ({len(self.buffered_rows)} clips). Flushing Parquet shard...")
            self.flush_shard()

        print(f"========================================================\n")
        return True

    def sync_audited_segments(self, audited_segments: List[Dict[str, Any]], shard_id: str = "") -> bool:
        """
        Ingests audited segments directly into the shard buffer (e.g. from segmented Parquet shards).
        """
        added_clips = 0
        for s in audited_segments:
            seg_id = s.get("segment_id")
            clean_text = (s.get("clean_text") or s.get("audited_text") or s.get("transcript") or "").strip()
            dur = float(s.get("duration", 0.0))
            emotion = s.get("emotion", "None")
            vid_id = s.get("video_id", "unknown")

            if not clean_text:
                continue

            audio_bytes = s.get("audio_bytes")
            if audio_bytes is None:
                chunk_path: Optional[Path] = s.get("audio_path")
                if not chunk_path or not chunk_path.exists() or chunk_path.stat().st_size < 500:
                    continue
                try:
                    audio_bytes = chunk_path.read_bytes()
                except Exception:
                    continue

            self.buffered_rows.append({
                "audio": {"bytes": audio_bytes, "path": f"{seg_id}.m4a"},
                "sentence": clean_text,
                "duration": dur,
                "video_id": vid_id,
                "segment_id": seg_id,
                "emotion": emotion
            })
            added_clips += 1

            if vid_id not in self.buffered_videos:
                self.buffered_videos[vid_id] = {
                    "title": s.get("title", vid_id),
                    "channel": s.get("channel_name", "Unknown"),
                    "duration_seconds": round(dur, 2),
                    "segments_count": 1,
                    "transcribed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "worker": self.worker_id
                }
            else:
                self.buffered_videos[vid_id]["segments_count"] += 1
                self.buffered_videos[vid_id]["duration_seconds"] = round(self.buffered_videos[vid_id]["duration_seconds"] + dur, 2)

        print(f"[PARQUET SYNCER] Ingested {added_clips} audited clips from shard '{shard_id}' (Buffer: {len(self.buffered_rows)}/{self.shard_threshold}).")
        if len(self.buffered_rows) >= self.shard_threshold:
            self.flush_shard()
        return True

    def flush_shard(self) -> Optional[str]:
        """
        Creates Apache Parquet shard from buffered clips, uploads to Hugging Face Dataset,
        updates Tier 1 tracking manifest, and cleans local disk.
        """
        if not self.buffered_rows:
            return None

        t0 = time.time()
        shard_filename = f"train-{self.worker_id}-{self.shard_index:05d}.parquet"
        local_parquet_path = self.shards_dir / shard_filename
        path_in_repo = f"data/{shard_filename}"

        total_clips = len(self.buffered_rows)
        shard_duration = sum(r["duration"] for r in self.buffered_rows)
        shard_hrs = shard_duration / 3600.0

        print(f"\n" + "#"*70)
        print(f"📦 [PARQUET SHARD BUILDER] Building {shard_filename} ({total_clips} clips, {shard_hrs:.2f} hrs)...")
        print("#"*70)

        # 1. Write PyArrow Table with Snappy compression
        try:
            table = pa.Table.from_pylist(self.buffered_rows, schema=self.schema)
            pq.write_table(table, str(local_parquet_path), compression="snappy")
            file_size_mb = local_parquet_path.stat().st_size / (1024 * 1024)
            print(f"[PARQUET SYNCER] Created Parquet file: {file_size_mb:.2f} MB")
        except Exception as e:
            print(f"[PARQUET SYNCER] Error serializing Parquet: {e}")
            return None

        # 2. Upload shard to Tier 2 Hugging Face Dataset
        try:
            print(f"[PARQUET SYNCER] Uploading {shard_filename} to Hugging Face Dataset '{self.dataset_repo}'...")
            self.api.upload_file(
                path_or_fileobj=str(local_parquet_path),
                path_in_repo=path_in_repo,
                repo_id=self.dataset_repo,
                repo_type="dataset",
                commit_message=f"Add shard {shard_filename} ({total_clips} clips, {shard_hrs:.2f} hrs)"
            )
            elapsed = time.time() - t0
            print(f"[PARQUET SYNCER] Successfully uploaded {shard_filename} in {elapsed:.1f}s!")
        except Exception as e:
            print(f"[PARQUET SYNCER] Error uploading shard to Hugging Face: {e}")
            if self.telegram_bot:
                self.telegram_bot.log_error(self.worker_id, shard_filename, f"Failed to upload shard: {e}")
            return None
        finally:
            # Purge local parquet file immediately
            if local_parquet_path.exists():
                local_parquet_path.unlink()

        # 3. Update Tier 1 tracking bucket manifest
        try:
            manifest = self.coordinator.load_manifest(force=True)
            manifest.setdefault("completed_videos", {})
            manifest.setdefault("shards", [])

            for vid_id, v_meta in self.buffered_videos.items():
                v_copy = dict(v_meta)
                v_copy["shard"] = path_in_repo
                manifest["completed_videos"][vid_id] = v_copy

            manifest["shards"].append({
                "shard": shard_filename,
                "path": path_in_repo,
                "clips_count": total_clips,
                "duration_seconds": round(shard_duration, 2),
                "duration_hours": round(shard_hrs, 2),
                "worker": self.worker_id,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            manifest["total_videos"] = len(manifest["completed_videos"])
            manifest["total_segments"] = sum(v.get("segments_count", 0) for v in manifest["completed_videos"].values())
            manifest["total_hours"] = round(sum(v.get("duration_seconds", 0) for v in manifest["completed_videos"].values()) / 3600.0, 2)
            manifest["total_shards"] = len(manifest["shards"])

            self.coordinator.save_manifest(manifest)
            total_harvested_hrs = manifest["total_hours"]
            print(f"[PARQUET SYNCER] Manifest updated: {manifest['total_videos']} videos | {manifest['total_segments']} segments | {total_harvested_hrs} hrs | {len(manifest['shards'])} shards.")
        except Exception as e:
            print(f"[PARQUET SYNCER] Notice updating manifest: {e}")
            total_harvested_hrs = shard_hrs

        # 4. Dispatch proactive Telegram alert
        if self.telegram_bot:
            try:
                self.telegram_bot.send_shard_alert(
                    shard_name=shard_filename,
                    num_clips=total_clips,
                    duration_sec=shard_duration,
                    total_hours=total_harvested_hrs,
                    dataset_repo=self.dataset_repo
                )
            except Exception as e:
                print(f"[PARQUET SYNCER] Notice sending Telegram shard alert: {e}")

        # 5. Reset buffer and increment shard index
        self.buffered_rows.clear()
        self.buffered_videos.clear()
        self.shard_index += 1

        print(f"[PARQUET SYNCER] Shard flush complete. Local storage clean and under 1.0 GB.\n")
        return shard_filename

# Backward compatibility alias
BucketSyncer = ParquetDatasetSyncer
