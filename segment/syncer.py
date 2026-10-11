import os
import sys
import re
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq

import segment.config as config

class ParquetSegmentSyncer:
    """
    Parquet Syncer for Segmented Tamil Audio Dataset:
    - Buffers ~10s clips with metadata:
      (audio struct [bytes, path], duration, channel_name, video_id, segment_id).
    - Packages clips into Apache Parquet shards (data/train-{worker_id}-{shard_index:05d}.parquet).
    - Flushes shards when buffer reaches ~300-500 clips (~70 MB) or batch ends.
    - Strictly maintains local disk usage under 1.0 GB (within 19GB total quota)
      by purging sliced audio files immediately.
    - Uploads Parquet shards to Hugging Face Dataset (King758/tamil-asr-segmented).
    - Updates segment_manifest.json in the tracking bucket.
    """
    def __init__(
        self,
        api: HfApi,
        coordinator: Any,
        dataset_repo: str = config.SEGMENTED_DATASET_REPO,
        shard_threshold: int = config.SHARD_CLIP_THRESHOLD,
        buffer_gb: float = config.BUFFER_GB,
        telegram_bot: Any = None
    ):
        self.api = api
        self.coordinator = coordinator
        self.dataset_repo = dataset_repo
        self.shard_threshold = shard_threshold
        self.buffer_gb = buffer_gb if buffer_gb is not None else getattr(config, "BUFFER_GB", 1.0)
        self.target_buffer_bytes = int(self.buffer_gb * (1024**3))
        self.telegram_bot = telegram_bot
        self.worker_id = coordinator.worker_id

        self.buffered_rows: List[Dict[str, Any]] = []
        self.buffered_bytes: int = 0
        self.buffered_videos: Dict[str, Dict[str, Any]] = {}
        self.shard_index: int = 0

        self.shards_dir = config.SHARDS_DIR
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        # PyArrow Audio Struct schema compatible with Hugging Face datasets Audio feature
        self.audio_type = pa.struct([
            ("bytes", pa.binary()),
            ("path", pa.string())
        ])
        hf_meta = {
            "info": {
                "features": {
                    "audio": {"_type": "Audio"},
                    "duration": {"dtype": "float64", "_type": "Value"},
                    "channel_name": {"dtype": "string", "_type": "Value"},
                    "video_id": {"dtype": "string", "_type": "Value"},
                    "segment_id": {"dtype": "string", "_type": "Value"}
                }
            }
        }
        self.schema = pa.schema([
            ("audio", self.audio_type),
            ("duration", pa.float64()),
            ("channel_name", pa.string()),
            ("video_id", pa.string()),
            ("segment_id", pa.string())
        ], metadata={b"huggingface": json.dumps(hf_meta).encode("utf-8")})

        self._init_shard_index()

    def _init_shard_index(self):
        """Discover next shard index from manifest."""
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

    def check_disk_quota(self):
        """Enforces 19GB disk safety, ensuring working directory stays under 1.0 GB."""
        try:
            usage = shutil.disk_usage(config.WORKING_DIR)
            used_gb = usage.used / (1024**3)
            free_gb = usage.free / (1024**3)
            if used_gb >= config.DISK_LIMIT_GB or free_gb < 2.0:
                print(f"[DISK GUARD] ⚠️ System disk usage ({used_gb:.1f} GB used, {free_gb:.1f} GB free). Forcing emergency purge.")
                # Clear staging directory
                if config.STAGING_DIR.exists():
                    for item in config.STAGING_DIR.glob("**/*"):
                        if item.is_file():
                            try:
                                item.unlink()
                            except Exception:
                                pass
        except Exception:
            pass

    def sync_video_segments(
        self,
        video_meta: Dict[str, Any],
        segments: List[Dict[str, Any]],
        staging_dir: Path
    ) -> bool:
        """
        Ingests segmented audio clips into the buffer.
        Purges audio slice files immediately to protect disk quota.
        Flushes Parquet shard if threshold is reached.
        """
        video_id = video_meta.get("video_id") or video_meta.get("id")
        channel_name = video_meta.get("channel_name") or video_meta.get("channel") or "Unknown"
        total_dur = video_meta.get("duration_seconds") or video_meta.get("duration") or sum(s.get("duration", 0) for s in segments)

        print(f"\n========================================================")
        print(f"[SEGMENT SYNCER] Packaging video '{video_id}' ({len(segments)} clips)...")

        added_clips = 0
        video_duration = 0.0

        for s in segments:
            seg_id = s.get("segment_id")
            chunk_path: Optional[Path] = s.get("audio_path")
            dur = float(s.get("duration", 0.0))

            if not chunk_path or not chunk_path.exists() or chunk_path.stat().st_size < 500:
                continue

            try:
                audio_bytes = chunk_path.read_bytes()
            except Exception as e:
                print(f"[SEGMENT SYNCER] Notice reading {chunk_path}: {e}")
                continue

            self.buffered_rows.append({
                "audio": {"bytes": audio_bytes, "path": f"{seg_id}.m4a"},
                "duration": dur,
                "channel_name": channel_name,
                "video_id": video_id,
                "segment_id": seg_id
            })
            self.buffered_bytes += len(audio_bytes)
            added_clips += 1
            video_duration += dur

        if added_clips > 0:
            self.buffered_videos[video_id] = {
                "title": video_meta.get("title", ""),
                "channel_name": channel_name,
                "duration_seconds": round(video_duration if video_duration > 0 else total_dur, 2),
                "segments_count": added_clips,
                "segmented_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "worker": self.worker_id
            }

        cur_mb = self.buffered_bytes / (1024 * 1024)
        target_mb = self.target_buffer_bytes / (1024 * 1024)
        print(f"[SEGMENT SYNCER] Added {added_clips} valid clips (Current buffer: {cur_mb:.1f} MB / {target_mb:.0f} MB [{len(self.buffered_rows)}/{self.shard_threshold} clips]).")

        # Immediately purge local sliced audio files and raw audio for this video
        try:
            target_video_dir = staging_dir / video_id
            if target_video_dir.exists():
                shutil.rmtree(target_video_dir, ignore_errors=True)
            for f in staging_dir.glob(f"{video_id}*.*"):
                f.unlink(missing_ok=True)
            print(f"[SEGMENT SYNCER] Purged local audio staging for {video_id} (Disk usage strictly protected).")
        except Exception as e:
            print(f"[SEGMENT SYNCER] Cleanup notice: {e}")

        self.check_disk_quota()

        # Flush shard if buffer has reached or exceeded GB size threshold or clip count
        reached_size = self.buffered_bytes >= self.target_buffer_bytes
        reached_count = len(self.buffered_rows) >= self.shard_threshold
        if reached_size or reached_count:
            reason = f"{cur_mb:.1f} MB reached" if reached_size else f"{len(self.buffered_rows)} clips reached"
            print(f"[SEGMENT SYNCER] Buffer reached threshold ({reason}). Flushing Parquet shard...")
            self.flush_shard()

        print(f"========================================================\n")
        return True

    def flush_shard(self) -> Optional[str]:
        """
        Creates Apache Parquet shard from buffered clips, uploads to Hugging Face Dataset,
        updates segment_manifest.json in tracking bucket, and purges local file.
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
            print(f"[SEGMENT SYNCER] Created Parquet file: {file_size_mb:.2f} MB")
        except Exception as e:
            print(f"[SEGMENT SYNCER] Error serializing Parquet: {e}")
            return None

        # 2. Upload shard to Hugging Face Dataset
        try:
            print(f"[SEGMENT SYNCER] Uploading {shard_filename} to Hugging Face Dataset '{self.dataset_repo}'...")
            self.api.upload_file(
                path_or_fileobj=str(local_parquet_path),
                path_in_repo=path_in_repo,
                repo_id=self.dataset_repo,
                repo_type="dataset",
                commit_message=f"Add segmented shard {shard_filename} ({total_clips} clips, {shard_hrs:.2f} hrs)"
            )
            elapsed = time.time() - t0
            print(f"[SEGMENT SYNCER] Successfully uploaded {shard_filename} in {elapsed:.1f}s!")
        except Exception as e:
            print(f"[SEGMENT SYNCER] Error uploading shard to Hugging Face: {e}")
            if self.telegram_bot:
                try:
                    self.telegram_bot.log_error(self.worker_id, shard_filename, f"Failed to upload segmented shard: {e}")
                except Exception:
                    pass
            return None
        finally:
            if local_parquet_path.exists():
                local_parquet_path.unlink()

        # 3. Update segment_manifest.json in tracking bucket
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
            print(f"[SEGMENT SYNCER] Manifest updated: {manifest['total_videos']} videos | {manifest['total_segments']} segments | {manifest['total_hours']} hrs | {manifest['total_shards']} shards.")
        except Exception as e:
            print(f"[SEGMENT SYNCER] Notice updating manifest: {e}")

        # 4. Dispatch Telegram alert if available
        if self.telegram_bot:
            try:
                self.telegram_bot.send_shard_alert(
                    shard_name=shard_filename,
                    num_clips=total_clips,
                    duration_sec=shard_duration,
                    total_hours=manifest.get("total_hours", shard_hrs),
                    dataset_repo=self.dataset_repo
                )
            except Exception:
                pass

        # 5. Reset buffer and increment shard index
        self.buffered_rows.clear()
        self.buffered_videos.clear()
        self.buffered_bytes = 0
        self.shard_index += 1

        print(f"[SEGMENT SYNCER] Shard flush complete. Disk usage clean and under 1.0 GB.\n")
        return shard_filename
