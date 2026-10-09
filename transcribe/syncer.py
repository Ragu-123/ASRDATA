import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from huggingface_hub import HfApi
import transcribe.config as config
from transcribe.coordinator import TranscriptionCoordinator

class BucketSyncer:
    """
    Uploads audited audio clips, full JSON transcripts, and manifest updates
    to the target Hugging Face Storage Bucket (King758/asr-transcripts-01).
    Purges local files to maintain free disk space on Kaggle.
    """
    def __init__(self, api: HfApi, coordinator: TranscriptionCoordinator):
        self.api = api
        self.coordinator = coordinator
        self.target_bucket = config.TARGET_BUCKET

    def sync_video_transcripts(
        self,
        video_meta: Dict[str, Any],
        audited_segments: List[Dict[str, Any]],
        staging_dir: Path
    ) -> bool:
        video_id = video_meta["video_id"]
        total_dur = video_meta.get("duration_seconds") or sum(s.get("duration", 0) for s in audited_segments)

        print(f"\n========================================================")
        print(f"[SYNCER] Syncing video '{video_id}' ({len(audited_segments)} segments) to '{self.target_bucket}'...")

        # 1. Prepare segment audio uploads
        upload_pairs = []
        clean_segments_meta = []

        for s in audited_segments:
            chunk_path: Path = s.get("audio_path")
            seg_id = s.get("segment_id")
            rem_audio_path = f"segments/{video_id}/{seg_id}.m4a"

            if chunk_path and chunk_path.exists():
                upload_pairs.append((chunk_path, rem_audio_path))

            clean_segments_meta.append({
                "id": seg_id,
                "start": s.get("start"),
                "end": s.get("end"),
                "duration": s.get("duration"),
                "whisper_draft": s.get("draft_text", ""),
                "transcript": s.get("clean_text") or s.get("audited_text", ""),
                "emotion": s.get("emotion", "None"),
                "audio_remote_path": rem_audio_path
            })

        # 2. Prepare full video transcript JSON
        video_transcript_payload = {
            "video_id": video_id,
            "title": video_meta.get("title", ""),
            "channel": video_meta.get("channel", "Unknown"),
            "total_duration": total_dur,
            "segments_count": len(clean_segments_meta),
            "worker_id": self.coordinator.worker_id,
            "transcribed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segments": clean_segments_meta
        }

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(video_transcript_payload, tmp, indent=2, ensure_ascii=False)
            json_tmp_path = Path(tmp.name)

        upload_pairs.append((json_tmp_path, f"transcripts/{video_id}.json"))

        # 3. Batch upload all files in ONE call
        try:
            t0 = time.time()
            self.api.batch_bucket_files(bucket_id=self.target_bucket, add=upload_pairs)
            elapsed = time.time() - t0
            print(f"[SYNCER] Uploaded {len(upload_pairs)} artifacts in {elapsed:.1f}s!")

            # 4. Update target manifest
            manifest = self.coordinator.load_target_manifest(force=True)
            manifest.setdefault("completed_videos", {})
            manifest["completed_videos"][video_id] = {
                "title": video_meta.get("title", ""),
                "duration_seconds": total_dur,
                "segments_count": len(clean_segments_meta),
                "transcribed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "worker": self.coordinator.worker_id
            }
            manifest["total_videos"] = len(manifest["completed_videos"])
            manifest["total_segments"] = sum(v.get("segments_count", 0) for v in manifest["completed_videos"].values())
            manifest["total_hours"] = round(sum(v.get("duration_seconds", 0) for v in manifest["completed_videos"].values()) / 3600.0, 2)

            self.coordinator.save_target_manifest(manifest)
            print(f"[SYNCER] Manifest updated: {manifest['total_videos']} videos | {manifest['total_segments']} segments | {manifest['total_hours']} hrs transcribed.")

            # 5. Local cleanup: Purge temporary sliced audios
            shutil.rmtree(staging_dir / video_id, ignore_errors=True)
            raw_files = list(staging_dir.glob(f"{video_id}.*"))
            for rf in raw_files:
                rf.unlink(missing_ok=True)

            print(f"[SYNCER] Purged local audio staging for {video_id}.")
            print(f"========================================================\n")
            return True

        except Exception as e:
            print(f"[SYNCER] ERROR syncing video {video_id}: {e}")
            return False
        finally:
            if json_tmp_path.exists():
                json_tmp_path.unlink()
