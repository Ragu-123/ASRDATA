import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from huggingface_hub import HfApi
import config
from coordinator import ClusterCoordinator

class BucketSync:
    """
    High-throughput batch uploader to Hugging Face Storage Bucket.
    Manages local staging cleanup and disk quota recovery.
    """
    def __init__(self, api: HfApi, coordinator: ClusterCoordinator):
        self.api = api
        self.coordinator = coordinator
        self.bucket_id = config.BUCKET_ID

    def flush_staging(self, manifest_updates: Dict[str, Any]) -> int:
        """
        Upload all audio files currently in staging to the bucket,
        merge updates into the global manifest, and purge local files.
        Strictly strips video files or drops non-audio containers.
        """
        raw_files = list(config.STAGING_DIR.glob("*.*"))
        audio_files = []
        for f in raw_files:
            if f.suffix.lower() in [".mp4", ".webm", ".mkv"]:
                audio_target = f.with_suffix(".m4a")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(f), "-vn", "-c:a", "copy", str(audio_target)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
                    )
                    f.unlink(missing_ok=True)
                    if audio_target.exists() and audio_target.stat().st_size > 10000:
                        audio_files.append(audio_target)
                except Exception:
                    f.unlink(missing_ok=True)
            elif f.suffix.lower() in [".m4a", ".opus", ".ogg", ".mp3", ".wav", ".aac"]:
                audio_files.append(f)
            else:
                f.unlink(missing_ok=True)

        if not audio_files:
            return 0

        print(f"\n========================================================")
        print(f"[BUCKET_SYNC] Syncing {len(audio_files)} audio files to '{self.bucket_id}'...")
        upload_pairs = [(f, f"audios/{f.name}") for f in audio_files]

        try:
            t0 = time.time()
            self.api.batch_bucket_files(bucket_id=self.bucket_id, add=upload_pairs)
            elapsed = time.time() - t0
            print(f"[BUCKET_SYNC] Uploaded {len(audio_files)} files in {elapsed:.1f}s!")

            # Load latest manifest to prevent overwriting peer nodes
            current_manifest = self.coordinator.load_manifest()
            current_manifest.setdefault("completed", {})
            current_manifest["completed"].update(manifest_updates)
            current_manifest["total_files"] = len(current_manifest["completed"])
            current_manifest["total_hours"] = sum(
                v.get("duration_seconds", 0) for v in current_manifest["completed"].values()
            ) / 3600

            self.coordinator.save_manifest(current_manifest)
            print(f"[BUCKET_SYNC] Global manifest updated: {current_manifest['total_files']} files | {current_manifest['total_hours']:.2f} hrs.")

            # Purge local files immediately to free disk space
            purged_bytes = 0
            for f in audio_files:
                try:
                    purged_bytes += f.stat().st_size
                    f.unlink()
                except Exception:
                    pass

            usage = shutil.disk_usage(config.WORKING_DIR)
            used_gb = usage.used / (1024**3)
            free_gb = usage.free / (1024**3)
            reclaimed_mb = purged_bytes / (1024**2)

            print(f"[BUCKET_SYNC] Purged {reclaimed_mb:.1f} MB. Local disk used: {used_gb:.2f} GB | Free: {free_gb:.2f} GB")
            print(f"========================================================\n")
            return len(audio_files)

        except Exception as e:
            print(f"[BUCKET_SYNC] ERROR during batch upload: {e}")
            return 0
