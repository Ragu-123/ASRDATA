import time
import shutil
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
        """
        audio_files = list(config.STAGING_DIR.glob("*.*"))
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
