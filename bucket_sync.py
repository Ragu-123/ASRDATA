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
    Batch uploader for Stage 1 Raw Audio.
    Uploads raw audio files to dedicated Raw Audio Storage (Dataset or separate bucket),
    completely separate from the tracking bucket.
    """
    def __init__(self, api: HfApi, coordinator: ClusterCoordinator):
        self.api = api
        self.coordinator = coordinator
        self.raw_storage = config.RAW_DATASET_REPO if getattr(config, "RAW_STORAGE_TYPE", "dataset") == "dataset" else config.RAW_BUCKET
        self.storage_type = getattr(config, "RAW_STORAGE_TYPE", "dataset")

    def flush_staging(self, manifest_updates: Dict[str, Any]) -> int:
        """
        Upload all audio files currently in staging to dedicated raw storage,
        merge updates into the global manifest in tracking bucket, and purge local files.
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
        print(f"[STAGE 1 INGEST] Syncing {len(audio_files)} audio files to separate {self.storage_type} '{self.raw_storage}'...")

        try:
            t0 = time.time()
            if self.storage_type == "dataset":
                self.api.create_repo(repo_id=self.raw_storage, repo_type="dataset", exist_ok=True)
                for f in audio_files:
                    self.api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=f"audios/{f.name}",
                        repo_id=self.raw_storage,
                        repo_type="dataset"
                    )
            else:
                self.api.create_bucket(bucket_id=self.raw_storage, private=False, exist_ok=True)
                upload_pairs = [(f, f"audios/{f.name}") for f in audio_files]
                self.api.batch_bucket_files(bucket_id=self.raw_storage, add=upload_pairs)
            elapsed = time.time() - t0
            print(f"[STAGE 1 INGEST] Uploaded {len(audio_files)} files in {elapsed:.1f}s!")

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
