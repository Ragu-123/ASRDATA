import os
import sys
import gc
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import transcribe.config as config

class DiskGuard:
    """
    Monitors ephemeral disk usage (especially Kaggle 20GB disk limit),
    proactively purges audio artifacts after syncing, and provides
    an auto-detector that triggers emergency cleanup if disk reaches 18-19GB.
    """
    def __init__(self, check_dir: Path = config.WORKING_DIR, max_used_gb: float = 18.0, min_free_gb: float = 2.0):
        self.check_dir = check_dir
        self.max_used_gb = max_used_gb
        self.min_free_gb = min_free_gb

    def get_disk_stats(self) -> Dict[str, float]:
        """Returns total, used, free GB and percent used."""
        try:
            total, used, free = shutil.disk_usage(self.check_dir)
            total_gb = round(total / (1024**3), 2)
            used_gb = round(used / (1024**3), 2)
            free_gb = round(free / (1024**3), 2)
            pct = round((used / total) * 100, 1) if total > 0 else 0.0
            return {
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "percent_used": pct
            }
        except Exception as e:
            return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 99.0, "percent_used": 0.0}

    def print_disk_status(self, prefix: str = "[DISK GUARD]"):
        stats = self.get_disk_stats()
        print(f"{prefix} Disk Status: {stats['used_gb']} GB used / {stats['total_gb']} GB total ({stats['free_gb']} GB free, {stats['percent_used']}%)")

    def check_and_clean(self, emergency_sync_cb: Optional[Callable[[], None]] = None) -> bool:
        """
        Auto-detector: If disk usage exceeds 18GB or free space < 2GB,
        triggers emergency cleanup and optional sync.
        """
        stats = self.get_disk_stats()
        if stats["used_gb"] >= self.max_used_gb or stats["free_gb"] <= self.min_free_gb:
            print("\n" + "!" * 70)
            print(f"🚨 [DISK GUARD] EMERGENCY WARNING: Disk limit approaching 19GB!")
            print(f"🚨 Used: {stats['used_gb']} GB | Free: {stats['free_gb']} GB ({stats['percent_used']}%)")
            print("!" * 70)

            # 1. Trigger emergency sync if provided
            if emergency_sync_cb:
                try:
                    print("[DISK GUARD] Flushing and syncing completed transcripts to Hugging Face bucket...")
                    emergency_sync_cb()
                except Exception as e:
                    print(f"[DISK GUARD] Notice during emergency sync: {e}")

            # 2. Purge staging directories
            purged_bytes = 0
            if config.STAGING_DIR.exists():
                for item in config.STAGING_DIR.iterdir():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                        else:
                            purged_bytes += item.stat().st_size
                            item.unlink(missing_ok=True)
                    except Exception:
                        pass

            # 3. Purge temp audio files in /tmp
            tmp_dir = Path("/tmp")
            if tmp_dir.exists():
                for tmp_file in tmp_dir.glob("tmp*"):
                    try:
                        if tmp_file.is_file():
                            tmp_file.unlink(missing_ok=True)
                    except Exception:
                        pass

            gc.collect()
            new_stats = self.get_disk_stats()
            print(f"[DISK GUARD] ✅ Purge complete. Disk freed! Current: {new_stats['used_gb']} GB used ({new_stats['free_gb']} GB free).\n")
            return True
        return False

    @staticmethod
    def cleanup_video_artifacts(video_id: str, staging_dir: Path):
        """Immediately removes local raw audio and sliced chunks for a video after syncing."""
        # Remove video segments folder
        vid_dir = staging_dir / video_id
        if vid_dir.exists():
            shutil.rmtree(vid_dir, ignore_errors=True)

        # Remove raw source audio file
        for f in staging_dir.glob(f"{video_id}.*"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
