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
    Monitors ephemeral disk usage (especially Kaggle 20GB disk limit and local 1.0GB quota),
    proactively purges audio artifacts after syncing, and provides
    an auto-detector that triggers emergency cleanup and Telegram alerts if disk limits approach.
    """
    def __init__(
        self,
        check_dir: Path = config.WORKING_DIR,
        max_used_gb: float = 18.0,
        min_free_gb: float = 2.0,
        max_staging_gb: float = config.MAX_LOCAL_DISK_GB,
        telegram_bot: Any = None
    ):
        self.check_dir = check_dir
        self.max_used_gb = max_used_gb
        self.min_free_gb = min_free_gb
        self.max_staging_gb = max_staging_gb
        self.telegram_bot = telegram_bot

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
        except Exception:
            return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 99.0, "percent_used": 0.0}

    def get_staging_size_gb(self) -> float:
        """Calculate total disk space consumed by local staging audio and temporary shards."""
        total_bytes = 0
        for target_dir in [config.STAGING_DIR, config.SHARDS_DIR]:
            if target_dir.exists():
                for root, _, files in os.walk(target_dir):
                    for f in files:
                        try:
                            total_bytes += (Path(root) / f).stat().st_size
                        except Exception:
                            pass
        return round(total_bytes / (1024**3), 3)

    def print_disk_status(self, prefix: str = "[DISK GUARD]"):
        stats = self.get_disk_stats()
        staging_gb = self.get_staging_size_gb()
        print(f"{prefix} Staging: {staging_gb:.2f} GB / {self.max_staging_gb:.1f} GB limit | System: {stats['used_gb']} GB used ({stats['free_gb']} GB free)")

    def check_and_clean(self, emergency_sync_cb: Optional[Callable[[], None]] = None) -> bool:
        """
        Auto-detector: If system disk usage exceeds 18GB or local staging exceeds 0.85 GB,
        triggers emergency flush to Hugging Face Dataset, purges local files, and alerts Telegram bot.
        """
        stats = self.get_disk_stats()
        staging_gb = self.get_staging_size_gb()

        system_danger = stats["used_gb"] >= self.max_used_gb or stats["free_gb"] <= self.min_free_gb
        staging_danger = staging_gb >= (self.max_staging_gb * 0.85)

        if system_danger or staging_danger:
            print("\n" + "!" * 70)
            if system_danger:
                print(f"🚨 [DISK GUARD] EMERGENCY WARNING: Disk limit approaching 19GB!")
                print(f"🚨 Used: {stats['used_gb']} GB | Free: {stats['free_gb']} GB ({stats['percent_used']}%)")
            else:
                print(f"⚠️ [DISK GUARD] LOCAL STORAGE QUOTA WARNING: Staging at {staging_gb:.2f} GB (Limit: {self.max_staging_gb} GB)!")
            print("!" * 70)

            # Send proactive Telegram alert
            if self.telegram_bot:
                try:
                    self.telegram_bot.send_disk_alert(
                        used_gb=stats["used_gb"],
                        total_gb=stats["total_gb"],
                        free_gb=stats["free_gb"],
                        percent=stats["percent_used"]
                    )
                except Exception:
                    pass

            # 1. Trigger emergency sync/flush if provided
            if emergency_sync_cb:
                try:
                    print("[DISK GUARD] Flushing buffered clips into Parquet shard and syncing...")
                    emergency_sync_cb()
                except Exception as e:
                    print(f"[DISK GUARD] Notice during emergency sync: {e}")

            # 2. Purge staging directories
            if config.STAGING_DIR.exists():
                for item in config.STAGING_DIR.iterdir():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                        else:
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
            new_staging_gb = self.get_staging_size_gb()
            print(f"[DISK GUARD] ✅ Purge complete. Staging freed to {new_staging_gb:.2f} GB | System used: {new_stats['used_gb']} GB.\n")
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
