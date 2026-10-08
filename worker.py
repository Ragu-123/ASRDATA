import time
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import yt_dlp
import config
from coordinator import ClusterCoordinator
from bucket_sync import BucketSync

class IngestionWorker:
    """
    Background worker that leases batches of videos, downloads native audio streams,
    tracks real-time telemetry, and triggers automatic bucket flushes.
    """
    def __init__(self, coordinator: ClusterCoordinator, bucket_sync: BucketSync, tunnel_url: str = ""):
        self.coordinator = coordinator
        self.bucket_sync = bucket_sync
        self.tunnel_url = tunnel_url

        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None

        # Real-time Telemetry State
        self.current_video: Dict[str, Any] = {}
        self.batch_info: str = "Idle"
        self.download_speed: str = "0 MB/s"
        self.session_count: int = 0
        self.telemetry_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Local completed buffer awaiting flush
        self.pending_manifest_updates: Dict[str, Any] = {}

    def register_callback(self, cb: Callable[[Dict[str, Any]], None]):
        self.telemetry_callbacks.append(cb)

    def _emit_telemetry(self):
        telemetry = self.get_status()
        for cb in self.telemetry_callbacks:
            try:
                cb(telemetry)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Return full snapshot of this worker's health and telemetry."""
        usage = shutil.disk_usage(config.WORKING_DIR)
        used_gb = round(usage.used / (1024**3), 2)
        total_gb = round(usage.total / (1024**3), 2)
        return {
            "node_id": config.NODE_ID,
            "status": "paused" if self.paused else ("downloading" if self.running else "idle"),
            "tunnel_url": self.tunnel_url,
            "current_video": self.current_video,
            "speed": self.download_speed,
            "batch_info": self.batch_info,
            "disk": {
                "used_gb": used_gb,
                "total_gb": total_gb,
                "percent": round((usage.used / usage.total) * 100, 1),
                "threshold_gb": config.DISK_THRESHOLD_GB
            },
            "session_downloaded": self.session_count,
            "total_harvested_hours": round(self.coordinator.manifest_cache.get("total_hours", 0.0), 2),
            "total_completed_videos": len(self.coordinator.manifest_cache.get("completed", {}))
        }

    def start(self):
        """Start the background ingestion thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def pause(self):
        self.paused = True
        print(f"[WORKER {config.NODE_ID}] Paused.")

    def resume(self):
        self.paused = False
        print(f"[WORKER {config.NODE_ID}] Resumed.")

    def stop(self):
        self.running = False
        print(f"[WORKER {config.NODE_ID}] Stopping...")

    def trigger_flush(self):
        """Manually trigger flush to HF Bucket."""
        if self.pending_manifest_updates or list(config.STAGING_DIR.glob("*.*")):
            self.bucket_sync.flush_staging(self.pending_manifest_updates)
            self.pending_manifest_updates.clear()

    def _run_loop(self):
        print(f"[WORKER {config.NODE_ID}] Ingestion loop started.")
        last_hb = 0.0

        # Heartbeat loop in background
        while self.running:
            # 1. Check for remote commands from dashboard/cluster
            cmd = self.coordinator.check_commands()
            if cmd == "pause":
                self.pause()
            elif cmd == "resume":
                self.resume()
            elif cmd == "flush":
                self.trigger_flush()
            elif cmd == "stop":
                self.stop()
                break

            # 2. Maintain Heartbeat (every 10s)
            now = time.time()
            if now - last_hb >= config.HEARTBEAT_INTERVAL_SEC:
                self.coordinator.update_heartbeat(
                    status="paused" if self.paused else "downloading",
                    current_video=self.current_video,
                    tunnel_url=self.tunnel_url,
                    speed_str=self.download_speed,
                    batch_info=self.batch_info
                )
                last_hb = now
                self._emit_telemetry()

            if self.paused:
                time.sleep(2)
                continue

            # 3. Fetch candidate videos from active channels
            channels = self.coordinator.load_channels()
            candidate_videos = []
            for ch in channels:
                if not ch.get("active", True):
                    continue
                # Scrape flat playlist for candidate IDs
                ch_url = ch["url"]
                try:
                    ydl_opts_meta = {'extract_flat': True, 'quiet': True}
                    with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
                        meta = ydl.extract_info(ch_url, download=False)
                        entries = meta.get("entries", [])
                        for e in entries:
                            if e.get("id"):
                                candidate_videos.append({
                                    "id": e["id"],
                                    "title": e.get("title", ""),
                                    "duration": e.get("duration", 0),
                                    "channel": ch.get("name", "Unknown")
                                })
                except Exception as e:
                    print(f"[WORKER] Error scraping channel {ch_url}: {e}")

            if not candidate_videos:
                print("[WORKER] No candidate channels or videos found. Retrying in 10s...")
                time.sleep(10)
                continue

            # 4. Atomically lease a batch of videos
            batch = self.coordinator.lease_batch(candidate_videos, batch_size=config.BATCH_LEASE_SIZE)
            if not batch:
                print("[WORKER] All current videos leased or completed. Checking again in 15s...")
                self.batch_info = "Waiting for new videos / channel updates"
                time.sleep(15)
                continue

            print(f"\n[WORKER {config.NODE_ID}] Successfully leased batch of {len(batch)} videos!")
            self._process_batch(batch)

        # Cleanup on exit
        self.coordinator.release_batch()
        self.trigger_flush()
        print(f"[WORKER {config.NODE_ID}] Worker exited cleanly.")

    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process and download all videos in the leased batch."""
        total_in_batch = len(batch)
        
        # yt-dlp configuration with progress hook
        def ydl_hook(d):
            if d['status'] == 'downloading':
                p_str = d.get('_percent_str', '0%').replace('%', '').strip()
                try:
                    pct = float(p_str)
                except ValueError:
                    pct = 0.0
                speed_str = d.get('_speed_str', 'N/A')
                eta_str = d.get('_eta_str', 'N/A')
                self.download_speed = speed_str
                self.current_video["progress"] = pct
                self.current_video["speed"] = speed_str
                self.current_video["eta"] = eta_str
                self._emit_telemetry()

        ydl_opts = {
            'format': 'ba[ext=opus]/ba[ext=m4a]/ba',
            'outtmpl': str(config.STAGING_DIR / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'concurrent_fragment_downloads': 4,
            'ignoreerrors': True,
            'progress_hooks': [ydl_hook]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for idx, item in enumerate(batch, 1):
                if not self.running or self.paused:
                    break

                vid_id = item["id"]
                self.batch_info = f"Video {idx}/{total_in_batch}"
                self.current_video = {
                    "id": vid_id,
                    "title": item["title"],
                    "duration_seconds": item.get("duration", 0),
                    "channel": item.get("channel", "Unknown"),
                    "progress": 0.0,
                    "speed": "0 MB/s",
                    "eta": "calculating..."
                }
                self._emit_telemetry()

                t0 = time.time()
                try:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid_id}", download=True)
                    if not info:
                        print(f"[WORKER] Video unavailable/private: {vid_id}")
                        continue

                    dur = info.get("duration") or item.get("duration", 0)
                    ext = info.get("ext", "m4a")
                    filename = f"{vid_id}.{ext}"
                    filepath = config.STAGING_DIR / filename
                    size_mb = filepath.stat().st_size / (1024**2) if filepath.exists() else 0
                    elapsed = time.time() - t0

                    self.session_count += 1
                    self.pending_manifest_updates[vid_id] = {
                        "title": item["title"],
                        "duration_seconds": dur,
                        "file": filename,
                        "size_mb": round(size_mb, 2),
                        "worker": config.NODE_ID,
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    running_hours = (
                        sum(v.get("duration_seconds", 0) for v in self.coordinator.manifest_cache.get("completed", {}).values())
                        + sum(v["duration_seconds"] for v in self.pending_manifest_updates.values())
                    ) / 3600

                    print(f"[{idx}/{total_in_batch}] ✓ [{vid_id}] {item['title'][:40]}... | {size_mb:.1f}MB | {dur/60:.1f}m | {elapsed:.1f}s | Cluster Total: {running_hours:.1f} hrs")

                except Exception as e:
                    print(f"[WORKER] Failed downloading {vid_id}: {e}")

                # Check Disk Threshold after each video
                used_gb = shutil.disk_usage(config.WORKING_DIR).used / (1024**3)
                if used_gb >= config.DISK_THRESHOLD_GB:
                    print(f"[WORKER] Disk reached threshold ({used_gb:.2f} GB). Triggering automatic bucket flush...")
                    self.trigger_flush()

        # Batch finished: flush all completed audios to bucket and release lease
        self.trigger_flush()
        self.coordinator.release_batch()
        self.current_video = {}
        self.download_speed = "0 MB/s"
        self._emit_telemetry()
