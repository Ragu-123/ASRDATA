import re
import time
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import yt_dlp
import config
from coordinator import ClusterCoordinator
from bucket_sync import BucketSync

def strip_ansi(s: str) -> str:
    """Remove ANSI color escape sequences from string."""
    if not isinstance(s, str):
        return str(s)
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', s).strip()

class IngestionWorker:
    """
    High-performance ingestion engine:
    - Dedicated daemon heartbeat thread (runs every 6s).
    - Concurrent stream downloader (2 parallel streams per node for clean non-throttled bandwidth).
    - YouTube player_client=['android', 'ios'] to completely eliminate 403 Forbidden errors.
    - Centralized coordinator leasing (sub-second, shared bucket catalog).
    - Accurate per-node session progress tracking.
    """
    def __init__(self, coordinator: ClusterCoordinator, bucket_sync: BucketSync, tunnel_url: str = ""):
        self.coordinator = coordinator
        self.bucket_sync = bucket_sync
        self.tunnel_url = tunnel_url

        self.running = False
        self.paused = False
        self.worker_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None

        # Real-time Telemetry State
        self.active_videos: Dict[str, Dict[str, Any]] = {}
        self.batch_info: str = "Idle"
        self.download_speed: str = "0 MB/s"
        self.session_count: int = 0
        self.session_seconds: float = 0.0
        self.session_hours: float = 0.0
        self.telemetry_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()

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
        """Return full snapshot of this worker's telemetry."""
        usage = shutil.disk_usage(config.WORKING_DIR)
        used_gb = round(usage.used / (1024**3), 2)
        total_gb = round(usage.total / (1024**3), 2)

        with self._lock:
            active_list = [dict(v) for v in self.active_videos.values()]
            current_vid = active_list[0] if active_list else {}

        return {
            "node_id": config.NODE_ID,
            "status": "paused" if self.paused else ("downloading" if self.running and active_list else "idle"),
            "tunnel_url": self.tunnel_url,
            "current_video": current_vid,
            "active_streams": active_list,
            "active_videos_count": len(active_list),
            "speed": self.download_speed,
            "batch_info": self.batch_info,
            "disk": {
                "used_gb": used_gb,
                "total_gb": total_gb,
                "percent": round((usage.used / usage.total) * 100, 1),
                "threshold_gb": config.DISK_THRESHOLD_GB
            },
            "session_downloaded": self.session_count,
            "session_hours": round(self.session_hours, 2),
            "total_harvested_hours": round(self.coordinator.manifest_cache.get("total_hours", 0.0), 2),
            "total_completed_videos": len(self.coordinator.manifest_cache.get("completed", {}))
        }

    def start(self):
        """Start both the ingestion engine and independent heartbeat daemon."""
        if self.running:
            return
        self.running = True

        # 1. Dedicated Heartbeat Daemon (Runs non-stop every 6s)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_daemon, daemon=True)
        self.heartbeat_thread.start()

        # 2. Ingestion Loop
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def _heartbeat_daemon(self):
        """Dedicated daemon thread sending heartbeats every 6 seconds."""
        print(f"[HEARTBEAT] Daemon started for {config.NODE_ID}")
        while self.running:
            try:
                status_str = "paused" if self.paused else ("downloading" if self.active_videos else "idle")
                with self._lock:
                    active_list = [dict(v) for v in self.active_videos.values()]
                
                self.coordinator.update_heartbeat(
                    status=status_str,
                    active_streams=active_list,
                    session_downloaded=self.session_count,
                    session_hours=self.session_hours,
                    tunnel_url=self.tunnel_url,
                    speed_str=self.download_speed,
                    batch_info=self.batch_info
                )
                self._emit_telemetry()
            except Exception as e:
                print(f"[HEARTBEAT] Error: {e}")
            time.sleep(config.HEARTBEAT_INTERVAL_SEC)

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
        """Flush staging files to HF Bucket."""
        with self._lock:
            updates = dict(self.pending_manifest_updates)
            self.pending_manifest_updates.clear()
        if updates or list(config.STAGING_DIR.glob("*.*")):
            self.bucket_sync.flush_staging(updates)

    def _run_loop(self):
        print(f"[WORKER {config.NODE_ID}] Ingestion loop started.")

        while self.running:
            # 1. Check for remote commands
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

            if self.paused:
                time.sleep(2)
                continue

            # 2. Atomically lease next batch from coordinator (sub-second, shared catalog)
            batch = self.coordinator.lease_next_batch(batch_size=config.BATCH_LEASE_SIZE)
            if not batch:
                self.batch_info = "Waiting for videos / peer leases"
                time.sleep(6)
                continue

            print(f"\n[WORKER {config.NODE_ID}] Successfully leased batch of {len(batch)} videos!")
            self._process_batch_concurrent(batch)

        # Cleanup on exit
        self.coordinator.release_batch()
        self.trigger_flush()
        print(f"[WORKER {config.NODE_ID}] Exited cleanly.")

    def _process_batch_concurrent(self, batch: List[Dict[str, Any]]):
        """Download videos in batch with high throughput concurrency."""
        total_in_batch = len(batch)
        completed_in_batch = 0

        def download_single(item):
            vid_id = item["id"]
            t0 = time.time()

            # Clean Progress Hook
            def ydl_hook(d):
                if d['status'] == 'downloading':
                    total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        pct = round((downloaded / total) * 100, 1)
                    else:
                        raw_pct = strip_ansi(d.get('_percent_str', '0%')).replace('%', '')
                        try:
                            pct = float(raw_pct)
                        except ValueError:
                            pct = 0.0

                    # Speed in pure MB/s
                    speed_bytes = d.get('speed') or 0
                    if speed_bytes > 0:
                        speed_mb = round(speed_bytes / (1024 * 1024), 2)
                        speed_clean = f"{speed_mb} MB/s"
                    else:
                        speed_clean = strip_ansi(d.get('_speed_str', '0 MB/s'))

                    # ETA
                    eta_sec = d.get('eta') or 0
                    if eta_sec > 0:
                        m, s = divmod(int(eta_sec), 60)
                        eta_clean = f"{m}m {s}s" if m > 0 else f"{s}s"
                    else:
                        eta_clean = strip_ansi(d.get('_eta_str', '—'))

                    with self._lock:
                        if vid_id in self.active_videos:
                            self.active_videos[vid_id]["progress"] = pct
                            self.active_videos[vid_id]["speed"] = speed_clean
                            self.active_videos[vid_id]["eta"] = eta_clean
                            self.download_speed = speed_clean

            ydl_opts = {
                'format': 'ba[ext=m4a]/ba[ext=opus]/ba',
                'outtmpl': str(config.STAGING_DIR / f"{vid_id}.%(ext)s"),
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 30,
                'retries': 5,
                'fragment_retries': 5,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'ios']
                    }
                },
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                    'preferredquality': '128',
                }],
                'progress_hooks': [ydl_hook]
            }

            with self._lock:
                self.active_videos[vid_id] = {
                    "id": vid_id,
                    "title": item.get("title", ""),
                    "duration_seconds": item.get("duration", 0),
                    "channel": item.get("channel", "Unknown"),
                    "progress": 0.0,
                    "speed": "0 MB/s",
                    "eta": "starting..."
                }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid_id}", download=True)
                    if not info:
                        return None

                    dur = info.get("duration") or item.get("duration", 0)

                    # Look for actual downloaded file on disk
                    matching = list(config.STAGING_DIR.glob(f"{vid_id}.*"))
                    if not matching:
                        return None

                    filepath = matching[0]

                    # STRICT AUDIO GUARD: Convert any video container (.mp4, .webm, .mkv) to .m4a audio
                    if filepath.suffix.lower() in [".mp4", ".webm", ".mkv", ".mov", ".avi"]:
                        audio_dest = config.STAGING_DIR / f"{vid_id}.m4a"
                        try:
                            # 1. Attempt fast stream copy of audio stream
                            ret = subprocess.run(
                                ["ffmpeg", "-y", "-i", str(filepath), "-vn", "-c:a", "copy", str(audio_dest)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=30
                            )
                            # 2. If copy failed or produces invalid file, transcode audio to 128k aac
                            if ret.returncode != 0 or not audio_dest.exists() or audio_dest.stat().st_size < 10000:
                                subprocess.run(
                                    ["ffmpeg", "-y", "-i", str(filepath), "-vn", "-c:a", "aac", "-b:a", "128k", str(audio_dest)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    timeout=60
                                )
                            if audio_dest.exists() and audio_dest.stat().st_size > 10000:
                                filepath.unlink(missing_ok=True)
                                filepath = audio_dest
                        except Exception as conv_e:
                            print(f"[WORKER] Video stripping notice for {vid_id}: {conv_e}")

                    # Reject and delete if still not a valid audio container
                    if filepath.suffix.lower() not in [".m4a", ".opus", ".ogg", ".mp3", ".wav", ".aac"]:
                        filepath.unlink(missing_ok=True)
                        print(f"[WORKER] Dropped non-audio file: {filepath.name}")
                        return None

                    filename = filepath.name
                    size_mb = filepath.stat().st_size / (1024**2)

                    elapsed = time.time() - t0

                    if size_mb < 0.02:
                        print(f"[WORKER] Warning: {vid_id} yielded empty file ({size_mb} MB). Skipping.")
                        return None

                    return {
                        "id": vid_id,
                        "title": item.get("title", ""),
                        "duration_seconds": dur,
                        "file": filename,
                        "size_mb": round(size_mb, 2),
                        "elapsed": elapsed
                    }
            except Exception as e:
                print(f"[WORKER] Failed {vid_id}: {e}")
                return None
            finally:
                with self._lock:
                    if vid_id in self.active_videos:
                        del self.active_videos[vid_id]

        # Execute concurrently across threads (2 streams per node)
        with ThreadPoolExecutor(max_workers=config.CONCURRENT_DOWNLOADS_PER_NODE) as executor:
            future_to_item = {executor.submit(download_single, it): it for it in batch}
            for future in as_completed(future_to_item):
                if not self.running or self.paused:
                    break

                res = future.result()
                completed_in_batch += 1
                self.batch_info = f"Video {completed_in_batch}/{total_in_batch}"

                if res and res.get("size_mb", 0) > 0.02:
                    vid_id = res["id"]
                    dur = res["duration_seconds"]
                    self.session_count += 1
                    self.session_seconds += dur
                    self.session_hours = round(self.session_seconds / 3600.0, 2)

                    with self._lock:
                        self.pending_manifest_updates[vid_id] = {
                            "title": res["title"],
                            "duration_seconds": dur,
                            "file": res["file"],
                            "size_mb": res["size_mb"],
                            "worker": config.NODE_ID,
                            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }

                    print(f"[{completed_in_batch}/{total_in_batch}] ✓ [{vid_id}] {res['title'][:38]}... | {res['size_mb']:.1f}MB | {res['duration_seconds']/60:.1f}m | in {res['elapsed']:.1f}s")

                # Check Disk Threshold after each completed video
                used_gb = shutil.disk_usage(config.WORKING_DIR).used / (1024**3)
                if used_gb >= config.DISK_THRESHOLD_GB:
                    print(f"[WORKER] Disk reached threshold ({used_gb:.2f} GB). Triggering automatic bucket flush...")
                    self.trigger_flush()

        # Batch finished: flush all completed audios to bucket and release lease
        self.trigger_flush()
        self.coordinator.release_batch()
        self.batch_info = "Batch Completed"
        self.download_speed = "0 MB/s"
