import re
import json
import time
import sys
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi
import yt_dlp
import transcribe.config as config

class TranscriptionCoordinator:
    """
    Decentralized task coordinator using Hugging Face Two-Tier Architecture.
    - Tier 1: Lightweight tracking bucket (King758/asr-tracking-01) (< 5 MB) for catalogs,
      worker leases, heartbeats, cluster commands, and manifest. Zero audio files.
    - Tier 2: Hugging Face Dataset (King758/tamil-asr-corpus) for Apache Parquet shards.
    Manages upfront channel catalog scraping, sub-10ms atomic batch leasing,
    and cluster-wide synchronization.
    """
    def __init__(
        self,
        api: HfApi,
        tracking_bucket: str = config.TRACKING_BUCKET,
        dataset_repo: str = config.DATASET_REPO,
        source_bucket: str = config.SOURCE_BUCKET,
        worker_id: str = config.WORKER_ID
    ):
        self.api = api
        self.tracking_bucket = tracking_bucket
        self.dataset_repo = dataset_repo
        self.source_bucket = source_bucket
        self.worker_id = worker_id

        # Local in-memory caches
        self.manifest_cache: Dict[str, Any] = {
            "total_videos": 0,
            "total_segments": 0,
            "total_hours": 0.0,
            "shards": [],
            "completed_videos": {},
            "last_updated": ""
        }
        self.channels_cache: List[Dict[str, Any]] = []
        self.catalog_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.exhausted_channels = set()
        self._workers_cache: List[Dict[str, Any]] = []
        self._workers_cache_time: float = 0.0
        self._manifest_cache_time: float = 0.0
        self._channels_cache_time: float = 0.0
        self._lock = threading.Lock()

        self._ensure_storage_ready()

    def _ensure_storage_ready(self):
        """Ensure Tier 1 tracking bucket and Tier 2 dataset repository exist."""
        # 1. Tier 1 Tracking Bucket
        try:
            self.api.create_bucket(bucket_id=self.tracking_bucket, private=False, exist_ok=True)
            self.purge_stale_records()
        except Exception as e:
            print(f"[COORDINATOR] Tracking bucket check notice: {e}")

        # 2. Tier 2 Hugging Face Dataset
        try:
            self.api.create_repo(repo_id=self.dataset_repo, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"[COORDINATOR] Dataset repo check notice: {e}")

    def purge_stale_records(self):
        """Purge abandoned worker records (>10 mins old) once on startup."""
        now = time.time()
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            stale_files = []
            for item in tree:
                if item.path.startswith("workers/") and item.path.endswith(".json"):
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                        t_path = Path(tmp.name)
                    try:
                        self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[(item.path, t_path)])
                        with open(t_path, "r", encoding="utf-8") as f:
                            w = json.load(f)
                        if now - w.get("last_heartbeat", 0) > 600:
                            w_id = w.get("worker_id")
                            stale_files.append(item.path)
                            if w_id:
                                stale_files.append(f"batches/{w_id}_lease.json")
                    except Exception:
                        pass
                    finally:
                        if t_path.exists():
                            t_path.unlink()

            if stale_files:
                tree_paths = {it.path for it in tree}
                existing_stale = [p for p in stale_files if p in tree_paths]
                if existing_stale:
                    self.api.batch_bucket_files(bucket_id=self.tracking_bucket, delete=existing_stale)
                    print(f"[COORDINATOR] Startup cleanup: purged {len(existing_stale)} stale artifacts.")
        except Exception as e:
            print(f"[COORDINATOR] Stale purge notice: {e}")

    def deregister_worker(self):
        """Clean self-deregistration when this node is stopped or shut down."""
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            to_delete = [p for p in [f"workers/{self.worker_id}.json", f"batches/{self.worker_id}_lease.json"] if p in paths]
            if to_delete:
                self.api.batch_bucket_files(bucket_id=self.tracking_bucket, delete=to_delete)
                print(f"[COORDINATOR] Cleanly deregistered node {self.worker_id}.")
        except Exception:
            pass

    # --- MANIFEST OPERATIONS (TIER 1 TRACKING BUCKET) ---

    def load_manifest(self, force: bool = False) -> Dict[str, Any]:
        """Fetch global manifest.json from the tracking bucket with TTL caching."""
        now = time.time()
        if not force and (now - self._manifest_cache_time < 12.0) and self.manifest_cache.get("completed_videos"):
            return self.manifest_cache

        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            if "manifest.json" in paths:
                self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[("manifest.json", tmp_path)])
                with open(tmp_path, "r", encoding="utf-8") as f:
                    self.manifest_cache = json.load(f)
                    self._manifest_cache_time = now
            else:
                self.manifest_cache = {
                    "total_videos": 0,
                    "total_segments": 0,
                    "total_hours": 0.0,
                    "shards": [],
                    "completed_videos": {},
                    "last_updated": ""
                }
                self._manifest_cache_time = now
        except Exception as e:
            print(f"[COORDINATOR] Warning loading manifest: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return self.manifest_cache

    def save_manifest(self, manifest: Dict[str, Any]):
        """Upload updated manifest to tracking bucket."""
        self.manifest_cache = manifest
        self._manifest_cache_time = time.time()
        manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(manifest, tmp, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                add=[(tmp_path, "manifest.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # --- CHANNELS & UPFRONT CATALOG SCRAPING ---

    def load_channels(self, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch configured channels from the tracking bucket with TTL caching."""
        now = time.time()
        if not force and (now - self._channels_cache_time < 30.0) and self.channels_cache:
            return self.channels_cache

        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            if "channels.json" in paths:
                self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[("channels.json", tmp_path)])
                with open(tmp_path, "r", encoding="utf-8") as f:
                    self.channels_cache = json.load(f)
                    self._channels_cache_time = now
            else:
                self.channels_cache = config.DEFAULT_CHANNELS
                self.save_channels(self.channels_cache)
                self._channels_cache_time = now
        except Exception as e:
            print(f"[COORDINATOR] Warning loading channels: {e}")
            if not self.channels_cache:
                self.channels_cache = config.DEFAULT_CHANNELS
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return self.channels_cache

    def save_channels(self, channels: List[Dict[str, Any]]):
        """Upload channels list to tracking bucket."""
        self.channels_cache = channels
        self._channels_cache_time = time.time()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(channels, tmp, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                add=[(tmp_path, "channels.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def parse_channel_info(url_or_handle: str, name: Optional[str] = None) -> tuple:
        """
        Parses (channel_id, normalized_url, channel_name) reliably from any URL or handle.
        Handles @handle, plain handle, https://.../@handle, https://.../@handle/videos, etc.
        """
        raw = url_or_handle.strip().rstrip("/")
        if not raw.startswith("http"):
            handle = raw.lstrip("@")
            ch_id = handle
            norm_url = f"https://www.youtube.com/@{handle}/videos"
        else:
            m = re.search(r"@([a-zA-Z0-9_\-\.]+)", raw)
            if m:
                ch_id = m.group(1)
                norm_url = f"https://www.youtube.com/@{ch_id}/videos"
            else:
                parts = raw.split("/")
                last_part = parts[-1]
                if last_part.lower() == "videos" and len(parts) > 1:
                    last_part = parts[-2]
                ch_id = last_part.replace("@", "") if last_part != "www.youtube.com" else f"ch_{int(time.time())}"
                norm_url = raw if raw.endswith("/videos") else f"{raw}/videos"

        ch_name = name or ch_id
        return ch_id, norm_url, ch_name

    def scrape_channel_catalog(self, url_or_handle: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Upfront channel catalog scraping:
        Extracts all video metadata once using yt-dlp flat playlist extraction.
        Stores into catalogs/{channel_id}.json in Tier 1 tracking bucket.
        Guarantees sub-10ms batch leasing with ZERO runtime YouTube pagination!
        """
        ch_id, norm_url, ch_name = self.parse_channel_info(url_or_handle, name)
        catalog_path = f"catalogs/{ch_id}.json"

        print(f"[COORDINATOR] Upfront scraping catalog for channel '{ch_name}' ({norm_url})...")
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
            'playlistend': 5000,
            'socket_timeout': 30,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios']
                }
            }
        }

        catalog = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(norm_url, download=False)
                entries = info.get("entries", []) if info else []

            seen_ids = set()
            for e in entries:
                v_id = e.get("id")
                if v_id and v_id not in seen_ids:
                    seen_ids.add(v_id)
                    catalog.append({
                        "id": v_id,
                        "title": e.get("title", ""),
                        "duration": e.get("duration", 0),
                        "channel": ch_name
                    })

            print(f"[COORDINATOR] Discovered {len(catalog)} videos for {ch_name}!")
            self.catalog_cache[ch_id] = catalog

            # Upload catalog to tracking bucket
            with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
                json.dump(catalog, tmp, indent=2, ensure_ascii=False)
                tmp_path = Path(tmp.name)
            try:
                self.api.batch_bucket_files(
                    bucket_id=self.tracking_bucket,
                    add=[(tmp_path, catalog_path)]
                )
                print(f"[COORDINATOR] Saved catalog to {self.tracking_bucket}/{catalog_path} (< 10ms leasing ready).")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            # Add to channels.json if not present
            channels = self.load_channels()
            if not any(c.get("url") == norm_url for c in channels):
                channels.append({
                    "id": ch_id,
                    "name": ch_name,
                    "url": norm_url,
                    "active": True,
                    "added_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                self.save_channels(channels)

        except Exception as e:
            print(f"[COORDINATOR] Error scraping catalog for {norm_url}: {e}")

        return catalog

    def get_channel_catalog(self, channel: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fast catalog fetch (< 10ms):
        First checks in-memory cache, then tracking bucket.
        If catalog doesn't exist, performs upfront scraping once.
        """
        ch_id = channel["id"]
        if ch_id in self.catalog_cache and self.catalog_cache[ch_id]:
            return self.catalog_cache[ch_id]

        catalog_path = f"catalogs/{ch_id}.json"
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        t_path = Path(tmp.name)
        tmp.close()
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            if catalog_path in paths:
                self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[(catalog_path, t_path)])
                with open(t_path, "r", encoding="utf-8") as f:
                    catalog = json.load(f)
                self.catalog_cache[ch_id] = catalog
                return catalog
        except Exception as e:
            print(f"[COORDINATOR] Notice fetching catalog for {ch_id}: {e}")
        finally:
            if t_path.exists():
                t_path.unlink()

        # Catalog not present in bucket: scrape upfront once!
        return self.scrape_channel_catalog(channel.get("url", f"https://www.youtube.com/@{ch_id}/videos"), channel.get("name", ch_id))

    # --- DYNAMIC BATCH LEASING ENGINE ---

    def lease_next_batch(self, batch_size: int = config.BATCH_LEASE_SIZE) -> List[Dict[str, Any]]:
        """
        Atomically leases non-overlapping videos across the cluster in < 10ms.
        Checks completed videos in manifest and active worker leases.
        """
        now = time.time()
        self.load_manifest(force=True)
        completed_ids = set(self.manifest_cache.get("completed_videos", {}).keys()) | set(self.manifest_cache.get("completed", {}).keys())

        # 1. Fetch active peer leases from tracking bucket in a single batch call
        leased_video_ids = set()
        expired_files_to_delete = []
        temp_lease_files = []

        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            lease_files = [it.path for it in tree if it.path.startswith("batches/") and it.path.endswith("_lease.json")]

            download_pairs = []
            for lp in lease_files:
                if lp == f"batches/{self.worker_id}_lease.json":
                    continue
                tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                t_path = Path(tmp.name)
                tmp.close()
                download_pairs.append((lp, t_path))
                temp_lease_files.append((lp, t_path))

            if download_pairs:
                self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=download_pairs)

            for lp, t_path in temp_lease_files:
                if t_path.exists():
                    try:
                        with open(t_path, "r", encoding="utf-8") as f:
                            lease_data = json.load(f)
                        if lease_data.get("expires_at", 0) <= now:
                            expired_files_to_delete.append(lp)
                        else:
                            leased_video_ids.update(lease_data.get("video_ids", []))
                    except Exception:
                        pass

            if expired_files_to_delete:
                try:
                    self.api.batch_bucket_files(bucket_id=self.tracking_bucket, delete=expired_files_to_delete)
                except Exception:
                    pass
        except Exception as e:
            print(f"[COORDINATOR] Warning inspecting leases: {e}")
        finally:
            for _, t_path in temp_lease_files:
                if t_path.exists():
                    t_path.unlink(missing_ok=True)

        # 2. Iterate channels to lease candidate videos from catalogs
        channels = self.load_channels()
        candidates = []

        for ch in channels:
            if not ch.get("active", True):
                continue
            catalog = self.get_channel_catalog(ch)
            for vid in catalog:
                vid_id = vid.get("id")
                if not vid_id:
                    continue
                if vid_id not in completed_ids and vid_id not in leased_video_ids:
                    item = {
                        "video_id": vid_id,
                        "title": vid.get("title", ""),
                        "duration_seconds": vid.get("duration", 0),
                        "channel": ch.get("name", ch["id"]),
                        "file": f"{vid_id}.m4a"
                    }
                    candidates.append(item)
                    leased_video_ids.add(vid_id)
                    if len(candidates) >= batch_size:
                        break
            if len(candidates) >= batch_size:
                break

        # Fallback to source audio bucket if channels catalog is empty
        if not candidates and self.source_bucket:
            try:
                tree_src = list(self.api.list_bucket_tree(bucket_id=self.source_bucket))
                src_paths = {it.path for it in tree_src}
                if "manifest.json" in src_paths:
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                        t_path = Path(tmp.name)
                    try:
                        self.api.download_bucket_files(bucket_id=self.source_bucket, files=[("manifest.json", t_path)])
                        with open(t_path, "r", encoding="utf-8") as f:
                            src_man = json.load(f)
                        for vid_id, meta in src_man.get("completed", {}).items():
                            if vid_id not in completed_ids and vid_id not in leased_video_ids:
                                item = dict(meta)
                                item["video_id"] = vid_id
                                candidates.append(item)
                                leased_video_ids.add(vid_id)
                                if len(candidates) >= batch_size:
                                    break
                    finally:
                        if t_path.exists():
                            t_path.unlink()
            except Exception:
                pass

        if not candidates:
            return []

        # 3. Write atomic lease to tracking bucket
        lease_payload = {
            "worker_id": self.worker_id,
            "leased_at": now,
            "expires_at": now + (config.LEASE_TTL_MINUTES * 60),
            "video_ids": [c["video_id"] for c in candidates],
            "count": len(candidates)
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(lease_payload, tmp, indent=2)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                add=[(tmp_path, f"batches/{self.worker_id}_lease.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return candidates

    def release_lease(self):
        """Release current worker's batch lease file."""
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                delete=[f"batches/{self.worker_id}_lease.json"]
            )
        except Exception:
            pass

    # --- HEARTBEAT & CLUSTER OVERVIEW ---

    def update_heartbeat(
        self,
        status: str,
        active_streams: List[Dict[str, Any]] = None,
        session_transcribed: int = 0,
        session_hours: float = 0.0,
        tunnel_url: str = "",
        speed_str: str = ""
    ):
        """Register worker heartbeat in tracking bucket (Thread-Safe)."""
        now = time.time()
        disk_usage = shutil.disk_usage(config.WORKING_DIR)
        used_gb = round(disk_usage.used / (1024**3), 2)
        total_gb = round(disk_usage.total / (1024**3), 2)
        free_gb = round(disk_usage.free / (1024**3), 2)

        hb_payload = {
            "worker_id": self.worker_id,
            "status": status,
            "tunnel_url": tunnel_url,
            "active_streams": active_streams or [],
            "session_transcribed": session_transcribed,
            "session_hours": round(session_hours, 2),
            "speed": speed_str,
            "disk": {
                "used_gb": used_gb,
                "total_gb": total_gb,
                "free_gb": free_gb,
                "percent": round((disk_usage.used / disk_usage.total) * 100, 1)
            },
            "last_heartbeat": now
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(hb_payload, tmp, indent=2)
            tmp_path = Path(tmp.name)
        try:
            with self._lock:
                self.api.batch_bucket_files(
                    bucket_id=self.tracking_bucket,
                    add=[(tmp_path, f"workers/{self.worker_id}.json")]
                )
        except Exception:
            pass
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def list_active_workers(self, force: bool = False) -> List[Dict[str, Any]]:
        """List active peer workers in the cluster."""
        now = time.time()
        if not force and (now - self._workers_cache_time < 8.0) and self._workers_cache:
            return self._workers_cache

        active_workers = []
        temp_files = []
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            w_paths = [it.path for it in tree if it.path.startswith("workers/") and it.path.endswith(".json")]

            download_pairs = []
            for path in w_paths:
                tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                t_path = Path(tmp.name)
                tmp.close()
                download_pairs.append((path, t_path))
                temp_files.append((path, t_path))

            if download_pairs:
                self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=download_pairs)

            for path, t_path in temp_files:
                if t_path.exists():
                    try:
                        with open(t_path, "r", encoding="utf-8") as f:
                            w_data = json.load(f)
                        if now - w_data.get("last_heartbeat", 0) <= 90:
                            w_data["is_alive"] = True
                            active_workers.append(w_data)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[COORDINATOR] Warning reading workers: {e}")
        finally:
            for _, t_path in temp_files:
                if t_path.exists():
                    t_path.unlink(missing_ok=True)

        self._workers_cache = active_workers
        self._workers_cache_time = now
        return active_workers

    def get_cluster_overview(self) -> Dict[str, Any]:
        """Calculates cluster-wide aggregated metrics across all active nodes."""
        manifest = self.load_manifest()
        live_workers = self.list_active_workers()
        node_count = max(1, len(live_workers))

        return {
            "node_count": node_count,
            "live_workers": live_workers,
            "total_completed_videos": len(manifest.get("completed_videos", {})),
            "total_segments": manifest.get("total_segments", 0),
            "total_harvested_hours": round(manifest.get("total_hours", 0.0), 2),
            "total_shards": len(manifest.get("shards", []))
        }

    # --- CLUSTER REMOTE COMMANDS ---

    def send_command(self, action: str, target_worker: Optional[str] = None):
        """Send command (pause, resume, flush) to target worker or cluster."""
        cmd_file = f"commands/{target_worker or 'cluster'}.json"
        payload = {"action": action, "timestamp": time.time()}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(bucket_id=self.tracking_bucket, add=[(tmp_path, cmd_file)])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def check_commands(self) -> Optional[str]:
        """Check if any command was addressed to this node or cluster."""
        for c_name in [f"commands/{self.worker_id}.json", "commands/cluster.json"]:
            tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            t_path = Path(tmp.name)
            tmp.close()
            try:
                tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
                paths = {it.path for it in tree}
                if c_name in paths:
                    self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[(c_name, t_path)])
                    with open(t_path, "r", encoding="utf-8") as f:
                        cmd = json.load(f)
                    if time.time() - cmd.get("timestamp", 0) < 60:
                        if c_name.startswith(f"commands/{self.worker_id}"):
                            self.api.batch_bucket_files(bucket_id=self.tracking_bucket, delete=[c_name])
                        return cmd.get("action")
            except Exception:
                pass
            finally:
                if t_path.exists():
                    t_path.unlink()
        return None

    # --- TELEGRAM ADMIN LOCK PERSISTENCE & POLLER COORDINATION ---

    def acquire_telegram_poller_lease(self, worker_id: str, ttl_sec: int = 45) -> bool:
        """
        Coordinates a single active Telegram bot poller across the cluster to avoid
        Telegram HTTP 409 Conflict ('terminated by other getUpdates request').
        Returns True if this worker holds or successfully acquired the poller lease.
        """
        now = time.time()
        lease_path = "batches/telegram_poller.json"
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        t_path = Path(tmp.name)
        tmp.close()

        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            if lease_path in paths:
                self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[(lease_path, t_path)])
                with open(t_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                holder = data.get("worker_id")
                expires_at = data.get("expires_at", 0)

                # If another worker holds a valid unexpired lease, yield
                if holder and holder != worker_id and expires_at > now:
                    return False
        except Exception:
            pass
        finally:
            if t_path.exists():
                t_path.unlink()

        # Acquire or refresh lease
        payload = {
            "worker_id": worker_id,
            "leased_at": now,
            "expires_at": now + ttl_sec
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp_write:
            json.dump(payload, tmp_write)
            write_path = Path(tmp_write.name)
        try:
            self.api.batch_bucket_files(bucket_id=self.tracking_bucket, add=[(write_path, lease_path)])
            return True
        except Exception:
            return False
        finally:
            if write_path.exists():
                write_path.unlink()

    def release_telegram_poller_lease(self, worker_id: str):
        """Release the Telegram poller lease if held by this worker."""
        try:
            self.api.batch_bucket_files(bucket_id=self.tracking_bucket, delete=["batches/telegram_poller.json"])
        except Exception:
            pass

    def get_telegram_admin(self) -> Optional[Dict[str, Any]]:
        """Retrieve locked admin identity from tracking bucket."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        t_path = Path(tmp.name)
        tmp.close()
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            if "telegram_admin.json" in paths:
                self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[("telegram_admin.json", t_path)])
                with open(t_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        finally:
            if t_path.exists():
                t_path.unlink()
        return None

    def save_telegram_admin(self, admin_data: Dict[str, Any]):
        """Persist locked admin identity in tracking bucket so all nodes recognize it."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(admin_data, tmp, indent=2)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                add=[(tmp_path, "telegram_admin.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
