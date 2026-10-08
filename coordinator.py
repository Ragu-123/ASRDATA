import json
import time
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from huggingface_hub import HfApi
import yt_dlp
import config

class ClusterCoordinator:
    """
    Decentralized task coordinator using Hugging Face Storage Bucket.
    Manages atomic batch leases, active worker heartbeats, centralized channel catalogs,
    and cluster-wide telemetry synchronization.
    """
    def __init__(self, api: HfApi, bucket_id: str, node_id: str):
        self.api = api
        self.bucket_id = bucket_id
        self.node_id = node_id
        self.manifest_cache = {"completed": {}, "total_hours": 0.0, "last_updated": ""}
        self.channels_cache = []
        self.catalog_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._ensure_bucket_ready()

    def _ensure_bucket_ready(self):
        try:
            self.api.create_bucket(bucket_id=self.bucket_id, private=False, exist_ok=True)
        except Exception as e:
            print(f"[COORDINATOR] Bucket ensure notice: {e}")

    # --- MANIFEST OPERATIONS ---
    def load_manifest(self) -> Dict[str, Any]:
        """Fetch global manifest.json from the bucket."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.bucket_id))
            paths = {it.path for it in tree}
            if "manifest.json" in paths:
                self.api.download_bucket_files(
                    bucket_id=self.bucket_id,
                    files=[("manifest.json", tmp_path)]
                )
                with open(tmp_path, "r", encoding="utf-8") as f:
                    self.manifest_cache = json.load(f)
            else:
                self.manifest_cache = {"completed": {}, "total_hours": 0.0, "last_updated": ""}
        except Exception as e:
            print(f"[COORDINATOR] Warning loading manifest: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return self.manifest_cache

    def save_manifest(self, manifest: Dict[str, Any]):
        """Upload updated manifest to the bucket."""
        self.manifest_cache = manifest
        manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(manifest, tmp, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.bucket_id,
                add=[(tmp_path, "manifest.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # --- CHANNELS QUEUE OPERATIONS ---
    def load_channels(self) -> List[Dict[str, Any]]:
        """Fetch configured channels from the bucket."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.bucket_id))
            paths = {it.path for it in tree}
            if "channels.json" in paths:
                self.api.download_bucket_files(
                    bucket_id=self.bucket_id,
                    files=[("channels.json", tmp_path)]
                )
                with open(tmp_path, "r", encoding="utf-8") as f:
                    self.channels_cache = json.load(f)
            else:
                self.channels_cache = config.DEFAULT_CHANNELS
                self.save_channels(self.channels_cache)
        except Exception as e:
            print(f"[COORDINATOR] Warning loading channels: {e}")
            if not self.channels_cache:
                self.channels_cache = config.DEFAULT_CHANNELS
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return self.channels_cache

    def save_channels(self, channels: List[Dict[str, Any]]):
        """Upload channels list to the bucket."""
        self.channels_cache = channels
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(channels, tmp, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.bucket_id,
                add=[(tmp_path, "channels.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def add_channel(self, url_or_handle: str, name: Optional[str] = None) -> bool:
        """Add a new YouTube channel to the persistent queue."""
        channels = self.load_channels()
        norm_url = url_or_handle.strip()
        if not norm_url.startswith("http"):
            if norm_url.startswith("@"):
                norm_url = f"https://www.youtube.com/{norm_url}/videos"
            else:
                norm_url = f"https://www.youtube.com/@{norm_url}/videos"
        
        for ch in channels:
            if ch.get("url") == norm_url:
                return False
                
        ch_id = norm_url.split("/")[-2].replace("@", "") if "@" in norm_url else f"ch_{int(time.time())}"
        channels.append({
            "id": ch_id,
            "name": name or ch_id,
            "url": norm_url,
            "active": True,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        self.save_channels(channels)
        return True

    # --- CENTRALIZED METADATA CATALOG ---
    def get_channel_catalog(self, channel: Dict[str, Any], needed_uncompleted: int = 120) -> List[Dict[str, Any]]:
        """
        Retrieves or expands the channel's video catalog.
        - First loads from the bucket (sub-second, shared across all nodes).
        - If uncompleted candidates < needed_uncompleted, fetches the next metadata slice via yt-dlp
          and saves back to bucket so peer nodes don't repeat the extraction.
        """
        ch_id = channel["id"]
        ch_url = channel["url"]
        catalog_path = f"catalogs/{ch_id}.json"

        # 1. Check local cache or fetch from bucket
        catalog = self.catalog_cache.get(ch_id, [])
        if not catalog:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                t_path = Path(tmp.name)
            try:
                tree = list(self.api.list_bucket_tree(bucket_id=self.bucket_id))
                paths = {it.path for it in tree}
                if catalog_path in paths:
                    self.api.download_bucket_files(bucket_id=self.bucket_id, files=[(catalog_path, t_path)])
                    with open(t_path, "r", encoding="utf-8") as f:
                        catalog = json.load(f)
                    self.catalog_cache[ch_id] = catalog
            except Exception as e:
                print(f"[COORDINATOR] Catalog fetch notice for {ch_id}: {e}")
            finally:
                if t_path.exists():
                    t_path.unlink()

        # 2. Count uncompleted videos currently in catalog
        completed_ids = set(self.manifest_cache.get("completed", {}).keys())
        uncompleted = [v for v in catalog if v.get("id") not in completed_ids]

        # 3. If uncompleted count is lower than needed, expand metadata via yt-dlp slice
        if len(uncompleted) < needed_uncompleted:
            start_idx = len(catalog) + 1
            chunk_size = max(config.CATALOG_FETCH_CHUNK, needed_uncompleted * 2)
            end_idx = len(catalog) + chunk_size
            print(f"[COORDINATOR] Expanding catalog for {channel.get('name', ch_id)}: fetching items {start_idx} to {end_idx}...")

            try:
                ydl_opts = {
                    'extract_flat': True,
                    'quiet': True,
                    'no_warnings': True,
                    'playliststart': start_idx,
                    'playlistend': end_idx,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android', 'ios']
                        }
                    }
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(ch_url, download=False)
                    entries = info.get("entries", []) if info else []

                existing_ids = {v["id"] for v in catalog}
                added_count = 0
                for e in entries:
                    v_id = e.get("id")
                    if v_id and v_id not in existing_ids:
                        catalog.append({
                            "id": v_id,
                            "title": e.get("title", ""),
                            "duration": e.get("duration", 0),
                            "channel": channel.get("name", ch_id)
                        })
                        existing_ids.add(v_id)
                        added_count += 1

                print(f"[COORDINATOR] Discovered {added_count} new videos for {ch_id} (Total catalog: {len(catalog)})")

                # Upload expanded catalog to bucket so all peer nodes immediately get it
                if added_count > 0:
                    self.catalog_cache[ch_id] = catalog
                    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
                        json.dump(catalog, tmp, indent=2)
                        t_path = Path(tmp.name)
                    try:
                        self.api.batch_bucket_files(bucket_id=self.bucket_id, add=[(t_path, catalog_path)])
                    except Exception as e:
                        print(f"[COORDINATOR] Warning saving catalog to bucket: {e}")
                    finally:
                        if t_path.exists():
                            t_path.unlink()

            except Exception as e:
                print(f"[COORDINATOR] Warning expanding channel {ch_url}: {e}")

        return catalog

    # --- DYNAMIC BATCH LEASING ENGINE ---
    def lease_next_batch(self, batch_size: int = config.BATCH_LEASE_SIZE) -> List[Dict[str, Any]]:
        """
        Dynamically calculates active node count, expands metadata buffer if needed,
        and atomically leases a non-overlapping batch for this node.
        Cleans expired or dead-node leases immediately so candidate videos are never locked.
        """
        now = time.time()
        self.load_manifest()
        completed_ids = set(self.manifest_cache.get("completed", {}).keys())

        # Determine active nodes count to scale metadata fetch buffer
        active_nodes = self.list_active_workers()
        live_workers = [w for w in active_nodes if w.get("is_alive", True)]
        live_node_ids = {w["worker_id"] for w in live_workers}
        live_node_count = max(1, len(live_workers))
        needed_buffer = live_node_count * batch_size * 2

        # 1. Read existing active leases from peer nodes & immediately release dead/expired leases
        leased_ids = set()
        expired_files_to_delete = []
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.bucket_id))
            lease_files = [it.path for it in tree if it.path.startswith("batches/") and it.path.endswith("_lease.json")]
            for l_path in lease_files:
                if l_path == f"batches/{self.node_id}_lease.json":
                    continue
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    t_path = Path(tmp.name)
                try:
                    self.api.download_bucket_files(bucket_id=self.bucket_id, files=[(l_path, t_path)])
                    with open(t_path, "r", encoding="utf-8") as f:
                        lease_data = json.load(f)
                    
                    l_worker = lease_data.get("worker_id")
                    is_expired = lease_data.get("expires_at", 0) <= now
                    is_dead_worker = l_worker not in live_node_ids

                    if is_expired or is_dead_worker:
                        # Dead worker or expired lease -> mark for immediate deletion & release videos!
                        expired_files_to_delete.append(l_path)
                    else:
                        leased_ids.update(lease_data.get("video_ids", []))
                except Exception:
                    pass
                finally:
                    if t_path.exists():
                        t_path.unlink()

            # Clean dead/expired lease files in bucket immediately
            if expired_files_to_delete:
                try:
                    self.api.batch_bucket_files(bucket_id=self.bucket_id, delete=expired_files_to_delete)
                    print(f"[COORDINATOR] Cleaned {len(expired_files_to_delete)} stale/dead leases from bucket.")
                except Exception:
                    pass
        except Exception as e:
            print(f"[COORDINATOR] Warning inspecting leases: {e}")

        # 2. Iterate channels to lease uncompleted, unleased videos
        channels = self.load_channels()
        selected = []

        for ch in channels:
            if not ch.get("active", True):
                continue
            catalog = self.get_channel_catalog(ch, needed_uncompleted=needed_buffer)
            for vid in catalog:
                v_id = vid["id"]
                if v_id not in completed_ids and v_id not in leased_ids:
                    selected.append(vid)
                    leased_ids.add(v_id) # Prevent duplicate assignment in this batch
                    if len(selected) >= batch_size:
                        break
            if len(selected) >= batch_size:
                break

        if not selected:
            return []

        # 3. Write this worker's lease to HF bucket
        lease_payload = {
            "worker_id": self.node_id,
            "leased_at": now,
            "expires_at": now + (config.LEASE_TTL_MINUTES * 60),
            "video_ids": [v["id"] for v in selected],
            "count": len(selected)
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(lease_payload, tmp, indent=2)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.bucket_id,
                add=[(tmp_path, f"batches/{self.node_id}_lease.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return selected

    def release_batch(self):
        """Release current worker's batch lease file."""
        try:
            self.api.batch_bucket_files(
                bucket_id=self.bucket_id,
                delete=[f"batches/{self.node_id}_lease.json"]
            )
        except Exception:
            pass

    # --- HEARTBEAT & WORKER MESH ---
    def update_heartbeat(
        self,
        status: str,
        active_streams: List[Dict[str, Any]],
        session_downloaded: int = 0,
        session_hours: float = 0.0,
        tunnel_url: str = "",
        speed_str: str = "",
        batch_info: str = ""
    ):
        """Register worker heartbeat in the bucket (Thread-Safe)."""
        now = time.time()
        disk_usage = shutil.disk_usage(config.WORKING_DIR)
        used_gb = round(disk_usage.used / (1024**3), 2)
        total_gb = round(disk_usage.total / (1024**3), 2)
        free_gb = round(disk_usage.free / (1024**3), 2)

        # Primary current video for legacy consumers
        primary_video = active_streams[0] if active_streams else {}

        hb_payload = {
            "worker_id": self.node_id,
            "status": status,
            "tunnel_url": tunnel_url,
            "active_streams": active_streams,
            "current_video": primary_video,
            "session_downloaded": session_downloaded,
            "session_hours": round(session_hours, 2),
            "speed": speed_str,
            "batch_info": batch_info,
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
                    bucket_id=self.bucket_id,
                    add=[(tmp_path, f"workers/{self.node_id}.json")]
                )
        except Exception:
            pass
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def list_active_workers(self) -> List[Dict[str, Any]]:
        """
        List all active peer workers in the cluster.
        IMMEDIATELY deletes and purges any offline workers from the bucket (older than 30s)
        so ghost/offline cards never remain and their leases are freed immediately.
        """
        now = time.time()
        active_workers = []
        dead_files_to_delete = []

        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.bucket_id))
            w_paths = [it.path for it in tree if it.path.startswith("workers/") and it.path.endswith(".json")]
            for path in w_paths:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    t_path = Path(tmp.name)
                try:
                    self.api.download_bucket_files(bucket_id=self.bucket_id, files=[(path, t_path)])
                    with open(t_path, "r", encoding="utf-8") as f:
                        w_data = json.load(f)
                    
                    time_since_hb = now - w_data.get("last_heartbeat", 0)
                    
                    if time_since_hb <= config.WORKER_OFFLINE_THRESHOLD_SEC:
                        w_data["is_alive"] = True
                        active_workers.append(w_data)
                    else:
                        # NODE IS OFFLINE -> Mark for immediate deletion from bucket!
                        w_id = w_data.get("worker_id")
                        if w_id and w_id != self.node_id:
                            dead_files_to_delete.extend([
                                path,
                                f"batches/{w_id}_lease.json",
                                f"commands/{w_id}.json"
                            ])
                except Exception:
                    pass
                finally:
                    if t_path.exists():
                        t_path.unlink()

            # Execute batch deletion of dead workers & their leases
            if dead_files_to_delete:
                try:
                    # Filter existing files before delete
                    existing_dead = [p for p in dead_files_to_delete if any(it.path == p for it in tree)]
                    if existing_dead:
                        self.api.batch_bucket_files(bucket_id=self.bucket_id, delete=existing_dead)
                        print(f"[COORDINATOR] Purged {len(existing_dead)} offline worker files from bucket.")
                except Exception as e:
                    print(f"[COORDINATOR] Notice purging offline files: {e}")

        except Exception as e:
            print(f"[COORDINATOR] Warning reading workers: {e}")
            
        # Ensure self is always present if alive
        self_present = any(w.get("worker_id") == self.node_id for w in active_workers)
        if not self_present:
            usage = shutil.disk_usage(config.WORKING_DIR)
            active_workers.append({
                "worker_id": self.node_id,
                "status": "online",
                "tunnel_url": "",
                "active_streams": [],
                "session_downloaded": 0,
                "session_hours": 0.0,
                "speed": "Active",
                "disk": {
                    "used_gb": round(usage.used / (1024**3), 2),
                    "total_gb": round(usage.total / (1024**3), 2),
                    "percent": round((usage.used / usage.total) * 100, 1)
                },
                "last_heartbeat": now,
                "is_alive": True
            })

        return active_workers

    def get_cluster_overview(self) -> Dict[str, Any]:
        """Calculates cluster-wide aggregated metrics across all active nodes."""
        active_workers = self.list_active_workers()
        live_workers = [w for w in active_workers if w.get("is_alive", True)]
        node_count = max(1, len(live_workers))

        # Aggregate Disk across all active nodes
        total_used_disk = sum(w.get("disk", {}).get("used_gb", 0.0) for w in live_workers)
        total_cluster_capacity = round(node_count * config.DISK_PER_NODE_GB, 1)
        total_cluster_trigger = round(node_count * config.DISK_THRESHOLD_GB, 1)

        # Aggregate active streams from all workers
        all_active_streams = []
        total_cluster_downloaded = 0

        for w in live_workers:
            w_id = w.get("worker_id", "unknown")
            total_cluster_downloaded += w.get("session_downloaded", 0)
            streams = w.get("active_streams", [])
            for st in streams:
                st_copy = dict(st)
                st_copy["worker_id"] = w_id
                all_active_streams.append(st_copy)

        return {
            "node_count": node_count,
            "live_workers": live_workers,
            "cluster_disk": {
                "used_gb": round(total_used_disk, 2),
                "total_gb": total_cluster_capacity,
                "trigger_gb": total_cluster_trigger,
                "percent": round((total_used_disk / total_cluster_capacity) * 100, 1) if total_cluster_capacity > 0 else 0.0
            },
            "all_active_streams": all_active_streams,
            "total_cluster_downloaded": total_cluster_downloaded,
            "total_harvested_hours": round(self.manifest_cache.get("total_hours", 0.0), 2),
            "total_completed_videos": len(self.manifest_cache.get("completed", {}))
        }

    # --- CLUSTER REMOTE COMMANDS ---
    def send_command(self, action: str, target_worker: Optional[str] = None):
        """Send command (pause, resume, flush) to target worker or entire cluster."""
        cmd_file = f"commands/{target_worker or 'cluster'}.json"
        payload = {"action": action, "timestamp": time.time()}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(bucket_id=self.bucket_id, add=[(tmp_path, cmd_file)])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def check_commands(self) -> Optional[str]:
        """Check if any command was addressed to this node or to the cluster."""
        for c_name in [f"commands/{self.node_id}.json", "commands/cluster.json"]:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                t_path = Path(tmp.name)
            try:
                tree = list(self.api.list_bucket_tree(bucket_id=self.bucket_id))
                paths = {it.path for it in tree}
                if c_name in paths:
                    self.api.download_bucket_files(bucket_id=self.bucket_id, files=[(c_name, t_path)])
                    with open(t_path, "r", encoding="utf-8") as f:
                        cmd = json.load(f)
                    if time.time() - cmd.get("timestamp", 0) < 60:
                        if c_name.startswith(f"commands/{self.node_id}"):
                            self.api.batch_bucket_files(bucket_id=self.bucket_id, delete=[c_name])
                        return cmd.get("action")
            except Exception:
                pass
            finally:
                if t_path.exists():
                    t_path.unlink()
        return None
