import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from huggingface_hub import HfApi
import config

class ClusterCoordinator:
    """
    Decentralized task coordinator using Hugging Face Storage Bucket.
    Manages atomic batch leases, active worker heartbeats, and channel queues.
    """
    def __init__(self, api: HfApi, bucket_id: str, node_id: str):
        self.api = api
        self.bucket_id = bucket_id
        self.node_id = node_id
        self.manifest_cache = {"completed": {}, "total_hours": 0.0, "last_updated": ""}
        self.channels_cache = []
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
        
        # Check if already in list
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

    # --- BATCH LEASING ENGINE ---
    def lease_batch(self, candidate_videos: List[Dict[str, Any]], batch_size: int = config.BATCH_LEASE_SIZE) -> List[Dict[str, Any]]:
        """
        Atomically lease a batch of videos for this node.
        Skips completed videos and videos leased by other active workers.
        """
        now = time.time()
        self.load_manifest()
        completed_ids = set(self.manifest_cache.get("completed", {}).keys())
        
        # Fetch active leases
        leased_ids = set()
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.bucket_id))
            lease_files = [it.path for it in tree if it.path.startswith("batches/") and it.path.endswith("_lease.json")]
            for l_path in lease_files:
                # If this node's own lease, release it first
                if l_path == f"batches/{self.node_id}_lease.json":
                    continue
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    t_path = Path(tmp.name)
                try:
                    self.api.download_bucket_files(bucket_id=self.bucket_id, files=[(l_path, t_path)])
                    with open(t_path, "r", encoding="utf-8") as f:
                        lease_data = json.load(f)
                    if lease_data.get("expires_at", 0) > now:
                        leased_ids.update(lease_data.get("video_ids", []))
                except Exception:
                    pass
                finally:
                    if t_path.exists():
                        t_path.unlink()
        except Exception as e:
            print(f"[COORDINATOR] Warning inspecting leases: {e}")

        # Pick candidate videos that are neither completed nor currently leased
        selected = []
        for vid in candidate_videos:
            v_id = vid["id"]
            if v_id not in completed_ids and v_id not in leased_ids:
                selected.append(vid)
                if len(selected) >= batch_size:
                    break

        if not selected:
            return []

        # Write this worker's lease
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
    def update_heartbeat(self, status: str, current_video: Optional[Dict[str, Any]] = None, tunnel_url: str = "", speed_str: str = "", batch_info: str = ""):
        """Register worker heartbeat in the bucket."""
        now = time.time()
        disk_usage = shutil.disk_usage(config.WORKING_DIR)
        used_gb = round(disk_usage.used / (1024**3), 2)
        free_gb = round(disk_usage.free / (1024**3), 2)

        hb_payload = {
            "worker_id": self.node_id,
            "status": status,
            "tunnel_url": tunnel_url,
            "current_video": current_video or {},
            "speed": speed_str,
            "batch_info": batch_info,
            "disk": {
                "used_gb": used_gb,
                "free_gb": free_gb,
                "percent": round((disk_usage.used / disk_usage.total) * 100, 1)
            },
            "last_heartbeat": now
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(hb_payload, tmp, indent=2)
            tmp_path = Path(tmp.name)
        try:
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
        """List all active peer workers in the cluster."""
        now = time.time()
        active_workers = []
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
                    
                    # Consider worker active if heartbeat is within threshold
                    if now - w_data.get("last_heartbeat", 0) <= config.WORKER_OFFLINE_THRESHOLD_SEC:
                        active_workers.append(w_data)
                    else:
                        # Auto-clean stale worker records
                        w_id = w_data.get("worker_id")
                        if w_id:
                            try:
                                self.api.batch_bucket_files(bucket_id=self.bucket_id, delete=[path, f"batches/{w_id}_lease.json"])
                            except Exception:
                                pass
                except Exception:
                    pass
                finally:
                    if t_path.exists():
                        t_path.unlink()
        except Exception as e:
            print(f"[COORDINATOR] Warning reading workers: {e}")
        return active_workers

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
                    # If command is recent (within 60s)
                    if time.time() - cmd.get("timestamp", 0) < 60:
                        # Clear command after reading if node-specific
                        if c_name.startswith(f"commands/{self.node_id}"):
                            self.api.batch_bucket_files(bucket_id=self.bucket_id, delete=[c_name])
                        return cmd.get("action")
            except Exception:
                pass
            finally:
                if t_path.exists():
                    t_path.unlink()
        return None
