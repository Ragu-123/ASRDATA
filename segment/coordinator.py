import os
import sys
import re
import json
import time
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
import segment.config as config

class SegmentCoordinator:
    """
    Decentralized Coordinator for Tamil ASR Audio Segmentation:
    - Multiple Kaggle sessions run in parallel without collisions using atomic batch leases.
    - Tracks completed videos in 'segment_manifest.json' in tracking bucket.
    - Leases batches of uncompleted and unleased videos via 'segment_leases/{worker_id}_lease.json'.
    - Maintains worker heartbeats in 'segment_workers/{worker_id}.json'.
    """
    def __init__(
        self,
        api: HfApi,
        tracking_bucket: str = config.TRACKING_BUCKET,
        dataset_repo: str = config.SEGMENTED_DATASET_REPO,
        worker_id: str = config.WORKER_ID
    ):
        self.api = api
        self.tracking_bucket = tracking_bucket
        self.dataset_repo = dataset_repo
        self.worker_id = worker_id

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
        self._manifest_cache_time: float = 0.0
        self._channels_cache_time: float = 0.0
        self._lock = threading.Lock()

        self._ensure_storage_ready()

    def _ensure_storage_ready(self):
        """Ensure tracking bucket and segmented dataset repository exist with Audio dataset card."""
        try:
            self.api.create_bucket(bucket_id=self.tracking_bucket, private=False, exist_ok=True)
        except Exception as e:
            print(f"[SEGMENT COORDINATOR] Tracking bucket check notice: {e}")

        try:
            self.api.create_repo(repo_id=self.dataset_repo, repo_type="dataset", exist_ok=True)
            card = (
                "---\n"
                "configs:\n"
                "- config_name: default\n"
                "  data_files:\n"
                "  - split: train\n"
                "    path: data/*.parquet\n"
                "dataset_info:\n"
                "  features:\n"
                "  - name: audio\n"
                "    dtype: audio\n"
                "  - name: duration\n"
                "    dtype: float64\n"
                "  - name: channel_name\n"
                "    dtype: string\n"
                "  - name: video_id\n"
                "    dtype: string\n"
                "  - name: segment_id\n"
                "    dtype: string\n"
                "  splits:\n"
                "  - name: train\n"
                "---\n\n"
                "# Tamil ASR Segmented Dataset\n"
                "High-quality segmented Tamil speech clips with embedded audio for ASR training.\n"
            )
            try:
                self.api.upload_file(
                    path_or_fileobj=card.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=self.dataset_repo,
                    repo_type="dataset"
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[SEGMENT COORDINATOR] Segmented dataset repo check notice: {e}")

    # --- MANIFEST OPERATIONS ---
    def load_manifest(self, force: bool = False) -> Dict[str, Any]:
        """Fetch segment_manifest.json from the tracking bucket with TTL caching."""
        now = time.time()
        if not force and (now - self._manifest_cache_time < 15.0) and self.manifest_cache.get("completed_videos"):
            return self.manifest_cache

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            if "segment_manifest.json" in paths:
                self.api.download_bucket_files(
                    bucket_id=self.tracking_bucket,
                    files=[("segment_manifest.json", tmp_path)]
                )
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
            print(f"[SEGMENT COORDINATOR] Warning loading segment_manifest: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return self.manifest_cache

    def save_manifest(self, manifest: Dict[str, Any]):
        """Upload updated segment_manifest.json to the tracking bucket."""
        self.manifest_cache = manifest
        self._manifest_cache_time = time.time()
        manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(manifest, tmp, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                add=[(tmp_path, "segment_manifest.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # --- CHANNELS QUEUE OPERATIONS ---
    def load_channels(self, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch configured channels from the tracking bucket."""
        now = time.time()
        if not force and (now - self._channels_cache_time < 30.0) and self.channels_cache:
            return self.channels_cache

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            if "channels.json" in paths:
                self.api.download_bucket_files(
                    bucket_id=self.tracking_bucket,
                    files=[("channels.json", tmp_path)]
                )
                with open(tmp_path, "r", encoding="utf-8") as f:
                    self.channels_cache = json.load(f)
                    self._channels_cache_time = now
            else:
                self.channels_cache = config.DEFAULT_CHANNELS
                self.save_channels(self.channels_cache)
                self._channels_cache_time = now
        except Exception as e:
            print(f"[SEGMENT COORDINATOR] Warning loading channels: {e}")
            if not self.channels_cache:
                self.channels_cache = config.DEFAULT_CHANNELS
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return self.channels_cache

    def save_channels(self, channels: List[Dict[str, Any]]):
        """Upload channels list to the bucket."""
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

    def add_channel(self, url_or_handle: str, name: Optional[str] = None) -> bool:
        """Add a YouTube channel to the persistent queue."""
        channels = self.load_channels()
        norm_url = url_or_handle.strip()
        if not norm_url.startswith("http"):
            if norm_url.startswith("@"):
                norm_url = f"https://www.youtube.com/{norm_url}/videos"
            else:
                norm_url = f"https://www.youtube.com/@{norm_url}/videos"

        for ch in channels:
            if ch.get("url") == norm_url:
                ch["active"] = True
                self.save_channels(channels)
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

    @staticmethod
    def parse_channel_info(url_or_handle: str, name: Optional[str] = None) -> tuple:
        """
        Parses (channel_id, normalized_url, channel_name) reliably from any URL or handle.
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

    def delete_channel(self, url_or_handle: str) -> Dict[str, Any]:
        """
        Deletes a channel, its catalog, and its manifest records from tracking bucket.
        """
        ch_id, _, ch_name = self.parse_channel_info(url_or_handle)

        # 1. Update channels.json
        channels = self.load_channels()
        remaining_channels = []
        removed_count = 0
        target_name = ch_name
        for ch in channels:
            if (ch.get("id") == ch_id or 
                ch.get("name", "").lower() == ch_id.lower() or 
                ch.get("name", "").lower() == url_or_handle.lower() or 
                ch_id in ch.get("url", "")):
                removed_count += 1
                target_name = ch.get("name", ch_id)
            else:
                remaining_channels.append(ch)

        if removed_count > 0 or len(channels) != len(remaining_channels):
            self.save_channels(remaining_channels)

        # 2. Delete catalog from bucket and find video IDs in it
        catalog = self.catalog_cache.get(ch_id, [])
        catalog_path = f"catalogs/{ch_id}.json"
        cat_vid_ids = {v.get("id") or v.get("video_id") for v in catalog if (v.get("id") or v.get("video_id"))}
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            tree_paths = {it.path for it in tree}
            if catalog_path in tree_paths and not cat_vid_ids:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    t_path = Path(tmp.name)
                try:
                    self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[(catalog_path, t_path)])
                    with open(t_path, "r", encoding="utf-8") as f:
                        c_data = json.load(f)
                    cat_vid_ids = {v.get("id") or v.get("video_id") for v in c_data if (v.get("id") or v.get("video_id"))}
                except Exception:
                    pass
                finally:
                    if t_path.exists():
                        t_path.unlink()

            if catalog_path in tree_paths:
                self.api.batch_bucket_files(bucket_id=self.tracking_bucket, delete=[catalog_path])
        except Exception as e:
            print(f"[SEGMENT COORDINATOR] Catalog deletion notice: {e}")

        self.catalog_cache.pop(ch_id, None)

        # 3. Clean segment_manifest.json
        manifest = self.load_manifest(force=True)
        completed_dict = manifest.get("completed_videos", {}) or manifest.get("completed", {})
        keys_to_remove = []
        for vid, v_info in list(completed_dict.items()):
            ch_val = v_info.get("channel") or v_info.get("channel_name") or ""
            if ch_val == target_name or ch_val == ch_id or vid in cat_vid_ids:
                keys_to_remove.append(vid)

        for k in keys_to_remove:
            if "completed_videos" in manifest and k in manifest["completed_videos"]:
                del manifest["completed_videos"][k]
            if "completed" in manifest and k in manifest["completed"]:
                del manifest["completed"][k]

        active_completed = manifest.get("completed_videos", manifest.get("completed", {}))
        manifest["total_videos"] = len(active_completed)
        manifest["total_hours"] = round(sum(v.get("duration", v.get("duration_seconds", 0)) for v in active_completed.values()) / 3600.0, 2)
        if "total_segments" in manifest:
            manifest["total_segments"] = sum(v.get("segments_count", 0) for v in active_completed.values())
        manifest["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self.save_manifest(manifest)

        # 4. Also clean manifest.json if present
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            if any(it.path == "manifest.json" for it in tree):
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    m_path = Path(tmp.name)
                try:
                    self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[("manifest.json", m_path)])
                    with open(m_path, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    m_completed = m_data.get("completed_videos", m_data.get("completed", {}))
                    m_rem = [vid for vid, v in m_completed.items() if (v.get("channel") == target_name or v.get("channel_name") == target_name or v.get("channel") == ch_id or vid in cat_vid_ids)]
                    for vid in m_rem:
                        m_completed.pop(vid, None)
                    m_data["total_videos"] = len(m_completed)
                    m_data["total_hours"] = round(sum(v.get("duration", v.get("duration_seconds", 0)) for v in m_completed.values()) / 3600.0, 2)
                    m_data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(m_path, "w", encoding="utf-8") as f:
                        json.dump(m_data, f, indent=2)
                    self.api.batch_bucket_files(bucket_id=self.tracking_bucket, add=[(m_path, "manifest.json")])
                finally:
                    if m_path.exists():
                        m_path.unlink()
        except Exception:
            pass

        return {
            "channel_id": ch_id,
            "channel_name": target_name,
            "channels_removed": removed_count,
            "catalog_videos_removed": len(cat_vid_ids),
            "manifest_videos_removed": len(keys_to_remove),
            "remaining_total_videos": manifest["total_videos"],
            "remaining_total_hours": manifest["total_hours"]
        }

    def delete_manifest(self) -> bool:
        """Resets segment_manifest.json and manifest.json in tracking bucket to empty."""
        empty_manifest = {
            "completed": {},
            "completed_videos": {},
            "total_videos": 0,
            "total_segments": 0,
            "total_hours": 0.0,
            "shards": [],
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_manifest(empty_manifest)
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            if any(it.path == "manifest.json" for it in tree):
                with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
                    json.dump(empty_manifest, tmp, indent=2)
                    m_p = Path(tmp.name)
                try:
                    self.api.batch_bucket_files(bucket_id=self.tracking_bucket, add=[(m_p, "manifest.json")])
                finally:
                    if m_p.exists():
                        m_p.unlink()
        except Exception:
            pass
        return True

    # --- CATALOG RETRIEVAL ---
    def get_channel_catalog(self, channel: Dict[str, Any], limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch or scrape catalog for a channel."""
        ch_id = channel["id"]
        ch_url = channel["url"]
        ch_name = channel.get("name", ch_id)

        # 1. Check local cache
        if ch_id in self.catalog_cache and len(self.catalog_cache[ch_id]) >= limit:
            return self.catalog_cache[ch_id]

        # 2. Check tracking bucket catalogs/
        cat_file = f"catalogs/{ch_id}.json"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            paths = {it.path for it in tree}
            if cat_file in paths:
                self.api.download_bucket_files(
                    bucket_id=self.tracking_bucket,
                    files=[(cat_file, tmp_path)]
                )
                with open(tmp_path, "r", encoding="utf-8") as f:
                    cat = json.load(f)
                    self.catalog_cache[ch_id] = cat
                    return cat
        except Exception:
            pass
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # 3. Fast scrape via yt-dlp flat playlist extraction
        print(f"[SEGMENT COORDINATOR] Extracting channel catalog for '{ch_name}' ({ch_url})...")
        ydl_opts = {
            'extract_flat': 'in_playlist',
            'quiet': True,
            'no_warnings': True,
            'playlistend': limit,
            'socket_timeout': 30,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'android_vr']
                }
            }
        }
        catalog = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(ch_url, download=False)
                if info and 'entries' in info:
                    for entry in info['entries']:
                        if entry and entry.get('id'):
                            catalog.append({
                                'video_id': entry['id'],
                                'title': entry.get('title', entry['id']),
                                'duration_seconds': entry.get('duration', 0.0),
                                'channel_id': ch_id,
                                'channel_name': ch_name
                            })
            if catalog:
                self.catalog_cache[ch_id] = catalog
                # Cache to bucket
                with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
                    json.dump(catalog, tmp, indent=2, ensure_ascii=False)
                    tmp_p = Path(tmp.name)
                try:
                    self.api.batch_bucket_files(
                        bucket_id=self.tracking_bucket,
                        add=[(tmp_p, cat_file)]
                    )
                finally:
                    if tmp_p.exists():
                        tmp_p.unlink()
        except Exception as e:
            print(f"[SEGMENT COORDINATOR] Catalog extraction error for {ch_name}: {e}")

        return catalog

    # --- ATOMIC DISTRIBUTED BATCH LEASING ---
    def lease_next_batch(self, batch_size: int = config.BATCH_LEASE_SIZE, channel_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Atomically leases an unsegmented batch of videos.
        Ensures multiple Kaggle nodes never process the same videos!
        """
        with self._lock:
            manifest = self.load_manifest(force=True)
            completed_vids = set(manifest.get("completed_videos", {}).keys())

            # Read all active segment leases in tracking bucket
            active_leased_vids = set()
            now = time.time()
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            lease_paths = [it.path for it in tree if it.path.startswith("segment_leases/") and it.path.endswith("_lease.json")]

            for lp in lease_paths:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_p = Path(tmp.name)
                try:
                    self.api.download_bucket_files(bucket_id=self.tracking_bucket, files=[(lp, tmp_p)])
                    with open(tmp_p, "r", encoding="utf-8") as f:
                        l_data = json.load(f)
                    lease_time = l_data.get("leased_at", 0)
                    # Check TTL (45 mins)
                    if now - lease_time < (config.LEASE_TTL_MINUTES * 60):
                        for v in l_data.get("videos", []):
                            active_leased_vids.add(v.get("video_id"))
                except Exception:
                    pass
                finally:
                    if tmp_p.exists():
                        tmp_p.unlink()

            # Find candidates across channels
            channels = self.load_channels()
            if channel_filter:
                channels = [c for c in channels if channel_filter.lower() in c.get("id", "").lower() or channel_filter.lower() in c.get("name", "").lower() or channel_filter.lower() in c.get("url", "").lower()]

            candidates = []
            for ch in channels:
                if not ch.get("active", True):
                    continue
                cat = self.get_channel_catalog(ch)
                for item in cat:
                    vid_id = item["video_id"]
                    if vid_id not in completed_vids and vid_id not in active_leased_vids:
                        candidates.append({
                            **item,
                            "channel_name": ch.get("name", ch.get("id", "Unknown"))
                        })
                        if len(candidates) >= batch_size:
                            break
                if len(candidates) >= batch_size:
                    break

            if not candidates:
                return []

            # Write atomic lease file
            lease_data = {
                "worker_id": self.worker_id,
                "leased_at": now,
                "expires_at": now + (config.LEASE_TTL_MINUTES * 60),
                "videos": candidates
            }
            lease_rel_path = f"segment_leases/{self.worker_id}_lease.json"
            with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
                json.dump(lease_data, tmp, indent=2, ensure_ascii=False)
                tmp_lease = Path(tmp.name)
            try:
                self.api.batch_bucket_files(
                    bucket_id=self.tracking_bucket,
                    add=[(tmp_lease, lease_rel_path)]
                )
            finally:
                if tmp_lease.exists():
                    tmp_lease.unlink()

            return candidates

    def release_lease(self):
        """Release this worker's current lease file in the tracking bucket."""
        lease_rel_path = f"segment_leases/{self.worker_id}_lease.json"
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                delete=[lease_rel_path]
            )
        except Exception:
            pass

    def update_heartbeat(self, status: str = "idle", active_streams: list = None):
        """Update worker heartbeat in tracking bucket workers/{worker_id}.json and segment_workers/{worker_id}.json."""
        active_list = active_streams or []
        primary_vid = active_list[0] if active_list else {}
        worker_info = {
            "worker_id": self.worker_id,
            "stage": "segment",
            "status": status,
            "current_video": primary_vid,
            "last_heartbeat": time.time(),
            "active_videos": [v.get("video_id") for v in active_list]
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(worker_info, tmp, indent=2)
            tmp_p = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                add=[
                    (tmp_p, f"workers/{self.worker_id}.json"),
                    (tmp_p, f"segment_workers/{self.worker_id}.json")
                ]
            )
        except Exception:
            pass
        finally:
            if tmp_p.exists():
                tmp_p.unlink()

    def deregister_worker(self):
        """Cleanly deregister worker on shutdown."""
        self.release_lease()
        try:
            self.api.batch_bucket_files(
                bucket_id=self.tracking_bucket,
                delete=[
                    f"workers/{self.worker_id}.json",
                    f"segment_workers/{self.worker_id}.json"
                ]
            )
        except Exception:
            pass

    def list_active_workers(self, force: bool = False) -> List[Dict[str, Any]]:
        """List active peer workers in the cluster."""
        now = time.time()
        active_workers = []
        temp_files = []
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.tracking_bucket))
            w_paths = [it.path for it in tree if (it.path.startswith("workers/") or it.path.startswith("segment_workers/")) and it.path.endswith(".json")]
            seen_ids = set()

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
                        w_id = w_data.get("worker_id")
                        if w_id and w_id not in seen_ids and now - w_data.get("last_heartbeat", 0) <= 90:
                            seen_ids.add(w_id)
                            w_data["is_alive"] = True
                            active_workers.append(w_data)
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            for _, t_path in temp_files:
                if t_path.exists():
                    t_path.unlink(missing_ok=True)

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

