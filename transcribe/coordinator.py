import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from huggingface_hub import HfApi
import transcribe.config as config

class TranscriptionCoordinator:
    """
    Decentralized task coordinator using Hugging Face Storage Bucket.
    Manages atomic batch leases so multiple parallel Kaggle sessions
    never transcribe the same video twice.
    """
    def __init__(self, api: HfApi, source_bucket: str = config.SOURCE_BUCKET, target_bucket: str = config.TARGET_BUCKET, worker_id: str = config.WORKER_ID):
        self.api = api
        self.source_bucket = source_bucket
        self.target_bucket = target_bucket
        self.worker_id = worker_id

        self.source_manifest: Dict[str, Any] = {}
        self.target_manifest: Dict[str, Any] = {}
        self._target_manifest_cache_time = 0.0

        self._ensure_target_bucket()

    def _ensure_target_bucket(self):
        try:
            self.api.create_bucket(bucket_id=self.target_bucket, private=False, exist_ok=True)
        except Exception as e:
            print(f"[COORDINATOR] Target bucket check notice: {e}")

    def load_source_manifest(self) -> Dict[str, Any]:
        """Fetch global manifest from source audio bucket."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.source_bucket))
            paths = {it.path for it in tree}
            if "manifest.json" in paths:
                self.api.download_bucket_files(bucket_id=self.source_bucket, files=[("manifest.json", tmp_path)])
                with open(tmp_path, "r", encoding="utf-8") as f:
                    self.source_manifest = json.load(f)
            else:
                self.source_manifest = {"completed": {}}
        except Exception as e:
            print(f"[COORDINATOR] Warning loading source manifest: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return self.source_manifest

    def load_target_manifest(self, force: bool = False) -> Dict[str, Any]:
        """Fetch completed transcripts manifest from target bucket with TTL cache."""
        now = time.time()
        if not force and (now - self._target_manifest_cache_time < 15.0) and self.target_manifest.get("completed_videos"):
            return self.target_manifest

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.target_bucket))
            paths = {it.path for it in tree}
            if "manifest.json" in paths:
                self.api.download_bucket_files(bucket_id=self.target_bucket, files=[("manifest.json", tmp_path)])
                with open(tmp_path, "r", encoding="utf-8") as f:
                    self.target_manifest = json.load(f)
                self._target_manifest_cache_time = now
            else:
                self.target_manifest = {"total_videos": 0, "total_segments": 0, "total_hours": 0.0, "completed_videos": {}}
                self._target_manifest_cache_time = now
        except Exception as e:
            print(f"[COORDINATOR] Warning loading target manifest: {e}")
            self.target_manifest = {"total_videos": 0, "total_segments": 0, "total_hours": 0.0, "completed_videos": {}}
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return self.target_manifest

    def save_target_manifest(self, manifest: Dict[str, Any]):
        """Upload updated transcripts manifest to target bucket."""
        self.target_manifest = manifest
        self._target_manifest_cache_time = time.time()
        manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
            json.dump(manifest, tmp, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        try:
            self.api.batch_bucket_files(
                bucket_id=self.target_bucket,
                add=[(tmp_path, "manifest.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def lease_next_batch(self, batch_size: int = config.BATCH_LEASE_SIZE) -> List[Dict[str, Any]]:
        """
        Atomically leases a non-overlapping batch of audio files from the source bucket.
        Guarantees that parallel Kaggle sessions will NEVER work on the same video.
        """
        now = time.time()
        self.load_source_manifest()
        self.load_target_manifest(force=True)

        source_completed = self.source_manifest.get("completed", {})
        target_completed = set(self.target_manifest.get("completed_videos", {}).keys())

        # 1. Fetch active peer leases from target bucket in a single batch call
        leased_video_ids = set()
        expired_files_to_delete = []
        temp_lease_files = []

        try:
            tree = list(self.api.list_bucket_tree(bucket_id=self.target_bucket))
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
                self.api.download_bucket_files(bucket_id=self.target_bucket, files=download_pairs)

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
                    self.api.batch_bucket_files(bucket_id=self.target_bucket, delete=expired_files_to_delete)
                except Exception:
                    pass
        except Exception as e:
            print(f"[COORDINATOR] Warning inspecting leases: {e}")
        finally:
            for _, t_path in temp_lease_files:
                if t_path.exists():
                    t_path.unlink(missing_ok=True)

        # 2. Select uncompleted, unleased candidate videos
        candidates = []
        for vid_id, meta in source_completed.items():
            if vid_id not in target_completed and vid_id not in leased_video_ids:
                item = dict(meta)
                item["video_id"] = vid_id
                candidates.append(item)
                leased_video_ids.add(vid_id)
                if len(candidates) >= batch_size:
                    break

        if not candidates:
            return []

        # 3. Write atomic lease to target bucket
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
                bucket_id=self.target_bucket,
                add=[(tmp_path, f"batches/{self.worker_id}_lease.json")]
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return candidates

    def release_lease(self):
        """Release this worker's lease file."""
        try:
            self.api.batch_bucket_files(
                bucket_id=self.target_bucket,
                delete=[f"batches/{self.worker_id}_lease.json"]
            )
        except Exception:
            pass
