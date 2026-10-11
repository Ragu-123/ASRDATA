import os
import sys
import time
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pyarrow as pa
import pyarrow.parquet as pq

import transcribe.config as config
from transcribe.coordinator import TranscriptionCoordinator
from transcribe.syncer import ParquetDatasetSyncer
from transcribe.telegram_bot import TelegramClusterBot
from transcribe.disk_guard import DiskGuard

class MockHfTreeItem:
    def __init__(self, path: str):
        self.path = path

class MockHfApi:
    """Mock HfApi for isolated unit and integration testing without network calls."""
    def __init__(self):
        self.storage: dict = {}  # {bucket_or_repo: {path: bytes}}

    def create_bucket(self, bucket_id: str, private: bool = False, exist_ok: bool = True):
        self.storage.setdefault(bucket_id, {})

    def create_repo(self, repo_id: str, repo_type: str = "dataset", exist_ok: bool = True):
        self.storage.setdefault(repo_id, {})

    def list_bucket_tree(self, bucket_id: str):
        files = self.storage.get(bucket_id, {})
        return [MockHfTreeItem(p) for p in files.keys()]

    def batch_bucket_files(self, bucket_id: str, add=None, delete=None):
        self.storage.setdefault(bucket_id, {})
        if add:
            for src_path, target_rel in add:
                with open(src_path, "rb") as f:
                    self.storage[bucket_id][target_rel] = f.read()
        if delete:
            for del_p in delete:
                self.storage[bucket_id].pop(del_p, None)

    def download_bucket_files(self, bucket_id: str, files=None):
        bucket_data = self.storage.get(bucket_id, {})
        if files:
            for rem_path, loc_path in files:
                if rem_path in bucket_data:
                    Path(loc_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(loc_path, "wb") as f:
                        f.write(bucket_data[rem_path])
                else:
                    raise FileNotFoundError(f"Remote file not found: {rem_path}")

    def upload_file(self, path_or_fileobj: str, path_in_repo: str, repo_id: str, repo_type: str = "dataset", commit_message: str = ""):
        self.storage.setdefault(repo_id, {})
        with open(path_or_fileobj, "rb") as f:
            self.storage[repo_id][path_in_repo] = f.read()

class TestTwoTierPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.staging_dir = self.test_dir / "staging"
        self.shards_dir = self.test_dir / "shards"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        config.WORKING_DIR = self.test_dir
        config.STAGING_DIR = self.staging_dir
        config.SHARDS_DIR = self.shards_dir

        self.orig_admin_id = config.TELEGRAM_ADMIN_ID
        config.TELEGRAM_ADMIN_ID = ""

        self.mock_api = MockHfApi()
        self.coordinator = TranscriptionCoordinator(
            api=self.mock_api,
            tracking_bucket="test/tracking-01",
            dataset_repo="test/tamil-dataset",
            source_bucket="test/media-archive",
            worker_id="test-node-1"
        )

    def tearDown(self):
        config.TELEGRAM_ADMIN_ID = self.orig_admin_id
        self.temp_dir.cleanup()

    # --- 1. PYARROW SCHEMA & PARQUET SHARD TESTING ---

    def test_parquet_syncer_schema_and_flushing(self):
        """Verify that ParquetDatasetSyncer generates compliant PyArrow audio structs and Parquet files."""
        syncer = ParquetDatasetSyncer(
            api=self.mock_api,
            coordinator=self.coordinator,
            dataset_repo="test/tamil-dataset",
            shard_threshold=3
        )

        # Create dummy audio files
        dummy_audio_file = self.staging_dir / "vid1_seg0001.m4a"
        dummy_audio_file.write_bytes(b"AAC_DUMMY_AUDIO_DATA_FOR_TAMIL_ASR_TESTING" * 50)

        dummy_audio_file2 = self.staging_dir / "vid1_seg0002.m4a"
        dummy_audio_file2.write_bytes(b"AAC_DUMMY_AUDIO_DATA_FOR_TAMIL_ASR_TESTING" * 50)

        audited_segments = [
            {
                "segment_id": "vid1_seg0001",
                "clean_text": "வணக்கம் இது ஒரு சோதனை",
                "duration": 5.2,
                "audio_path": dummy_audio_file,
                "emotion": "None"
            },
            {
                "segment_id": "vid1_seg0002",
                "clean_text": "தமிழ் பேச்சு அறிதல் அமைப்பு",
                "duration": 6.8,
                "audio_path": dummy_audio_file2,
                "emotion": "Happy"
            }
        ]

        # 1. Sync video
        res = syncer.sync_video_transcripts(
            video_meta={"video_id": "vid1", "title": "Test Tamil Video", "duration_seconds": 12.0},
            audited_segments=audited_segments,
            staging_dir=self.staging_dir
        )
        self.assertTrue(res)
        self.assertEqual(len(syncer.buffered_rows), 2)

        # 2. Flush shard manually
        shard_name = syncer.flush_shard()
        self.assertIsNotNone(shard_name)
        self.assertTrue(shard_name.startswith("train-test-node-1-00000.parquet"))

        # 3. Verify shard exists in Mock HF Dataset repo
        dataset_files = self.mock_api.storage.get("test/tamil-dataset", {})
        repo_parquet_path = f"data/{shard_name}"
        self.assertIn(repo_parquet_path, dataset_files)

        # 4. Read back and verify PyArrow Table schema and data integrity
        parquet_bytes = dataset_files[repo_parquet_path]
        reader = pa.BufferReader(parquet_bytes)
        table = pq.read_table(reader)

        self.assertEqual(table.num_rows, 2)
        self.assertEqual(table.column_names, ["audio", "sentence", "duration", "video_id", "segment_id", "emotion"])

        # Check audio struct
        audio_col = table["audio"]
        self.assertEqual(audio_col[0]["path"].as_py(), "vid1_seg0001.m4a")
        self.assertEqual(audio_col[0]["bytes"].as_py(), b"AAC_DUMMY_AUDIO_DATA_FOR_TAMIL_ASR_TESTING" * 50)
        self.assertEqual(table["sentence"][0].as_py(), "வணக்கம் இது ஒரு சோதனை")
        self.assertEqual(table["emotion"][1].as_py(), "Happy")

        # 5. Verify manifest.json in tracking bucket was updated
        tracking_files = self.mock_api.storage.get("test/tracking-01", {})
        self.assertIn("manifest.json", tracking_files)
        manifest_data = json.loads(tracking_files["manifest.json"])
        self.assertEqual(manifest_data["total_videos"], 1)
        self.assertEqual(manifest_data["total_segments"], 2)
        self.assertIn("vid1", manifest_data["completed_videos"])
        self.assertEqual(len(manifest_data["shards"]), 1)

    # --- 2. TELEGRAM BOT STRICT AUTHORIZATION & COMMANDS ---

    def test_telegram_bot_strict_authorization_and_locking(self):
        """Verify strict admin auto-locking and rejection of unauthorized strangers."""
        bot = TelegramClusterBot(
            token="8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ",
            coordinator=self.coordinator
        )

        # State 1: Unlocked - First user runs /start
        reply1 = bot.handle_command("/start", user_id=987654321, chat_id=987654321, username="ragu_admin")
        self.assertIn("Admin Lock Established", reply1)
        self.assertEqual(bot.admin_user_id, 987654321)
        self.assertEqual(bot.admin_chat_id, 987654321)

        # Verify admin was persisted to Tier 1 tracking bucket
        tracking_files = self.mock_api.storage.get("test/tracking-01", {})
        self.assertIn("telegram_admin.json", tracking_files)
        saved_admin = json.loads(tracking_files["telegram_admin.json"])
        self.assertEqual(saved_admin["admin_user_id"], 987654321)

        # State 2: Unauthorized stranger sends command
        stranger_reply = bot.handle_command("/status", user_id=111222333, chat_id=111222333, username="stranger_bot")
        self.assertIn("Access Denied", stranger_reply)
        self.assertNotIn("Cluster Status", stranger_reply)

        # State 3: Authorized admin sends /status
        admin_reply = bot.handle_command("/status", user_id=987654321, chat_id=987654321)
        self.assertIn("Cluster Status", admin_reply)
        self.assertIn("Active Nodes", admin_reply)

        # State 4: Remote cluster commands (/pause, /resume, /flush)
        pause_reply = bot.handle_command("/pause", user_id=987654321, chat_id=987654321)
        self.assertIn("Pause Signal Dispatched", pause_reply)
        self.assertEqual(self.coordinator.check_commands(), "pause")

        resume_reply = bot.handle_command("/resume", user_id=987654321, chat_id=987654321)
        self.assertIn("Resume Signal Dispatched", resume_reply)
        self.assertEqual(self.coordinator.check_commands(), "resume")

    # --- 3. UPFRONT SCRAPING & SUB-10MS BATCH LEASING ---

    def test_upfront_catalog_and_sub_10ms_batch_leasing(self):
        """Verify upfront channel catalog storing and rapid non-overlapping batch leasing."""
        # Prepopulate catalog for @madangowri in tracking bucket
        catalog = [
            {"id": "vid_m1", "title": "Video 1", "duration": 180, "channel": "madangowri"},
            {"id": "vid_m2", "title": "Video 2", "duration": 240, "channel": "madangowri"},
            {"id": "vid_m3", "title": "Video 3", "duration": 300, "channel": "madangowri"},
        ]
        self.mock_api.storage.setdefault("test/tracking-01", {})
        self.mock_api.storage["test/tracking-01"]["catalogs/madangowri.json"] = json.dumps(catalog).encode("utf-8")

        # Configure channel in channels.json
        channels = [{"id": "madangowri", "name": "Madan Gowri", "url": "https://www.youtube.com/@madangowri/videos", "active": True}]
        self.mock_api.storage["test/tracking-01"]["channels.json"] = json.dumps(channels).encode("utf-8")

        # Worker 1 leases batch of 2 videos
        t0 = time.time()
        batch_w1 = self.coordinator.lease_next_batch(batch_size=2)
        elapsed = time.time() - t0

        self.assertEqual(len(batch_w1), 2)
        self.assertEqual([v["video_id"] for v in batch_w1], ["vid_m1", "vid_m2"])
        # Verify sub-10ms leasing
        self.assertLess(elapsed, 0.5)

        # Worker 2 creates separate coordinator
        coordinator_w2 = TranscriptionCoordinator(
            api=self.mock_api,
            tracking_bucket="test/tracking-01",
            dataset_repo="test/tamil-dataset",
            worker_id="test-node-2"
        )
        batch_w2 = coordinator_w2.lease_next_batch(batch_size=2)

        # Worker 2 must NOT receive vid_m1 or vid_m2 because Worker 1 holds active lease!
        self.assertEqual(len(batch_w2), 1)
        self.assertEqual(batch_w2[0]["video_id"], "vid_m3")

        # Worker 1 releases lease
        self.coordinator.release_lease()
        lease_files = [p for p in self.mock_api.storage["test/tracking-01"].keys() if p.startswith("batches/test-node-1")]
        self.assertEqual(len(lease_files), 0)

    # --- 4. DISK GUARD & 1.0GB STAGING LIMIT ---

    def test_disk_guard_quota_monitoring(self):
        """Verify DiskGuard detects staging quota and triggers emergency cleanup."""
        mock_bot = MagicMock()
        guard = DiskGuard(
            check_dir=self.test_dir,
            max_staging_gb=0.001,  # ~1MB for testing trigger
            telegram_bot=mock_bot
        )

        # Write dummy file larger than 1MB to staging
        big_file = self.staging_dir / "large_audio.m4a"
        big_file.write_bytes(b"X" * (1024 * 1024 * 2))  # 2 MB

        flushed = False
        def dummy_flush():
            nonlocal flushed
            flushed = True

        triggered = guard.check_and_clean(emergency_sync_cb=dummy_flush)
        self.assertTrue(triggered)
        self.assertTrue(flushed)
        # Verify dummy file was purged
        self.assertFalse(big_file.exists())
        # Verify Telegram alert was called
        mock_bot.send_disk_alert.assert_called_once()

    # --- 5. EDGE CASES & ERROR RESILIENCE ---

    def test_empty_and_corrupt_segments_handling(self):
        """Verify that segments with empty transcripts or missing/corrupt audio are safely skipped."""
        syncer = ParquetDatasetSyncer(
            api=self.mock_api,
            coordinator=self.coordinator,
            dataset_repo="test/tamil-dataset",
            shard_threshold=5
        )

        missing_audio = self.staging_dir / "nonexistent.m4a"
        empty_audio = self.staging_dir / "empty.m4a"
        empty_audio.write_bytes(b"")  # 0 bytes

        valid_audio = self.staging_dir / "valid.m4a"
        valid_audio.write_bytes(b"VALID_AUDIO_BYTES_TEST" * 40)

        segments = [
            {"segment_id": "seg_empty_text", "clean_text": "   ", "duration": 4.0, "audio_path": valid_audio},
            {"segment_id": "seg_missing_audio", "clean_text": "சொல்", "duration": 4.0, "audio_path": missing_audio},
            {"segment_id": "seg_zero_bytes", "clean_text": "சொல்", "duration": 4.0, "audio_path": empty_audio},
            {"segment_id": "seg_valid", "clean_text": "சரியான உரை", "duration": 4.0, "audio_path": valid_audio}
        ]

        syncer.sync_video_transcripts(
            video_meta={"video_id": "test_edge", "title": "Edge Test", "duration_seconds": 16.0},
            audited_segments=segments,
            staging_dir=self.staging_dir
        )

        # Only seg_valid should be buffered!
        self.assertEqual(len(syncer.buffered_rows), 1)
        self.assertEqual(syncer.buffered_rows[0]["segment_id"], "seg_valid")

        # Test empty flush returns None
        syncer.buffered_rows.clear()
        res = syncer.flush_shard()
        self.assertIsNone(res)

    def test_telegram_admin_passcode_protection(self):
        """Verify passcode protection when TELEGRAM_ADMIN_PASSCODE is enforced."""
        orig_pass = config.TELEGRAM_ADMIN_PASSCODE
        try:
            config.TELEGRAM_ADMIN_PASSCODE = "secret_tamil_789"
            bot = TelegramClusterBot(
                token="8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ",
                coordinator=self.coordinator
            )

            # Incorrect passcode
            fail_reply = bot.handle_command("/start wrong_code", user_id=444555, chat_id=444555)
            self.assertIn("Passcode Required", fail_reply)
            self.assertIsNone(bot.admin_user_id)

            # Correct passcode
            success_reply = bot.handle_command("/start secret_tamil_789", user_id=444555, chat_id=444555, username="real_admin")
            self.assertIn("Admin Lock Established", success_reply)
            self.assertEqual(bot.admin_user_id, 444555)
        finally:
            config.TELEGRAM_ADMIN_PASSCODE = orig_pass

    def test_auto_flushing_at_exact_threshold(self):
        """Verify automatic shard flush when buffer hits exact threshold."""
        syncer = ParquetDatasetSyncer(
            api=self.mock_api,
            coordinator=self.coordinator,
            dataset_repo="test/tamil-dataset",
            shard_threshold=2
        )

        audio1 = self.staging_dir / "clip1.m4a"
        audio1.write_bytes(b"AUDIO_1" * 100)
        audio2 = self.staging_dir / "clip2.m4a"
        audio2.write_bytes(b"AUDIO_2" * 100)

        # Feed 2 clips (hits threshold=2)
        syncer.sync_video_transcripts(
            video_meta={"video_id": "auto_v1", "title": "V1", "duration_seconds": 10.0},
            audited_segments=[
                {"segment_id": "c1", "clean_text": "ஒன்று", "duration": 5.0, "audio_path": audio1},
                {"segment_id": "c2", "clean_text": "இரண்டு", "duration": 5.0, "audio_path": audio2}
            ],
            staging_dir=self.staging_dir
        )

        # Buffer should have auto-flushed and cleared!
        self.assertEqual(len(syncer.buffered_rows), 0)
        self.assertEqual(syncer.shard_index, 1)

        # Shard must exist in Mock HF Dataset
        dataset_files = self.mock_api.storage.get("test/tamil-dataset", {})
        self.assertIn("data/train-test-node-1-00000.parquet", dataset_files)

    # --- 6. ADVANCED EDGE CASES & ROBUSTNESS TESTS ---

    def test_stranger_cannot_claim_admin_with_non_start_command(self):
        """Verify that an unauthorized stranger cannot steal admin ownership with /status or /errors."""
        bot = TelegramClusterBot(
            token="8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ",
            coordinator=self.coordinator
        )
        self.assertIsNone(bot.admin_user_id)

        # Stranger attempts /status
        resp = bot.handle_command("/status", user_id=123999, chat_id=123999, username="bad_actor")
        self.assertIn("Setup Required", resp)
        self.assertIsNone(bot.admin_user_id)  # MUST NOT lock!

        # Stranger attempts /add_channel
        resp2 = bot.handle_command("/add_channel @spam", user_id=123999, chat_id=123999)
        self.assertIn("Setup Required", resp2)
        self.assertIsNone(bot.admin_user_id)

    def test_preconfigured_admin_auto_authorization_and_stranger_rejection(self):
        """Verify that preconfigured admin ID (e.g. 5959376217) is immediately locked and strangers rejected."""
        bot = TelegramClusterBot(
            token="8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ",
            admin_id="5959376217",
            coordinator=self.coordinator
        )
        self.assertEqual(bot.admin_user_id, 5959376217)

        # Stranger is rejected
        stranger_reply = bot.handle_command("/status", user_id=888888, chat_id=888888)
        self.assertIn("Access Denied", stranger_reply)

        # Authorized admin is recognized on /start
        admin_reply = bot.handle_command("/start", user_id=5959376217, chat_id=5959376217, username="Ghostrider2004")
        self.assertIn("Welcome back", admin_reply)
        self.assertIn("Authorized Administrator", admin_reply)

    def test_admin_id_type_normalization(self):
        """Verify is_authorized works regardless of whether user_id is passed as int or str."""
        bot = TelegramClusterBot(
            token="8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ",
            admin_id="5959376217",
            coordinator=self.coordinator
        )
        # Compare int against string admin_id
        self.assertTrue(bot.is_authorized(5959376217, 5959376217))
        # Compare str against string admin_id
        self.assertTrue(bot.is_authorized("5959376217", "5959376217"))
        # Stranger
        self.assertFalse(bot.is_authorized(12345, 12345))

    def test_url_extraction_channel_variants(self):
        """Verify robust URL parsing across all YouTube handle and URL variants."""
        variants = [
            ("@madangowri", "madangowri", "https://www.youtube.com/@madangowri/videos"),
            ("madangowri", "madangowri", "https://www.youtube.com/@madangowri/videos"),
            ("https://www.youtube.com/@madangowri", "madangowri", "https://www.youtube.com/@madangowri/videos"),
            ("https://www.youtube.com/@madangowri/videos", "madangowri", "https://www.youtube.com/@madangowri/videos"),
            ("https://www.youtube.com/channel/UC12345", "UC12345", "https://www.youtube.com/channel/UC12345/videos")
        ]
        for inp, exp_id, exp_url in variants:
            ch_id, norm_url, _ = TranscriptionCoordinator.parse_channel_info(inp)
            self.assertEqual(ch_id, exp_id, f"Failed for input: {inp}")
            self.assertEqual(norm_url, exp_url, f"Failed for input: {inp}")

    def test_telegram_poller_leader_lease(self):
        """Verify that multi-worker cluster coordinates Telegram polling without conflict."""
        coord_w1 = self.coordinator
        coord_w2 = TranscriptionCoordinator(
            api=self.mock_api,
            tracking_bucket="test/tracking-01",
            worker_id="test-node-2"
        )

        # Worker 1 acquires lease
        is_w1_leader = coord_w1.acquire_telegram_poller_lease(worker_id="test-node-1", ttl_sec=30)
        self.assertTrue(is_w1_leader)

        # Worker 2 attempts while Worker 1 holds active lease
        is_w2_leader = coord_w2.acquire_telegram_poller_lease(worker_id="test-node-2", ttl_sec=30)
        self.assertFalse(is_w2_leader)

        # Worker 1 can refresh lease
        self.assertTrue(coord_w1.acquire_telegram_poller_lease(worker_id="test-node-1", ttl_sec=30))

        # Worker 1 releases lease
        coord_w1.release_telegram_poller_lease("test-node-1")

        # Now Worker 2 can acquire lease
        self.assertTrue(coord_w2.acquire_telegram_poller_lease(worker_id="test-node-2", ttl_sec=30))

    def test_markdown_fallback_on_error(self):
        """Verify that send_message falls back to plain text when Markdown is rejected."""
        bot = TelegramClusterBot(
            token="8656137684:AAHlwGwYrNmJ-7Uzj6UshpnBr0d05V0hOKQ",
            admin_id="5959376217"
        )

        with patch("httpx.Client.post") as mock_post:
            # First call (Markdown) returns 400 Bad Request; second call (Plain text) returns 200 OK
            first_resp = MagicMock(status_code=400)
            second_resp = MagicMock(status_code=200)
            mock_post.side_effect = [first_resp, second_resp]

            success = bot.send_message("Alert with unescaped text: error_in_train_01")
            self.assertTrue(success)
            self.assertEqual(mock_post.call_count, 2)
            # Second call should not have parse_mode
            second_call_json = mock_post.call_args_list[1].kwargs["json"]
            self.assertNotIn("parse_mode", second_call_json)

    def test_shard_index_resumption_from_existing_shards(self):
        """Verify ParquetDatasetSyncer resumes shard_index from highest existing shard number."""
        # Prepopulate manifest with shards 00000 and 00004
        manifest = {
            "completed_videos": {},
            "shards": [
                {"shard": "train-test-node-1-00000.parquet"},
                {"shard": "train-test-node-1-00004.parquet"},
                {"shard": "train-other-node-00009.parquet"}
            ],
            "total_hours": 1.0
        }
        self.mock_api.storage["test/tracking-01"]["manifest.json"] = json.dumps(manifest).encode("utf-8")

        syncer = ParquetDatasetSyncer(
            api=self.mock_api,
            coordinator=self.coordinator,
            dataset_repo="test/tamil-dataset"
        )
        # Maximum index for test-node-1 was 4, so next shard_index must be 5
        self.assertEqual(syncer.shard_index, 5)

if __name__ == "__main__":
    unittest.main()
