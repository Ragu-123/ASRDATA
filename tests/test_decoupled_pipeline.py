import os
import sys
import time
import json
import shutil
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pyarrow as pa
import pyarrow.parquet as pq

from secret_resolver import resolve_secret
import segment.config as seg_config
from segment.segmenter import AudioSegmenter
from segment.syncer import ParquetSegmentSyncer
from segment.coordinator import SegmentCoordinator
import transcribe.config as trans_config
from transcribe.coordinator import TranscriptionCoordinator
from transcribe.syncer import ParquetDatasetSyncer
from transcribe.rules import SYSTEM_PROMPT, build_user_prompt, load_rules_text

class MockHfTreeItem:
    def __init__(self, path: str):
        self.path = path

class MockHfApi:
    """Mock HfApi for isolated testing without network calls."""
    def __init__(self):
        self.storage: dict = {}  # {repo_or_bucket: {path: bytes}}

    def create_bucket(self, bucket_id: str, private: bool = False, exist_ok: bool = True):
        self.storage.setdefault(bucket_id, {})

    def create_repo(self, repo_id: str, repo_type: str = "dataset", exist_ok: bool = True):
        self.storage.setdefault(repo_id, {})

    def list_bucket_tree(self, bucket_id: str):
        files = self.storage.get(bucket_id, {})
        return [MockHfTreeItem(p) for p in files.keys()]

    def list_repo_files(self, repo_id: str, repo_type: str = "dataset"):
        files = self.storage.get(repo_id, {})
        return list(files.keys())

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

class TestDecoupledPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.staging_dir = self.test_dir / "segment_staging"
        self.shards_dir = self.test_dir / "shards"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        seg_config.WORKING_DIR = self.test_dir
        seg_config.STAGING_DIR = self.staging_dir
        seg_config.SHARDS_DIR = self.shards_dir

        trans_config.WORKING_DIR = self.test_dir
        trans_config.STAGING_DIR = self.staging_dir
        trans_config.SHARDS_DIR = self.shards_dir

        self.mock_api = MockHfApi()
        self.coordinator = SegmentCoordinator(
            api=self.mock_api,
            tracking_bucket="test/tracking-01",
            dataset_repo="test/tamil-segmented",
            worker_id="test-segmenter-1"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- 1. SECRET RESOLVER TESTING ---
    def test_secret_resolver_env_and_fallback(self):
        """Verify secret_resolver retrieves from env, kaggle mock, and .env."""
        os.environ["TEST_SECRET_KEY"] = "super_secret_value_123"
        self.assertEqual(resolve_secret("TEST_SECRET_KEY"), "super_secret_value_123")
        del os.environ["TEST_SECRET_KEY"]

        # Default fallback
        self.assertEqual(resolve_secret("NON_EXISTENT_KEY", default="fallback_val"), "fallback_val")

    def test_secret_resolver_dot_env_file(self):
        """Verify secret_resolver reads from local .env without crashing."""
        test_env_file = self.test_dir / ".env"
        test_env_file.write_text("DUMMY_HF_TOKEN=hf_mock_token_abc\n# Comment\nOTHER_KEY=123", encoding="utf-8")
        with patch("secret_resolver.Path.cwd", return_value=self.test_dir):
            val = resolve_secret("DUMMY_HF_TOKEN")
            self.assertEqual(val, "hf_mock_token_abc")

    # --- 2. MUSIC RULES & ACOUSTIC DIRECTIVES TESTING ---
    def test_music_rules_in_prompt_and_rules_md(self):
        """Verify that rules.py and RULES.md include [music] and <music> rules."""
        self.assertIn("[music]", SYSTEM_PROMPT)
        self.assertIn("<music>", SYSTEM_PROMPT)
        rules_text = load_rules_text()
        self.assertIn("[music]", rules_text)
        self.assertIn("<music>...</music>", rules_text)

    def test_build_user_prompt(self):
        """Verify prompt formatting for clean vs draft inputs."""
        p1 = build_user_prompt("")
        self.assertIn("Audio clip is provided.", p1)
        p2 = build_user_prompt("Draft audio speech")
        self.assertIn("Draft audio speech", p2)

    # --- 3. SEGMENTER SILENCE CHUNKING LOGIC ---
    def test_compute_chunk_boundaries_targeting_10sec(self):
        """Verify boundary calculation slices audio targeting ~10s clips (5-15s bounds)."""
        segmenter = AudioSegmenter(api=self.mock_api)
        # 45 seconds total duration with natural pauses
        pauses = [
            {"start": 9.5, "end": 10.2, "mid": 9.85},
            {"start": 19.8, "end": 20.3, "mid": 20.05},
            {"start": 30.1, "end": 30.6, "mid": 30.35},
            {"start": 39.5, "end": 40.1, "mid": 39.8}
        ]
        boundaries = segmenter.compute_chunk_boundaries(total_duration=45.0, pauses=pauses)
        self.assertTrue(len(boundaries) >= 4)
        for st, en in boundaries:
            dur = en - st
            self.assertGreaterEqual(dur, seg_config.VAD_MIN_CHUNK_SEC)
            self.assertLessEqual(dur, seg_config.VAD_MAX_CHUNK_SEC + 0.5)

    # --- 4. PARQUET SEGMENT SYNCER SCHEMA & METADATA ---
    def test_parquet_segment_syncer_schema_and_flushing(self):
        """Verify ParquetSegmentSyncer produces parquet shards with duration, audio struct, and channel_name."""
        syncer = ParquetSegmentSyncer(
            api=self.mock_api,
            coordinator=self.coordinator,
            dataset_repo="test/tamil-segmented",
            shard_threshold=2
        )

        dummy_audio1 = self.staging_dir / "vidA_seg0001.m4a"
        dummy_audio1.write_bytes(b"MOCK_AAC_AUDIO_DATA_FOR_SEGMENT_1" * 30)
        dummy_audio2 = self.staging_dir / "vidA_seg0002.m4a"
        dummy_audio2.write_bytes(b"MOCK_AAC_AUDIO_DATA_FOR_SEGMENT_2" * 30)

        segments = [
            {
                "segment_id": "vidA_seg0001",
                "video_id": "vidA",
                "channel_name": "Madan Gowri",
                "duration": 9.85,
                "audio_path": dummy_audio1
            },
            {
                "segment_id": "vidA_seg0002",
                "video_id": "vidA",
                "channel_name": "Madan Gowri",
                "duration": 10.2,
                "audio_path": dummy_audio2
            }
        ]

        # Ingest segments
        res = syncer.sync_video_segments(
            video_meta={"video_id": "vidA", "channel_name": "Madan Gowri", "duration_seconds": 20.05},
            segments=segments,
            staging_dir=self.staging_dir
        )
        self.assertTrue(res)

        # Local audio files must be immediately purged
        self.assertFalse(dummy_audio1.exists())
        self.assertFalse(dummy_audio2.exists())

        # Verify shard was built and uploaded
        dataset_files = self.mock_api.storage.get("test/tamil-segmented", {})
        self.assertEqual(len(dataset_files), 1)
        shard_path = list(dataset_files.keys())[0]
        self.assertTrue(shard_path.startswith("data/train-test-segmenter-1-"))

        # Verify PyArrow Table Schema & Data
        parquet_bytes = dataset_files[shard_path]
        table = pq.read_table(pa.BufferReader(parquet_bytes))
        self.assertEqual(table.num_rows, 2)
        self.assertEqual(
            table.column_names,
            ["audio", "duration", "channel_name", "video_id", "segment_id"]
        )
        self.assertEqual(table["channel_name"][0].as_py(), "Madan Gowri")
        self.assertEqual(table["video_id"][0].as_py(), "vidA")
        self.assertEqual(table["segment_id"][0].as_py(), "vidA_seg0001")
        self.assertAlmostEqual(table["duration"][0].as_py(), 9.85, places=2)
        self.assertEqual(table["audio"][0]["path"].as_py(), "vidA_seg0001.m4a")
        self.assertEqual(table["audio"][0]["bytes"].as_py(), b"MOCK_AAC_AUDIO_DATA_FOR_SEGMENT_1" * 30)

        # Verify tracking bucket segment_manifest.json
        tracking_files = self.mock_api.storage.get("test/tracking-01", {})
        self.assertIn("segment_manifest.json", tracking_files)
        manifest = json.loads(tracking_files["segment_manifest.json"])
        self.assertEqual(manifest["total_videos"], 1)
        self.assertEqual(manifest["total_segments"], 2)
        self.assertIn("vidA", manifest["completed_videos"])

    # --- 5. PARALLEL KAGGLE SESSIONS CONCURRENCY & ZERO COLLISIONS ---
    def test_segment_coordinator_multi_worker_non_overlapping_leases(self):
        """Verify two parallel workers never receive the same video lease."""
        worker1 = SegmentCoordinator(
            api=self.mock_api,
            tracking_bucket="test/tracking-01",
            dataset_repo="test/tamil-segmented",
            worker_id="node-alpha"
        )
        worker2 = SegmentCoordinator(
            api=self.mock_api,
            tracking_bucket="test/tracking-01",
            dataset_repo="test/tamil-segmented",
            worker_id="node-beta"
        )

        channels = [
            {
                "id": "test_ch",
                "name": "Test Channel",
                "url": "https://www.youtube.com/@test/videos",
                "active": True
            }
        ]
        worker1.save_channels(channels)
        catalog = [
            {"video_id": f"vid_{i:03d}", "title": f"Video {i}", "duration_seconds": 60.0}
            for i in range(10)
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump(catalog, tmp)
            tmp_p = Path(tmp.name)
        self.mock_api.batch_bucket_files("test/tracking-01", add=[(tmp_p, "catalogs/test_ch.json")])
        tmp_p.unlink()

        # Worker 1 leases batch of 3
        batch1 = worker1.lease_next_batch(batch_size=3)
        self.assertEqual(len(batch1), 3)
        vids1 = {v["video_id"] for v in batch1}

        # Worker 2 leases batch of 3 concurrently
        batch2 = worker2.lease_next_batch(batch_size=3)
        self.assertEqual(len(batch2), 3)
        vids2 = {v["video_id"] for v in batch2}

        # Zero overlap guaranteed!
        self.assertEqual(len(vids1.intersection(vids2)), 0)

        # Worker 1 completes and releases lease
        worker1.release_lease()
        worker2.release_lease()

    # --- 6. TRANSCRIBE PULLING FROM SEGMENTED SHARDS ---
    def test_transcribe_coordinator_leasing_and_auditing_from_segmented_shard(self):
        """Verify transcribe coordinator leases segmented shards and syncs audited records."""
        # Setup a segmented shard in mock dataset repo
        rows = [
            {
                "audio": {"bytes": b"CLIP_1_AUDIO_BYTES_TESTING", "path": "seg1.m4a"},
                "duration": 8.5,
                "channel_name": "Channel X",
                "video_id": "vidX",
                "segment_id": "vidX_seg0001"
            },
            {
                "audio": {"bytes": b"CLIP_2_AUDIO_BYTES_TESTING", "path": "seg2.m4a"},
                "duration": 9.2,
                "channel_name": "Channel X",
                "video_id": "vidX",
                "segment_id": "vidX_seg0002"
            }
        ]
        schema = pa.schema([
            ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("duration", pa.float64()),
            ("channel_name", pa.string()),
            ("video_id", pa.string()),
            ("segment_id", pa.string())
        ])
        table = pa.Table.from_pylist(rows, schema=schema)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_parquet = Path(tmp.name)
        pq.write_table(table, str(tmp_parquet), compression="snappy")

        # Upload to mock segmented dataset
        self.mock_api.upload_file(
            path_or_fileobj=str(tmp_parquet),
            path_in_repo="data/train-source-00000.parquet",
            repo_id="test/segmented-input",
            repo_type="dataset"
        )
        tmp_parquet.unlink()

        # Initialize TranscriptionCoordinator
        trans_coord = TranscriptionCoordinator(
            api=self.mock_api,
            tracking_bucket="test/tracking-01",
            dataset_repo="test/tamil-audited-corpus",
            worker_id="transcriber-1"
        )

        # Lease shard
        lease = trans_coord.lease_segmented_shard(segmented_repo="test/segmented-input")
        self.assertIsNotNone(lease)
        self.assertEqual(lease["shard_file"], "data/train-source-00000.parquet")

        # Worker 2 trying to lease gets None (already leased)
        trans_coord2 = TranscriptionCoordinator(
            api=self.mock_api,
            tracking_bucket="test/tracking-01",
            dataset_repo="test/tamil-audited-corpus",
            worker_id="transcriber-2"
        )
        lease2 = trans_coord2.lease_segmented_shard(segmented_repo="test/segmented-input")
        self.assertIsNone(lease2)

        # Syncer ingests audited segments with music rules
        trans_syncer = ParquetDatasetSyncer(
            api=self.mock_api,
            coordinator=trans_coord,
            dataset_repo="test/tamil-audited-corpus",
            shard_threshold=5
        )

        audited_clips = [
            {
                "segment_id": "vidX_seg0001",
                "video_id": "vidX",
                "audio_bytes": b"CLIP_1_AUDIO_BYTES_TESTING",
                "clean_text": "<music>வணக்கம் நண்பர்களே</music>",
                "emotion": "None",
                "duration": 8.5
            },
            {
                "segment_id": "vidX_seg0002",
                "video_id": "vidX",
                "audio_bytes": b"CLIP_2_AUDIO_BYTES_TESTING",
                "clean_text": "[music]",
                "emotion": "None",
                "duration": 9.2
            }
        ]

        trans_syncer.sync_audited_segments(audited_clips, shard_id=lease["shard_file"])
        shard_out = trans_syncer.flush_shard()
        self.assertIsNotNone(shard_out)

        # Verify audited repo output
        audited_storage = self.mock_api.storage.get("test/tamil-audited-corpus", {})
        self.assertIn(f"data/{shard_out}", audited_storage)

        audited_table = pq.read_table(pa.BufferReader(audited_storage[f"data/{shard_out}"]))
        self.assertEqual(audited_table.num_rows, 2)
        self.assertEqual(audited_table["sentence"][0].as_py(), "<music>வணக்கம் நண்பர்களே</music>")
        self.assertEqual(audited_table["sentence"][1].as_py(), "[music]")

        # Clean release
        trans_coord.release_shard_lease()

    # --- 7. CLI COMMAND LAUNCHERS AND ALIASES ---
    def test_cli_help_flags(self):
        """Verify that main.py, ingest.py, segment.py, segement.py, transcribe.py, trascribe.py execute --help successfully."""
        scripts = ["main.py", "ingest.py", "segment.py", "segement.py", "transcribe.py", "trascribe.py"]
        for s in scripts:
            script_path = ROOT_DIR / s
            self.assertTrue(script_path.exists(), f"File {s} does not exist!")
            res = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.assertEqual(res.returncode, 0, f"Error running {s} --help: {res.stderr}")

    # --- 8. DELETE CHANNEL AND DELETE MANIFEST ---
    def test_coordinator_delete_channel_and_manifest(self):
        """Verify deleting a channel removes catalog, channels.json entry, and manifest videos."""
        channels = [
            {"id": "del_ch", "name": "To Delete", "url": "https://www.youtube.com/@del_ch/videos", "active": True},
            {"id": "keep_ch", "name": "To Keep", "url": "https://www.youtube.com/@keep_ch/videos", "active": True}
        ]
        self.coordinator.save_channels(channels)

        catalog = [{"id": "vid_del_1", "title": "Del 1"}, {"id": "vid_del_2", "title": "Del 2"}]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump(catalog, tmp)
            tmp_p = Path(tmp.name)
        self.mock_api.batch_bucket_files("test/tracking-01", add=[(tmp_p, "catalogs/del_ch.json")])
        tmp_p.unlink()

        manifest = {
            "total_videos": 2,
            "total_segments": 10,
            "total_hours": 1.5,
            "shards": [],
            "completed_videos": {
                "vid_del_1": {"duration_seconds": 1800.0, "segments_count": 5, "channel_name": "To Delete"},
                "vid_keep": {"duration_seconds": 3600.0, "segments_count": 5, "channel_name": "To Keep"}
            }
        }
        self.coordinator.save_manifest(manifest)

        # Delete channel
        res = self.coordinator.delete_channel("@del_ch")
        self.assertEqual(res["channel_id"], "del_ch")
        self.assertEqual(res["manifest_videos_removed"], 1)

        # Verify catalog removed
        bucket_files = self.mock_api.storage.get("test/tracking-01", {})
        self.assertNotIn("catalogs/del_ch.json", bucket_files)

        # Verify channels.json updated
        updated_channels = self.coordinator.load_channels(force=True)
        self.assertEqual(len(updated_channels), 1)
        self.assertEqual(updated_channels[0]["id"], "keep_ch")

        # Verify manifest updated
        updated_manifest = self.coordinator.load_manifest(force=True)
        self.assertNotIn("vid_del_1", updated_manifest["completed_videos"])
        self.assertIn("vid_keep", updated_manifest["completed_videos"])
        self.assertEqual(updated_manifest["total_videos"], 1)

        # Test delete_manifest
        self.coordinator.delete_manifest()
        cleared_manifest = self.coordinator.load_manifest(force=True)
        self.assertEqual(cleared_manifest["total_videos"], 0)
        self.assertEqual(cleared_manifest["total_hours"], 0.0)

    # --- 9. TELEGRAM BOT COMMANDS & 3-STAGE STATUS ---
    def test_telegram_bot_delete_commands_and_status(self):
        """Verify Telegram bot handles /delete_channel, /delete_manifest, and 3-stage /status."""
        from transcribe.telegram_bot import TelegramClusterBot
        bot = TelegramClusterBot(
            token="test_bot_token_mock",
            admin_id="12345678",
            coordinator=self.coordinator
        )

        manifest = {
            "total_videos": 1,
            "total_segments": 5,
            "total_hours": 0.5,
            "shards": [{"shard": "train-01.parquet"}],
            "completed_videos": {}
        }
        self.coordinator.save_manifest(manifest)

        # Status output contains 3 stages
        status_reply = bot.handle_command("/status", user_id=12345678, chat_id=12345678)
        self.assertIn("Stage 1: Raw Ingest", status_reply)
        self.assertIn("Stage 2: Silence Segmentation", status_reply)
        self.assertIn("Stage 3: Transcription", status_reply)

        # Delete channel command
        del_reply = bot.handle_command("/delete_channel @test_ch", user_id=12345678, chat_id=12345678)
        self.assertIn("Channel & Manifest Cleaned", del_reply)

        # Reset manifest command
        reset_reply = bot.handle_command("/delete_manifest", user_id=12345678, chat_id=12345678)
        self.assertIn("Manifest Reset Complete", reset_reply)

    # --- 10. WATCHDOG STALL ALERT ---
    def test_telegram_bot_watchdog_stall_alert(self):
        """Verify watchdog alerts when a worker has been inactive for > 10 minutes."""
        from transcribe.telegram_bot import TelegramClusterBot
        bot = TelegramClusterBot(
            token="test_bot_token_mock",
            admin_id="12345678",
            coordinator=self.coordinator
        )

        stalled_worker = {
            "worker_id": "stalled-node-1",
            "stage": "segment",
            "status": "segmenting",
            "current_video": {"video_id": "stalled_vid_xyz"},
            "last_heartbeat": time.time() - 750  # 12.5 mins ago
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump(stalled_worker, tmp)
            tmp_p = Path(tmp.name)
        self.mock_api.batch_bucket_files("test/tracking-01", add=[(tmp_p, "workers/stalled-node-1.json")])
        tmp_p.unlink()

        with patch.object(bot, "send_message", return_value=True) as mock_send:
            bot.check_watchdog_stalls()
            self.assertTrue(mock_send.called)
            alert_text = mock_send.call_args[0][0]
            self.assertIn("STALL ALERT", alert_text)
            self.assertIn("stalled-node-1", alert_text)
            self.assertIn("stalled_vid_xyz", alert_text)

    # --- 11. ARIA2C OPTIMIZATION CHECK ---
    def test_aria2c_downloader_opts(self):
        """Verify aria2c downloader flags are configured when aria2c is available."""
        from transcribe.ingest import get_ytdlp_audio_opts
        with patch("shutil.which", return_value="/usr/bin/aria2c"):
            opts = get_ytdlp_audio_opts("test_template.%(ext)s")
            self.assertIn("external_downloader", opts)
            self.assertEqual(opts["external_downloader"]["default"], "aria2c")
            self.assertIn("aria2c", opts["external_downloader_args"])

        with patch("shutil.which", return_value=None):
            opts_native = get_ytdlp_audio_opts("test_template.%(ext)s")
            self.assertNotIn("external_downloader", opts_native)

if __name__ == "__main__":
    unittest.main()
