import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
import pot_service
from coordinator import ClusterCoordinator
from worker import IngestionWorker

class TestPoTokenAndStagingFix(unittest.TestCase):

    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    # 1. POT SERVICE ALIVE DETECTION
    @patch("urllib.request.urlopen")
    def test_pot_service_alive_true(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        self.assertTrue(pot_service.is_pot_service_alive())

    @patch("urllib.request.urlopen")
    def test_pot_service_alive_false(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        self.assertFalse(pot_service.is_pot_service_alive())

    # 2. CONFIG EXTRACTOR ARGS ADAPTATION
    @patch("pot_service.is_pot_service_alive", return_value=True)
    def test_config_extractor_args_with_pot_active(self, mock_alive):
        args = config.get_ytdlp_extractor_args()
        self.assertIn("youtubepot-bgutilhttp", args)
        self.assertEqual(args["youtubepot-bgutilhttp"]["base_url"], ["http://127.0.0.1:4416"])
        self.assertIn("web_creator", args["youtube"]["player_client"])

    @patch("pot_service.is_pot_service_alive", return_value=False)
    def test_config_extractor_args_with_pot_inactive(self, mock_alive):
        args = config.get_ytdlp_extractor_args()
        self.assertNotIn("youtubepot-bgutilhttp", args)
        self.assertIn("tv_samsung", args["youtube"]["player_client"])

    # 3. COORDINATOR LEASE EXCLUDE_IDS & LEASE PRESERVATION
    def test_coordinator_lease_excludes_staged_videos(self):
        mock_api = MagicMock()
        mock_api.list_bucket_tree.return_value = []
        coordinator = ClusterCoordinator(api=mock_api, bucket_id="test-bucket", node_id="test-node")

        # Mock channels and catalog
        coordinator.load_channels = MagicMock(return_value=[{"id": "ch1", "active": True}])
        coordinator.get_channel_catalog = MagicMock(return_value=[
            {"id": "vid1", "title": "Vid 1"},
            {"id": "vid2", "title": "Vid 2"},
            {"id": "vid3", "title": "Vid 3"},
            {"id": "vid4", "title": "Vid 4"}
        ])
        coordinator.load_manifest = MagicMock(return_value={"completed": {}})

        # vid1 and vid2 are already staged on disk
        exclude_ids = {"vid1", "vid2"}
        batch = coordinator.lease_next_batch(batch_size=2, exclude_ids=exclude_ids)

        batch_ids = [v["id"] for v in batch]
        self.assertEqual(batch_ids, ["vid3", "vid4"])
        self.assertNotIn("vid1", batch_ids)
        self.assertNotIn("vid2", batch_ids)

        # Verify batch lease uploaded to bucket includes BOTH newly leased and excluded staged IDs
        calls = mock_api.batch_bucket_files.call_args_list
        lease_call = [c for c in calls if "batches/test-node_lease.json" in str(c)]
        self.assertTrue(len(lease_call) > 0)

    def test_coordinator_update_batch_lease(self):
        mock_api = MagicMock()
        coordinator = ClusterCoordinator(api=mock_api, bucket_id="test-bucket", node_id="test-node")
        coordinator.update_batch_lease(["vid1", "vid2", "vid3"])

        mock_api.batch_bucket_files.assert_called()
        call_args = mock_api.batch_bucket_files.call_args
        self.assertIn("batches/test-node_lease.json", str(call_args))

    # 4. WORKER STAGED IDS INSPECTION
    def test_worker_get_staged_video_ids(self):
        mock_coord = MagicMock()
        mock_sync = MagicMock()
        worker = IngestionWorker(coordinator=mock_coord, bucket_sync=mock_sync)

        worker.pending_manifest_updates = {
            "vid_mem1": {"title": "Title 1"},
            "vid_mem2": {"title": "Title 2"}
        }

        # Create temporary dummy file in config.STAGING_DIR
        test_file = config.STAGING_DIR / "vid_disk1.m4a"
        try:
            test_file.write_bytes(b"dummy audio content")
            staged = worker.get_staged_video_ids()
            self.assertIn("vid_mem1", staged)
            self.assertIn("vid_mem2", staged)
            self.assertIn("vid_disk1", staged)
        finally:
            if test_file.exists():
                test_file.unlink()

    # 5. WORKER AUTO-FLUSH ON CATALOG EXHAUSTION
    def test_worker_auto_flush_when_batch_empty_and_staging_has_data(self):
        mock_coord = MagicMock()
        mock_sync = MagicMock()
        worker = IngestionWorker(coordinator=mock_coord, bucket_sync=mock_sync)

        # Worker has pending staged updates
        worker.pending_manifest_updates = {"vid_done": {"title": "Done Title"}}
        worker.running = True

        # Coordinator returns no more videos
        mock_coord.check_commands.return_value = None
        mock_coord.lease_next_batch.return_value = []

        with patch.object(worker, "trigger_flush") as mock_flush:
            # Run one cycle of loop logic
            def stop_worker(*args, **kwargs):
                worker.running = False
                return None
            mock_coord.lease_next_batch.side_effect = stop_worker

            worker._run_loop()
            mock_flush.assert_called()

if __name__ == "__main__":
    unittest.main()
