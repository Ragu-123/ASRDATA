import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from secret_resolver import resolve_secret, ALIAS_MAP
from transcribe.telegram_bot import TelegramClusterBot
import config
from worker import IngestionWorker

class TestBotAndYtdlpResilience(unittest.TestCase):

    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    # 1. SECRET RESOLUTION ALIASES
    def test_secret_resolver_aliases_env(self):
        """Verify resolve_secret checks aliases such as bot_token, telegram_admin_id, etc."""
        # Test bot_token alias
        os.environ["bot_token"] = "test_bot_token_123"
        val = resolve_secret("TELEGRAM_BOT_TOKEN")
        self.assertEqual(val, "test_bot_token_123")

        # Test telegram_admin_id alias
        os.environ["chat_id"] = "987654321"
        admin_val = resolve_secret("TELEGRAM_ADMIN_ID")
        self.assertEqual(admin_val, "987654321")

        # Test lowercase hf_token alias
        os.environ["hf_token"] = "hf_test_secret_token"
        hf_val = resolve_secret("HF_TOKEN")
        self.assertEqual(hf_val, "hf_test_secret_token")

    def test_secret_resolver_aliases_in_env_file(self):
        """Verify resolve_secret checks aliases inside .env files."""
        temp_env = Path.cwd() / ".env"
        try:
            temp_env.write_text("telegram_bot_token=tok_from_env_file\nADMIN_USER_ID=555666777\n", encoding="utf-8")
            # Clear matching os.environ
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            os.environ.pop("telegram_bot_token", None)
            os.environ.pop("TELEGRAM_ADMIN_ID", None)
            os.environ.pop("ADMIN_USER_ID", None)

            tok = resolve_secret("TELEGRAM_BOT_TOKEN")
            self.assertEqual(tok, "tok_from_env_file")

            admin_id = resolve_secret("TELEGRAM_ADMIN_ID")
            self.assertEqual(admin_id, "555666777")
        finally:
            if temp_env.exists():
                temp_env.unlink()

    # 2. TELEGRAM BOT SEND_MESSAGE AND FALLBACKS
    def test_telegram_bot_send_message_target_chat_id_fallback(self):
        """Verify send_message sends to explicit chat_id or falls back to admin_chat_id or admin_user_id."""
        bot = TelegramClusterBot(
            token="123456789:AAFakeTestTokenForUnitTestingOnly123",
            admin_id="5959376217"
        )
        self.assertEqual(bot.admin_user_id, 5959376217)
        self.assertEqual(bot.admin_chat_id, 5959376217)
        self.assertEqual(bot.chat_id, 5959376217)

        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)

            # Case 1: Default to admin_chat_id
            bot.send_message("Hello Admin!")
            payload = mock_post.call_args.kwargs["json"]
            self.assertEqual(payload["chat_id"], 5959376217)
            self.assertEqual(payload["text"], "Hello Admin!")

            # Case 2: Override with explicit chat_id
            bot.send_message("Custom alert", chat_id=11223344)
            payload2 = mock_post.call_args.kwargs["json"]
            self.assertEqual(payload2["chat_id"], 11223344)

            # Case 3: When admin_chat_id is None, fall back to admin_user_id
            bot.admin_chat_id = None
            bot.send_message("User fallback alert")
            payload3 = mock_post.call_args.kwargs["json"]
            self.assertEqual(payload3["chat_id"], 5959376217)

    def test_telegram_bot_send_alert(self):
        """Verify send_alert formats title and details and dispatches to Telegram."""
        bot = TelegramClusterBot(
            token="123456789:AAFakeTestTokenForUnitTestingOnly123",
            admin_id="5959376217"
        )

        with patch.object(bot, "send_message", return_value=True) as mock_send:
            bot.send_alert("Disk Alert", "Staging directory at 95%")
            mock_send.assert_called_once()
            args, kwargs = mock_send.call_args
            self.assertIn("Disk Alert", kwargs.get("text", args[0] if args else ""))
            self.assertIn("95%", kwargs.get("text", args[0] if args else ""))

    # 3. WORKER TELEGRAM STARTUP PING & SHUTDOWN
    def test_worker_telegram_startup_and_stop_notifications(self):
        """Verify IngestionWorker notifies Telegram on start() and stop()."""
        mock_coordinator = MagicMock()
        mock_coordinator.manifest_cache = {"total_hours": 0.0, "completed": {}}
        mock_bucket_sync = MagicMock()
        mock_bot = MagicMock()

        worker = IngestionWorker(
            coordinator=mock_coordinator,
            bucket_sync=mock_bucket_sync,
            telegram_bot=mock_bot
        )

        # Patch worker and heartbeat loop so they don't block
        with patch.object(worker, "_run_loop"):
            worker.start()
            self.assertTrue(mock_bot.send_message.called)
            start_msg = mock_bot.send_message.call_args[0][0]
            self.assertIn("started", start_msg)
            self.assertIn("Ingestion", start_msg)

            # Stop worker
            mock_bot.send_message.reset_mock()
            worker.stop()
            self.assertTrue(mock_bot.send_message.called)
            stop_msg = mock_bot.send_message.call_args[0][0]
            self.assertIn("stopped", stop_msg)

    # 4. WORKER FAILURE ALERT ON DOWNLOAD ERROR
    def test_worker_failure_alert_dispatched_on_download_error(self):
        """Verify IngestionWorker dispatches send_error_alert to Telegram when download fails."""
        mock_coordinator = MagicMock()
        mock_coordinator.manifest_cache = {"total_hours": 0.0, "completed": {}}
        mock_coordinator.lease_next_batch.return_value = [{"id": "bad_video_123", "title": "Test Video"}]
        mock_coordinator.release_batch.return_value = None
        mock_coordinator.check_commands.return_value = "stop"
        mock_bucket_sync = MagicMock()
        mock_bot = MagicMock()

        worker = IngestionWorker(
            coordinator=mock_coordinator,
            bucket_sync=mock_bucket_sync,
            telegram_bot=mock_bot
        )
        worker.running = True

        with patch("worker.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = mock_ydl_cls.return_value.__enter__.return_value
            mock_ydl.extract_info.side_effect = Exception("Bot detection error")

            worker._process_batch_concurrent([{"id": "bad_video_123", "title": "Test Video"}])

            self.assertTrue(mock_bot.send_error_alert.called)
            call_kwargs = mock_bot.send_error_alert.call_args.kwargs
            self.assertEqual(call_kwargs["video_id"], "bad_video_123")
            self.assertIn("Bot detection error", call_kwargs["error_msg"])

    # 5. YT-DLP PLAYER_CLIENT CONFIGURATION ACROSS MODULES
    def test_ytdlp_player_clients_include_android_and_ios(self):
        """Verify yt-dlp options across all downloaders include android, ios, and android_vr."""
        from transcribe.ingest import get_ytdlp_audio_opts
        opts = get_ytdlp_audio_opts("test.%(ext)s")
        clients = opts["extractor_args"]["youtube"]["player_client"]
        skips = opts["extractor_args"]["youtube"]["player_skip"]
        self.assertIn("android", clients)
        self.assertIn("ios", clients)
        self.assertIn("android_vr", clients)
        self.assertIn("web", skips)
        self.assertIn("ba/b", opts["format"])

    def test_segmenter_player_clients_configured(self):
        """Verify AudioSegmenter in segment/segmenter.py configures player_client and accepts telegram_bot."""
        from segment.segmenter import AudioSegmenter
        mock_api = MagicMock()
        mock_bot = MagicMock()
        segmenter = AudioSegmenter(api=mock_api, telegram_bot=mock_bot)
        self.assertIs(segmenter.telegram_bot, mock_bot)

    def test_transcribe_segmenter_player_clients_configured(self):
        """Verify AudioSegmenter in transcribe/segmenter.py configures player_client and accepts telegram_bot."""
        from transcribe.segmenter import AudioSegmenter
        mock_api = MagicMock()
        mock_bot = MagicMock()
        segmenter = AudioSegmenter(api=mock_api, telegram_bot=mock_bot)
        self.assertIs(segmenter.telegram_bot, mock_bot)

    # 6. TELEGRAM BOT COMMAND MENU REGISTRATION & DEFAULT ADMIN
    def test_telegram_bot_command_menu_registration(self):
        """Verify setup_bot_commands registers commands and menu button with Telegram."""
        with patch("httpx.Client.post") as mock_post, patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: {"ok": True, "result": {"id": 123, "username": "test_bot"}})
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})

            bot = TelegramClusterBot(token="123456:fake_token")
            # Called during init
            endpoints_called = [call.args[0] for call in mock_post.call_args_list]
            self.assertTrue(any("setMyCommands" in ep for ep in endpoints_called))
            self.assertTrue(any("setChatMenuButton" in ep for ep in endpoints_called))

    def test_telegram_bot_default_admin_id_fallback(self):
        """Verify TelegramClusterBot send_message defaults to 5959376217 when no admin is specified or cached."""
        import transcribe.config as t_config
        orig = t_config.TELEGRAM_ADMIN_ID
        try:
            t_config.TELEGRAM_ADMIN_ID = ""
            bot = TelegramClusterBot(token="123456:fake_token", admin_id="")
            self.assertIsNone(bot.admin_user_id)
            with patch("httpx.Client.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200)
                bot.send_message("Alert without explicit admin")
                payload = mock_post.call_args.kwargs["json"]
                self.assertEqual(payload["chat_id"], "5959376217")
        finally:
            t_config.TELEGRAM_ADMIN_ID = orig

    def test_telegram_bot_send_error_alert_buffers_recent_errors(self):
        """Verify send_error_alert buffers errors in recent_errors for the /errors command."""
        bot = TelegramClusterBot(token="123456:fake_token", admin_id="5959376217")
        with patch.object(bot, "send_message", return_value=True):
            bot.send_error_alert("node-1", "vid_xyz", "HTTP 403 Forbidden")
            self.assertEqual(len(bot.recent_errors), 1)
            self.assertEqual(bot.recent_errors[0]["video_id"], "vid_xyz")
            self.assertIn("HTTP 403", bot.recent_errors[0]["error"])

    def test_segmenter_audio_transcode_resilience(self):
        """Verify AudioSegmenter fetch_audio handles non-m4a formats and transcodes them safely."""
        from segment.segmenter import AudioSegmenter
        import tempfile
        import subprocess

        mock_api = MagicMock()
        mock_api.download_bucket_files.side_effect = Exception("Not in bucket")
        segmenter = AudioSegmenter(api=mock_api)

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            vid = "test_transcode_vid"

            with patch("segment.segmenter.yt_dlp.YoutubeDL") as mock_ydl_cls:
                def fake_download(*args, **kwargs):
                    # Simulate yt-dlp downloading an opus webm file
                    webm_file = dest / f"{vid}.webm"
                    # Generate a 3s audio webm file with ffmpeg (>10KB)
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "libopus", str(webm_file)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                mock_ydl = mock_ydl_cls.return_value.__enter__.return_value
                mock_ydl.extract_info.side_effect = fake_download

                result = segmenter.fetch_audio(vid, dest)
                self.assertIsNotNone(result)
                self.assertTrue(result.exists())
                self.assertEqual(result.suffix.lower(), ".m4a")
                self.assertGreater(result.stat().st_size, 10000)

if __name__ == "__main__":
    unittest.main()

