import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yt_dlp
from huggingface_hub import HfApi
import segment.config as config

class AudioSegmenter:
    """
    Audio Segmenter for Tamil ASR:
    1. Downloads raw audio from source HF bucket or direct YouTube via yt-dlp.
    2. Runs fast ffmpeg silence detection to find natural phrase pauses.
    3. Crops clips into ~10s range (constrained between 4-15s).
    4. Extracts metadata (duration, audio, channel_name, video_id, segment_id).
    """
    def __init__(self, api: HfApi, telegram_bot: Optional[Any] = None):
        self.api = api
        self.telegram_bot = telegram_bot

    def fetch_audio(self, video_id: str, dest_dir: Path, filename: Optional[str] = None) -> Optional[Path]:
        """
        Retrieves raw audio for a video:
        1. Checks source bucket if configured.
        2. If not found in bucket, downloads audio directly from YouTube via yt-dlp
           (using player_client=['android', 'ios'] to eliminate 403 Forbidden errors).
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or f"{video_id}.m4a"
        dest_path = dest_dir / fname

        # 1. Try downloading from source bucket
        if config.SOURCE_BUCKET:
            try:
                remote_path = f"audios/{fname}"
                self.api.download_bucket_files(
                    bucket_id=config.SOURCE_BUCKET,
                    files=[(remote_path, dest_path)]
                )
                if dest_path.exists() and dest_path.stat().st_size > 10000:
                    return dest_path
            except Exception:
                pass

        # 2. Download directly from YouTube
        print(f"[SEGMENTER] Downloading audio from YouTube: {video_id}...")
        ydl_opts = {
            'format': 'ba[ext=m4a]/ba[ext=opus]/ba/b',
            'outtmpl': str(dest_dir / f"{video_id}.%(ext)s"),
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 4,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'android_vr']
                }
            },
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
                'preferredquality': '128',
            }],
        }
        if shutil.which("aria2c"):
            ydl_opts["external_downloader"] = {"default": "aria2c"}
            ydl_opts["external_downloader_args"] = {
                "aria2c": ["-s", "8", "-x", "8", "-k", "1M", "-j", "8"]
            }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)

            matching = list(dest_dir.glob(f"{video_id}.*"))
            if matching:
                filepath = matching[0]
                if filepath.suffix.lower() != ".m4a":
                    audio_target = dest_dir / f"{video_id}.m4a"
                    ret = subprocess.run(
                        ["ffmpeg", "-y", "-i", str(filepath), "-vn", "-c:a", "copy", str(audio_target)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
                    )
                    if ret.returncode != 0 or not audio_target.exists() or audio_target.stat().st_size < 10000:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", str(filepath), "-vn", "-c:a", "aac", "-b:a", "128k", str(audio_target)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
                        )
                    if audio_target.exists() and audio_target.stat().st_size > 10000:
                        filepath.unlink(missing_ok=True)
                        filepath = audio_target
                if filepath.exists() and filepath.stat().st_size > 10000:
                    return filepath
        except Exception as e:
            print(f"[SEGMENTER] Error downloading {video_id} from YouTube: {e}")
            if self.telegram_bot:
                try:
                    self.telegram_bot.send_error_alert(
                        worker_id=config.WORKER_ID,
                        video_id=video_id,
                        error_msg=f"Segmenter download failed: {e}"
                    )
                except Exception:
                    pass
        return None

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using ffprobe or ffmpeg."""
        try:
            res = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            val = float(res.stdout.strip())
            if val > 0:
                return val
        except Exception:
            pass

        try:
            res = subprocess.run(
                ["ffmpeg", "-i", str(audio_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
            if m:
                hours, mins, secs = int(m.group(1)), int(m.group(2)), float(m.group(3))
                return hours * 3600 + mins * 60 + secs
        except Exception:
            pass

        return 0.0

    def find_silence_pauses(self, audio_path: Path, noise_db: str = "-30dB", min_silence_sec: float = 0.35) -> List[Dict[str, float]]:
        """
        Uses ffmpeg silencedetect filter to find natural speech pauses.
        Returns a list of dicts: [{'start': ..., 'end': ..., 'mid': ...}, ...]
        """
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-af", f"silencedetect=noise={noise_db}:d={min_silence_sec}",
            "-f", "null", "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        pauses = []
        current_start = None

        for line in proc.stderr.splitlines():
            if "silence_start:" in line:
                m = re.search(r"silence_start:\s*([0-9.]+)", line)
                if m:
                    current_start = float(m.group(1))
            elif "silence_end:" in line:
                m_end = re.search(r"silence_end:\s*([0-9.]+)", line)
                if m_end:
                    s_end = float(m_end.group(1))
                    s_start = current_start if current_start is not None else max(0.0, s_end - min_silence_sec)
                    pauses.append({
                        "start": s_start,
                        "end": s_end,
                        "mid": round((s_start + s_end) / 2.0, 3)
                    })
                    current_start = None
        return pauses

    def compute_chunk_boundaries(self, total_duration: float, pauses: List[Dict[str, float]]) -> List[tuple]:
        """
        Calculates cut boundaries (start, end) targeting ~10 seconds,
        constrained strictly between VAD_MIN_CHUNK_SEC (4s) and VAD_MAX_CHUNK_SEC (15s).
        """
        min_sec = config.VAD_MIN_CHUNK_SEC
        max_sec = config.VAD_MAX_CHUNK_SEC
        target_sec = config.TARGET_CHUNK_SEC

        if total_duration <= max_sec:
            return [(0.0, round(total_duration, 2))]

        chunks = []
        cur_time = 0.0

        while cur_time < total_duration:
            rem = total_duration - cur_time
            if rem <= max_sec:
                if rem >= 2.0:
                    chunks.append((round(cur_time, 2), round(total_duration, 2)))
                elif chunks:
                    prev_start, _ = chunks[-1]
                    chunks[-1] = (prev_start, round(total_duration, 2))
                break

            earliest_allowed = cur_time + min_sec
            latest_allowed = cur_time + max_sec

            candidates = [p for p in pauses if earliest_allowed <= p["mid"] <= latest_allowed]

            if candidates:
                target_mid = cur_time + target_sec
                best_pause = min(candidates, key=lambda p: abs(p["mid"] - target_mid))
                cut_point = best_pause["mid"]
                chunks.append((round(cur_time, 2), round(cut_point, 2)))
                cur_time = cut_point
            else:
                cut_point = min(cur_time + target_sec, total_duration)
                chunks.append((round(cur_time, 2), round(cut_point, 2)))
                cur_time = cut_point

        return chunks

    def segment_audio_by_silence(
        self,
        audio_path: Path,
        video_id: str,
        channel_name: str,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Splits audio file based on silence detection into ~10s clips.
        Extracts duration, channel_name, video_id, and segment_id.
        """
        total_duration = self.get_audio_duration(audio_path)
        print(f"[SEGMENTER] Audio duration: {total_duration:.2f}s ({total_duration/60:.1f} mins)")

        if total_duration <= 0.0:
            print(f"[SEGMENTER] Warning: Unable to determine duration for {audio_path.name}")
            return []

        # Find speech pauses
        pauses = self.find_silence_pauses(audio_path, noise_db="-30dB", min_silence_sec=0.35)
        if len(pauses) < (total_duration / 30.0):
            more_pauses = self.find_silence_pauses(audio_path, noise_db="-25dB", min_silence_sec=0.25)
            if len(more_pauses) > len(pauses):
                pauses = more_pauses

        boundaries = self.compute_chunk_boundaries(total_duration, pauses)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _slice_one(idx: int, st: float, en: float) -> Optional[Dict[str, Any]]:
            dur = round(en - st, 2)
            seg_id = f"{video_id}_seg{idx:04d}"
            chunk_file = output_dir / f"{seg_id}.m4a"

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{st:.2f}",
                "-i", str(audio_path),
                "-t", f"{dur:.2f}",
                "-vn", "-c:a", "aac", "-b:a", "64k",
                str(chunk_file)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if chunk_file.exists() and chunk_file.stat().st_size > 500:
                return {
                    "segment_id": seg_id,
                    "video_id": video_id,
                    "channel_name": channel_name,
                    "index": idx,
                    "start": st,
                    "end": en,
                    "duration": dur,
                    "audio_path": chunk_file
                }
            return None

        segments_data = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_slice_one, idx, st, en) for idx, (st, en) in enumerate(boundaries, start=1)]
            for fut in futures:
                res = fut.result()
                if res:
                    segments_data.append(res)

        segments_data.sort(key=lambda s: s["index"])
        print(f"[SEGMENTER] Created {len(segments_data)} audio clips (~10s each).")
        return segments_data
