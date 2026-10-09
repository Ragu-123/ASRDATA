import os
import sys
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi
import transcribe.config as config

class AudioSegmenter:
    """
    Downloads raw YouTube audio from source HF bucket,
    detects speech pauses via fast ffmpeg silence detection,
    and extracts clean 5-15s clips for Gemini Canvas Proxy without Whisper.
    """
    def __init__(self, api: HfApi):
        self.api = api

    def download_source_audio(self, filename: str, dest_dir: Path) -> Optional[Path]:
        """Download raw audio file from King758/media-archive-01."""
        dest_path = dest_dir / filename
        remote_path = f"audios/{filename}"
        try:
            self.api.download_bucket_files(
                bucket_id=config.SOURCE_BUCKET,
                files=[(remote_path, dest_path)]
            )
            if dest_path.exists() and dest_path.stat().st_size > 10000:
                return dest_path
        except Exception as e:
            print(f"[SEGMENTER] Error downloading {remote_path}: {e}")
        return None

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using ffprobe or ffmpeg."""
        # 1. ffprobe method
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

        # 2. ffmpeg banner parse method
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
        Calculates cut boundaries (start, end) targeting ~8-12 seconds,
        constrained strictly between VAD_MIN_CHUNK_SEC and VAD_MAX_CHUNK_SEC.
        """
        min_sec = config.VAD_MIN_CHUNK_SEC
        max_sec = config.VAD_MAX_CHUNK_SEC
        target_sec = 10.0

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
                    # Merge tiny tail into previous chunk
                    prev_start, _ = chunks[-1]
                    chunks[-1] = (prev_start, round(total_duration, 2))
                break

            # Find candidate pauses within [cur_time + min_sec, cur_time + max_sec]
            earliest_allowed = cur_time + min_sec
            latest_allowed = cur_time + max_sec

            candidates = [p for p in pauses if earliest_allowed <= p["mid"] <= latest_allowed]

            if candidates:
                # Pick the pause closest to cur_time + target_sec
                target_mid = cur_time + target_sec
                best_pause = min(candidates, key=lambda p: abs(p["mid"] - target_mid))
                cut_point = best_pause["mid"]
                chunks.append((round(cur_time, 2), round(cut_point, 2)))
                cur_time = cut_point
            else:
                # No pause found in the window (e.g. continuous fast speech)
                # Split cleanly at target_sec
                cut_point = min(cur_time + target_sec, total_duration)
                chunks.append((round(cur_time, 2), round(cut_point, 2)))
                cur_time = cut_point

        return chunks

    def segment_audio_by_silence(self, audio_path: Path, video_id: str, output_dir: Path) -> List[Dict[str, Any]]:
        """
        Pure silence-based audio segmenter.
        Scans for natural pauses with ffmpeg and cuts into clean 5-15s clips.
        Zero Whisper overhead, zero model loading!
        """
        total_duration = self.get_audio_duration(audio_path)
        print(f"[SEGMENTER] Audio duration: {total_duration:.2f}s ({total_duration/60:.1f} mins)")

        if total_duration <= 0.0:
            print(f"[SEGMENTER] Warning: Unable to determine duration for {audio_path.name}")
            return []

        # Find speech pauses using ffmpeg silencedetect
        print(f"[SEGMENTER] Detecting silence pauses with ffmpeg (noise=-30dB)...")
        pauses = self.find_silence_pauses(audio_path, noise_db="-30dB", min_silence_sec=0.35)

        # If very few pauses detected (e.g. loud music/background), try a more relaxed noise threshold
        if len(pauses) < (total_duration / 30.0):
            more_pauses = self.find_silence_pauses(audio_path, noise_db="-25dB", min_silence_sec=0.25)
            if len(more_pauses) > len(pauses):
                pauses = more_pauses

        print(f"[SEGMENTER] Detected {len(pauses)} natural pauses. Computing cut boundaries...")
        boundaries = self.compute_chunk_boundaries(total_duration, pauses)
        print(f"[SEGMENTER] Generated {len(boundaries)} chunks (~10s each). Slicing audio files in parallel...")

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
                    "index": idx,
                    "start": st,
                    "end": en,
                    "duration": dur,
                    "audio_path": chunk_file,
                    "draft_text": ""
                }
            return None

        # Slice all chunks in parallel using ThreadPoolExecutor
        segments_data = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_slice_one, idx, st, en) for idx, (st, en) in enumerate(boundaries, start=1)]
            for fut in futures:
                res = fut.result()
                if res:
                    segments_data.append(res)

        segments_data.sort(key=lambda s: s["index"])
        print(f"[SEGMENTER] Successfully created {len(segments_data)} audio segments ready for Gemini!")
        return segments_data

    def segment_with_whisper_and_vad(self, audio_path: Path, video_id: str, output_dir: Path) -> List[Dict[str, Any]]:
        """Alias for backward compatibility - directs to fast silence segmenter."""
        return self.segment_audio_by_silence(audio_path, video_id, output_dir)
