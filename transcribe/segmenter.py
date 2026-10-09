import os
import sys
import re
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi
import transcribe.config as config

class AudioSegmenter:
    """
    Downloads raw YouTube audio from source HF bucket,
    detects speech pauses via VAD / Whisper, and extracts clean 5-15s clips.
    """
    def __init__(self, api: HfApi, whisper_model_name: str = config.WHISPER_MODEL):
        self.api = api
        self.whisper_model_name = whisper_model_name
        self._whisper_model = None

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

    def _get_whisper_model(self):
        """Lazy loader for faster-whisper model."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                print(f"[SEGMENTER] Initializing faster-whisper ({self.whisper_model_name}) on {device.upper()} ({compute_type})...")
                self._whisper_model = WhisperModel(
                    self.whisper_model_name,
                    device=device,
                    compute_type=compute_type
                )
            except Exception as e:
                print(f"[SEGMENTER] faster-whisper not available or failed to load: {e}")
                self._whisper_model = False
        return self._whisper_model

    def segment_with_whisper_and_vad(self, audio_path: Path, video_id: str, output_dir: Path) -> List[Dict[str, Any]]:
        """
        Stage 1: Uses faster-whisper with native Silero VAD to segment audio at natural pauses
        and generate initial draft Tamil transcripts simultaneously.
        """
        model = self._get_whisper_model()
        if not model:
            return self.segment_with_ffmpeg_vad(audio_path, video_id, output_dir)

        print(f"[SEGMENTER] Transcribing & segmenting {audio_path.name} with faster-whisper VAD...")
        segments_data = []

        try:
            # Native VAD filter prevents hallucinations during music/silence and cuts at speech pauses
            segments_gen, info = model.transcribe(
                str(audio_path),
                language="ta",
                beam_size=3,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=config.VAD_SILENCE_MS,
                    speech_pad_ms=250
                )
            )

            seg_list = list(segments_gen)
            print(f"[SEGMENTER] Discovered {len(seg_list)} speech segments (Language: {info.language}, Prob: {info.language_probability:.2f})")

            # Slice and merge short segments into ideal 4s - 15s ranges
            merged = []
            cur_start = None
            cur_end = None
            cur_text = []

            for seg in seg_list:
                s_start = seg.start
                s_end = seg.end
                s_text = seg.text.strip()
                s_dur = s_end - s_start

                if cur_start is None:
                    cur_start = s_start
                    cur_end = s_end
                    cur_text = [s_text]
                else:
                    proposed_dur = s_end - cur_start
                    if proposed_dur <= config.VAD_MAX_CHUNK_SEC:
                        cur_end = s_end
                        cur_text.append(s_text)
                    else:
                        merged.append((cur_start, cur_end, " ".join(cur_text)))
                        cur_start = s_start
                        cur_end = s_end
                        cur_text = [s_text]

            if cur_start is not None:
                merged.append((cur_start, cur_end, " ".join(cur_text)))

            # Export individual sliced audio chunks
            output_dir.mkdir(parents=True, exist_ok=True)
            for idx, (st, en, txt) in enumerate(merged, start=1):
                dur = round(en - st, 2)
                if dur < 1.0:
                    continue  # skip tiny blips

                seg_id = f"{video_id}_seg{idx:04d}"
                chunk_file = output_dir / f"{seg_id}.m4a"

                # Extract audio chunk via ffmpeg stream copy/fast re-encode
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(st),
                    "-to", str(en),
                    "-i", str(audio_path),
                    "-vn", "-c:a", "aac", "-b:a", "64k",
                    str(chunk_file)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if chunk_file.exists() and chunk_file.stat().st_size > 1000:
                    segments_data.append({
                        "segment_id": seg_id,
                        "video_id": video_id,
                        "index": idx,
                        "start": round(st, 2),
                        "end": round(en, 2),
                        "duration": dur,
                        "audio_path": chunk_file,
                        "draft_text": txt
                    })

        except Exception as e:
            print(f"[SEGMENTER] Error in Whisper VAD segmentation: {e}")
            return self.segment_with_ffmpeg_vad(audio_path, video_id, output_dir)

        return segments_data

    def segment_with_ffmpeg_vad(self, audio_path: Path, video_id: str, output_dir: Path) -> List[Dict[str, Any]]:
        """Fallback silence-detection segmenter using ffmpeg."""
        print(f"[SEGMENTER] Using ffmpeg silence detection on {audio_path.name}...")
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-af", "silencedetect=noise=-30dB:d=0.4",
            "-f", "null", "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = proc.stderr.splitlines()

        silence_ends = []
        for line in lines:
            if "silence_end:" in line:
                m = re.search(r"silence_end:\s*([0-9.]+)", line)
                if m:
                    silence_ends.append(float(m.group(1)))

        output_dir.mkdir(parents=True, exist_ok=True)
        segments_data = []
        cur_pos = 0.0

        for idx, s_end in enumerate(silence_ends, start=1):
            dur = s_end - cur_pos
            if dur >= config.VAD_MIN_CHUNK_SEC:
                chunk_end = s_end
                seg_id = f"{video_id}_seg{idx:04d}"
                chunk_file = output_dir / f"{seg_id}.m4a"

                c_cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(cur_pos),
                    "-to", str(chunk_end),
                    "-i", str(audio_path),
                    "-vn", "-c:a", "aac", "-b:a", "64k",
                    str(chunk_file)
                ]
                subprocess.run(c_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if chunk_file.exists() and chunk_file.stat().st_size > 1000:
                    segments_data.append({
                        "segment_id": seg_id,
                        "video_id": video_id,
                        "index": idx,
                        "start": round(cur_pos, 2),
                        "end": round(chunk_end, 2),
                        "duration": round(chunk_end - cur_pos, 2),
                        "audio_path": chunk_file,
                        "draft_text": ""
                    })
                cur_pos = s_end

        return segments_data
