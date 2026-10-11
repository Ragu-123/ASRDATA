#!/usr/bin/env python3
"""
High-Speed Ingestion & Audio Downloader Engine with aria2c optimization.
Provides 5x-10x accelerated YouTube audio extraction for Kaggle and distributed clusters.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yt_dlp
import transcribe.config as config

def ensure_aria2() -> bool:
    """
    Checks if aria2c is installed. If on Linux/Kaggle and missing, attempts automatic installation.
    """
    if shutil.which("aria2c"):
        return True

    if sys.platform.startswith("linux"):
        try:
            print("[INGEST] Attempting automatic aria2 installation on Linux/Kaggle...")
            subprocess.run(
                ["sudo", "apt-get", "update", "-qq"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45
            )
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "aria2"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
            )
            if shutil.which("aria2c"):
                print("[INGEST] Successfully installed aria2c!")
                return True
        except Exception as e:
            print(f"[INGEST] Notice: automatic aria2 install skipped: {e}")

    return False

def get_ytdlp_audio_opts(output_template: str, quiet: bool = True) -> Dict[str, Any]:
    """
    Builds optimized yt-dlp options:
    - Strictly ba[ext=m4a]/ba[ext=opus]/ba with FFmpegExtractAudio (preferredcodec: m4a).
    - Checks if aria2c is available; if present, enables multi-connection aria2c (-s 8 -x 8 -k 1M -j 8).
    - Falls back seamlessly to native yt-dlp multi-fragment downloader.
    """
    opts: Dict[str, Any] = {
        'format': 'ba[ext=m4a]/ba[ext=opus]/ba',
        'outtmpl': output_template,
        'quiet': quiet,
        'no_warnings': True,
        'socket_timeout': 30,
        'retries': 5,
        'fragment_retries': 5,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios']
            }
        },
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '128',
        }],
    }

    if shutil.which("aria2c"):
        opts["external_downloader"] = {"default": "aria2c"}
        opts["external_downloader_args"] = {
            "aria2c": ["-s", "8", "-x", "8", "-k", "1M", "-j", "8"]
        }

    return opts

def download_audio(video_id: str, dest_dir: Path, filename: Optional[str] = None) -> Optional[Path]:
    """
    Fast audio-only download from YouTube using aria2c or native downloader.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / f"{video_id}.%(ext)s")
    opts = get_ytdlp_audio_opts(out_template)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)

        matching = list(dest_dir.glob(f"{video_id}.*"))
        if matching:
            filepath = matching[0]
            if filepath.suffix.lower() != ".m4a":
                audio_target = dest_dir / f"{video_id}.m4a"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(filepath), "-vn", "-c:a", "copy", str(audio_target)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
                )
                filepath.unlink(missing_ok=True)
                filepath = audio_target
            if filepath.exists() and filepath.stat().st_size > 10000:
                return filepath
    except Exception as e:
        print(f"[INGEST] Notice downloading {video_id}: {e}")

    return None

if __name__ == "__main__":
    ensure_aria2()
    from main import main
    main()
