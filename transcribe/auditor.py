import re
import sys
import json
import base64
import asyncio
import random
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
from tqdm import tqdm
from openai import AsyncOpenAI
import transcribe.config as config
from transcribe.rules import SYSTEM_PROMPT, build_user_prompt, build_bundle_prompt

def parse_emotion_and_clean_text(raw_output: str) -> Tuple[str, str]:
    """Extract dominant emotion tag from brackets e.g. [None], [Happy] and return (clean_text, emotion)."""
    emotion = "None"
    clean_text = (raw_output or "").strip()
    m = re.search(r"\[(None|Happy|Angry|Sad|Fear|Sarcastic|Disgust|Surprise|Excited)\]\s*$", raw_output, re.IGNORECASE)
    if m:
        emotion = m.group(1).capitalize()
        clean_text = raw_output[:m.start()].strip()

    # Handle pure music acoustic tag
    if clean_text.lower() in ["[music]", "music"] or raw_output.lower() in ["[music]", "[music] [none]"]:
        clean_text = "[music]"
        emotion = "None"

    return clean_text, emotion


def concatenate_audio_clips(clips: List[Dict[str, Any]], temp_dir: Path) -> Tuple[Optional[bytes], str, List[Tuple[float, float]]]:
    """
    Concatenates multiple audio clips into a single continuous AAC audio stream using ffmpeg.
    Returns (merged_bytes, format, timestamps).
    """
    if not clips:
        return None, "m4a", []

    if len(clips) == 1:
        c = clips[0]
        audio_bytes = c.get("audio_bytes")
        if audio_bytes is None and c.get("audio_path"):
            p = Path(c["audio_path"])
            if p.exists():
                audio_bytes = p.read_bytes()
        dur = float(c.get("duration", 0.0))
        fmt = c.get("audio_format") or "m4a"
        return audio_bytes, fmt, [(0.0, dur)]

    timestamps = []
    file_list = []
    curr_time = 0.0

    try:
        for idx, c in enumerate(clips):
            dur = float(c.get("duration", 0.0))
            audio_bytes = c.get("audio_bytes")
            if audio_bytes is None and c.get("audio_path"):
                p = Path(c["audio_path"])
                if p.exists():
                    audio_bytes = p.read_bytes()

            if not audio_bytes:
                return None, "m4a", []

            clip_file = temp_dir / f"bundle_clip_{idx:03d}.m4a"
            clip_file.write_bytes(audio_bytes)
            file_list.append(clip_file)
            timestamps.append((curr_time, curr_time + dur))
            curr_time += dur

        concat_txt = temp_dir / "concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for cf in file_list:
                esc_path = cf.as_posix().replace("'", "'\\''")
                f.write(f"file '{esc_path}'\n")

        merged_out = temp_dir / "merged_bundle.m4a"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:a", "aac", "-b:a", "64k", "-ar", "16000", "-ac", "1",
            str(merged_out)
        ]
        ret = subprocess.run(cmd, capture_output=True, timeout=30)
        if ret.returncode == 0 and merged_out.exists() and merged_out.stat().st_size > 0:
            return merged_out.read_bytes(), "m4a", timestamps
        else:
            return None, "m4a", []
    except Exception:
        return None, "m4a", []


class GeminiAuditor:
    """
    Stage 2: Sends segmented audio clips to Gemini Canvas Proxy (/v1),
    adhering to the 23 Tamil ASR rules with strictly sequential 1-by-1 requests.
    """
    def __init__(self, proxy_url: str, token: str = config.PROXY_TOKEN, model: str = config.MODEL_NAME, concurrency: int = config.CONCURRENCY):
        raw_url = proxy_url.strip()
        # Clean query strings and fragments (e.g. ?autoconnect=true&resize=scale, #...)
        if "?" in raw_url:
            raw_url = raw_url.split("?")[0].strip()
        if "#" in raw_url:
            raw_url = raw_url.split("#")[0].strip()

        # If user inadvertently pasted the noVNC viewer URL
        if "vnc.html" in raw_url.lower():
            print(f"[WARN] ⚠️ Detected noVNC desktop viewer page in URL ('{proxy_url}').", file=sys.stderr)
            print("[WARN] ⚠️ The Gemini Canvas Proxy must be the HTTP API tunnel endpoint (ending in /v1), NOT the noVNC web viewer.", file=sys.stderr)
            raw_url = re.sub(r"/vnc\.html(/v1)?", "", raw_url, flags=re.IGNORECASE).rstrip("/")

        # Normalize proxy URL to point to /v1
        norm_url = raw_url.rstrip("/")
        if not norm_url.endswith("/v1"):
            norm_url = f"{norm_url}/v1"
            
        self.proxy_url = norm_url
        self.token = token
        self.model = model
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

        self.client = AsyncOpenAI(
            base_url=self.proxy_url,
            api_key=self.token or "51dfb957-7347-449b-b153-df797a72c9d5",
            timeout=180.0
        )

    async def test_connection(self) -> bool:
        """Verify that the Gemini Canvas Proxy is live and operational with a fast 12s timeout."""
        try:
            models = await asyncio.wait_for(self.client.models.list(), timeout=12.0)
            model_ids = [m.id for m in models.data]
            print(f"[AUDITOR] Connected to Gemini Canvas Proxy! Available models: {model_ids}")
            return True
        except asyncio.TimeoutError:
            print(f"[AUDITOR] ❌ Connection test timed out (12s) for {self.proxy_url}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[AUDITOR] ❌ Connection test failed for {self.proxy_url}: {e}", file=sys.stderr)
            return False

    async def audit_single_segment(self, segment: Dict[str, Any], max_retries: int = 4) -> Dict[str, Any]:
        """
        Transcribes/audits a single 5-15s audio segment with Gemini 3 Flash.
        Strictly applies the 23 Tamil ASR rules.
        """
        audio_path: Optional[Path] = segment.get("audio_path")
        audio_bytes: Optional[bytes] = segment.get("audio_bytes")
        draft_text: str = segment.get("draft_text", "")
        seg_id: str = segment.get("segment_id", "seg_unknown")

        if audio_bytes is None:
            if not audio_path or not Path(audio_path).exists():
                return {**segment, "audited_text": draft_text, "clean_text": draft_text, "emotion": "None", "error": "Audio file not found"}
            audio_bytes = Path(audio_path).read_bytes()

        # Read audio bytes and convert to base64
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        fmt = segment.get("audio_format") or (Path(audio_path).suffix.lstrip(".").lower() if audio_path else "m4a")
        if fmt not in ["m4a", "mp3", "wav", "aac", "ogg", "opus"]:
            fmt = "m4a"

        user_content = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": b64_audio,
                    "format": fmt
                }
            },
            {
                "type": "text",
                "text": build_user_prompt(draft_text)
            }
        ]

        async with self.semaphore:
            for attempt in range(1, max_retries + 1):
                try:
                    extra_body_params = {}
                    if config.THINKING_BUDGET:
                        extra_body_params["thinking_budget"] = config.THINKING_BUDGET

                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.2,
                        extra_body=extra_body_params if extra_body_params else None
                    )

                    raw_output = (resp.choices[0].message.content or "").strip()
                    clean_text, emotion = parse_emotion_and_clean_text(raw_output)

                    return {
                        **segment,
                        "audited_text": raw_output,
                        "clean_text": clean_text,
                        "emotion": emotion,
                        "status": "ok"
                    }

                except Exception as exc:
                    err_str = str(exc)
                    is_gateway_err = any(code in err_str for code in ["502", "504", "Bad gateway", "overloaded", "origin_bad_gateway", "Connection error"])
                    if is_gateway_err:
                        sleep_sec = min(20.0, 3.5 * attempt + random.uniform(0.5, 2.0))
                        tqdm.write(f"[AUDITOR] [{seg_id}] Cloudflare/Proxy transient notice (Attempt {attempt}/{max_retries}). Retrying in {sleep_sec:.1f}s...")
                    else:
                        sleep_sec = 2.0 * attempt
                        tqdm.write(f"[AUDITOR] [{seg_id}] Attempt {attempt}/{max_retries} notice: {exc}")

                    if attempt == max_retries:
                        # Fallback if Gemini proxy fails after all retries
                        return {
                            **segment,
                            "audited_text": draft_text,
                            "clean_text": draft_text,
                            "emotion": "None",
                            "status": "fallback",
                            "error": str(exc)
                        }
                    await asyncio.sleep(sleep_sec)

    async def audit_clip_bundle(self, clips: List[Dict[str, Any]], max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Transcribes a bundle of up to 10 clips in a single Gemini request.
        Gemini separates transcripts using '[#cut]'.
        If delimiter count mismatches, retries once, then falls back to 1-by-1 processing.
        """
        N = len(clips)
        if N == 0:
            return []
        if N == 1:
            res = await self.audit_single_segment(clips[0])
            return [res]

        with tempfile.TemporaryDirectory() as tmp_d:
            merged_bytes, fmt, timestamps = concatenate_audio_clips(clips, Path(tmp_d))

        if not merged_bytes or not timestamps:
            tqdm.write(f"[AUDITOR] Notice: Audio concat failed for bundle of {N} clips. Processing 1-by-1.")
            results = []
            for c in clips:
                results.append(await self.audit_single_segment(c))
                await asyncio.sleep(0.5)
            return results

        b64_audio = base64.b64encode(merged_bytes).decode("utf-8")
        prompt_text = build_bundle_prompt(timestamps, clips)

        user_content = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": b64_audio,
                    "format": fmt
                }
            },
            {
                "type": "text",
                "text": prompt_text
            }
        ]

        async with self.semaphore:
            # Attempt 1: Standard bundle request
            for attempt in range(1, max_retries + 1):
                try:
                    extra_body_params = {}
                    if config.THINKING_BUDGET:
                        extra_body_params["thinking_budget"] = config.THINKING_BUDGET

                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.2,
                        extra_body=extra_body_params if extra_body_params else None
                    )

                    raw_output = (resp.choices[0].message.content or "").strip()
                    parts = [p.strip() for p in re.split(r"\[#cut\]", raw_output, flags=re.IGNORECASE) if p.strip()]

                    if len(parts) == N:
                        results = []
                        for idx, clip in enumerate(clips):
                            c_raw = parts[idx]
                            c_clean, c_emotion = parse_emotion_and_clean_text(c_raw)
                            results.append({
                                **clip,
                                "audited_text": c_raw,
                                "clean_text": c_clean,
                                "emotion": c_emotion,
                                "status": "ok"
                            })
                        return results
                    else:
                        tqdm.write(f"[AUDITOR] ⚠️ Bundle cut mismatch: expected {N} parts, received {len(parts)} parts. Retrying bundle once...")
                        break
                except Exception as exc:
                    err_str = str(exc)
                    is_gw = any(c in err_str for c in ["502", "504", "Bad gateway", "overloaded", "origin_bad_gateway", "Connection error"])
                    sleep_sec = min(20.0, 3.5 * attempt + random.uniform(0.5, 2.0)) if is_gw else 2.0 * attempt
                    tqdm.write(f"[AUDITOR] Bundle transient notice (Attempt {attempt}/{max_retries}): {exc}. Retrying in {sleep_sec:.1f}s...")
                    await asyncio.sleep(sleep_sec)

            # Retry with emphatic correction instruction
            try:
                retry_content = [
                    user_content[0],
                    {
                        "type": "text",
                        "text": prompt_text + f"\n\nCRITICAL: In your last response you did not separate with [#cut] correctly. There are exactly {N} segments. Output EXACTLY {N} transcripts separated by '[#cut]'."
                    }
                ]
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": retry_content}
                    ],
                    temperature=0.1,
                    extra_body=extra_body_params if extra_body_params else None
                )
                raw_output = (resp.choices[0].message.content or "").strip()
                parts = [p.strip() for p in re.split(r"\[#cut\]", raw_output, flags=re.IGNORECASE) if p.strip()]
                if len(parts) == N:
                    results = []
                    for idx, clip in enumerate(clips):
                        c_raw = parts[idx]
                        c_clean, c_emotion = parse_emotion_and_clean_text(c_raw)
                        results.append({
                            **clip,
                            "audited_text": c_raw,
                            "clean_text": c_clean,
                            "emotion": c_emotion,
                            "status": "ok"
                        })
                    return results
            except Exception as e:
                tqdm.write(f"[AUDITOR] Notice on bundle retry: {e}")

            # Mismatch persisted -> Fallback to processing these N clips individually (one by one)
            tqdm.write(f"[AUDITOR] ⚠️ Fallback to 1-by-1 sequential processing for {N} clips to guarantee exact alignment.")
            results = []
            for c in clips:
                res = await self.audit_single_segment(c)
                results.append(res)
                await asyncio.sleep(1.0)
            return results

    async def audit_batch(
        self,
        segments: List[Dict[str, Any]],
        desc: str = "Transcribing clips",
        bundle_size: int = config.BUNDLE_SIZE,
        request_delay: float = config.REQUEST_DELAY_SECONDS,
        cooldown_every: int = config.COOLDOWN_EVERY,
        cooldown_seconds: float = config.COOLDOWN_SECONDS,
        cooldown_callback: Optional[Callable[[], None]] = None,
        on_chunk_done: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Audits audio segments strictly sequentially:
        - Bundles up to 10 clips per Gemini request (separated by [#cut])
        - Paces requests with 2.0s delay between requests
        - Real-time tqdm progress bar
        - Periodically calls cooldown_callback to renew shard lease
        - Calls on_chunk_done to incrementally buffer results
        """
        if not segments:
            return []

        total_segs = len(segments)
        results = [None] * total_segs
        pbar = tqdm(total=total_segs, desc=f"🎙️  {desc}", unit="clip", dynamic_ncols=True)
        ok_count = 0
        fallback_count = 0
        req_count = 0

        chunk_start = 0
        while chunk_start < total_segs:
            chunk_end = min(chunk_start + bundle_size, total_segs)
            current_bundle = segments[chunk_start:chunk_end]

            # Strictly sequential call to Gemini
            audited_group = await self.audit_clip_bundle(current_bundle)
            req_count += 1

            for idx, res in enumerate(audited_group):
                results[chunk_start + idx] = res
                if res.get("status") == "ok":
                    ok_count += 1
                else:
                    fallback_count += 1

            pbar.set_postfix({
                "OK": ok_count,
                "Fallback": fallback_count,
                "Bundles": f"{req_count}"
            })
            pbar.update(len(current_bundle))

            if on_chunk_done:
                try:
                    on_chunk_done(audited_group)
                except Exception as e:
                    tqdm.write(f"[AUDITOR] Notice in on_chunk_done: {e}")

            chunk_start = chunk_end

            # Pacing delay between sequential requests (e.g. 2.0s + slight micro jitter)
            if chunk_start < total_segs:
                if request_delay > 0:
                    delay = request_delay + random.uniform(0.05, 0.25)
                    await asyncio.sleep(delay)

                # Periodic lease renewal / heartbeat callback
                if cooldown_callback and (req_count % 10 == 0 or (cooldown_every > 0 and req_count % cooldown_every == 0)):
                    try:
                        cooldown_callback()
                    except Exception:
                        pass

                # Optional cooldown if enabled
                if cooldown_every > 0 and (req_count % cooldown_every == 0):
                    tqdm.write(
                        f"\n[AUDITOR] ☕ Cooldown pause: Completed {req_count} requests "
                        f"({chunk_start}/{total_segs} clips, {ok_count} OK). Pausing {cooldown_seconds:.1f}s..."
                    )
                    await asyncio.sleep(cooldown_seconds)

        pbar.close()
        return results
