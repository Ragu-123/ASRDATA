import re
import sys
import json
import base64
import asyncio
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
from tqdm import tqdm
from openai import AsyncOpenAI
import transcribe.config as config
from transcribe.rules import SYSTEM_PROMPT, build_user_prompt

class GeminiAuditor:
    """
    Stage 2: Sends segmented audio clips to Gemini Canvas Proxy (/v1),
    adhering to the 23 Tamil ASR rules with 3-5 concurrent requests.
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

                    # Extract dominant emotion tag from brackets e.g. [None], [Happy], [Angry]
                    emotion = "None"
                    clean_text = raw_output
                    m = re.search(r"\[(None|Happy|Angry|Sad|Fear|Sarcastic|Disgust|Surprise|Excited)\]\s*$", raw_output, re.IGNORECASE)
                    if m:
                        emotion = m.group(1).capitalize()
                        clean_text = raw_output[:m.start()].strip()

                    # Handle pure music acoustic tag
                    if clean_text.lower() in ["[music]", "music"] or raw_output.lower() in ["[music]", "[music] [none]"]:
                        clean_text = "[music]"
                        emotion = "None"

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

    async def audit_batch(
        self,
        segments: List[Dict[str, Any]],
        desc: str = "Transcribing clips",
        cooldown_every: int = config.COOLDOWN_EVERY,
        cooldown_seconds: float = config.COOLDOWN_SECONDS,
        cooldown_jitter: float = config.COOLDOWN_JITTER,
        request_jitter: float = config.REQUEST_JITTER,
        cooldown_callback: Optional[Callable[[], None]] = None,
        on_chunk_done: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Audits multiple segments concurrently using the Semaphore with real-time tqdm progress bar.
        Incorporates rate protection cooldowns: every ~250 requests (randomized), pauses for
        cooldown_seconds (with random jitter) to keep Gemini web sessions safe from anti-abuse flagging.
        """
        if not segments:
            return []

        total_segs = len(segments)
        results = [None] * total_segs
        pbar = tqdm(total=total_segs, desc=f"🎙️  {desc}", unit="clip", dynamic_ncols=True)
        ok_count = 0
        fallback_count = 0

        chunk_start = 0
        chunk_idx = 1
        while chunk_start < total_segs:
            # Pick a randomized chunk size around cooldown_every (e.g. 210-265) to look organic
            if cooldown_every > 0:
                low = max(40, int(cooldown_every * 0.85))
                high = max(low + 10, int(cooldown_every * 1.15))
                chunk_len = random.randint(low, high)
            else:
                chunk_len = total_segs

            chunk_end = min(chunk_start + chunk_len, total_segs)
            current_slice = segments[chunk_start:chunk_end]

            async def _run_indexed(global_idx: int, local_idx: int, seg: Dict[str, Any]):
                nonlocal ok_count, fallback_count
                # Stagger startup of initial parallel requests to avoid thundering herd on Cloudflare tunnel
                if local_idx < self.concurrency:
                    await asyncio.sleep(local_idx * 0.25)
                elif request_jitter > 0:
                    # Subtle human-like micro-delay between requests to avoid robotic burst patterns
                    await asyncio.sleep(random.uniform(0.05, request_jitter))

                res = await self.audit_single_segment(seg)
                results[global_idx] = res
                if res.get("status") == "ok":
                    ok_count += 1
                else:
                    fallback_count += 1
                pbar.set_postfix({"OK": ok_count, "Fallback": fallback_count})
                pbar.update(1)
                return res

            tasks = [
                _run_indexed(chunk_start + i, i, seg)
                for i, seg in enumerate(current_slice)
            ]
            chunk_results = await asyncio.gather(*tasks)

            if on_chunk_done:
                try:
                    on_chunk_done(chunk_results)
                except Exception as e:
                    tqdm.write(f"[AUDITOR] Notice in on_chunk_done callback: {e}")

            chunk_start = chunk_end

            # Rate protection cooldown: pause between chunks if more clips remain
            if chunk_start < total_segs and cooldown_every > 0:
                pause_sec = random.uniform(cooldown_seconds, cooldown_seconds + cooldown_jitter)
                tqdm.write(
                    f"\n[AUDITOR] ☕ Rate protection cooldown: Completed {chunk_end}/{total_segs} clips "
                    f"({ok_count} Gemini OK). Pausing {pause_sec:.1f}s to keep Gemini session safe..."
                )
                if cooldown_callback:
                    try:
                        cooldown_callback()
                    except Exception:
                        pass
                await asyncio.sleep(pause_sec)
                chunk_idx += 1

        pbar.close()
        return results
