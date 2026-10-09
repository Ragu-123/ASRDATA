import re
import sys
import json
import base64
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

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
        # Normalize proxy URL to point to /v1
        norm_url = proxy_url.strip().rstrip("/")
        if not norm_url.endswith("/v1"):
            norm_url = f"{norm_url}/v1"
            
        self.proxy_url = norm_url
        self.token = token
        self.model = model
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

        self.client = AsyncOpenAI(
            base_url=self.proxy_url,
            api_key=self.token,
            timeout=180.0
        )

    async def test_connection(self) -> bool:
        """Verify that the Gemini Canvas Proxy is live and operational."""
        try:
            models = await self.client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"[AUDITOR] Connected to Gemini Canvas Proxy! Available models: {model_ids}")
            return True
        except Exception as e:
            print(f"[AUDITOR] Connection test failed for {self.proxy_url}: {e}")
            return False

    async def audit_single_segment(self, segment: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Transcribes/audits a single 5-15s audio segment with Gemini 3 Flash.
        Strictly applies the 23 Tamil ASR rules.
        """
        audio_path: Path = segment["audio_path"]
        draft_text: str = segment.get("draft_text", "")
        seg_id: str = segment["segment_id"]

        if not audio_path.exists():
            return {**segment, "audited_text": draft_text, "clean_text": draft_text, "emotion": "None", "error": "Audio file not found"}

        # Read audio bytes and convert to base64
        audio_bytes = audio_path.read_bytes()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        fmt = audio_path.suffix.lstrip(".").lower()
        if fmt not in ["m4a", "mp3", "wav", "aac"]:
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

                    return {
                        **segment,
                        "audited_text": raw_output,
                        "clean_text": clean_text,
                        "emotion": emotion,
                        "status": "ok"
                    }

                except Exception as exc:
                    tqdm.write(f"[AUDITOR] [{seg_id}] Attempt {attempt} notice: {exc}")
                    if attempt == max_retries:
                        # Fallback to Whisper draft if Gemini proxy fails after max retries
                        return {
                            **segment,
                            "audited_text": draft_text,
                            "clean_text": draft_text,
                            "emotion": "None",
                            "status": "fallback",
                            "error": str(exc)
                        }
                    await asyncio.sleep(2 * attempt)

    async def audit_batch(self, segments: List[Dict[str, Any]], desc: str = "Transcribing clips") -> List[Dict[str, Any]]:
        """Audits multiple segments concurrently using the Semaphore with real-time tqdm progress bar."""
        if not segments:
            return []

        results = [None] * len(segments)
        pbar = tqdm(total=len(segments), desc=f"🎙️  {desc}", unit="clip", dynamic_ncols=True)
        ok_count = 0
        fallback_count = 0

        async def _run_indexed(idx: int, seg: Dict[str, Any]):
            nonlocal ok_count, fallback_count
            res = await self.audit_single_segment(seg)
            results[idx] = res
            if res.get("status") == "ok":
                ok_count += 1
            else:
                fallback_count += 1
            pbar.set_postfix({"OK": ok_count, "Fallback": fallback_count})
            pbar.update(1)
            return res

        tasks = [_run_indexed(i, s) for i, s in enumerate(segments)]
        await asyncio.gather(*tasks)
        pbar.close()
        return results
