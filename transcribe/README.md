# 🎙️ Tamil ASR Distributed Transcription Engine (Stage 1 & 2)

Modular, distributed, multi-node transcription pipeline for Tamil Speech-to-Text datasets adhering strictly to the **23 Tamil ASR Rules** in [`RULES.md`](./RULES.md).

---

## ⚡ How It Works (2-Stage Hybrid Architecture)

```
[Raw Audio in King758/media-archive-01]
                  │
                  ▼ (Stage 1: Local Kaggle GPU)
   faster-whisper large-v3-turbo + Silero VAD
   - Detects speech pauses (>400ms) without cutting words
   - Slices into clean 5-15s speech clips
   - Generates initial Tamil draft transcript in seconds
                  │
                  ▼ (Stage 2: Gemini Canvas Proxy /v1)
   Gemini 3 Flash Reasoning Auditor (3 to 5 parallel requests)
   - Injects full 23 Tamil ASR Rules from RULES.md
   - Audits draft text against audio chunk
   - Fixes spoken verbs (வந்திருக்கான் vs வந்திருக்கிறார்)
   - Fixes English Latin script common nouns (side-ல, window-க்கு)
   - Expands digits to words (ஒன்பது, சதவீதம்)
   - Labels acoustic emotions ([None], [Happy], etc.)
                  │
                  ▼ (Stage 3: Auto-Sync)
[Target Bucket: King758/asr-transcripts-01]
   - Uploads 5-15s sliced audio clips (segments/<video_id>/...)
   - Uploads full JSON transcripts (transcripts/<video_id>.json)
   - Updates global manifest.json (total hours, segments, videos)
```

---

## 🚀 How to Run on Kaggle

In your Kaggle notebook (or parallel "Save Version" notebook instances across accounts):

```bash
!git clone https://github.com/Ragu-123/ASRDATA.git && cd ASRDATA && pip install -q -r requirements.txt && python transcribe/main.py --proxy-url "YOUR_CANVAS_V1_URL"
```

### Or Interactive Prompt:
If you just run:
```bash
python transcribe/main.py
```
It will ask in the console:
```text
👉 Enter Gemini Canvas /v1 URL:
```
Paste your Cloudflare tunnel link (e.g. `https://pill-ultra-invitation-varying.trycloudflare.com/v1`) and press **Enter**.

---

## 🛡️ Multi-Node Parallel Safety (Zero Duplicate Work)

You can launch **4 to 5 Kaggle notebook sessions simultaneously** without any risk of overlap:
1. Each node acquires an **atomic batch lease** in the bucket (`batches/<worker_id>_lease.json`).
2. Peer nodes automatically inspect active leases and the global manifest.
3. No two nodes will ever transcribe the same video.
4. If a node disconnects, its lease automatically expires so other workers pick up the remaining work.

---

## 📦 Output Target Structure (`King758/asr-transcripts-01`)

```text
King758/asr-transcripts-01/
├── manifest.json                  # Global dataset registry (hours, completed videos)
├── batches/                       # Distributed worker leases
│   └── transcriber-a8f3b2_lease.json
├── transcripts/                   # Full structured JSON transcripts per video
│   └── pPQ2PSWlY6g.json
└── segments/                      # Sliced 5-15s audio clips
    └── pPQ2PSWlY6g/
        ├── pPQ2PSWlY6g_seg0001.m4a
        ├── pPQ2PSWlY6g_seg0002.m4a
        └── ...
```
