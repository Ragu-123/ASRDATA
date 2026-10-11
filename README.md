# 🎙️ ASRDATA: Distributed Tamil ASR Audio Harvesting & Transcription Engine

A high-throughput, distributed ASR data engineering pipeline designed to scale across multiple parallel **Kaggle** instances without collisions, producing foundational speech training datasets in Apache Parquet format.

The architecture is cleanly decoupled into 3 modular entry points:
1. **Raw Ingestion (`python main.py`)**: Streams raw audio from YouTube channels into archive storage.
2. **Silence Segmentation (`python segment.py`)**: Runs ffmpeg silence detection, cuts audio into ~10s clips (5-15s range), extracts metadata (`duration`, `audio` struct, `channel_name`, `video_id`, `segment_id`), buffers into PyArrow Parquet shards (~300-500 clips, ~70MB), respects the 19GB disk guard (<1GB temp disk usage), and uploads Parquet shards to Hugging Face (`King758/tamil-asr-segmented`).
3. **Transcription & Auditing (`python transcribe.py`)**: Connects to Gemini Canvas Proxy (`--url https://.../v1`), pulls clips from the segmented dataset, executes with 3-5 concurrent parallel requests, applies 23 acoustic rules (including `[music]` and `<music>...text...</music>`), syncs tracking manifests in the tracking bucket, and writes the final audited dataset (`King758/tamil-asr-corpus`).

---

## ⚡ 3-Stage Pipeline Commands

### Stage 1: Ingestion
Download raw audio from YouTube based on channel input or catalog:
```bash
# Ingest specific channel
python main.py --channel @madangowri --no-web

# Run cluster ingestion node with live dashboard
python main.py
```

### Stage 2: Silence Segmentation
Detect natural pauses via ffmpeg and crop into ~10s clips, saving metadata (`duration`, `audio`, `channel_name`, `video_id`, `segment_id`) into Parquet shards:
```bash
# Runs distributed silence segmenter (also alias: python segement.py)
python segment.py

# Or segment a specific channel / video directly:
python segment.py --channel @madangowri
python segment.py --video-id WLUErZVZQkQ
```

### Stage 3: Distributed Transcription
Pulls clips from the segmented dataset and transcribes via Gemini Canvas Proxy:
```bash
# Transcribe with Cloudflare /v1 URL (also alias: python trascribe.py)
python transcribe.py --url https://your-cloudflare-tunnel.trycloudflare.com/v1 --concurrency 4
```

### Fast Download Acceleration (aria2c):
For 5x-10x download speedup on Kaggle, install aria2:
```bash
!sudo apt-get update -qq && sudo apt-get install -y aria2
```
`yt-dlp` automatically detects `aria2c` and downloads with `--external-downloader aria2c --external-downloader-args "aria2c:-s 8 -x 8 -k 1M -j 8"`.

---

## 🤖 Telegram Cluster Bot Remote Management

Control your cluster and tracking bucket directly from Telegram:
- `📊 /status`: Cluster overview, active worker nodes, harvested hours.
- `➕ /add_channel <url>`: Upfront scraping and indexing of channel videos.
- `🗑️ /delete_channel <url>`: Delete a channel, its catalog, and all its manifest records saved in buckets.
- `🗑️ /delete_manifest`: Reset all completed video records in `manifest.json` and `segment_manifest.json` in tracking bucket.
- `⏸️ /pause` & `▶️ /resume`: Pause or resume cluster operations.
- `💾 /flush`: Force immediate Parquet shard upload.
- `🛑 /stop`: Graceful shutdown across all nodes.

---

## 🔐 Authentication & Secrets

In Kaggle notebooks, add your tokens into **Add-ons → Secrets**:
- `HF_TOKEN`: Hugging Face write token
- `TELEGRAM_BOT_TOKEN`: Telegram cluster monitor bot token
- `TELEGRAM_ADMIN_ID`: Your Telegram numeric user ID

All scripts automatically resolve secrets using `secret_resolver.py` across:
1. Environment variables (`os.getenv`)
2. Kaggle Notebook Secrets (`UserSecretsClient().get_secret(...)`)
3. Local `.env` files

Zero hardcoded tokens exist anywhere in the repository.

---

## 🎵 Acoustic Rules Update

1. **Pure Music (No Speech)**: If there are NO words and ONLY music or songs are playing, output:
   ```
   [music]
   ```
2. **Background Music (Speech Occurring)**: If music is playing in the background while speech is occurring, wrap the spoken words with `<music>...</music>`:
   ```
   <music> இன்று வானிலை நன்றாக இருக்கிறது </music> [None]
   ```
3. **Dominant Emotion Tag**: End spoken segments with exactly one bracketed emotion tag (e.g. `[None]`, `[Happy]`, `[Angry]`). Pure music clips output `[music]` without emotion tags.

---

## 📁 Repository Structure

```
ASRDATA/
├── main.py                     # Stage 1: Audio ingestion entry point
├── segment.py                  # Stage 2: Silence segmenter entry point (alias: segement.py)
├── segement.py                 # Typo alias for segment.py
├── transcribe.py               # Stage 3: Transcription entry point (alias: trascribe.py)
├── trascribe.py                # Typo alias for transcribe.py
├── secret_resolver.py          # Unified safe secret retrieval
├── config.py                   # Ingestion configuration
├── coordinator.py              # Ingestion cluster coordinator
├── bucket_sync.py              # Raw audio uploader
├── worker.py                   # Ingestion stream worker
│
├── segment/                    # Audio Segmentation Module
│   ├── config.py               # Segmentation settings & thresholds
│   ├── coordinator.py          # Atomic batch leasing across Kaggle sessions
│   ├── segmenter.py            # FFmpeg silencedetect & ~10s clipping
│   ├── syncer.py               # Parquet shard builder (audio, duration, channel)
│   └── main.py                 # Segmentation runner loop
│
├── transcribe/                 # Transcription & Auditing Module
│   ├── config.py               # Transcriber settings & model parameters
│   ├── coordinator.py          # Segmented shard leasing & tracking bucket sync
│   ├── auditor.py              # Multi-session concurrent Gemini Canvas proxy client
│   ├── syncer.py               # Audited Parquet corpus builder
│   ├── disk_guard.py           # Strict < 1.0 GB local disk protector
│   ├── rules.py                # 23 Tamil ASR rules with [music] and <music>
│   ├── RULES.md                # Full 23 transcription rules documentation
│   └── telegram_bot.py         # Cluster Telegram monitor & alert bot
│
├── tests/
│   ├── test_two_tier_pipeline.py    # Two-tier architecture tests
│   └── test_decoupled_pipeline.py   # Decoupled 3-command pipeline tests
└── requirements.txt
```

