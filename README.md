# 🎙️ ASRDATA: Distributed Audio Harvesting Hub

A high-throughput, distributed YouTube audio harvesting system designed to scale across multiple parallel **Kaggle** instances (interactive or background "Save Version" runs) to collect 4,000–5,000 hours of speech audio for foundational ASR model training.

The audio is streamed in native compressed format (Opus/M4A) and continuously synchronized into a Hugging Face Storage Bucket (`King758/media-archive-01`).

---

## ⚡ 1-Line Kaggle Quickstart

In any Kaggle notebook, add your `HF_TOKEN` into **Add-ons → Secrets**, then run:

```bash
!git clone https://github.com/Ragu-123/ASRDATA.git && cd ASRDATA && pip install -q -r requirements.txt && python main.py
```

### What Happens Automatically:
1. **Auto-Authentication:** Connects to Hugging Face using your Kaggle Secret `HF_TOKEN`.
2. **Cloudflare Tunnel:** Automatically downloads `cloudflared` and outputs a live, public HTTPS dashboard link (`https://*.trycloudflare.com`).
3. **Cluster Auto-Discovery:** Discovers other running parallel Kaggle instances and coordinates via the HF Bucket.
4. **Batch Leasing:** Leases 30 videos per batch without locking overhead or collisions.
5. **Kaggle 20GB Disk Safety:** When local disk usage reaches **14.0 GB**, it automatically batch-uploads all files to the bucket, updates `manifest.json`, and purges local storage back to **0.0 GB**.

---

## 🚀 Key Features

* **Distributed Multi-Instance Mesh:** Run up to 5+ parallel Kaggle "Save Version" instances concurrently across one or multiple accounts. They automatically share the queue with zero duplicate downloads.
* **Unified Single-URL Dashboard:** Open **ANY** single worker's Cloudflare link to control the entire cluster:
  * View all parallel nodes side-by-side with live progress bars and download speeds.
  * Monitor total harvested hours against the 5,000-hour foundational goal.
  * Track local Kaggle disk space (used vs 14 GB limit).
* **Dynamic Channel Queue:** Add new YouTube channels or playlists on the fly directly from the web interface. Newly added channels sync to `channels.json` in the bucket and are immediately picked up by all running nodes.
* **1 Audio Per Video:** Saves each video as an intact, single audio file (`audios/{video_id}.m4a`) without splitting into arbitrary chunks.
* **Fault Tolerant Resumption:** Interrupted sessions or expired notebook runs resume instantly from `manifest.json`.

---

## 📁 Repository Structure

```
ASRDATA/
├── main.py              # CLI entrypoint: bootstraps secrets, tunnel, server, and worker
├── config.py            # Global settings (bucket ID, 14GB threshold, batch size)
├── coordinator.py       # Distributed lease coordinator & worker heartbeat registry
├── bucket_sync.py       # HF Storage Bucket uploader & disk quota recovery
├── worker.py            # Ingestion worker with yt-dlp callbacks and disk monitor
├── tunnel.py            # Cloudflare Tunnel supervisor (captures public HTTPS URL)
├── app.py               # FastAPI application with REST APIs and WebSocket telemetry
├── templates/
│   └── index.html       # Full Unified Cluster Mesh Dashboard (Tailwind CSS)
├── requirements.txt     # Python dependencies
└── README.md            # Quickstart guide
```

---

## 🎮 Web Dashboard Controls

From the web dashboard:
* **Add Channel:** Paste any YouTube channel URL or `@handle` to queue all its videos across all workers.
* **Flush All to HF:** Force all running Kaggle instances to upload their staging files to the bucket immediately.
* **Pause / Resume Cluster:** Pause or resume downloading across all nodes.
