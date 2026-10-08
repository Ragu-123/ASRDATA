import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import config
from coordinator import ClusterCoordinator
from worker import IngestionWorker

# Shared references initialized in main.py
coordinator_ref: Optional[ClusterCoordinator] = None
worker_ref: Optional[IngestionWorker] = None

app = FastAPI(title="Tamil ASR Audio Harvest Hub")

class ChannelRequest(BaseModel):
    url: str
    name: Optional[str] = None

class ControlRequest(BaseModel):
    action: str
    target_worker: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = Path(__file__).parent / "templates" / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Tamil ASR Dashboard - Index template missing</h1>")

@app.get("/api/status")
async def get_status():
    if worker_ref:
        return worker_ref.get_status()
    return {"status": "initializing", "node_id": config.NODE_ID}

@app.get("/api/channels")
async def get_channels():
    if coordinator_ref:
        return coordinator_ref.load_channels()
    return []

@app.post("/api/channels")
async def add_channel(req: ChannelRequest):
    if not coordinator_ref:
        return JSONResponse({"status": "error", "message": "Coordinator not ready"}, status_code=500)
    added = coordinator_ref.add_channel(req.url, req.name)
    if added:
        return {"status": "ok", "message": "Channel added successfully"}
    return {"status": "notice", "message": "Channel already exists in queue"}

@app.get("/api/workers")
async def get_workers():
    if coordinator_ref:
        return coordinator_ref.list_active_workers()
    return []

@app.get("/api/cluster")
async def get_cluster():
    if coordinator_ref:
        return coordinator_ref.get_cluster_overview()
    return {"node_count": 1, "live_workers": [], "all_active_streams": []}

@app.post("/api/flush")
async def trigger_flush():
    if worker_ref:
        worker_ref.trigger_flush()
        return {"status": "ok", "message": "Flush triggered"}
    return JSONResponse({"status": "error", "message": "Worker not ready"}, status_code=500)

@app.post("/api/control")
async def control_cluster(req: ControlRequest):
    if not coordinator_ref:
        return JSONResponse({"status": "error", "message": "Coordinator not ready"}, status_code=500)
    coordinator_ref.send_command(req.action, req.target_worker)
    return {"status": "ok", "message": f"Command '{req.action}' sent to {req.target_worker or 'all nodes'}"}

@app.websocket("/ws")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = {}
            if worker_ref:
                payload["local"] = worker_ref.get_status()
            if coordinator_ref:
                payload["cluster"] = coordinator_ref.get_cluster_overview()
            await websocket.send_json(payload)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
