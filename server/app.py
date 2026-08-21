"""
app.py —— tsunami-lab GeoClaw 科研后端服务(FastAPI)。

端点:
  GET  /health              环境与任务概览
  POST /jobs                提交模拟任务(body: sim_time/nframes/fault/bathy_b64)
  GET  /jobs/{job_id}       任务状态(queued/running/done/error)
  GET  /jobs/{job_id}/frames          帧索引(frames.json)
  GET  /jobs/{job_id}/frames/{k}      第 k 帧二进制(float32 小端)

启动:
  python3 -m pip install -r server/requirements.txt
  python3 -m uvicorn server.app:app --port 8100
  (Web 端默认连接 http://localhost:8100)
"""

from __future__ import annotations

import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import geoclaw_runner as runner

JOBS_DIR = Path(__file__).resolve().parent / "_jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="tsunami-lab GeoClaw backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- 数据模型


class FaultParams(BaseModel):
    """Okada 断层参数(教学默认值接近 2011 东北海域事件)。"""

    longitude: float | None = None
    latitude: float | None = None
    depth: float = Field(20e3, description="震源深度 (m)")
    length: float = Field(200e3, description="断层长度 (m)")
    width: float = Field(80e3, description="断层宽度 (m)")
    strike: float = Field(20.0, description="走向 (°)")
    dip: float = Field(10.0, description="倾角 (°)")
    rake: float = Field(90.0, description="滑动角 (°)")
    slip: float = Field(10.0, description="滑动量 (m)")


class JobRequest(BaseModel):
    sim_time: float = Field(3600, ge=60, le=43200, description="模拟时长 (s)")
    nframes: int = Field(24, ge=2, le=200, description="输出帧数")
    fault: FaultParams = FaultParams()
    bathy_b64: str | None = Field(None, description="base64 编码的 .tsunami 地形")


# ---------------------------------------------------------------- 任务存储

JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _run_job(job_id: str) -> None:
    jobdir = _job_dir(job_id)
    job = JOBS[job_id]
    job["status"] = "running"
    try:
        runner.run(jobdir)
        frames = runner.export_frames(jobdir, jobdir / "frames")
        job["frames"] = frames
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = f"{exc}\n{traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------- 端点


@app.get("/health")
def health() -> JSONResponse:
    env = runner.check_environment()
    with JOB_LOCK:
        counts = {"total": len(JOBS),
                  "done": sum(1 for j in JOBS.values() if j["status"] == "done"),
                  "running": sum(1 for j in JOBS.values() if j["status"] == "running")}
    return JSONResponse({"ok": True, "geoclaw_env": env, "jobs": counts})


@app.post("/jobs")
def create_job(req: JobRequest) -> dict:
    env = runner.check_environment()
    if not env["geoclaw"]:
        raise HTTPException(
            status_code=503,
            detail="GeoClaw 环境不可用:" + (env["note"] or "clawpack 未安装")
            + "。安装方法:brew install gfortran && python3 -m pip install clawpack",
        )

    job_id = uuid.uuid4().hex[:12]
    jobdir = _job_dir(job_id)
    params = req.model_dump()
    params["fault"] = params["fault"] or {}
    try:
        region = runner.write_job(jobdir, params)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"任务生成失败:{exc}") from exc

    job = {"id": job_id, "status": "queued", "region": region,
           "frames": [], "error": None}
    with JOB_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "region": region}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    out = {k: job[k] for k in ("id", "status", "region", "error")}
    out["nframes"] = len(job["frames"])
    return out


@app.get("/jobs/{job_id}/frames")
def job_frames(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"任务尚未完成:{job['status']}")
    return FileResponse(_job_dir(job_id) / "frames" / "frames.json",
                        media_type="application/json")


@app.get("/jobs/{job_id}/frames/{k}")
def job_frame(job_id: str, k: int):
    job = JOBS.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(status_code=404, detail="任务不存在或未完成")
    frames = job["frames"]
    if k < 0 or k >= len(frames):
        raise HTTPException(status_code=404, detail=f"帧 {k} 不存在")
    fp = _job_dir(job_id) / "frames" / frames[k]["file"]
    return FileResponse(fp, media_type="application/octet-stream")
