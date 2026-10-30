"""smaller-comfyui - local diffusion API server."""

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import jobs, models
from .config import FRONTEND_DIR, OUTPUT_DIR

app = FastAPI(title="smaller-comfyui", version="0.1.0")


class GenerateRequest(BaseModel):
    model: str = Field(..., description="Model id from GET /api/models")
    kind: str = Field("image", description="image | video")
    prompt: str = ""
    negative_prompt: str = ""
    width: int = Field(0, description="Width in pixels (0 = model default)")
    height: int = Field(0, description="Height in pixels (0 = model default)")
    num_inference_steps: int = Field(25, ge=1, le=50)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    seed: int = Field(0, description="0 = random seed")
    num_frames: int = Field(16, ge=1, le=64)
    fps: int = Field(7, ge=1, le=60)
    init_image: str = ""


@app.get("/api/health")
def health():
    from . import engine

    return {"status": "ok", "device": engine.get_device()}


@app.get("/api/models")
def api_models():
    found = models.discover_models()
    return [
        {
            "id": m.id,
            "path": str(m.path),
            "kind": m.kind,
            "pipeline": m.pipeline,
            "estimated_gb": m.estimated_gb,
        }
        for m in found
    ]


@app.post("/api/jobs")
async def api_create_job(request: GenerateRequest, background_tasks: BackgroundTasks):
    job = jobs.create_job(request.kind, request.model, request.model_dump())
    background_tasks.add_task(jobs.run, job)
    return {"id": job.id, "status": job.status}


@app.get("/api/jobs")
def api_list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return {
        "id": job.id,
        "kind": job.kind,
        "model_id": job.model_id,
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result": job.result,
    }


@app.get("/api/jobs/{job_id}/result")
def api_job_result(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    if job.status != "done" or job.result is None:
        return JSONResponse(status_code=409, content={"error": "job_not_done", "status": job.status})
    return {"id": job.id, "status": job.status, "result": job.result}


@app.get("/api/outputs")
def api_list_outputs():
    if not OUTPUT_DIR.exists():
        return []
    return sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())


@app.get("/api/outputs/{filename}")
def api_get_output(filename: str):
    if "/" in filename or "\\" in filename or not filename:
        return JSONResponse(status_code=400, content={"error": "invalid_filename"})
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not_found"})
    media = "video/mp4" if filename.lower().endswith(".mp4") else "image/png"
    return FileResponse(path=str(path), media_type=media)


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
