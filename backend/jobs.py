"""In-process job queue for generation tasks."""

import time
import uuid

from . import engine

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class Job:
    def __init__(self, job_id: str, kind: str, model_id: str, params: dict):
        self.id = job_id
        self.kind = kind
        self.model_id = model_id
        self.params = params
        self.status = STATUS_QUEUED
        self.error: str = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.result: dict = None


_jobs: dict = {}
_job_seq = 0


def create_job(kind: str, model_id: str, params: dict) -> Job:
    global _job_seq
    _job_seq += 1
    job = Job(f"job_{_job_seq}_{uuid.uuid4().hex[:6]}", kind, model_id, params)
    _jobs[job.id] = job
    return job


def get_job(job_id: str):
    return _jobs.get(job_id)


def list_jobs(limit: int = 20) -> list:
    jobs_list = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return [
        {
            "id": j.id,
            "kind": j.kind,
            "model_id": j.model_id,
            "status": j.status,
            "error": j.error,
            "created_at": j.created_at,
            "updated_at": j.updated_at,
        }
        for j in jobs_list[:limit]
    ]


def run(job: Job) -> None:
    job.status = STATUS_RUNNING
    job.updated_at = time.time()
    try:
        from .config import MODELS_DIR

        model_path = MODELS_DIR / job.model_id
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {job.model_id}")
        job.result = engine.run_generation(str(model_path), job.kind, job.params, job.id)
        job.status = STATUS_DONE
    except Exception as exc:  # noqa: BLE001 - surface any inference failure to the client
        job.status = STATUS_FAILED
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.updated_at = time.time()
