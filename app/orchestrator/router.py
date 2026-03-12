from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/internal", tags=["internal"])


@router.get("/jobs")
async def job_status() -> dict:
    try:
        from app.orchestrator.scheduler import get_job_status

        return {"ok": True, "jobs": get_job_status()}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


@router.post("/trigger/{job_id}")
async def manual_trigger(job_id: str) -> dict:
    try:
        from app.orchestrator.scheduler import trigger_job_now

        ok = await trigger_job_now(job_id)
        return {"ok": True, "triggered": ok, "job_id": job_id}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "job_id": job_id}
