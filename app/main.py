from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.settings import settings


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log = logging.getLogger("main")
    log.info("Starting NewsAgent...")

    try:
        settings.validate()
    except Exception:
        log.exception("Settings validation failed")
        raise

    try:
        from app.orchestrator.runtime import init_graph

        await init_graph()
    except Exception:
        log.exception("Failed to initialize orchestrator graph")
        raise

    # Scheduler is optional during early development; don't crash if missing/broken.
    start_scheduler = None
    shutdown_scheduler = None
    try:
        from app.orchestrator.scheduler import start_scheduler as _start, shutdown_scheduler as _shutdown

        start_scheduler = _start
        shutdown_scheduler = _shutdown
    except Exception:
        log.warning("Scheduler not started (module missing or import error).")

    if start_scheduler:
        try:
            await start_scheduler()
        except Exception:
            log.exception("Failed to start scheduler")

    # Register Telegram webhook in production if RENDER_EXTERNAL_URL is set.
    if settings.env == "production" and settings.render_external_url:
        try:
            from app.telegram.service import register_webhook

            webhook_url = f"{settings.render_external_url.rstrip('/')}/webhook/telegram"
            await register_webhook(webhook_url)
            log.info("Telegram webhook registration attempted: %s", webhook_url)
        except Exception:
            log.exception("Telegram webhook registration failed")

    yield  # ← app runs here

    try:
        from app.orchestrator.runtime import shutdown_graph

        await shutdown_graph()
    except Exception:
        log.exception("Failed to shutdown orchestrator graph")

    if shutdown_scheduler:
        try:
            await shutdown_scheduler()
        except Exception:
            log.exception("Failed to shutdown scheduler")

    log.info("NewsAgent shut down")


app = FastAPI(lifespan=lifespan, title="NewsAgent", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    jobs: list[dict] | None = None
    try:
        from app.orchestrator.scheduler import get_job_status

        jobs = get_job_status()
    except Exception:
        jobs = None

    return {"status": "ok", "jobs": jobs}


def _try_include_routers(fastapi_app: FastAPI) -> None:
    log = logging.getLogger("main")

    try:
        from app.telegram.router import router as telegram_router

        fastapi_app.include_router(telegram_router)
    except Exception:
        log.warning("Telegram router not included (missing or import error).")

    try:
        from app.orchestrator.router import router as orchestrator_router

        fastapi_app.include_router(orchestrator_router)
    except Exception:
        log.warning("Orchestrator internal router not included (missing or import error).")


_try_include_routers(app)