from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from app.core.settings import settings
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log = logging.getLogger("main")
    log.info("StartiAUTO_WARMUPng NewsAgent...")

    try:
        settings.validate()
    except Exception:
        log.exception("Settings validation failed")
        raise

    # Lazy-init: keep boot lightweight so Render can bind the port before heavy init.
    app.state.orchestrator_ready = False
    app.state.scheduler_started = False

    # Register Telegram webhook in production if RENDER_EXTERNAL_URL is set.
    if settings.env == "production" and settings.render_external_url:
        try:
            from app.telegram.service import register_webhook

            webhook_url = f"{settings.render_external_url.rstrip('/')}/webhook/telegram"
            await register_webhook(webhook_url)
            log.info("Telegram webhook registration attempted: %s", webhook_url)
        except Exception:
            log.exception("Telegram webhook registration failed")

    auto = str(os.getenv("AUTO_WARMUP", "")).strip().lower()
    print("auto: ",auto)
    if auto in {"1", "true", "yes"}:
        async def _auto_warmup() -> None:
            try:
                from app.orchestrator.runtime import ensure_graph_initialized

                await ensure_graph_initialized()
                app.state.orchestrator_ready = True
            except Exception:
                log.exception("Auto warmup: orchestrator init failed")
                return

            try:
                from app.orchestrator.scheduler import start_scheduler

                if not getattr(app.state, "scheduler_started", False):
                    await start_scheduler()
                    app.state.scheduler_started = True
            except Exception as exc:
                log.warning("Auto warmup: scheduler start skipped/failed: %s", exc)

        asyncio.create_task(_auto_warmup())

    yield  # ← app runs here

    try:
        from app.orchestrator.runtime import shutdown_graph

        await shutdown_graph()
    except Exception:
        log.exception("Failed to shutdown orchestrator graph")

    try:
        from app.orchestrator.scheduler import shutdown_scheduler

        if getattr(app.state, "scheduler_started", False):
            await shutdown_scheduler()
    except Exception:
        log.exception("Failed to shutdown scheduler")

    log.info("NewsAgent shut down")


app = FastAPI(lifespan=lifespan, title="NewsAgent", version="1.0.0")


@app.get("/health")
@app.head("/health")
async def health() -> dict:
    jobs: list[dict] | None = None
    try:
        from app.orchestrator.scheduler import get_job_status

        jobs = get_job_status()
    except Exception:
        jobs = None

    return {"status": "ok", "jobs": jobs}


@app.post("/warmup")
async def warmup() -> dict:
    """Initialize orchestrator graph and scheduler on-demand.

    Use this in production after deploy to avoid OOM during cold start.
    """

    log = logging.getLogger("main")
    try:
        from app.orchestrator.runtime import ensure_graph_initialized

        await ensure_graph_initialized()
        app.state.orchestrator_ready = True
    except Exception as exc:
        log.exception("Warmup: orchestrator init failed")
        return {"ok": False, "stage": "orchestrator", "error": type(exc).__name__}

    # Scheduler is optional; start if available.
    try:
        from app.orchestrator.scheduler import start_scheduler

        if not getattr(app.state, "scheduler_started", False):
            await start_scheduler()
            app.state.scheduler_started = True
    except Exception as exc:
        # Do not fail warmup if scheduler can't start.
        log.warning("Warmup: scheduler start skipped/failed: %s", exc)

    return {
        "ok": True,
        "orchestrator_ready": bool(getattr(app.state, "orchestrator_ready", False)),
        "scheduler_started": bool(getattr(app.state, "scheduler_started", False)),
        "lazy_init": True,
        "port": os.getenv("PORT"),
    }


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