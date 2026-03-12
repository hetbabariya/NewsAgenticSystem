"""
scheduler.py  —  NewsAgent Production Scheduler
------------------------------------------------
Runs INSIDE FastAPI process (same Render web service).
Timezone: Asia/Kolkata (IST)

Jobs:
  1. ingest_job       — every 30 min
  2. daily_news_job   — every day at 6:55 AM IST (GitHub fetch before newspaper)
  3. newspaper_job    — every day at 7:00 AM IST
  4. health_ping_job  — every 14 min (keeps Render free tier awake)
  5. cleanup_job      — every day at 2:00 AM IST (prune old raw articles)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
)
from zoneinfo import ZoneInfo

# ── Internal imports (adjust paths to your project) ──────────────────────────
from app.db.neon import execute
from app.core.settings import settings
from app.telegram.bot import send_message

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("scheduler")

# ── Timezone ──────────────────────────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def _run_with_guard(job_name: str, trigger_type: str, payload: dict | None = None) -> None:
    """
    Central wrapper for every cron job.

    Responsibilities:
      - Run the LangGraph graph
      - Catch and log ALL exceptions so APScheduler never sees an unhandled error
        (an unhandled error in APScheduler silently kills the job permanently)
      - Alert via Telegram if a job fails in production
    """
    trace = str(uuid.uuid4())[:8]
    log.info("[%s] %s started trace=%s", job_name, trigger_type, trace)

    try:
        try:
            from app.orchestrator.runtime import run_graph

            await run_graph(trigger_type, payload)
        except Exception:
            log.warning("run_graph not available yet; skipping job=%s trigger=%s", job_name, trigger_type)
        log.info("[%s] finished ok trace=%s", job_name, trace)

    except Exception as exc:
        log.exception("[%s] FAILED trace=%s error=%s", job_name, trace, exc)

        # Alert yourself on Telegram if this is production
        if settings.env == "production" and settings.telegram_admin_chat_id:
            try:
                await send_message(
                    settings.telegram_admin_chat_id,
                    f"⚠️ *Job failed*: `{job_name}`\nTrace: `{trace}`\nError: `{exc}`",
                )
            except Exception:
                pass  # don't let the alert itself crash anything


# ══════════════════════════════════════════════════════════════════════════════
# JOB FUNCTIONS
# Each is a thin async function. All real logic is in the graph.
# ══════════════════════════════════════════════════════════════════════════════

async def ingest_job() -> None:
    """
    Runs every 30 min.
    Triggers: Fanout → all MCP sources → Preference Agent → Summarizer.
    """
    await _run_with_guard("ingest_job", "ingest_cron")


async def daily_news_job() -> None:
    """
    Runs at 6:55 AM IST — 5 min before newspaper.
    Fetches GitHub trending (daily-only source) so its articles
    are scored + summarized before the newspaper agent runs.
    """
    await _run_with_guard(
        "daily_news_job",
        "ingest_cron",
        payload={"sources": ["github"], "reason": "daily_prefetch"},
    )


async def newspaper_job() -> None:
    """
    Runs at 7:00 AM IST.
    Triggers: Newspaper Agent → PDF → Telegram sendDocument.
    """
    await _run_with_guard("newspaper_job", "daily_cron")


async def cleanup_job() -> None:
    """
    Runs at 2:00 AM IST.
    Prunes raw_articles older than 7 days with status = 'discarded'.
    Keeps DB lean on NeonDB free tier.
    """
    log.info("[cleanup_job] running DB prune")
    try:
        deleted = await execute(
            """
            DELETE FROM raw_articles
            WHERE status = 'discarded'
              AND fetched_at < NOW() - INTERVAL '7 days'
            """,
        )
        log.info("[cleanup_job] pruned discarded articles: %s", deleted)

        # Also prune conversation_log older than 30 days
        deleted2 = await execute(
            """
            DELETE FROM conversation_log
            WHERE created_at < NOW() - INTERVAL '30 days'
            """,
        )
        log.info("[cleanup_job] pruned conversation_log: %s", deleted2)

    except Exception as exc:
        log.exception("[cleanup_job] failed: %s", exc)


async def health_ping_job() -> None:
    """
    Runs every 14 min.
    Pings own /health endpoint to prevent Render free tier from sleeping.
    Only active when ENV = 'production'.
    Skip entirely in development to avoid noise.
    """
    if settings.env != "production":
        return

    if not settings.render_external_url:
        return

    url = f"{settings.render_external_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                log.warning("[health_ping_job] unexpected status %s", resp.status_code)

    except Exception as exc:
        log.warning("[health_ping_job] ping failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# APScheduler EVENT LISTENERS
# ══════════════════════════════════════════════════════════════════════════════

def _on_job_executed(event) -> None:
    log.debug("job executed: %s runtime=%s", event.job_id, event.retval)


def _on_job_error(event) -> None:
    # This fires if an exception escapes _run_with_guard (it shouldn't, but belt+suspenders)
    log.error("APScheduler caught job error: job=%s exc=%s", event.job_id, event.exception)


def _on_job_missed(event) -> None:
    """
    Fires when a job was scheduled but couldn't run (e.g. server was asleep).
    Log it — you'll want to know if Render is sleeping too often.
    """
    log.warning(
        "MISSED JOB: %s scheduled=%s — server may have been asleep",
        event.job_id,
        event.scheduled_run_time,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def create_scheduler() -> AsyncIOScheduler:
    """
    Build and return a configured AsyncIOScheduler.
    Does NOT start it — call scheduler.start() in FastAPI lifespan.

    misfire_grace_time:
      If a job was scheduled at T but server only woke at T+5min,
      APScheduler will still run the job if the delay < misfire_grace_time.
      Set to 10 min for ingest (tolerable), 15 min for newspaper (must run).
    """
    scheduler = AsyncIOScheduler(timezone=IST)

    # ── 1. Ingest — every 30 min ────────────────────────────────────────────
    scheduler.add_job(
        ingest_job,
        trigger=IntervalTrigger(minutes=30, timezone=IST),
        id="ingest_job",
        name="News Ingest (30-min)",
        misfire_grace_time=10 * 60,   # 10 min grace
        coalesce=True,                # if 2+ runs were missed, only run once
        max_instances=1,              # never run two ingests simultaneously
        replace_existing=True,
    )

    # ── 2. Daily GitHub fetch — 6:55 AM IST ────────────────────────────────
    scheduler.add_job(
        daily_news_job,
        trigger=CronTrigger(hour=6, minute=55, timezone=IST),
        id="daily_news_job",
        name="Daily GitHub Fetch (6:55 AM IST)",
        misfire_grace_time=5 * 60,
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )

    # ── 3. Newspaper — 7:00 AM IST ──────────────────────────────────────────
    scheduler.add_job(
        newspaper_job,
        trigger=CronTrigger(hour=7, minute=0, timezone=IST),
        id="newspaper_job",
        name="Morning Newspaper (7:00 AM IST)",
        misfire_grace_time=15 * 60,   # 15 min grace — newspaper must run
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )

    # ── 4. Health ping — every 14 min ──────────────────────────────────────
    scheduler.add_job(
        health_ping_job,
        trigger=IntervalTrigger(minutes=14, timezone=IST),
        id="health_ping_job",
        name="Render Keep-Alive Ping",
        misfire_grace_time=2 * 60,
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )

    # ── 5. Cleanup — 2:00 AM IST ────────────────────────────────────────────
    scheduler.add_job(
        cleanup_job,
        trigger=CronTrigger(hour=2, minute=0, timezone=IST),
        id="cleanup_job",
        name="DB Cleanup (2:00 AM IST)",
        misfire_grace_time=30 * 60,
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )

    # ── Event listeners ─────────────────────────────────────────────────────
    scheduler.add_listener(_on_job_executed, EVENT_JOB_EXECUTED)
    scheduler.add_listener(_on_job_error,    EVENT_JOB_ERROR)
    scheduler.add_listener(_on_job_missed,   EVENT_JOB_MISSED)

    return scheduler


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL TRIGGER  (for Telegram commands like "run ingest now")
# ══════════════════════════════════════════════════════════════════════════════

_scheduler_instance: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    """Return the running scheduler instance (set after start_scheduler() called)."""
    if _scheduler_instance is None:
        raise RuntimeError("Scheduler not started yet. Call start_scheduler() first.")
    return _scheduler_instance


async def trigger_job_now(job_id: str) -> bool:
    """
    Manually fire a specific job immediately.
    Used by Orchestrator when user sends Telegram command like 'fetch news now'.

    Returns True if job was found and triggered, False otherwise.
    """
    job_map = {
        "ingest":    ingest_job,
        "newspaper": newspaper_job,
        "cleanup":   cleanup_job,
    }
    fn = job_map.get(job_id)
    if fn is None:
        log.warning("trigger_job_now: unknown job_id=%s", job_id)
        return False

    log.info("Manual trigger: %s", job_id)
    asyncio.create_task(fn())   # fire-and-forget, don't await (would block Telegram response)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN  (called from FastAPI lifespan)
# ══════════════════════════════════════════════════════════════════════════════

async def start_scheduler() -> AsyncIOScheduler:
    """
    Create and start the scheduler.
    Call this inside FastAPI lifespan AFTER DB pool is ready.

    Usage in main.py:
        from scheduler import start_scheduler, shutdown_scheduler

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await get_pool()              # DB first
            await start_scheduler()       # scheduler second
            yield
            await shutdown_scheduler()    # clean shutdown
    """
    global _scheduler_instance

    if _scheduler_instance is not None and _scheduler_instance.running:
        log.warning("start_scheduler called but scheduler already running — skipping")
        return _scheduler_instance

    _scheduler_instance = create_scheduler()
    _scheduler_instance.start()

    _log_schedule()
    return _scheduler_instance


async def shutdown_scheduler() -> None:
    """
    Gracefully shut down on FastAPI exit.
    wait=False so it doesn't block Render's 30-second shutdown window.
    """
    global _scheduler_instance
    if _scheduler_instance and _scheduler_instance.running:
        _scheduler_instance.shutdown(wait=False)
        log.info("Scheduler shut down cleanly")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _log_schedule() -> None:
    """Log all scheduled jobs and their next run times at startup."""
    log.info("━━━ Scheduler started ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for job in _scheduler_instance.get_jobs():
        next_run = job.next_run_time
        next_str = next_run.strftime("%Y-%m-%d %H:%M:%S %Z") if next_run else "not scheduled"
        log.info("  %-35s next: %s", job.name, next_str)
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def get_job_status() -> list[dict]:
    """
    Return job status for /health or status Telegram command.
    Example: user asks 'when is next ingest?' — call this.
    """
    scheduler = get_scheduler()
    return [
        {
            "id":       job.id,
            "name":     job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "pending":  job.pending,
        }
        for job in scheduler.get_jobs()
    ]