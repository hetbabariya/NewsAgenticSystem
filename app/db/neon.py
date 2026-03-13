"""
db/neon.py — NeonDB (asyncpg) connection pool.
All agents call these helpers. Never write raw asyncpg elsewhere.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg

from app.core.settings import settings
from app.semantic_memory import init_semantic_memory, semantic_upsert

log = logging.getLogger("db")

_pool: asyncpg.Pool | None = None


def _normalize_database_url_for_asyncpg(database_url: str) -> tuple[str, dict[str, Any]]:
    connect_kwargs: dict[str, Any] = {}

    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    parts = urlsplit(database_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))

    sslmode = query.pop("sslmode", None)
    if sslmode and sslmode.lower() in {"require", "verify-ca", "verify-full"}:
        connect_kwargs["ssl"] = True

    # libpq-only parameters that asyncpg does not accept
    query.pop("channel_binding", None)

    normalized = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    return normalized, connect_kwargs


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        database_url, connect_kwargs = _normalize_database_url_for_asyncpg(settings.database_url)
        _pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=5,
            command_timeout=30,
            statement_cache_size=0,
            **connect_kwargs,
        )
        log.info("NeonDB pool created")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("NeonDB pool closed")


async def fetch_one(query: str, *args: Any) -> asyncpg.Record | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetch_all(query: str, *args: Any) -> list[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *args)
        return list(rows)


async def execute(query: str, *args: Any) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetch_val(query: str, *args: Any) -> Any:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def get_preferences() -> dict:
    """Load user preferences from Pinecone (semantic memory). Returns {} if not set."""
    # We use a deterministic ID for preferences in Pinecone
    pref_id = "user-preferences"
    index, _ = init_semantic_memory()
    if index is None:
        return {}

    def _do_fetch() -> dict:
        try:
            res = index.fetch(ids=[pref_id])
            if res and "vectors" in res and pref_id in res["vectors"]:
                vec_data = res["vectors"][pref_id]
                metadata = vec_data.get("metadata", {})
                # Preferences are stored as JSON strings in the 'prefs_json' metadata field
                prefs_raw = metadata.get("prefs_json")
                if prefs_raw:
                    if isinstance(prefs_raw, str):
                        return json.loads(prefs_raw)
                    return dict(prefs_raw)
            return {}
        except Exception as exc:
            log.warning("Pinecone preference fetch failed: %s", exc)
            return {}

    return await asyncio.to_thread(_do_fetch)


async def save_preferences(new_prefs: dict) -> bool:
    """Save user preferences directly to Pinecone."""
    pref_id = "user-preferences"
    prefs_text = (
        "User preference profile. "
        f"Topics: {', '.join(new_prefs.get('topics', []))}. "
        f"Keywords: {', '.join(new_prefs.get('keywords', []))}. "
        f"Excluded: {', '.join(new_prefs.get('excluded_topics', []))}."
    )

    try:
        await semantic_upsert(
            item_id=pref_id,
            text=prefs_text,
            metadata={
                "type": "preference",
                "importance": 5,
                "prefs_json": json.dumps(new_prefs)
            },
        )
        return True
    except Exception as exc:
        log.error("Failed to save preferences to Pinecone: %s", exc)
        return False


async def ensure_key_usage_row(provider: str, key_index: int) -> None:
    row = await fetch_one(
        "SELECT id FROM key_usage WHERE provider = $1 AND key_index = $2 LIMIT 1",
        provider,
        int(key_index),
    )
    if row:
        return

    await execute(
        """
        INSERT INTO key_usage (provider, key_index, calls_today, is_blocked, updated_at)
        VALUES ($1, $2, 0, FALSE, NOW())
        """,
        provider,
        int(key_index),
    )


async def bump_key_usage(provider: str, key_index: int) -> None:
    """Increment calls_today for a provider/key_index. Resets the counter when the date changes."""
    await ensure_key_usage_row(provider, key_index)

    row = await fetch_one(
        "SELECT calls_today, updated_at FROM key_usage WHERE provider = $1 AND key_index = $2 LIMIT 1",
        provider,
        int(key_index),
    )
    if not row:
        return

    calls_today = int(row["calls_today"] or 0)
    updated_at = row["updated_at"]
    if updated_at is not None:
        try:
            if updated_at.date() != datetime.date.today():
                calls_today = 0
        except Exception:
            pass

    await execute(
        """
        UPDATE key_usage
        SET calls_today = $1,
            is_blocked = FALSE,
            updated_at = NOW()
        WHERE provider = $2 AND key_index = $3
        """,
        calls_today + 1,
        provider,
        int(key_index),
    )


async def report_key_429(provider: str, key_index: int) -> None:
    """Mark a provider/key_index as blocked due to 429."""
    await ensure_key_usage_row(provider, key_index)
    await execute(
        """
        UPDATE key_usage
        SET last_429_at = NOW(),
            is_blocked = TRUE,
            updated_at = NOW()
        WHERE provider = $1 AND key_index = $2
        """,
        provider,
        int(key_index),
    )


async def write_episodic(event_type: str, description: str, metadata: dict | None = None) -> None:
    await execute(
        """
        INSERT INTO episodic_memory (event_type, description, metadata)
        VALUES ($1, $2, $3)
        """,
        event_type,
        description,
        json.dumps(metadata or {}),
    )


async def log_conversation(role: str, content: str) -> None:
    await execute(
        "INSERT INTO conversation_log (role, content) VALUES ($1, $2)",
        role,
        content,
    )


async def get_recent_conversation(limit: int = 10) -> list[dict]:
    rows = await fetch_all(
        """
        SELECT role, content FROM conversation_log
        ORDER BY created_at DESC LIMIT $1
        """,
        limit,
    )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
