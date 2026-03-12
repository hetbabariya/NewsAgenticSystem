"""
tools/llm.py — LLM caller with automatic key rotation.
All agents call call_llm(). Never call Groq/OpenRouter directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import settings
from app.keys.rotation import RateLimit429Error, call_with_key_rotation

log = logging.getLogger("llm")


_groq_idx = 0
_openrouter_idx = 0


def _next_groq_key() -> str:
    global _groq_idx
    keys = settings.groq_keys
    if not keys:
        raise RuntimeError("No Groq API keys configured.")
    key = keys[_groq_idx % len(keys)]
    _groq_idx += 1
    return key


def _next_openrouter_key() -> str:
    global _openrouter_idx
    keys = settings.openrouter_keys
    if not keys:
        raise RuntimeError("No OpenRouter API keys configured.")
    key = keys[_openrouter_idx % len(keys)]
    _openrouter_idx += 1
    return key


async def call_llm(
    messages: list[dict[str, Any]],
    *,
    model: str = "llama3-8b-8192",
    max_tokens: int = 512,
    temperature: float = 0.2,
    retries: int = 3,
    db: AsyncSession | None = None,
) -> str:
    """
    Call LLM with automatic key rotation on 429.

    Order:
      - Try Groq keys first
      - Fall back to OpenRouter keys

    If `db` is provided, uses the DB-backed key rotation tables.
    """

    if not settings.groq_keys and not settings.openrouter_keys:
        raise RuntimeError("No LLM API keys configured.")

    last_exc: Exception | None = None

    # Prefer DB-backed rotation if available
    if db is not None:
        if settings.groq_keys:
            try:
                return await call_with_key_rotation(
                    db=db,
                    provider="groq",
                    max_attempts=retries,
                    make_call=lambda pk: _call_groq(
                        key=pk.key,
                        messages=messages,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
            except Exception as exc:
                last_exc = exc

        if settings.openrouter_keys:
            try:
                return await call_with_key_rotation(
                    db=db,
                    provider="openrouter",
                    max_attempts=retries,
                    make_call=lambda pk: _call_openrouter(
                        key=pk.key,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"All LLM keys exhausted. Last error: {last_exc}")

    # In-memory rotation (no DB)
    attempts = retries * (max(len(settings.groq_keys), 1) + max(len(settings.openrouter_keys), 1))

    for attempt in range(attempts):
        provider = "groq" if settings.groq_keys and attempt < retries * max(len(settings.groq_keys), 1) else "openrouter"

        try:
            if provider == "groq" and settings.groq_keys:
                key = _next_groq_key()
                return await _call_groq(
                    key=key,
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            if provider == "openrouter" and settings.openrouter_keys:
                key = _next_openrouter_key()
                return await _call_openrouter(
                    key=key,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            last_exc = RuntimeError("No keys available for provider selection")

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                log.warning("Rate limited on %s key — rotating", provider)
                last_exc = exc
                await asyncio.sleep(0.5)
                continue
            raise
        except Exception as exc:
            log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            last_exc = exc
            await asyncio.sleep(1)

    raise RuntimeError(f"All LLM keys exhausted. Last error: {last_exc}")


async def _call_groq(
    *,
    key: str,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        if resp.status_code == 429:
            raise RateLimit429Error("Groq rate limited")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _call_openrouter(
    *,
    key: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://newsagent.app",
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct:free",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        if resp.status_code == 429:
            raise RateLimit429Error("OpenRouter rate limited")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
