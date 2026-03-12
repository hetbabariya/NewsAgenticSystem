from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.keys.manager import ProviderKey, acquire_key, report_429

T = TypeVar("T")


class RateLimit429Error(RuntimeError):
    pass


async def call_with_key_rotation(
    *,
    db: AsyncSession,
    provider: str,
    make_call: Callable[[ProviderKey], Awaitable[T]],
    max_attempts: int = 3,
) -> T:
    last_exc: Exception | None = None

    for _ in range(max_attempts):
        provider_key = await acquire_key(db, provider=provider)

        try:
            return await make_call(provider_key)
        except RateLimit429Error as exc:
            last_exc = exc
            await report_429(db, provider=provider, key_index=provider_key.key_index)

    raise RateLimit429Error(f"All keys rate-limited for provider='{provider}'.") from last_exc
