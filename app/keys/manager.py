from __future__ import annotations

import datetime
from dataclasses import dataclass

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import settings
from app.db.models import KeyUsage


class NoAvailableKeysError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderKey:
    provider: str
    key: str
    key_index: int


def _parse_keys(raw: str | None) -> list[str]:
    if not raw:
        return []

    # Accept either comma-separated or newline-separated values
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def get_provider_keys(provider: str) -> list[str]:
    provider = provider.lower()
    if provider == "groq":
        return _parse_keys(settings.groq_api_keys)
    if provider == "openrouter":
        return _parse_keys(settings.openrouter_api_keys)
    return []


async def ensure_key_rows(db: AsyncSession, provider: str, key_count: int) -> None:
    if key_count <= 0:
        return

    existing = await db.execute(select(KeyUsage.key_index).where(KeyUsage.provider == provider))
    existing_indices = {row[0] for row in existing.all() if row[0] is not None}

    now = datetime.datetime.now(datetime.timezone.utc)

    created = False
    for idx in range(key_count):
        if idx in existing_indices:
            continue
        created = True
        db.add(
            KeyUsage(
                provider=provider,
                key_index=idx,
                calls_today=0,
                is_blocked=False,
                last_429_at=None,
                updated_at=now,
            )
        )

    if created:
        await db.commit()


async def acquire_key(db: AsyncSession, provider: str) -> ProviderKey:
    keys = get_provider_keys(provider)
    if not keys:
        raise NoAvailableKeysError(f"No API keys configured for provider='{provider}'.")

    await ensure_key_rows(db, provider=provider, key_count=len(keys))

    now = datetime.datetime.now(datetime.timezone.utc)
    cooldown = datetime.timedelta(seconds=settings.key_rotation_cooldown_seconds)
    cutoff = now - cooldown

    async with db.begin():
        # Pick least-used key today, ignoring blocked keys unless cooldown expired.
        stmt = (
            select(KeyUsage)
            .where(
                and_(
                    KeyUsage.provider == provider,
                    or_(
                        KeyUsage.is_blocked.is_(False),
                        KeyUsage.last_429_at.is_(None),
                        KeyUsage.last_429_at < cutoff,
                    ),
                )
            )
            .order_by(KeyUsage.calls_today.asc(), KeyUsage.key_index.asc())
            .with_for_update()
            .limit(1)
        )

        res = await db.execute(stmt)
        row = res.scalar_one_or_none()
        if not row or row.key_index is None:
            raise NoAvailableKeysError(f"No available keys for provider='{provider}' (all blocked).")

        # Reset daily usage if row.updated_at is not today (UTC).
        if row.updated_at and row.updated_at.date() != now.date():
            row.calls_today = 0

        row.calls_today = int(row.calls_today or 0) + 1
        row.is_blocked = False
        row.updated_at = now

        key_index = int(row.key_index)
        key = keys[key_index]

    return ProviderKey(provider=provider, key=key, key_index=key_index)


async def report_429(db: AsyncSession, provider: str, key_index: int) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)

    res = await db.execute(
        select(KeyUsage).where(
            and_(KeyUsage.provider == provider, KeyUsage.key_index == key_index)
        )
    )
    row = res.scalar_one_or_none()
    if not row:
        return

    row.last_429_at = now
    row.is_blocked = True
    row.updated_at = now
    await db.commit()
