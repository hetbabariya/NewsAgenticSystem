from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Bot, Update

from app.core.settings import settings
from app.db.session import get_db
from app.telegram.service import log_message

router = APIRouter(tags=["telegram"])


@router.post("/webhook/telegram")
async def telegram_webhook(
    update_json: dict,
    db: AsyncSession = Depends(get_db),
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict:
    if settings.telegram_webhook_secret_token:
        if x_telegram_bot_api_secret_token != settings.telegram_webhook_secret_token:
            raise HTTPException(status_code=403, detail="Invalid webhook secret token")

    bot = Bot(token=settings.telegram_bot_token)

    try:
        update = Update.de_json(update_json, bot)
    except Exception as exc:
        await log_message(db, role="assistant", content=f"[telegram webhook parse error] {type(exc).__name__}")
        raise HTTPException(status_code=400, detail="Invalid Telegram payload") from exc

    message = update.effective_message
    if not message:
        return {"ok": True}

    text = message.text or message.caption
    if not text:
        await log_message(db, role="user", content=f"[non-text message] chat_id={message.chat_id}")
        return {"ok": True}

    await log_message(db, role="user", content=text)

    try:
        from app.orchestrator.runtime import handle_telegram_message

        asyncio.create_task(handle_telegram_message(text, str(message.chat_id)))
    except Exception as exc:
        await log_message(db, role="assistant", content=f"[orchestrator error] {type(exc).__name__}")

    return {"ok": True}