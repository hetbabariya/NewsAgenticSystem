from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Bot

from app.core.settings import settings
from app.db.models import ConversationLog


async def log_message(db: AsyncSession, role: str, content: str) -> None:
    try:
        db.add(ConversationLog(role=role, content=content))
        await db.commit()
    except Exception:
        await db.rollback()
        raise


async def send_echo_reply(bot: Bot, chat_id: int, text: str) -> str:
    reply_text = f"Echo: {text}"
    await bot.send_message(chat_id=chat_id, text=reply_text)
    return reply_text


async def register_webhook(webhook_url: str) -> dict:
    bot = Bot(token=settings.telegram_bot_token)
    ok = await bot.set_webhook(url=webhook_url, secret_token=settings.telegram_webhook_secret_token)
    return {"ok": bool(ok), "url": webhook_url}