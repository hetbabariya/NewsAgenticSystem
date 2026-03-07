from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Bot

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
