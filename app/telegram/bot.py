"""
telegram/bot.py — Outbound Telegram helpers.
All agents use send_message() / send_document(). Never call the API directly.
"""

from __future__ import annotations

import logging
from telegram import Bot

from app.core.settings import settings

log = logging.getLogger("telegram")


def _get_bot() -> Bot:
    return Bot(token=settings.telegram_bot_token)


async def send_message(chat_id: str, text: str, parse_mode: str = "Markdown") -> bool:
    """Send a text message. Returns True on success."""
    try:
        bot = _get_bot()
        try:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        except Exception as e:
            log.error("send_message failed: %s", e)
            # Don't raise, just log, so the process can continue during testing
            return False
        return True
    except Exception as exc:
        log.error("send_message failed: %s", exc)
        return False


async def send_document(
    chat_id: str,
    file_path: str,
    filename: str,
    caption: str = "",
) -> bool:
    """Send a file (PDF). Returns True on success."""
    try:
        bot = _get_bot()
        with open(file_path, "rb") as f:
            await bot.send_document(chat_id=chat_id, document=f, filename=filename, caption=caption)
        return True
    except Exception as exc:
        log.error("send_document failed: %s", exc)
        return False
