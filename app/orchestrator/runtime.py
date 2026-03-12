from __future__ import annotations

import json
import logging
import uuid

from langchain_core.messages import HumanMessage

from app.db.neon import get_recent_conversation, log_conversation, write_episodic
from app.telegram.bot import send_message

from app.orchestrator.graph import build_coordinator, build_registry, build_shared_tools, close_mcp, init_mcp

log = logging.getLogger("orchestrator.runtime")


_coordinator = None
_mcp_client = None
_mcp_tools: list = []


async def init_graph() -> None:
    global _coordinator, _mcp_client, _mcp_tools

    _mcp_client, _mcp_tools = await init_mcp()
    _coordinator = build_coordinator(mcp_client=_mcp_client, mcp_tools=_mcp_tools)


async def shutdown_graph() -> None:
    global _mcp_client
    await close_mcp(_mcp_client)


def get_graph():
    if _coordinator is None:
        raise RuntimeError("Graph not initialised. Call init_graph() first.")
    return _coordinator


from app.core.logger import agent_logger

async def run_graph(trigger: str, payload: dict | None = None) -> dict:
    trace = str(uuid.uuid4())[:8]
    graph = get_graph()

    agent_logger.log_agent_start("COORDINATOR", f"Trigger: {trigger} | Payload: {payload}")
    log.info("[%s] run_graph trigger=%s", trace, trigger)
    payload = payload or {}

    # Deterministic cron routing: do not rely on coordinator prompt.
    if trigger in {"ingest_cron", "daily_cron"}:
        shared_tools = build_shared_tools(mcp_tools=_mcp_tools)
        registry = build_registry()

        if trigger == "ingest_cron":
            sources = payload.get("sources") or ["tavily", "twitter", "reddit"]

            collector = registry.get("collector").build(shared_tools)
            filtr = registry.get("filter").build(shared_tools)
            summarizer = registry.get("summarizer").build(shared_tools)

            agent_logger.log_thought("INGEST_FLOW", f"Starting deterministic ingest sequence. Sources: {sources}")

            agent_logger.log_agent_start("COLLECTOR", "Collecting news articles from sources.")
            r1 = await collector.ainvoke({"messages": [HumanMessage(content=f"Collect news now. sources={sources}")], "context": {"trigger": trigger, "trace_id": trace}})
            agent_logger.log_final_answer("COLLECTOR", str(r1.get("messages", [-1])[-1].content if r1.get("messages") else "Done"))

            agent_logger.log_agent_start("FILTER", "Scoring and filtering collected articles.")
            r2 = await filtr.ainvoke({"messages": [HumanMessage(content="Score/filter newly collected articles.")], "context": {"trigger": trigger, "trace_id": trace}})
            agent_logger.log_final_answer("FILTER", str(r2.get("messages", [-1])[-1].content if r2.get("messages") else "Done"))

            agent_logger.log_agent_start("SUMMARIZER", "Summarizing relevant filtered articles.")
            summarizer_input = "Summarize the relevant articles that were just filtered."
            r3 = await summarizer.ainvoke({"messages": [HumanMessage(content=summarizer_input)], "context": {"trigger": trigger, "trace_id": trace}})
            agent_logger.log_final_answer("SUMMARIZER", str(r3.get("messages", [-1])[-1].content if r3.get("messages") else "Done"))

            result = {"collector": r1, "filter": r2, "summarizer": r3}

        else:
            agent_logger.log_agent_start("PUBLISHER", "Generating daily newspaper PDF.")
            publisher = registry.get("publisher").build(shared_tools)
            result = await publisher.ainvoke({"messages": [HumanMessage(content="Generate today's newspaper.")], "context": {"trigger": trigger, "trace_id": trace}})
            agent_logger.log_final_answer("PUBLISHER", str(result.get("messages", [-1])[-1].content if result.get("messages") else "Done"))

    else:
        msg = f"Trigger={trigger}. Payload={payload}"
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=msg)],
                "context": {"trigger": trigger, "payload": payload, "trace_id": trace},
            }
        )
        agent_logger.log_final_answer("COORDINATOR", str(result.get("messages", [-1])[-1].content if result.get("messages") else "Done"))

    await write_episodic(
        f"{trigger}_completed",
        f"Trigger: {trigger}",
        {"trace_id": trace},
    )

    return result


async def handle_telegram_message(text: str, chat_id: str) -> None:
    trace = str(uuid.uuid4())[:8]
    graph = get_graph()

    log.info("[%s] Telegram: %s", trace, text[:80])
    await log_conversation("user", text)

    history = await get_recent_conversation(10)
    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

    context = {"trigger": "telegram", "chat_id": chat_id, "trace_id": trace, "history": history_str}

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=text)],
            "context": context,
        }
    )

    messages = result.get("messages")
    final_text = None
    if messages:
        last = messages[-1]
        final_text = getattr(last, "content", None)

    if not final_text:
        final_text = "OK"

    # Escape markdown special characters for Telegram's MarkdownV2 if needed,
    # or just use HTML/Plain text if preferred for stability.
    # For now, let's just log and send.
    try:
        await send_message(chat_id, final_text)
    except Exception as e:
        log.error("Failed to send telegram message: %s", e)
        # Fallback to plain text without any formatting if it failed
        import re
        plain_text = re.sub(r'[*_`\[\]()]', '', final_text)
        await send_message(chat_id, f"Error sending formatted message, falling back to plain:\n\n{plain_text}")

    await log_conversation("assistant", final_text)

    await write_episodic(
        "telegram_handled",
        f"Message: {text[:80]}",
        {"trace_id": trace},
    )
