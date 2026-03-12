from __future__ import annotations

import json
import logging
import os
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from app.db.neon import get_recent_conversation, log_conversation, write_episodic
from app.telegram.bot import send_document, send_message

from app.core.settings import settings

from app.orchestrator.graph import build_coordinator, build_registry, build_shared_tools, close_mcp, init_mcp
from app.orchestrator.tools import (
    run_news_collector,
    run_preference_scoring,
    run_summarizer,
    fetch_summaries_for_newspaper,
    generate_newspaper_pdf,
    mark_newspaper_sent,
    store_user_fact,
    update_user_preferences,
)

log = logging.getLogger("orchestrator.runtime")


_coordinator = None
_mcp_client = None
_mcp_tools: list = []


async def init_graph() -> None:
    global _coordinator, _mcp_client, _mcp_tools

    _mcp_client, _mcp_tools = await init_mcp()
    log.info("MCP tools loaded: %d tools", len(_mcp_tools))
    if _mcp_tools:
        for t in _mcp_tools:
            log.info("  - MCP tool: %s", getattr(t, "name", str(t)))
    else:
        log.warning("No MCP tools loaded. Check TAVILY_API_KEYS, GITHUB_TOKEN, TWITTER_API_KEY in .env")
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
            sources = payload.get("sources") or ["reddit"]  # Default to reddit only since MCP tools are passed separately

            def _truncate(s: str, max_len: int) -> str:
                return s if len(s) <= max_len else s[:max_len]

            def _pick_tool(name_contains: list[str]) -> object | None:
                for t in _mcp_tools or []:
                    tool_name = str(getattr(t, "name", "") or "").lower()
                    if all(n in tool_name for n in name_contains):
                        return t
                return None

            def _normalize_mcp_items(res: object) -> list[dict]:
                if res is None:
                    return []
                if isinstance(res, list):
                    return [x for x in res if isinstance(x, dict)]
                if isinstance(res, dict):
                    for k in ["results", "items", "data"]:
                        v = res.get(k)
                        if isinstance(v, list):
                            return [x for x in v if isinstance(x, dict)]
                # Handle text-based responses (e.g., Tavily returns formatted text)
                if isinstance(res, str):
                    # Parse Tavily-style "Detailed Results:" text format
                    items = []
                    blocks = res.split("\n\n")
                    for block in blocks:
                        item = {}
                        for line in block.strip().split("\n"):
                            if line.startswith("Title:"):
                                item["title"] = line[6:].strip()
                            elif line.startswith("URL:"):
                                item["url"] = line[4:].strip()
                            elif line.startswith("Content:"):
                                item["content"] = line[8:].strip()
                        if item.get("url"):
                            items.append(item)
                    return items
                return []

            def _to_article(item: dict, source: str) -> dict | None:
                url = (
                    item.get("url")
                    or item.get("link")
                    or item.get("html_url")
                    or item.get("permalink")
                    or ""
                )
                url = str(url).strip()
                if not url:
                    return None
                title = item.get("title") or item.get("name") or item.get("full_name") or ""
                content = (
                    item.get("content")
                    or item.get("snippet")
                    or item.get("description")
                    or item.get("text")
                    or ""
                )
                return {
                    "url": url,
                    "title": str(title),
                    "content": str(content),
                    "source": source,
                }

            mcp_articles: list[dict] = []
            if settings.mcp_ingest_enabled and _mcp_tools:
                call_budget = max(0, int(settings.mcp_max_calls_per_run))

                async def _call_mcp(tool: object, args: dict, source: str) -> None:
                    nonlocal call_budget, mcp_articles
                    if call_budget <= 0:
                        return
                    call_budget -= 1
                    tool_name = getattr(tool, "name", str(tool))
                    try:
                        log.info("Calling MCP tool %s with args: %s", tool_name, args)
                        res = await tool.ainvoke(args)
                        log.info("MCP tool %s returned: %s...", tool_name, str(res)[:500])
                        items = _normalize_mcp_items(res)
                        log.info("Normalized %d items from %s", len(items), tool_name)
                        for it in items:
                            art = _to_article(it, source=source)
                            if art:
                                mcp_articles.append(art)
                    except Exception as e:
                        import traceback
                        log.warning("MCP tool call failed for %s: %s\n%s", source, e, traceback.format_exc())

                topics = payload.get("topics") or ["AI", "Machine learning"]
                topics = [str(t).strip() for t in (topics or []) if str(t).strip()]
                topics = topics[: max(1, int(settings.collector_max_topics))]

                if "tavily" in [str(s).lower() for s in sources]:
                    tavily_tool = _pick_tool(["tavily", "search"])
                    log.info("Tavily tool found: %s", bool(tavily_tool))
                    if tavily_tool:
                        for q in topics:
                            await _call_mcp(
                                tavily_tool,
                                {
                                    "query": _truncate(q, settings.max_query_chars),
                                    "max_results": int(settings.tavily_max_results),
                                },
                                source="tavily",
                            )

                if "twitter" in [str(s).lower() for s in sources]:
                    twitter_tool = _pick_tool(["search", "tweets"]) or _pick_tool(["tweets"])
                    log.info("Twitter tool found: %s", bool(twitter_tool))
                    if twitter_tool:
                        for q in topics:
                            await _call_mcp(
                                twitter_tool,
                                {
                                    "query": _truncate(q, settings.max_query_chars),
                                    "count": int(settings.twitter_max_results),
                                },
                                source="twitter",
                            )

                if "github" in [str(s).lower() for s in sources]:
                    gh_tool = _pick_tool(["search", "repositories"]) or _pick_tool(["trending"])
                    log.info("GitHub tool found: %s", bool(gh_tool))
                    if gh_tool:
                        for q in topics:
                            await _call_mcp(
                                gh_tool,
                                {
                                    "query": _truncate(q, settings.max_query_chars),
                                    "per_page": int(settings.github_max_results),
                                    "limit": int(settings.github_max_results),
                                },
                                source="github",
                            )

            log.info("MCP articles collected: %d", len(mcp_articles))
            agent_logger.log_thought("INGEST_FLOW", f"Starting deterministic ingest sequence (tool-first). Sources: {sources}, MCP articles: {len(mcp_articles)}")

            # IMPORTANT: For cron ingest, avoid Groq tool-calling agents.
            # Groq can fail with `tool_use_failed` if the agent's *final answer* is invalid JSON.
            # Instead, call the tools directly (they already enforce structured output via Pydantic).

            agent_logger.log_agent_start("COLLECTOR", "Collecting news articles from sources (direct tool call).")
            r1 = await run_news_collector.ainvoke({"sources": sources, "topics": None, "articles": (mcp_articles or None)})
            agent_logger.log_final_answer("COLLECTOR", str(r1))

            agent_logger.log_agent_start("FILTER", "Scoring and filtering collected articles (direct tool call).")
            r2 = await run_preference_scoring.ainvoke({})
            agent_logger.log_final_answer("FILTER", str(r2))

            agent_logger.log_agent_start("SUMMARIZER", "Summarizing relevant filtered articles (direct tool call).")
            r3 = await run_summarizer.ainvoke({})
            agent_logger.log_final_answer("SUMMARIZER", str(r3))

            result = {"collector": r1, "filter": r2, "summarizer": r3}

        else:
            agent_logger.log_agent_start("PUBLISHER", "Generating daily newspaper PDF (direct tool call).")

            fetched = await fetch_summaries_for_newspaper.ainvoke({})
            summaries = fetched.get("summaries") or []

            pdf_res = await generate_newspaper_pdf.ainvoke({"summaries": summaries})

            # Send the PDF to Telegram admin (if configured)
            if summaries and settings.telegram_admin_chat_id:
                try:
                    storage = (pdf_res or {}).get("storage") or {}
                    url = storage.get("public_url") or storage.get("signed_url")
                    caption = "Your daily newspaper is ready."
                    if url:
                        caption = f"{caption}\n\nDownload: {url}"

                    pdf_path = (pdf_res or {}).get("pdf_path")
                    if isinstance(pdf_path, str) and pdf_path.lower().endswith(".pdf") and os.path.exists(pdf_path):
                        await send_document(
                            settings.telegram_admin_chat_id,
                            pdf_path,
                            filename=os.path.basename(pdf_path),
                            caption=caption,
                        )
                    else:
                        await send_message(settings.telegram_admin_chat_id, caption)
                except Exception as exc:
                    log.warning("Failed to send newspaper to Telegram: %s", exc)

            sent_res = None
            if summaries:
                sent_res = await mark_newspaper_sent.ainvoke({})

            result = {"fetched": fetched, "pdf": pdf_res, "marked_sent": sent_res}
            agent_logger.log_final_answer("PUBLISHER", str(result))

    # Deterministic Telegram routing: avoid coordinator loops / repeated tool calls.
    elif trigger == "telegram" and isinstance(payload, dict) and payload.get("text"):
        shared_tools = build_shared_tools(mcp_tools=_mcp_tools)
        registry = build_registry()

        text = str(payload.get("text") or "")
        text_l = text.lower()
        await log_conversation("user", text)
        context = {
            "trigger": trigger,
            "payload": payload,
            "trace_id": trace,
            "chat_id": payload.get("chat_id"),
        }

        # Heuristic: preference/fact updates -> memory, otherwise -> support.
        is_pref_update = any(
            k in text_l
            for k in [
                "i am interested",
                "i like",
                "i prefer",
                "my preference",
                "i want to avoid",
                "remember that",
                "i am ",
            ]
        )

        if is_pref_update:
            agent_logger.log_agent_start("MEMORY", "Handling preference/fact update (deterministic telegram routing).")
            # Deterministic memory updates: avoid LLM tool-calling issues (400) by invoking tools directly.
            pref_res = await update_user_preferences.ainvoke({"change_description": text})

            fact_res = None
            if any(k in text_l for k in ["remember", "fact", "i am ", "my name", "my job", "my role"]):
                fact_res = await store_user_fact.ainvoke({"fact": text})

            result = {"preferences": pref_res, "fact": fact_res}
            agent_logger.log_final_answer("MEMORY", str(result))

            try:
                await log_conversation("assistant", json.dumps(result))
            except Exception:
                await log_conversation("assistant", str(result))
        else:
            agent_logger.log_agent_start("SUPPORT", "Handling user news question (deterministic telegram routing).")
            support = registry.get("support").build(shared_tools)
            result = await support.ainvoke(
                {"messages": [HumanMessage(content=text)], "context": context},
                config=RunnableConfig(recursion_limit=10),
            )
            agent_logger.log_final_answer(
                "SUPPORT",
                str(result.get("messages", [-1])[-1].content if result.get("messages") else "Done"),
            )

            messages = result.get("messages")
            final_text = None
            if messages:
                last = messages[-1]
                final_text = getattr(last, "content", None)
            if final_text:
                await log_conversation("assistant", str(final_text))

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
