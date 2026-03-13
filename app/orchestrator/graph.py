from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from app.core.settings import settings
from app.orchestrator.agents.collector import build_collector_agent
from app.orchestrator.agents.filter import build_filter_agent
from app.orchestrator.agents.memory import build_memory_agent
from app.orchestrator.agents.publisher import build_publisher_agent
from app.orchestrator.agents.summarizer import build_summarizer_agent
from app.orchestrator.agents.support import build_support_agent
from app.orchestrator.dispatcher import AgentRegistry, build_dispatch_tools
from app.orchestrator.mcp import mcp_config
from app.orchestrator.registry import AgentSpec
from app.db.neon import bump_key_usage, report_key_429
from app.orchestrator.tools import (
    fetch_summaries_for_newspaper,
    generate_newspaper_pdf,
    get_system_stats,
    get_user_preferences,
    mark_newspaper_sent,
    query_recent_summaries,
    run_news_collector,
    run_preference_scoring,
    run_summarizer,
    store_user_fact,
    update_user_preferences,
    fetch_reddit_posts,
    semantic_search_memory,
    plan_interest_queries,
)


def _local_tools() -> list:
    return [
        get_user_preferences,
        query_recent_summaries,
        get_system_stats,
        fetch_summaries_for_newspaper,
        mark_newspaper_sent,
        run_news_collector,
        run_preference_scoring,
        run_summarizer,
        update_user_preferences,
        store_user_fact,
        generate_newspaper_pdf,
        fetch_reddit_posts,
        semantic_search_memory,
        plan_interest_queries,
    ]


def local_tools() -> list:
    return _local_tools()


def _build_registry() -> AgentRegistry:
    specs = [
        AgentSpec(
            name="collector",
            description="Collects fresh news articles from configured sources and stores them.",
            build=lambda shared: build_collector_agent(model=_openrouter_model(settings.model_collector), tools=shared),
        ),
        AgentSpec(
            name="filter",
            description="Scores and filters collected articles based on user preferences.",
            build=lambda shared: build_filter_agent(model=_openrouter_model(settings.model_filter), tools=shared),
        ),
        AgentSpec(
            name="summarizer",
            description="Summarizes relevant scored articles and stores summaries.",
            build=lambda shared: build_summarizer_agent(model=_openrouter_model(settings.model_summarizer), tools=shared),
        ),
        AgentSpec(
            name="memory",
            description="Updates user preferences and stores user facts.",
            build=lambda shared: build_memory_agent(model=_openrouter_model(settings.model_memory), tools=shared),
        ),
        AgentSpec(
            name="support",
            description="Answers user questions about news using stored summaries and (optionally) web search.",
            build=lambda shared: build_support_agent(model=_openrouter_model(settings.model_support), tools=shared),
        ),
        AgentSpec(
            name="publisher",
            description="Generates daily newspaper (fetch summaries -> generate PDF -> mark sent).",
            build=lambda shared: build_publisher_agent(model=_openrouter_model(settings.model_publisher), tools=shared),
        ),
    ]
    return AgentRegistry(specs)


def build_registry() -> AgentRegistry:
    return _build_registry()


_cached_groq_model = None
_cached_openrouter_model = None
_groq_key_idx = 0
_openrouter_key_idx = 0


def _is_rate_limit_429(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return " 429" in msg or "rate limit" in msg or "rate limited" in msg


class RotatingGroqChatModel(Runnable):
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        tools: list[BaseTool] | None = None,
        max_attempts: int = 3,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._tools = tools or []
        self._max_attempts = max_attempts

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> "RotatingGroqChatModel":
        # Keep the same wrapper, but remember tool schema so each per-call ChatGroq instance
        # is tool-enabled.
        return RotatingGroqChatModel(
            model=self._model,
            temperature=self._temperature,
            tools=list(tools),
            max_attempts=self._max_attempts,
        )

    def _next_key(self) -> tuple[str, int]:
        global _groq_key_idx
        keys = settings.groq_keys
        if not keys:
            raise RuntimeError("Groq keys not configured (GROQ_KEY_0...).")
        idx = _groq_key_idx % len(keys)
        key = keys[idx]
        _groq_key_idx += 1
        return key, idx

    def _new_client(self, *, key: str) -> ChatGroq:
        client = ChatGroq(
            model=self._model,
            api_key=key,
            temperature=self._temperature,
        )

        if self._tools:
            client = client.bind_tools(self._tools)

        return client

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        attempts = max(self._max_attempts, len(settings.groq_keys) or 1)
        for _ in range(attempts):
            try:
                key, idx = self._next_key()
                await bump_key_usage("groq", idx)
                response = await self._new_client(key=key).ainvoke(input, config=config, **kwargs)
                from app.core.logger import agent_logger
                agent_logger.log_llm_io(self._model, input, response)
                return response
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_429(exc):
                    try:
                        await report_key_429("groq", idx)
                    except Exception:
                        pass
                    continue
                raise
        raise RuntimeError(f"All Groq keys rate-limited/exhausted. Last error: {last_exc}") from last_exc


class RotatingOpenRouterChatModel(Runnable):
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        tools: list[BaseTool] | None = None,
        max_attempts: int = 3,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._tools = tools or []
        self._max_attempts = max_attempts

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> "RotatingOpenRouterChatModel":
        return RotatingOpenRouterChatModel(
            model=self._model,
            temperature=self._temperature,
            tools=list(tools),
            max_attempts=self._max_attempts,
        )

    def _next_key(self) -> tuple[str, int]:
        global _openrouter_key_idx
        keys = settings.openrouter_keys
        if not keys:
            raise RuntimeError("OpenRouter keys not configured (OR_KEY_0... or OPENROUTER_API_KEYS).")
        idx = _openrouter_key_idx % len(keys)
        key = keys[idx]
        _openrouter_key_idx += 1
        return key, idx

    def _new_client(self, *, key: str) -> ChatOpenAI:
        client = ChatOpenAI(
            model=self._model,
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            temperature=self._temperature,
        )

        if self._tools:
            client = client.bind_tools(self._tools)

        return client

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        attempts = max(self._max_attempts, len(settings.openrouter_keys) or 1)
        for _ in range(attempts):
            try:
                key, idx = self._next_key()
                try:
                    import asyncio
                    asyncio.get_event_loop().create_task(bump_key_usage("openrouter", idx))
                except Exception:
                    pass
                response = self._new_client(key=key).invoke(input, config=config, **kwargs)
                from app.core.logger import agent_logger
                agent_logger.log_llm_io(self._model, input, response)
                return response
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_429(exc):
                    try:
                        import asyncio
                        asyncio.get_event_loop().create_task(report_key_429("openrouter", idx))
                    except Exception:
                        pass
                    continue
                raise
        raise RuntimeError(
            f"All OpenRouter keys rate-limited/exhausted. Last error: {last_exc}"
        ) from last_exc

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        attempts = max(self._max_attempts, len(settings.openrouter_keys) or 1)
        for _ in range(attempts):
            try:
                key, idx = self._next_key()
                await bump_key_usage("openrouter", idx)
                response = await self._new_client(key=key).ainvoke(input, config=config, **kwargs)
                from app.core.logger import agent_logger
                agent_logger.log_llm_io(self._model, input, response)
                return response
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_429(exc):
                    try:
                        await report_key_429("openrouter", idx)
                    except Exception:
                        pass
                    continue
                raise
        raise RuntimeError(
            f"All OpenRouter keys rate-limited/exhausted. Last error: {last_exc}"
        ) from last_exc


def _groq_model(model_name: str | None = None) -> Any:
    return RotatingGroqChatModel(
        model=model_name or settings.model_coordinator,
        temperature=0,
        max_attempts=3,
    )


def _openrouter_model(model_name: str) -> Any:
    return RotatingOpenRouterChatModel(
        model=model_name,
        temperature=0,
        max_attempts=3,
    )


def build_coordinator(*, mcp_client: MultiServerMCPClient | None, mcp_tools: list) -> Any:
    shared_tools = _local_tools() + mcp_tools
    registry = _build_registry()
    dispatch_tools = build_dispatch_tools(registry=registry, shared_tools=shared_tools)

    prompt = SystemMessage(
        content=(
            "You are the autonomous Coordinator in a hierarchical news multi-agent system.\n"
            "Your goal is to satisfy the user's request completely by delegating to specialized sub-agents.\n\n"
            "Operational Loop (Autonomous Behavior):\n"
            "1. Read the user's request and evaluate the current state.\n"
            "2. Decide which sub-agent (collector, filter, summarizer, memory, support, publisher) is needed.\n"
            "3. Call task('<agent_name>', ...) to delegate.\n"
            "4. EVALUATE the result from the sub-agent. If the goal is not yet met (e.g., articles fetched but not yet filtered or summarized), CONTINUE the loop by calling the next agent.\n"
            "5. If a sub-agent suggests a follow-up or if you are not satisfied with the depth of the answer, run another loop cycle.\n"
            "6. Once the entire workflow is truly complete and you are satisfied, provide a comprehensive 'Final Answer:'.\n\n"
            "Hierarchy & tools:\n"
            "- Use list_agents to discover available sub-agents.\n"
            "- Use task('<agent_name>', ...) for delegation.\n"
            "- You are responsible for the FINAL response to the user. Do not just relay sub-agent snippets; synthesize them.\n\n"
            "Decision rules:\n"
            "- News Ingestion: You MUST run the sequence: collector -> filter -> summarizer. Do not stop until summarization is done.\n"
            "- Preferences: Use the 'memory' agent for any changes to interests or facts. Preferences are stored semantically in Pinecone.\n"
            "- News Questions: Use 'support' for complex research, then summarize the findings yourself.\n\n"
            "Constraint: You MUST conclude with 'Final Answer:' followed by your synthesis. If you are not satisfied, do NOT give a Final Answer; instead, call another tool."
        )
    )

    return create_react_agent(model=_openrouter_model(settings.model_coordinator), tools=dispatch_tools, prompt=prompt)


def build_shared_tools(*, mcp_tools: list) -> list:
    return _local_tools() + (mcp_tools or [])


async def init_mcp() -> tuple[MultiServerMCPClient | None, list]:
    cfg = mcp_config()
    if not cfg:
        return None, []
    client = MultiServerMCPClient(cfg)
    tools = await client.get_tools()
    return client, tools


async def close_mcp(client: MultiServerMCPClient | None) -> None:
    if client is None:
        return
    # Best-effort cleanup. The adapter/client API has changed across versions,
    # so we defensively call whichever close method exists.
    try:
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            await aclose()
            return

        close = getattr(client, "close", None)
        if callable(close):
            res = close()
            if hasattr(res, "__await__"):
                await res
            return
    except Exception:
        # Avoid raising on shutdown; this is purely a cleanup path.
        return
