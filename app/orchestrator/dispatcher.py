from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from app.orchestrator.registry import AgentSpec, format_agent_list

log = logging.getLogger("orchestrator.dispatcher")


class AgentRegistry:
    def __init__(self, specs: list[AgentSpec]):
        self._specs = {s.name: s for s in specs}

    @property
    def specs(self) -> list[AgentSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> AgentSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown agent_name={name}")
        return spec


def build_dispatch_tools(*, registry: AgentRegistry, shared_tools: list[BaseTool]) -> list[BaseTool]:
    @tool
    def list_agents(query: str = "") -> str:
        """List available subagents, optionally filtered by a query."""
        return format_agent_list(registry.specs, query=query)

    @tool
    async def task(agent_name: str, description: str, context_json: str = "{}") -> str:
        """Launch an ephemeral subagent for a task.

        Use list_agents to discover available agent_name values.

        Args:
            agent_name: Name of the subagent.
            description: The task to perform.
            context_json: JSON string containing runtime context (e.g. chat_id, trigger).
        """
        try:
            context: dict[str, Any] = json.loads(context_json or "{}")
        except Exception:
            context = {}

        spec = registry.get(agent_name)
        agent = spec.build(shared_tools)

        result = await agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=f"You are subagent '{spec.name}'. {spec.description}"),
                    HumanMessage(content=description),
                ],
                "context": context,
            }
        )

        messages = result.get("messages")
        if messages:
            last = messages[-1]
            return getattr(last, "content", str(last))
        return ""

    return [list_agents, task]
