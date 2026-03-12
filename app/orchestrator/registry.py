from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    build: Callable[[list[BaseTool]], "RunnableLike"]


class RunnableLike:
    async def ainvoke(self, input: dict) -> dict: ...  # pragma: no cover


def _matches_query(spec: AgentSpec, query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return True
    return q in spec.name.lower() or q in spec.description.lower()


def format_agent_list(specs: Iterable[AgentSpec], query: str = "") -> str:
    lines: list[str] = []
    for spec in specs:
        if not _matches_query(spec, query):
            continue
        lines.append(f"- {spec.name}: {spec.description}")
    if not lines:
        return "(no agents found)"
    return "\n".join(lines)
