from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import SupportResponse
from app.core.settings import settings

def build_support_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Support agent. Answer user questions about news.\n"
            "Rules:\n"
            "- Call query_recent_summaries FIRST to find local news info.\n"
            "- If local info is insufficient, use AT MOST ONE additional source tool (e.g. fetch_reddit_posts or MCP search).\n"
            f"  - If you use Tavily/GitHub/Twitter MCP search tools, keep query SHORT (max {settings.max_query_chars} chars) and request a small number of results (Tavily <= {settings.tavily_max_results}, GitHub <= {settings.github_max_results}, Twitter <= {settings.twitter_max_results}) if supported by that tool (e.g., `max_results`, `limit`, `per_page`).\n"
            "- Do NOT call the same tool repeatedly. Max tool calls total = 2.\n"
            "- Then STOP and produce the final answer JSON.\n"
            "- Your final response MUST be a valid JSON OBJECT (an instance), matching this EXACT structure:\n"
            '{"answer": "<string>", "sources_used": ["<string>"]}\n'
            "Example valid response:\n"
            '{"answer": "Today on Reddit: ...", "sources_used": ["reddit/r/MachineLearning/top"]}\n'
            "- Do NOT output JSON schema (do NOT include keys like 'properties', '$defs', 'title').\n"
            "- Do not wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
