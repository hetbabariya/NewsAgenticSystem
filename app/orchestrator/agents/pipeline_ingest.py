from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from app.core.settings import settings


def build_pipeline_ingest_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You run the ingest pipeline. Execute in order:\n"
            f"1) run_news_collector(sources=['tavily','twitter','reddit']) with safety limits: max_sources={settings.collector_max_sources}, max_topics={settings.collector_max_topics}, max_total_articles={settings.collector_max_articles_total}.\n"
            "2) run_preference_scoring\n"
            "3) run_summarizer\n"
            "Do NOT loop or retry tools indefinitely. If a step returns empty, continue to the next step.\n"
            "Return a short status summary."
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
