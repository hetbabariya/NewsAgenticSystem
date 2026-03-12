from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent


def build_pipeline_ingest_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You run the ingest pipeline. Execute in order:\n"
            "1) run_news_collector(sources=['tavily','twitter','reddit'])\n"
            "2) run_preference_scoring\n"
            "3) run_summarizer\n"
            "Return a short status summary."
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
