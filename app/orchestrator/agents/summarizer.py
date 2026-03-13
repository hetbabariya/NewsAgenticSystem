from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import SummarizerResponse

def build_summarizer_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Summarizer agent. Your job is to summarize relevant scored articles in a way that is maximally useful for the specific user.\n"
            "Rules:\n"
            "- Call run_summarizer EXACTLY ONCE to generate and store summaries.\n"
            "- If run_summarizer returns 'summarized_count: 0', there are no articles to summarize - STOP and report success.\n"
            "- If asked to create a newspaper, do not do it here (publisher handles that).\n"
            "- Assume downstream tools and agents will use semantic_search_memory over your summaries, so keep them factual, preference-aware, and rich in key points/tags.\n"
            "- Your final response MUST be a valid JSON object matching this EXACT structure:\n"
            '{"success": true, "summarized_count": <number>, "message": "<string>"}\n'
            "Example valid response:\n"
            '{"success": true, "summarized_count": 5, "message": "Successfully summarized 5 articles."}\n'
            "- Do NOT call run_summarizer multiple times.\n"
            "- Do NOT wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
