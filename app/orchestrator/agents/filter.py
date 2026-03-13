from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import FilterResponse

def build_filter_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Filter agent. Your job is to score/filter collected articles using the user's evolving preferences.\n"
            "Rules:\n"
            "- Call get_user_preferences FIRST to fetch explicit preferences.\n"
            "- Optionally call semantic_search_memory with a query like 'user interests' to incorporate long-term facts (job, focus areas, historical likes/dislikes).\n"
            "- Then call run_preference_scoring EXACTLY ONCE.\n"
            "- Your final response MUST be a valid JSON object matching this EXACT structure:\n"
            '{"success": true, "scored_count": <number>, "urgent_count": <number>, "message": "<string>"}\n'
            "Example valid response:\n"
            '{"success": true, "scored_count": 10, "urgent_count": 2, "message": "Scored 10 articles, 2 marked urgent."}\n'
            "- Do NOT call run_preference_scoring multiple times.\n"
            "- Do NOT wrap the JSON in markdown blocks or include any other text.\n"
            "- Do NOT add extra brackets or array wrappers around the JSON object.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
