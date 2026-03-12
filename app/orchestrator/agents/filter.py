from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import FilterResponse

def build_filter_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Filter agent. Your job is to score/filter collected articles using preferences.\n"
            "Rules:\n"
            "- Always call get_user_preferences first if needed.\n"
            "- Then call run_preference_scoring.\n"
            f"- Your final response MUST be a valid JSON object matching this structure: {json.dumps(FilterResponse.model_json_schema())}\n"
            "- Do not wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
