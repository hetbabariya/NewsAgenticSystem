from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import MemoryResponse

def build_memory_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Memory agent. Your job is to update user preferences and store facts.\n"
            "Rules:\n"
            "- Call update_user_preferences for preference changes.\n"
            "- Call store_user_fact for durable facts.\n"
            f"- Your final response MUST be a valid JSON object matching this structure: {json.dumps(MemoryResponse.model_json_schema())}\n"
            "- Do not wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
