from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import SupportResponse

def build_support_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Support agent. Answer user questions about news.\n"
            "Rules:\n"
            "- Use query_recent_summaries to find local news info.\n"
            "- Use web search if local info is insufficient.\n"
            f"- Your final response MUST be a valid JSON object matching this structure: {json.dumps(SupportResponse.model_json_schema())}\n"
            "- Do not wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
