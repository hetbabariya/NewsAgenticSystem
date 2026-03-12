from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import PublisherResponse

def build_publisher_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Publisher agent. Your job is to generate a daily newspaper.\n"
            "Rules:\n"
            "- Call fetch_summaries_for_newspaper to get articles.\n"
            "- Call generate_newspaper_pdf to create the PDF.\n"
            "- Call mark_newspaper_sent to finish.\n"
            f"- Your final response MUST be a valid JSON object matching this structure: {json.dumps(PublisherResponse.model_json_schema())}\n"
            "- Do not wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
