from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import SummarizerResponse

def build_summarizer_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Summarizer agent. Your job is to summarize relevant scored articles.\n"
            "Rules:\n"
            "- Call run_summarizer to generate and store summaries.\n"
            "- If asked to create a newspaper, do not do it here (publisher handles that).\n"
            f"- Your final response MUST be a valid JSON object matching this structure: {json.dumps(SummarizerResponse.model_json_schema())}\n"
            "- Do not wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
