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
            "- Call fetch_summaries_for_newspaper FIRST to get articles.\n"
            "- Then call generate_newspaper_pdf to create the PDF.\n"
            "- Finally call mark_newspaper_sent to finish.\n"
            "- Your final response MUST be a valid JSON object matching this EXACT structure:\n"
            '{"success": true, "pdf_path": "<string>", "articles_count": <number>, "message": "<string>"}\n'
            "Example valid response:\n"
            '{"success": true, "pdf_path": "/tmp/newspaper.pdf", "articles_count": 10, "message": "Generated newspaper with 10 articles."}\n'
            "- Do NOT wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
