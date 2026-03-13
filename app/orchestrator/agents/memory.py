from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import MemoryResponse

def build_memory_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Memory agent. Your job is to keep the user's profile and long-term facts up to date.\n"
            "Rules:\n"
            "- Call update_user_preferences for preference changes; this will also refresh the semantic preference profile.\n"
            "- Call store_user_fact for durable facts; this will write to episodic memory AND semantic memory (Pinecone).\n"
            "- If the user says what they are interested in (e.g. 'my main interest is AI/tech/innovation', 'I like X', 'I prefer Y') you MUST call update_user_preferences.\n"
            "- If the user states their role/job (e.g. 'I am an AI engineer') you SHOULD store it as a fact AND also update preferences if interests are mentioned.\n"
            "- Your final response MUST be a valid JSON OBJECT (an instance), matching this EXACT structure:\n"
            '{"updated": <true|false>, "summary": "<string>"}\n'
            "Example valid response:\n"
            '{"updated": true, "summary": "Stored user fact: user_interests=quantum computing"}\n'
            "- Do NOT output JSON schema (do NOT include keys like 'properties', '$defs', 'title').\n"
            "- Do not wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
