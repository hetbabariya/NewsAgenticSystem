from __future__ import annotations
import json
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.orchestrator.models import CollectorResponse
from app.core.settings import settings

def build_collector_agent(*, model, tools: list):
    prompt = SystemMessage(
        content=(
            "You are the Collector agent. Your job is to collect fresh news articles.\n"
            "Rules:\n"
            "- For Reddit: use the local `fetch_reddit_posts` tool with subreddit names like 'MachineLearning', 'Programming'.\n"
            f"- For Tavily web search: use MCP tool `tavily-search` with a SHORT query (max {settings.max_query_chars} chars) and request at most {settings.tavily_max_results} results if the tool supports it (e.g., `max_results`).\n"
            f"- For GitHub: use MCP tool `search_repositories`/`get_trending_repos` and request at most {settings.github_max_results} results if the tool supports it (e.g., `per_page`, `limit`).\n"
            f"- For Twitter: use MCP tool `twitter_search` (or similar) with a SHORT query (max {settings.max_query_chars} chars) and request at most {settings.twitter_max_results} results if the tool supports it (e.g., `max_results`, `limit`).\n"
            "- After fetching from sources, call `run_news_collector(...)` EXACTLY ONCE to store articles.\n"
            "  - If you fetched articles via MCP tools, pass them using `run_news_collector(articles=[...], sources=[...])`.\n"
            "  - Each item in `articles` MUST be an object with: url (required), title, content, source.\n"
            "  - If you also used Reddit, still call `run_news_collector` only once total (include sources like ['reddit','tavily','twitter'] as needed).\n"
            "- Do NOT loop or retry tool calls indefinitely. If a tool errors or returns empty, move on and continue with what you have.\n"
            "- Your final response MUST be a valid JSON object matching this EXACT structure:\n"
            '{"success": true, "collected_count": <number>, "sources_used": ["<string>"], "message": "<string>"}\n'
            "Example valid response:\n"
            '{"success": true, "collected_count": 15, "sources_used": ["reddit", "tavily"], "message": "Collected 15 articles from 2 sources."}\n'
            "- Do NOT call run_news_collector multiple times.\n"
            "- Do NOT wrap the JSON in markdown blocks or include any other text.\n"
        )
    )
    return create_react_agent(model=model, tools=tools, prompt=prompt)
