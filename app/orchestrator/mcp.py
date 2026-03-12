from __future__ import annotations

from app.core.settings import settings


def mcp_config() -> dict:
    cfg: dict = {}

    if settings.tavily_keys:
        cfg["tavily"] = {
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.1.4"],
            "env": {"TAVILY_API_KEY": settings.tavily_keys[0]},
            "transport": "stdio",
        }

    if settings.github_token:
        cfg["github"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token},
            "transport": "stdio",
        }

    if settings.twitter_api_key:
        cfg["twitter"] = {
            "command": "npx",
            "args": ["-y", "@enescinar/twitter-mcp"],
            "env": {
                "API_KEY": settings.twitter_api_key,
                "API_SECRET_KEY": settings.twitter_api_secret,
                "ACCESS_TOKEN": settings.twitter_access_token,
                "ACCESS_TOKEN_SECRET": settings.twitter_access_token_secret,
            },
            "transport": "stdio",
        }

    return cfg
