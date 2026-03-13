from __future__ import annotations

import os

from app.core.settings import settings


def mcp_config() -> dict:
    cfg: dict = {}

    # Remote MCP (preferred for low-memory deployments): configure by environment variables.
    # Examples:
    #   MCP_TAVILY_URL=https://your-mcp-host/tavily/mcp
    #   MCP_GITHUB_URL=https://your-mcp-host/github/mcp
    #   MCP_TWITTER_URL=https://your-mcp-host/twitter/mcp
    # Optional:
    #   MCP_TRANSPORT=http|sse  (default: http)
    #   MCP_AUTH_HEADER=Authorization
    #   MCP_AUTH_TOKEN=Bearer ...
    transport = str(os.getenv("MCP_TRANSPORT", "http") or "http").strip().lower()
    auth_header = str(os.getenv("MCP_AUTH_HEADER", "Authorization") or "Authorization").strip()
    auth_token = str(os.getenv("MCP_AUTH_TOKEN", "") or "").strip()
    common_headers = {auth_header: auth_token} if auth_token else {}

    tavily_url = str(os.getenv("MCP_TAVILY_URL", "") or "").strip()
    if tavily_url:
        cfg["tavily"] = {
            "transport": transport,
            "url": tavily_url,
            "headers": common_headers,
        }

    github_url = str(os.getenv("MCP_GITHUB_URL", "") or "").strip()
    if github_url:
        cfg["github"] = {
            "transport": transport,
            "url": github_url,
            "headers": common_headers,
        }

    twitter_url = str(os.getenv("MCP_TWITTER_URL", "") or "").strip()
    if twitter_url:
        cfg["twitter"] = {
            "transport": transport,
            "url": twitter_url,
            "headers": common_headers,
        }

    # If remote MCP is configured, do not start local stdio servers.
    if cfg:
        return cfg

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
