import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    database_url: str

    env: str = Field(default="development", validation_alias="ENV")
    render_external_url: str | None = Field(default=None, validation_alias="RENDER_EXTERNAL_URL")

    telegram_bot_token: str = Field(validation_alias="TELEGRAM_BOT_TOKEN")
    telegram_webhook_secret_token: str | None = Field(default=None, validation_alias="TELEGRAM_WEBHOOK_SECRET_TOKEN")
    telegram_admin_chat_id: str | None = Field(default=None, validation_alias="TELEGRAM_CHAT_ID")

    pinecone_api_key: str | None = Field(default=None, validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="newsagent", validation_alias="PINECONE_INDEX_NAME")

    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    groq_api_keys: str | None = Field(default=None, validation_alias="GROQ_API_KEYS")
    openrouter_api_keys: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEYS")
    tavily_api_keys: str | None = Field(default=None, validation_alias="TAVILY_API_KEYS")

    newsapi_key: str | None = Field(default=None, validation_alias="NEWSAPI_KEY")
    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")

    twitter_api_key: str | None = Field(default=None, validation_alias="TWITTER_API_KEY")
    twitter_api_secret: str | None = Field(default=None, validation_alias="TWITTER_API_SECRET")
    twitter_access_token: str | None = Field(default=None, validation_alias="TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret: str | None = Field(default=None, validation_alias="TWITTER_ACCESS_TOKEN_SECRET")

    reddit_client_id: str | None = Field(default=None, validation_alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str | None = Field(default=None, validation_alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="newsagent:v1.0", validation_alias="REDDIT_USER_AGENT")

    supabase_url: str | None = Field(default=None, validation_alias="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(default=None, validation_alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = Field(default="newspapers", validation_alias="SUPABASE_STORAGE_BUCKET")
    supabase_storage_public: bool = Field(default=False, validation_alias="SUPABASE_STORAGE_PUBLIC")

    score_urgent: float = Field(default=0.7, validation_alias="SCORE_URGENT")
    score_minimum: float = Field(default=0.4, validation_alias="SCORE_MINIMUM")

    key_rotation_cooldown_seconds: int = Field(default=60, validation_alias="KEY_ROTATION_COOLDOWN_SECONDS")

    # --- External tool/source limits (robustness) ---
    max_query_chars: int = Field(default=240, validation_alias="MAX_QUERY_CHARS")
    tavily_max_results: int = Field(default=5, validation_alias="TAVILY_MAX_RESULTS")
    github_max_results: int = Field(default=5, validation_alias="GITHUB_MAX_RESULTS")
    twitter_max_results: int = Field(default=10, validation_alias="TWITTER_MAX_RESULTS")

    mcp_ingest_enabled: bool = Field(default=True, validation_alias="MCP_INGEST_ENABLED")
    mcp_max_calls_per_run: int = Field(default=6, validation_alias="MCP_MAX_CALLS_PER_RUN")

    reddit_max_posts: int = Field(default=5, validation_alias="REDDIT_MAX_POSTS")
    reddit_timeout_seconds: float = Field(default=20.0, validation_alias="REDDIT_TIMEOUT_SECONDS")
    reddit_max_retries: int = Field(default=2, validation_alias="REDDIT_MAX_RETRIES")
    reddit_retry_base_delay_seconds: float = Field(default=1.5, validation_alias="REDDIT_RETRY_BASE_DELAY_SECONDS")

    # Collector safety caps (prevents loops / huge payloads)
    collector_max_articles_total: int = Field(default=50, validation_alias="COLLECTOR_MAX_ARTICLES_TOTAL")
    collector_max_mcp_articles: int = Field(default=30, validation_alias="COLLECTOR_MAX_MCP_ARTICLES")
    collector_max_topics: int = Field(default=6, validation_alias="COLLECTOR_MAX_TOPICS")
    collector_max_sources: int = Field(default=6, validation_alias="COLLECTOR_MAX_SOURCES")

    # Stored field sizes (truncate to avoid DB bloat)
    max_article_url_chars: int = Field(default=2048, validation_alias="MAX_ARTICLE_URL_CHARS")
    max_article_title_chars: int = Field(default=500, validation_alias="MAX_ARTICLE_TITLE_CHARS")
    max_article_content_chars: int = Field(default=6000, validation_alias="MAX_ARTICLE_CONTENT_CHARS")
    max_article_source_chars: int = Field(default=120, validation_alias="MAX_ARTICLE_SOURCE_CHARS")

    # Model Configuration
    model_coordinator: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct", validation_alias="MODEL_COORDINATOR")
    model_collector: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct", validation_alias="MODEL_COLLECTOR")
    model_filter: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct", validation_alias="MODEL_FILTER")
    model_memory: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct", validation_alias="MODEL_MEMORY")

    # OpenRouter long-context models
    model_summarizer: str = Field(default="arcee-ai/trinity-large-preview:free", validation_alias="MODEL_SUMMARIZER")
    model_support: str = Field(default="arcee-ai/trinity-large-preview:free", validation_alias="MODEL_SUPPORT")
    model_publisher: str = Field(default="arcee-ai/trinity-large-preview:free", validation_alias="MODEL_PUBLISHER")


    def _split_csv(self, value: str | None) -> list[str]:
        if not value:
            return []
        return [v.strip() for v in value.split(",") if v.strip()]


    @property
    def groq_keys(self) -> list[str]:
        return self._split_csv(self.groq_api_keys)


    @property
    def openrouter_keys(self) -> list[str]:
        return self._split_csv(self.openrouter_api_keys)


    @property
    def tavily_keys(self) -> list[str]:
        return self._split_csv(self.tavily_api_keys)


    def validate(self) -> None:
        log = logging.getLogger("settings")

        if not self.groq_keys and not self.openrouter_keys:
            raise RuntimeError("No LLM API keys found. Set at least GROQ_API_KEYS or OPENROUTER_API_KEYS.")

        if self.env == "production":
            if not self.render_external_url:
                log.warning("RENDER_EXTERNAL_URL missing — Telegram webhook registration will be skipped.")
            if not self.telegram_admin_chat_id:
                log.warning("TELEGRAM_CHAT_ID missing — scheduler alerts will be disabled.")

        if not self.tavily_keys:
            log.warning("No Tavily keys found — ingest will have no web search source.")

        if not self.github_token:
            log.warning("GITHUB_TOKEN missing — GitHub daily fetch will be skipped.")

        if not self.twitter_api_key:
            log.warning("Twitter keys missing — Twitter source will be skipped.")

        if not self.reddit_client_id:
            log.warning("Reddit keys missing — Reddit source will be skipped.")

        if self.supabase_url and not self.supabase_service_role_key:
            log.warning("SUPABASE_URL set but SUPABASE_SERVICE_ROLE_KEY missing — PDF upload will be disabled.")

        log.info(
            "Settings OK | env=%s groq_keys=%d or_keys=%d tavily_keys=%d",
            self.env,
            len(self.groq_keys),
            len(self.openrouter_keys),
            len(self.tavily_keys),
        )


settings = Settings()
