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

    score_urgent: int = Field(default=8, validation_alias="SCORE_URGENT")
    score_minimum: int = Field(default=4, validation_alias="SCORE_MINIMUM")

    key_rotation_cooldown_seconds: int = Field(default=60, validation_alias="KEY_ROTATION_COOLDOWN_SECONDS")

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

        log.info(
            "Settings OK | env=%s groq_keys=%d or_keys=%d tavily_keys=%d",
            self.env,
            len(self.groq_keys),
            len(self.openrouter_keys),
            len(self.tavily_keys),
        )


settings = Settings()
