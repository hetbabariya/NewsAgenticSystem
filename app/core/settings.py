from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    telegram_bot_token: str = Field(validation_alias="TELEGRAM_BOT_TOKEN")
    telegram_webhook_secret_token: str | None = Field(default=None, validation_alias="TELEGRAM_WEBHOOK_SECRET_TOKEN")


settings = Settings()
