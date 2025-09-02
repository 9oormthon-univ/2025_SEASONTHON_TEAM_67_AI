
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    MODEL_NAME: str = "gpt-5-mini"
    MAX_BODY_CHARS: int = 200_000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
