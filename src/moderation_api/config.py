from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONTENT_MODERATION_", env_file=".env")

    model_dir: str = Field(default="JungleLee/bert-toxic-comment-classification")
    backend: str = Field(default="torch", pattern="^(torch|onnx)$")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_length: int = Field(default=192, ge=16, le=512)
    onnx_provider: str = "CPUExecutionProvider"


@lru_cache
def get_settings() -> Settings:
    return Settings()
