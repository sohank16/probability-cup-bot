from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    sportspredict_api_key: str = Field(default="", alias="SPORTSPREDICT_API_KEY")
    sportspredict_base_url: str = Field(
        default="https://api.sportspredict.com/api/v1",
        alias="SPORTSPREDICT_BASE_URL",
    )
    database_path: Path = Field(
        default=Path("data/probability_cup.sqlite"),
        alias="DATABASE_PATH",
    )
    raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
    dry_run: bool = Field(default=True, alias="DRY_RUN")

    @property
    def has_api_key(self) -> bool:
        return bool(self.sportspredict_api_key.strip())


def load_settings() -> Settings:
    """Load settings after reading local .env values."""

    load_dotenv()
    return Settings()
