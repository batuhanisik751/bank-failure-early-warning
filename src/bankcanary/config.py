"""Typed settings loaded from config/settings.yaml plus environment variables (.env)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class FdicSettings(BaseModel):
    base_url: str = "https://api.fdic.gov/banks/"
    docs_url: str = "https://api.fdic.gov/banks/docs/"
    page_size: int = 10_000
    requests_per_second: float = 4.0
    timeout_seconds: float = 120.0
    max_retries: int = 6


class FixedSplit(BaseModel):
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date


class Settings(BaseModel):
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    start_quarter: dt.date = dt.date(2001, 3, 31)
    end_quarter: dt.date | None = None
    availability_lag_days: int = 60
    horizons_quarters: list[int] = Field(default_factory=lambda: [4, 8])
    fixed_split: FixedSplit
    peer_asset_buckets_thousands: list[int] = Field(
        default_factory=lambda: [100_000, 1_000_000, 10_000_000, 100_000_000]
    )
    fdic: FdicSettings = Field(default_factory=FdicSettings)

    def resolve(self, root: Path = PROJECT_ROOT) -> Settings:
        """Return a copy whose relative directories are anchored at ``root``."""
        return self.model_copy(
            update={
                "data_dir": _anchor(self.data_dir, root),
                "models_dir": _anchor(self.models_dir, root),
                "reports_dir": _anchor(self.reports_dir, root),
            }
        )


class Secrets(BaseSettings):
    """Values that come from the environment or a git-ignored .env file, never from YAML."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    fdic_api_key: str | None = None
    fred_api_key: str | None = None
    database_url: str | None = None


def _anchor(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_settings(path: Path | None = None, root: Path = PROJECT_ROOT) -> Settings:
    """Load and validate settings.yaml, anchoring relative paths at the project root."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    with open(settings_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Settings.model_validate(raw).resolve(root)


def load_secrets() -> Secrets:
    return Secrets()
