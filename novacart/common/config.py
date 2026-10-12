"""Central configuration for the NovaCart MLOps platform.

Uses pydantic-settings so every value is overridable via environment variables
(the 12-factor way). In Docker each service gets its own env; locally the
defaults point at the repo's ./data directory.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (novacart/common/config.py -> repo)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical product-category vocabulary — single source of truth shared by the
# data engine (generation) and the RAG intent parser (NL detection).
CATEGORIES = [
    "electronics", "home", "beauty", "grocery", "fashion", "toys", "sports", "books",
]


class Settings(BaseSettings):
    """Runtime settings shared across data engine, training and serving."""

    model_config = SettingsConfigDict(env_prefix="NOVACART_", env_file=".env", extra="ignore")

    # --- storage locations -------------------------------------------------
    data_dir: Path = Field(default=REPO_ROOT / "data")
    # SQLite locally; set NOVACART_DB_URL=postgresql://... in compose.
    db_url: str = Field(default=f"sqlite:///{REPO_ROOT / 'data' / 'raw' / 'novacart.db'}")

    # --- experiment / label config ----------------------------------------
    seed: int = 42
    n_customers: int = 4000
    # The "today" the whole platform reasons relative to. Point-in-time features
    # are computed as-of this date; the churn label looks 30 days forward from it.
    reference_date: str = "2026-06-01"
    churn_horizon_days: int = 30
    # A customer counts as "active before" if they purchased within this lookback.
    active_lookback_days: int = 90
    # Feature-window lengths (days) used by the shared feature computation.
    freq_window_days: int = 90
    sessions_window_days: int = 30

    # --- policy knobs a business owner would tune ---------------------------
    risk_band_high: float = 0.66   # churn prob >= high -> "high" band
    risk_band_medium: float = 0.33
    clv_margin: float = 0.30       # gross margin assumption for CLV
    clv_lifespan_years: float = 3.0

    # --- model registry ----------------------------------------------------
    registry_dir: Path = Field(default=REPO_ROOT / "data" / "models" / "registry")
    model_name: str = "novacart_churn"
    experiment_name: str = "churn_v1"

    # --- serving -----------------------------------------------------------
    feature_service_url: str = "http://127.0.0.1:8101"
    feature_service_host: str = "127.0.0.1"
    feature_service_port: int = 8101
    prediction_service_host: str = "127.0.0.1"
    prediction_service_port: int = 8102
    http_timeout_seconds: float = 5.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def sqlite_path(self) -> Path:
        """Filesystem path behind db_url when using sqlite."""
        assert self.db_url.startswith("sqlite:///"), "sqlite_path only valid for sqlite db_url"
        return Path(self.db_url.replace("sqlite:///", "", 1))

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.features_dir, self.registry_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
