"""A minimal, MLflow-shaped model registry backed by the filesystem.

Why not MLflow directly? Python 3.14 currently lacks reliable mlflow wheels,
and a portfolio must actually *run*. So this implements the same mental model
MLflow uses — experiments, runs (params + metrics), model versions, and stage
promotion (None -> Staging -> Production) — in ~120 lines you can read
end to end. docker-compose ships a real MLflow server for the production story;
the `Registry` API is intentionally swappable for `mlflow.*`.

On-disk layout:
    registry/
      <model_name>/
        runs.jsonl                 # append-only log of every training run
        versions/
          1/model.joblib
          1/meta.json              # {version, stage, metrics, params, created}
          2/...
        production -> pointer in each meta.json (stage field)
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

from novacart.common.config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelVersion:
    version: int
    stage: str  # "None" | "Staging" | "Production" | "Archived"
    metrics: dict[str, float]
    params: dict[str, Any]
    feature_columns: list[str]
    created_at: str
    python_version: str = field(default_factory=platform.python_version)
    sklearn_version: str = ""


class Registry:
    """Filesystem-backed model registry with MLflow-like semantics."""

    def __init__(self, model_name: str | None = None, root: Path | None = None):
        self.model_name = model_name or settings.model_name
        self.root = (root or settings.registry_dir) / self.model_name
        self.versions_dir = self.root / "versions"
        self.runs_log = self.root / "runs.jsonl"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    # -- runs ---------------------------------------------------------------
    def log_run(self, params: dict[str, Any], metrics: dict[str, float]) -> None:
        """Append an experiment run (analogous to mlflow.log_params/metrics)."""
        entry = {
            "timestamp": _now(),
            "experiment": settings.experiment_name,
            "params": params,
            "metrics": metrics,
        }
        with self.runs_log.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    # -- model versions -----------------------------------------------------
    def _next_version(self) -> int:
        existing = [int(p.name) for p in self.versions_dir.iterdir() if p.name.isdigit()]
        return (max(existing) + 1) if existing else 1

    def register(
        self,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any],
        feature_columns: list[str],
        sklearn_version: str = "",
    ) -> ModelVersion:
        """Persist a trained model as a new version (stage=None)."""
        version = self._next_version()
        vdir = self.versions_dir / str(version)
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, vdir / "model.joblib")
        mv = ModelVersion(
            version=version,
            stage="None",
            metrics=metrics,
            params=params,
            feature_columns=feature_columns,
            created_at=_now(),
            sklearn_version=sklearn_version,
        )
        (vdir / "meta.json").write_text(json.dumps(asdict(mv), indent=2))
        return mv

    def transition_stage(self, version: int, stage: str) -> ModelVersion:
        """Promote/demote a version. Only one Production version at a time."""
        assert stage in {"None", "Staging", "Production", "Archived"}
        if stage == "Production":
            # Archive any currently-Production version first.
            for mv in self.list_versions():
                if mv.stage == "Production" and mv.version != version:
                    self._write_meta(mv.version, {"stage": "Archived"})
        return self._write_meta(version, {"stage": stage})

    def _write_meta(self, version: int, updates: dict[str, Any]) -> ModelVersion:
        path = self.versions_dir / str(version) / "meta.json"
        meta = json.loads(path.read_text())
        meta.update(updates)
        path.write_text(json.dumps(meta, indent=2))
        return ModelVersion(**meta)

    def list_versions(self) -> list[ModelVersion]:
        out = []
        for p in sorted(self.versions_dir.iterdir(), key=lambda x: x.name):
            if p.name.isdigit():
                out.append(ModelVersion(**json.loads((p / "meta.json").read_text())))
        return out

    def get_production(self) -> Optional[ModelVersion]:
        for mv in self.list_versions():
            if mv.stage == "Production":
                return mv
        return None

    def load_production_model(self) -> tuple[Any, ModelVersion]:
        """Load the current Production model + its metadata (used by serving)."""
        mv = self.get_production()
        if mv is None:
            raise RuntimeError(
                f"No Production model for '{self.model_name}'. Run training first."
            )
        model = joblib.load(self.versions_dir / str(mv.version) / "model.joblib")
        return model, mv
