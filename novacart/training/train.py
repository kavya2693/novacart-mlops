"""Training + registration (Project 2).

Trains a churn classifier, evaluates it with ROC-AUC and PR-AUC, logs the run,
registers the model, and promotes it to Production in the registry.

Model choice: a StandardScaler -> LogisticRegression Pipeline. Logistic
regression is deliberate — it gives exact, signed per-feature contributions in
log-odds (coef * standardized_value) with no SHAP dependency, so the prediction
API can explain every score. We also report a HistGradientBoosting benchmark so
the choice is defensible ("LR is within X of GBM and fully interpretable").
"""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from novacart.common.config import settings
from novacart.common.registry import Registry
from novacart.common.schemas import FEATURE_COLUMNS
from novacart.feature_pipeline.build_features import load_feature_table


def _split(df):
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["churn"].astype(int)
    # One row per customer at a single snapshot, so a stratified split cannot
    # leak a customer across folds. (Production uses rolling as-of snapshots;
    # see MODEL_CARD.md.) Stratify to preserve the churn base rate.
    return train_test_split(X, y, test_size=0.25, random_state=settings.seed, stratify=y)


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
            ),
        ]
    )


def train() -> dict:
    df = load_feature_table()
    X_train, X_test, y_train, y_test = _split(df)

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    p_test = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "base_rate": float(y_test.mean()),
    }

    # GBM benchmark (reported, not shipped) to justify the interpretable choice.
    gbm = HistGradientBoostingClassifier(random_state=settings.seed)
    gbm.fit(X_train, y_train)
    metrics["gbm_roc_auc"] = float(roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1]))

    params = {
        "model": "LogisticRegression",
        "C": 1.0,
        "class_weight": "balanced",
        "features": FEATURE_COLUMNS,
        "reference_date": settings.reference_date,
        "churn_horizon_days": settings.churn_horizon_days,
    }

    reg = Registry()
    reg.log_run(params=params, metrics=metrics)
    mv = reg.register(
        model=pipe,
        metrics=metrics,
        params=params,
        feature_columns=FEATURE_COLUMNS,
        sklearn_version=sklearn.__version__,
    )
    reg.transition_stage(mv.version, "Production")

    print(f"trained novacart_churn v{mv.version} -> Production")
    print(
        f"  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
        f"(GBM benchmark ROC-AUC={metrics['gbm_roc_auc']:.3f}, base_rate={metrics['base_rate']:.3f})"
    )
    return metrics


if __name__ == "__main__":
    train()
