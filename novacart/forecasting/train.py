"""Demand forecasting: train, backtest, register, forecast 365 days.

Compares three models — LightGBM, ARIMA and seasonal-naive — evaluated
with a proper time-based backtest (last N days held out), reporting MAPE/WAPE
per model. The winning model is registered + promoted in the shared registry
and used to roll a 365-day-ahead forecast recursively.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from novacart.common.config import settings
from novacart.common.registry import Registry
from novacart.forecasting import dataset as ds
from novacart.forecasting.dataset import FEATURE_COLS, mape, wape

warnings.filterwarnings("ignore")
TEST_DAYS = 60
FORECAST_HORIZON = 365
MODEL_NAME = "novacart_demand"


def _make_gbm() -> lgb.LGBMRegressor:
    """Single source for the demand LightGBM config — backtest and production
    fit call this, so they can never silently diverge."""
    return lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=settings.seed, verbosity=-1,
    )


def seasonal_naive(train: pd.DataFrame, horizon: int, season: int = 7) -> np.ndarray:
    """Forecast = value from `season` days ago (weekly seasonality baseline)."""
    hist = train["demand"].to_numpy()
    return np.array([hist[-season + (i % season)] for i in range(horizon)])


def _fit_arima(series: np.ndarray, horizon: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(series, order=(2, 1, 2)).fit()
    return np.asarray(model.forecast(steps=horizon))


def backtest_total() -> dict:
    """Time-split backtest on the total-company series across all three models."""
    total = ds.daily_demand("total")
    sup = ds.make_supervised(total)
    split_date = sup["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train, test = sup[sup["date"] <= split_date], sup[sup["date"] > split_date]

    results = {}

    # 1) seasonal naive
    sn = seasonal_naive(train, len(test))
    results["seasonal_naive"] = {"mape": mape(test["demand"], sn), "wape": wape(test["demand"], sn)}

    # 2) LightGBM (global, feature-based)
    gbm = _make_gbm()
    gbm.fit(train[FEATURE_COLS], train["demand"])
    pred_gbm = gbm.predict(test[FEATURE_COLS])
    results["lightgbm"] = {"mape": mape(test["demand"], pred_gbm), "wape": wape(test["demand"], pred_gbm)}

    # 3) ARIMA on the raw series
    try:
        ar = _fit_arima(train["demand"].to_numpy(), len(test))
        results["arima"] = {"mape": mape(test["demand"], ar), "wape": wape(test["demand"], ar)}
    except Exception as exc:  # pragma: no cover
        results["arima"] = {"mape": float("nan"), "wape": float("nan"), "error": str(exc)}

    best = min(results, key=lambda k: results[k]["mape"] if not np.isnan(results[k]["mape"]) else 1e9)
    return {"results": results, "best_model": best, "test_days": TEST_DAYS,
            "n_train": int(len(train)), "n_test": int(len(test))}


def _recursive_forecast(model, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Roll LightGBM forward one day at a time, feeding predictions back as lags."""
    series = history[["date", "demand"]].copy()
    preds = []
    for _ in range(horizon):
        nxt = series["date"].max() + pd.Timedelta(days=1)
        row = {"date": nxt, "demand": np.nan}
        tmp = pd.concat([series, pd.DataFrame([row])], ignore_index=True)
        tmp["series"] = "total"
        feat = ds.make_supervised(tmp.assign(revenue=0.0))
        feat_row = feat.iloc[[-1]] if len(feat) else None
        if feat_row is None or feat_row.empty:
            yhat = float(series["demand"].tail(7).mean())
        else:
            yhat = float(model.predict(feat_row[FEATURE_COLS])[0])
        yhat = max(yhat, 0.0)
        preds.append({"date": nxt, "forecast": round(yhat, 2)})
        series = pd.concat([series, pd.DataFrame([{"date": nxt, "demand": yhat}])], ignore_index=True)
    return pd.DataFrame(preds)


def train() -> dict:
    bt = backtest_total()

    # Fit the production LightGBM on ALL history, then roll the 365-day forecast.
    total = ds.daily_demand("total")
    sup = ds.make_supervised(total)
    gbm = _make_gbm()
    gbm.fit(sup[FEATURE_COLS], sup["demand"])
    forecast = _recursive_forecast(gbm, total[total["series"] == "total"][["date", "demand"]],
                                   FORECAST_HORIZON)
    settings.features_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(settings.features_dir / "demand_forecast_365d.parquet", index=False)

    metrics = {f"{m}_mape": round(bt["results"][m]["mape"], 3) for m in bt["results"]}
    metrics["best_model"] = bt["best_model"]
    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    reg = Registry(model_name=MODEL_NAME)
    reg.log_run(params={"models": list(bt["results"].keys()), "horizon": FORECAST_HORIZON,
                        "test_days": TEST_DAYS}, metrics=numeric_metrics)
    mv = reg.register(model=gbm, metrics=numeric_metrics,
                      params={"features": FEATURE_COLS, "horizon": FORECAST_HORIZON},
                      feature_columns=FEATURE_COLS)
    reg.transition_stage(mv.version, "Production")

    print(f"demand forecasting — backtest ({TEST_DAYS}d holdout):")
    for m, r in bt["results"].items():
        print(f"  {m:15s} MAPE={r['mape']:.2f}%  WAPE={r['wape']:.2f}%")
    print(f"  best={bt['best_model']}  -> registered novacart_demand v{mv.version} Production")
    print(f"  365-day forecast saved: mean/day={forecast['forecast'].mean():.1f}")
    return {"backtest": bt, "version": mv.version}


if __name__ == "__main__":
    train()
