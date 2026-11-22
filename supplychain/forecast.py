"""Forecasting trade value per route.

We only have 5 yearly points per route (2019-2023), so a big model would
just overfit. Instead I compare a few small models with a rolling-origin
backtest (train on years up to T, predict T+1) and use the one with the
lowest error across all routes. The backtest error is also used to build
the forecast range, so the app is honest about how unsure it is.

Models:
  naive        - next year = this year
  cagr         - continue the average growth rate
  log_trend    - linear regression on log(value) vs year (exponential trend)
  blend        - simple average of naive, log_trend and pooled_ar
  pooled_ar    - Ridge regression trained on ALL routes together: predicts
                 next year's growth from this year's growth (an AR(1) on
                 growth). Pooling gives it more training rows than any
                 single route has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

METHODS = ["naive", "cagr", "log_trend", "pooled_ar", "blend"]
METHOD_LABELS = {
    "naive": "Naive (last value)",
    "cagr": "Average growth (CAGR)",
    "log_trend": "Log-linear trend",
    "pooled_ar": "Pooled Ridge AR(1) on growth",
    "blend": "Blend (mean of naive + log trend + pooled AR)",
}


def _series(routes: pd.DataFrame, importer: str, supplier: str) -> pd.Series:
    s = routes[(routes["importer"] == importer) & (routes["supplier"] == supplier)]
    return s.set_index("year")["value"].sort_index().astype(float)


def _growth_rows(routes: pd.DataFrame, max_target_year: int) -> pd.DataFrame:
    """(previous log-growth -> log-growth) pairs from every route."""
    rows = []
    for _, grp in routes.groupby(["importer", "supplier"]):
        g = np.log(grp.set_index("year")["value"].sort_index()).diff().dropna()
        for (y1, g1), (y2, g2) in zip(g.items(), list(g.items())[1:]):
            if y2 <= max_target_year:
                rows.append({"x": g1, "y": g2})
    return pd.DataFrame(rows)


def fit_pooled_ar(routes: pd.DataFrame, max_target_year: int) -> Ridge | None:
    rows = _growth_rows(routes, max_target_year)
    if len(rows) < 5:
        return None
    return Ridge(alpha=1.0).fit(rows[["x"]].values, rows["y"].values)


def predict_series(history: pd.Series, method: str, horizon: int = 1, ar_model: Ridge | None = None) -> list[float]:
    """Predict the next `horizon` years from a yearly history."""
    vals = history.values
    last = vals[-1]

    if method == "naive" or len(vals) < 2:
        return [last] * horizon

    if method == "cagr":
        years = history.index[-1] - history.index[0]
        g = (vals[-1] / vals[0]) ** (1 / years) - 1
        return [last * (1 + g) ** h for h in range(1, horizon + 1)]

    if method == "log_trend":
        X = history.index.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, np.log(vals))
        future = np.arange(history.index[-1] + 1, history.index[-1] + horizon + 1).reshape(-1, 1)
        return list(np.exp(reg.predict(future)))

    if method == "pooled_ar":
        if ar_model is None:
            return [last] * horizon
        g_prev = np.log(vals[-1] / vals[-2])
        out, level = [], last
        for _ in range(horizon):
            g_next = float(ar_model.predict([[g_prev]])[0])
            level = level * np.exp(g_next)
            out.append(level)
            g_prev = g_next
        return out

    if method == "blend":
        parts = [predict_series(history, m, horizon, ar_model) for m in ("naive", "log_trend", "pooled_ar")]
        return list(np.mean(parts, axis=0))

    raise ValueError(f"unknown method: {method}")


def backtest(routes: pd.DataFrame, origins: list[int] | None = None) -> pd.DataFrame:
    """One-step-ahead rolling backtest for every route and method."""
    years = sorted(routes["year"].unique())
    origins = origins or [y for y in years[2:-1]]  # need 3 years of history
    rows = []
    for origin in origins:
        train = routes[routes["year"] <= origin]
        ar = fit_pooled_ar(train, max_target_year=origin)
        for (imp, sup), _ in routes.groupby(["importer", "supplier"]):
            hist = _series(train, imp, sup)
            actual = _series(routes, imp, sup).get(origin + 1)
            if actual is None or len(hist) < 2:
                continue
            for m in METHODS:
                pred = predict_series(hist, m, 1, ar)[0]
                rows.append({
                    "route": f"{sup} → {imp}",
                    "origin": int(origin),
                    "target_year": int(origin + 1),
                    "method": m,
                    "actual": actual,
                    "predicted": pred,
                    "ape_pct": abs(pred - actual) / actual * 100,
                })
    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> pd.DataFrame:
    summary = bt.groupby("method")["ape_pct"].agg(MAPE="mean", median_APE="median", worst="max", n="count")
    summary = summary.sort_values("MAPE").round(2)
    summary.index = [METHOD_LABELS[m] + f" ({m})" for m in summary.index]
    return summary


@dataclass
class Forecast:
    route: str
    method: str
    history: pd.Series
    table: pd.DataFrame       # year, forecast, low, high
    backtest_mape: float
    interval_pct: float       # +/- used for year 1


class Forecaster:
    def __init__(self, routes: pd.DataFrame, coverage: float = 0.8):
        self.routes = routes
        self.coverage = coverage
        self.bt = backtest(routes)
        mape = self.bt.groupby("method")["ape_pct"].mean()
        self.best_method = str(mape.idxmin())
        self.mape = mape.to_dict()
        # error quantile per method -> used as +/- range
        self.err_q = self.bt.groupby("method")["ape_pct"].quantile(coverage).to_dict()
        self.ar_model = fit_pooled_ar(routes, max_target_year=int(routes["year"].max()))

    def summary(self) -> pd.DataFrame:
        return backtest_summary(self.bt)

    def forecast(self, importer: str, supplier: str, horizon: int = 2, method: str | None = None) -> Forecast:
        method = method or self.best_method
        hist = _series(self.routes, importer, supplier)
        if hist.empty:
            raise ValueError(f"no data for {supplier} → {importer}")
        preds = predict_series(hist, method, horizon, self.ar_model)
        band = self.err_q[method] / 100
        last_year = int(hist.index[-1])
        rows = []
        for h, p in enumerate(preds, start=1):
            b = band * np.sqrt(h)  # uncertainty grows with the horizon
            rows.append({"year": last_year + h, "forecast": p, "low": p * max(0.0, 1 - b), "high": p * (1 + b)})
        return Forecast(
            route=f"{supplier} → {importer}",
            method=method,
            history=hist,
            table=pd.DataFrame(rows),
            backtest_mape=round(self.mape[method], 2),
            interval_pct=round(band * 100, 1),
        )

    def forecast_all(self, horizon: int = 2, method: str | None = None) -> pd.DataFrame:
        out = []
        for (imp, sup), _ in self.routes.groupby(["importer", "supplier"]):
            fc = self.forecast(imp, sup, horizon, method)
            t = fc.table.copy()
            t.insert(0, "route", fc.route)
            t.insert(1, "importer", imp)
            t.insert(2, "supplier", sup)
            out.append(t)
        return pd.concat(out, ignore_index=True)
