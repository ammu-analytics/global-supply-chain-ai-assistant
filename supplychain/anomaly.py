"""Finding unusual route-years with an Isolation Forest.

Each route-year (2020 onwards) becomes one sample with three features:

  growth        - log change vs. the previous year
  trend_gap     - how far the value is from the route's own trend line,
                  fitted WITHOUT that year (leave-one-out), in log terms
  vs_market     - this route's growth minus the average growth of all
                  routes that year (separates "everyone fell in 2020"
                  from "only this route fell")

Isolation Forest is unsupervised: there are no labels for "anomaly" in
this data, so it scores points by how easy they are to isolate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

FEATURES = ["growth", "trend_gap", "vs_market"]
FEATURE_TEXT = {
    "growth": "unusual change from the previous year",
    "trend_gap": "far off the route's own trend",
    "vs_market": "moved differently from the other routes that year",
}


def build_features(routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (imp, sup), grp in routes.groupby(["importer", "supplier"]):
        s = grp.set_index("year")["value"].sort_index()
        logv = np.log(s)
        for yr in s.index[1:]:
            others = logv.drop(yr)
            slope, intercept = np.polyfit(others.index.values, others.values, 1)
            rows.append({
                "route": f"{sup} → {imp}",
                "importer": imp,
                "supplier": sup,
                "year": int(yr),
                "value": float(s[yr]),
                "growth": float(logv[yr] - logv[yr - 1]) if (yr - 1) in logv.index else np.nan,
                "trend_gap": float(logv[yr] - (slope * yr + intercept)),
            })
    df = pd.DataFrame(rows).dropna(subset=["growth"])
    df["vs_market"] = df["growth"] - df.groupby("year")["growth"].transform("mean")
    return df


def detect(routes: pd.DataFrame, contamination: float = 0.12, seed: int = 42) -> pd.DataFrame:
    df = build_features(routes)
    X = RobustScaler().fit_transform(df[FEATURES])

    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    model.fit(X)
    df["anomaly_score"] = -model.score_samples(X)  # higher = more unusual
    df["is_anomaly"] = model.predict(X) == -1

    # explain each point by its most extreme (scaled) feature
    z = pd.DataFrame(X, columns=FEATURES, index=df.index).abs()
    df["main_reason"] = z.idxmax(axis=1).map(FEATURE_TEXT)
    df["growth_pct"] = (np.exp(df["growth"]) - 1) * 100

    cols = ["route", "importer", "supplier", "year", "value", "growth_pct", "trend_gap", "vs_market",
            "anomaly_score", "is_anomaly", "main_reason"]
    return df[cols].sort_values("anomaly_score", ascending=False).reset_index(drop=True)
