"""Plain pandas analytics on the cleaned route table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fmt_value(v: float) -> str:
    """175605000000 -> '175.61 B'"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.2f} {unit}"
    return f"{v:,.2f}"


def fmt_pct(v: float) -> str:
    if v is None or np.isnan(v):
        return "n/a"
    return f"{v:+.1f}%"


def filter_routes(routes: pd.DataFrame, importer=None, supplier=None, year=None) -> pd.DataFrame:
    df = routes
    if importer:
        df = df[df["importer"] == importer]
    if supplier:
        df = df[df["supplier"] == supplier]
    if year is not None:
        years = year if isinstance(year, (list, tuple, set)) else [year]
        df = df[df["year"].isin(years)]
    return df


def total_value(routes, importer=None, supplier=None, year=None) -> float:
    return float(filter_routes(routes, importer, supplier, year)["value"].sum())


def rank(routes, by="supplier", importer=None, supplier=None, year=None, ascending=False) -> pd.Series:
    df = filter_routes(routes, importer, supplier, year)
    return df.groupby(by)["value"].sum().sort_values(ascending=ascending)


def yearly(routes, importer=None, supplier=None) -> pd.Series:
    return filter_routes(routes, importer, supplier).groupby("year")["value"].sum().sort_index()


def route_pivot(routes: pd.DataFrame) -> pd.DataFrame:
    """year x route table, handy for line charts."""
    return routes.pivot_table(index="year", columns="route", values="value", aggfunc="sum").sort_index()


def growth(routes: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Year-over-year growth per route for one year (latest by default)."""
    df = routes.sort_values(["importer", "supplier", "year"]).copy()
    df["prev"] = df.groupby(["importer", "supplier"])["value"].shift(1)
    df["growth_pct"] = (df["value"] / df["prev"] - 1) * 100
    year = year or int(df["year"].max())
    out = df[df["year"] == year].dropna(subset=["growth_pct"])
    return out[["route", "importer", "supplier", "year", "prev", "value", "growth_pct"]].sort_values(
        "growth_pct", ascending=False
    )


def all_growth(routes: pd.DataFrame) -> pd.DataFrame:
    df = routes.sort_values(["importer", "supplier", "year"]).copy()
    df["growth_pct"] = df.groupby(["importer", "supplier"])["value"].pct_change() * 100
    return df


def cagr(routes, importer=None, supplier=None) -> float:
    s = yearly(routes, importer, supplier)
    if len(s) < 2 or s.iloc[0] <= 0:
        return float("nan")
    n = s.index[-1] - s.index[0]
    return ((s.iloc[-1] / s.iloc[0]) ** (1 / n) - 1) * 100


def hhi_label(hhi: float) -> str:
    # usual competition-policy thresholds, on a 0-1 scale
    if hhi >= 0.25:
        return "highly concentrated"
    if hhi >= 0.15:
        return "moderately concentrated"
    return "diversified"


def dependency(routes: pd.DataFrame, importer: str, year: int | None = None) -> tuple[pd.DataFrame, float]:
    """Share of each supplier in an importer's imports + Herfindahl index.

    Note: shares are only among the suppliers present in this dataset,
    not among all of the country's real trading partners.
    """
    year = year or int(routes["year"].max())
    df = filter_routes(routes, importer=importer, year=year)
    shares = df.groupby("supplier")["value"].sum().sort_values(ascending=False)
    shares = shares / shares.sum()
    hhi = float((shares ** 2).sum())
    table = pd.DataFrame({"share_pct": (shares * 100).round(1)})
    return table, hhi


def dependency_overview(routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for imp in sorted(routes["importer"].unique()):
        for yr in sorted(routes["year"].unique()):
            table, hhi = dependency(routes, imp, int(yr))
            if table.empty:
                continue
            rows.append({
                "importer": imp,
                "year": int(yr),
                "top_supplier": table.index[0],
                "top_share_pct": float(table["share_pct"].iloc[0]),
                "hhi": round(hhi, 3),
                "level": hhi_label(hhi),
            })
    return pd.DataFrame(rows)


def product_concentration(product_lines: pd.DataFrame) -> pd.DataFrame:
    """How much of a route depends on a few product lines (Germany detail data)."""
    rows = []
    for (imp, sup, yr), grp in product_lines.groupby(["importer", "supplier", "year"]):
        v = grp["value"].sort_values(ascending=False)
        share = v / v.sum()
        rows.append({
            "route": f"{sup} → {imp}",
            "year": int(yr),
            "product_lines": int(len(v)),
            "top1_share_pct": round(float(share.iloc[0] * 100), 1),
            "top5_share_pct": round(float(share.iloc[:5].sum() * 100), 1),
            "lines_for_80pct": int((share.cumsum() < 0.8).sum() + 1),
            "hhi": round(float((share ** 2).sum()), 3),
        })
    return pd.DataFrame(rows)
