"""Loading and cleaning the trade data.

The raw CSV mixes two kinds of rows:

* one summary row per (importer, supplier, year) - China, USA and Germany 2019
* for Germany 2020-2023: one all-products TOTAL row plus many product-line
  rows that add up to that same total

If you just sum the "Trade Value" column,
Germany gets counted twice. `clean()` separates the two kinds of rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "global_supply_chain.csv"

# product lines must add up to the biggest row within 1% to call it a total row
TOTAL_MATCH_TOLERANCE = 0.01

KEYS = ["importer", "supplier", "year"]


@dataclass
class TradeData:
    routes: pd.DataFrame          # one row per importer-supplier-year
    product_lines: pd.DataFrame   # detailed rows (Germany 2020-2023 only)
    report: dict = field(default_factory=dict)

    @property
    def importers(self) -> list[str]:
        return sorted(self.routes["importer"].unique())

    @property
    def suppliers(self) -> list[str]:
        return sorted(self.routes["supplier"].unique())

    @property
    def years(self) -> list[int]:
        return sorted(int(y) for y in self.routes["year"].unique())

    @property
    def countries(self) -> list[str]:
        return sorted(set(self.importers) | set(self.suppliers))


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"trade_value": "value"})

    for col in ("importer", "supplier", "trade_flow"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def clean(raw: pd.DataFrame) -> TradeData:
    df = raw.dropna(subset=["year", "value"]).copy()
    dropped_invalid = len(raw) - len(df)
    df = df[df["value"] > 0]
    df["year"] = df["year"].astype(int)
    df["row_type"] = "total"

    groups_without_total = []

    for key, grp in df.groupby(KEYS):
        if len(grp) == 1:
            continue
        top_idx = grp["value"].idxmax()
        biggest = grp.at[top_idx, "value"]
        rest = grp["value"].drop(top_idx).sum()

        if abs(rest - biggest) / biggest <= TOTAL_MATCH_TOLERANCE:
            # biggest row is the total, the others are its breakdown
            df.loc[grp.index.drop(top_idx), "row_type"] = "product_line"
        else:
            # no total row in this group, so we build the total ourselves
            df.loc[grp.index, "row_type"] = "product_line"
            groups_without_total.append(key)

    product_lines = df.loc[df["row_type"] == "product_line", KEYS + ["value"]]
    totals = df.loc[df["row_type"] == "total", KEYS + ["value"]]

    derived = (
        product_lines.groupby(KEYS, as_index=False)["value"].sum()
        .merge(totals[KEYS], on=KEYS, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )

    routes = (
        pd.concat([totals, derived], ignore_index=True)
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    routes["route"] = routes["supplier"] + " → " + routes["importer"]

    raw_sum = float(df["value"].sum())
    clean_sum = float(routes["value"].sum())

    report = {
        "raw_rows": int(len(raw)),
        "invalid_rows_dropped": int(dropped_invalid),
        "summary_rows": int((df["row_type"] == "total").sum()),
        "product_line_rows": int(len(product_lines)),
        "groups_without_total_row": [list(map(str, k)) for k in groups_without_total],
        "routes": int(routes[["importer", "supplier"]].drop_duplicates().shape[0]),
        "years": sorted(int(y) for y in routes["year"].unique()),
        "naive_sum": raw_sum,
        "clean_sum": clean_sum,
        "double_counted": raw_sum - clean_sum,
        "trade_flows": sorted(df["trade_flow"].unique()) if "trade_flow" in df else [],
    }

    return TradeData(routes=routes, product_lines=product_lines.reset_index(drop=True), report=report)


def load_data(path: str | Path | None = None) -> TradeData:
    return clean(load_raw(path))
