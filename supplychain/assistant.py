"""The assistant: question -> intent + entities -> analytics/ML -> answer."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import analytics as A
from . import anomaly, llm, nlp
from .data import TradeData, load_data
from .forecast import METHOD_LABELS, Forecaster

EXAMPLES = [
    "Who is the biggest supplier?",
    "How much does Germany import from China in 2023?",
    "Show USA imports over the years",
    "Which route grew the fastest in 2023?",
    "Forecast Japan to USA trade",
    "How dependent is Germany on China?",
    "Any anomalies in the data?",
    "Was anything double counted?",
]


@dataclass
class Answer:
    text: str
    intent: str = "unknown"
    confidence: float = 0.0
    entities: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None
    chart: pd.DataFrame | None = None
    chart_type: str | None = None  # "line" or "bar"
    llm_used: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "intent": self.intent,
            "confidence": round(self.confidence, 3),
            "entities": self.entities,
            "table": None if self.table is None else self.table.reset_index().to_dict(orient="records"),
            "llm_used": self.llm_used,
        }


def _money_table(series_or_df, value_cols=("value",)) -> pd.DataFrame:
    df = series_or_df.to_frame("value") if isinstance(series_or_df, pd.Series) else series_or_df.copy()
    for c in value_cols:
        if c in df.columns:
            df[c + "_readable"] = df[c].map(A.fmt_value)
    return df


class SupplyChainAssistant:
    def __init__(self, data: TradeData | None = None, model=None):
        self.data = data or load_data()
        self.routes = self.data.routes
        self.model = model or nlp.load_model()
        self.forecaster = Forecaster(self.routes)
        self.anomalies = anomaly.detect(self.routes)
        self.latest_year = max(self.data.years)

    # ---------- helpers ----------

    def _resolve_route(self, countries: list[str], text: str) -> tuple[str | None, str | None]:
        """Work out (importer, supplier) from the countries in the question."""
        imps, sups = set(self.data.importers), set(self.data.suppliers)
        existing = set(map(tuple, self.routes[["importer", "supplier"]].drop_duplicates().values))
        low = text.lower()

        if len(countries) >= 2:
            a, b = countries[0], countries[1]
            if (a, b) in existing and (b, a) not in existing:
                return a, b
            if (b, a) in existing and (a, b) not in existing:
                return b, a
            # both directions exist (doesn't happen in this data yet): use word order
            if f"from {b.lower()}" in low or " to " not in low:
                return a, b
            return b, a

        if len(countries) == 1:
            c = countries[0]
            if c in imps and c not in sups:
                return c, None
            if c in sups and c not in imps:
                return None, c
            # China is both an importer and a supplier here
            idx = low.find(c.lower()) if c.lower() in low else -1
            before = low[max(0, idx - 12): idx] if idx >= 0 else ""
            if "from" in before or "with" in before or "supplier" in low or "export" in low or "suppl" in low:
                return None, c
            return c, None

        return None, None

    def _year(self, ents: nlp.Entities) -> int | None:
        valid = [y for y in ents.years if y in self.data.years]
        return valid[0] if valid else None

    def _scope_text(self, importer, supplier, year) -> str:
        parts = []
        if supplier and importer:
            parts.append(f"{supplier} → {importer}")
        elif importer:
            parts.append(f"imports of {importer}")
        elif supplier:
            parts.append(f"supplies from {supplier}")
        if year:
            parts.append(f"in {year}")
        return " ".join(parts)

    # ---------- main entry ----------

    def ask(self, question: str, use_llm: bool = False) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer(text="Please type a question.")

        pred = nlp.predict_intent(self.model, question)
        ents = nlp.extract_entities(question)

        if pred.confidence < nlp.MIN_CONFIDENCE:
            options = ", ".join(i.replace("_", " ") for i, _ in [(pred.intent, 0)] + pred.alternatives)
            ans = Answer(
                text=(
                    "I'm not sure what you mean. It could be about: "
                    f"{options}.\n\nTry for example: *\"{EXAMPLES[1]}\"*"
                ),
                intent="unsure",
                confidence=pred.confidence,
                entities=ents.to_dict(),
            )
            return ans

        # a year was asked for but none of them are in the data -> say so instead of guessing
        if ents.years and not self._year(ents) and pred.intent not in ("forecast", "help", "data_quality"):
            if all(y > self.latest_year for y in ents.years):
                pred.intent = "forecast"
            else:
                years = ", ".join(map(str, ents.years))
                return Answer(
                    text=f"There's no data for {years}. The data covers {min(self.data.years)}–{self.latest_year}.",
                    intent=pred.intent, confidence=pred.confidence, entities=ents.to_dict(),
                )

        handler = getattr(self, f"_h_{pred.intent}", self._h_help)
        ans = handler(question, ents)
        ans.intent = pred.intent
        ans.confidence = pred.confidence
        ans.entities = ents.to_dict()

        if use_llm:
            table_text = "" if ans.table is None else ans.table.head(12).to_string()
            better = llm.rephrase(question, ans.text, table_text)
            if better:
                ans.text, ans.llm_used = better, True
        return ans

    # ---------- intent handlers ----------

    def _h_total_trade(self, q, ents):
        imp, sup = self._resolve_route(ents.countries, q)
        year = self._year(ents)
        total = A.total_value(self.routes, imp, sup, year)
        scope = self._scope_text(imp, sup, year)
        text = f"**Total trade value{' (' + scope + ')' if scope else ''}:** {A.fmt_value(total)}"
        if not (imp or sup or year):
            text += (
                f"\n\nThis is after removing {self.data.report['product_line_rows']:,} Germany product-line rows "
                f"that were counted twice in the raw file (a naive sum would say "
                f"{A.fmt_value(self.data.report['naive_sum'])})."
            )
        chart = A.yearly(self.routes, imp, sup)
        return Answer(text=text, table=_money_table(chart), chart=chart.to_frame("value"), chart_type="bar")

    def _rank(self, q, ents, by: str):
        imp, sup = self._resolve_route(ents.countries, q)
        year = self._year(ents)
        filt = {"importer": imp} if by == "supplier" else {"supplier": sup or (ents.countries[0] if ents.countries else None)}
        result = A.rank(self.routes, by=by, year=year, ascending=ents.ascending, **filt)
        if result.empty:
            return Answer(text="I couldn't find matching data for that.")

        word = "smallest" if ents.ascending else "biggest"
        role = "supplier" if by == "supplier" else "importer"
        scope = []
        if by == "supplier" and imp:
            scope.append(f"to {imp}")
        if by == "importer" and filt["supplier"]:
            scope.append(f"from {filt['supplier']}")
        if year:
            scope.append(f"in {year}")
        scope_txt = (" " + " ".join(scope)) if scope else ""

        n = ents.top_n or 1
        if n == 1:
            text = f"**{result.index[0]}** is the {word} {role}{scope_txt}, with **{A.fmt_value(result.iloc[0])}**."
        else:
            names = ", ".join(f"{name} ({A.fmt_value(v)})" for name, v in result.head(n).items())
            text = f"The {n} {word} {role}s{scope_txt}: {names}."
        if len(result) > n:
            text += f"\n\nFull ranking of {len(result)} {role}s below."
        return Answer(text=text, table=_money_table(result), chart=result.to_frame("value"), chart_type="bar")

    def _h_rank_suppliers(self, q, ents):
        return self._rank(q, ents, "supplier")

    def _h_rank_importers(self, q, ents):
        return self._rank(q, ents, "importer")

    def _h_route_value(self, q, ents):
        imp, sup = self._resolve_route(ents.countries, q)
        if not (imp and sup):
            if imp or sup:
                return self._h_total_trade(q, ents)
            routes = ", ".join(sorted(self.routes["route"].unique()))
            return Answer(text=f"Which route do you mean? Available routes: {routes}.")
        series = A.yearly(self.routes, imp, sup)
        if series.empty:
            return Answer(text=f"There is no {sup} → {imp} trade in this data.")
        year = self._year(ents)
        if year:
            v = series.get(year)
            text = f"**{sup} → {imp} in {year}:** {A.fmt_value(v)}"
            prev = series.get(year - 1)
            if prev:
                text += f" ({A.fmt_pct((v / prev - 1) * 100)} vs {year - 1})"
        else:
            text = (
                f"**{sup} → {imp}:** {A.fmt_value(series.iloc[-1])} in {series.index[-1]}, "
                f"{A.fmt_value(series.sum())} over {series.index[0]}–{series.index[-1]} "
                f"(average growth {A.cagr(self.routes, imp, sup):+.1f}% per year)."
            )
        return Answer(text=text, table=_money_table(series), chart=series.to_frame(f"{sup} → {imp}"), chart_type="line")

    def _h_trend(self, q, ents):
        imp, sup = self._resolve_route(ents.countries, q)
        if imp or sup:
            series = A.yearly(self.routes, imp, sup)
            label = self._scope_text(imp, sup, None)
            first, last = series.iloc[0], series.iloc[-1]
            peak = int(series.idxmax())
            text = (
                f"**Trend for {label}:** from {A.fmt_value(first)} ({series.index[0]}) to {A.fmt_value(last)} "
                f"({series.index[-1]}), {A.fmt_pct((last / first - 1) * 100)} overall. Peak year: {peak}."
            )
            sub = A.filter_routes(self.routes, imp, sup)
            chart = A.route_pivot(sub)
            return Answer(text=text, table=_money_table(series), chart=chart, chart_type="line")

        series = A.yearly(self.routes)
        text = (
            f"**Trade value by year (all routes):** peak in {int(series.idxmax())} with "
            f"{A.fmt_value(series.max())}; {self.latest_year} was {A.fmt_value(series.iloc[-1])}."
        )
        return Answer(text=text, table=_money_table(series), chart=A.route_pivot(self.routes), chart_type="line")

    def _h_growth(self, q, ents):
        imp, sup = self._resolve_route(ents.countries, q)
        year = self._year(ents) or self.latest_year
        if year == min(self.data.years):
            year = min(self.data.years) + 1

        if imp or sup:
            g = A.all_growth(A.filter_routes(self.routes, imp, sup))
            g = g.dropna(subset=["growth_pct"])
            yearly = A.yearly(self.routes, imp, sup)
            yoy = (yearly.pct_change() * 100).dropna()
            text = (
                f"**Growth for {self._scope_text(imp, sup, None)}:** {A.fmt_pct(yoy.get(year, float('nan')))} in {year}. "
                f"Average growth {A.cagr(self.routes, imp, sup):+.1f}% per year since {min(self.data.years)}."
            )
            table = g[["route", "year", "value", "growth_pct"]].round({"growth_pct": 1})
            return Answer(text=text, table=table, chart=yoy.to_frame("growth %"), chart_type="bar")

        g = A.growth(self.routes, year)
        best, worst = g.iloc[0], g.iloc[-1]
        text = (
            f"**Year-over-year growth in {year}:** fastest was **{best['route']}** ({A.fmt_pct(best['growth_pct'])}), "
            f"biggest drop was **{worst['route']}** ({A.fmt_pct(worst['growth_pct'])})."
        )
        table = g[["route", "prev", "value", "growth_pct"]].round({"growth_pct": 1}).set_index("route")
        return Answer(text=text, table=table, chart=table[["growth_pct"]], chart_type="bar")

    MAX_HORIZON = 3  # with 5 years of history, going further out isn't meaningful

    def _h_forecast(self, q, ents):
        imp, sup = self._resolve_route(ents.countries, q)
        f = self.forecaster
        future = [y for y in ents.years if y > self.latest_year]
        target = min(future) if future else self.latest_year + 1
        too_far = ""
        if target > self.latest_year + self.MAX_HORIZON:
            too_far = (f"\n\n{target} is too far ahead to forecast with only {len(self.data.years)} years of history, "
                       f"so I'm showing {self.latest_year + self.MAX_HORIZON} instead.")
            target = self.latest_year + self.MAX_HORIZON
        horizon = max(2, target - self.latest_year)
        method_note = (
            f"Model: **{METHOD_LABELS[f.best_method]}**. It had the lowest error in the backtest "
            f"(average error {f.mape[f.best_method]:.1f}% when predicting 2022 and 2023 from earlier years)."
        )

        if imp and sup:
            fc = f.forecast(imp, sup, horizon=horizon)
            r1 = fc.table.set_index("year").loc[target]
            r1["year"] = target
            text = (
                f"**Forecast {fc.route} for {int(r1['year'])}:** about {A.fmt_value(r1['forecast'])} "
                f"(likely range {A.fmt_value(r1['low'])} – {A.fmt_value(r1['high'])}).\n\n{method_note}{too_far}"
            )
            chart = pd.concat([fc.history.rename("actual"), fc.table.set_index("year")[["forecast", "low", "high"]]], axis=1)
            chart.loc[fc.history.index[-1], ["forecast", "low", "high"]] = fc.history.iloc[-1]
            return Answer(text=text, table=_money_table(fc.table.set_index("year"), ("forecast", "low", "high")),
                          chart=chart, chart_type="line")

        all_fc = f.forecast_all(horizon=horizon)
        if imp:
            all_fc = all_fc[all_fc["importer"] == imp]
        elif sup:
            all_fc = all_fc[all_fc["supplier"] == sup]
        nxt = target
        sub = all_fc[all_fc["year"] == nxt]
        total = sub["forecast"].sum()
        label = self._scope_text(imp, sup, None) or "all routes"
        text = (
            f"**Forecast for {label} in {nxt}:** about {A.fmt_value(total)} in total "
            f"(sum of {len(sub)} route forecasts).\n\n{method_note}\n\n"
            "Ask about one route (e.g. *forecast Japan to USA*) to see its range." + too_far
        )
        table = sub.set_index("route")[["forecast", "low", "high"]]
        return Answer(text=text, table=_money_table(table, ("forecast", "low", "high")),
                      chart=table[["forecast"]], chart_type="bar")

    def _h_dependency(self, q, ents):
        imp, _ = self._resolve_route(ents.countries[:1], q) if ents.countries else (None, None)
        if imp not in self.data.importers:
            imp = next((c for c in ents.countries if c in self.data.importers), None)
        year = self._year(ents) or self.latest_year

        if not imp:
            ov = A.dependency_overview(self.routes)
            ov = ov[ov["year"] == year].set_index("importer")
            worst = ov["hhi"].idxmax()
            text = (
                f"**Supplier concentration in {year}:** {worst} is the most concentrated "
                f"({ov.loc[worst, 'top_share_pct']:.0f}% from {ov.loc[worst, 'top_supplier']}, "
                f"HHI {ov.loc[worst, 'hhi']:.2f}).\n\n"
                "HHI goes from 0 (spread over many suppliers) to 1 (one supplier). Above 0.25 counts as highly concentrated."
            )
            return Answer(text=text, table=ov, chart=ov[["hhi"]], chart_type="bar")

        table, hhi = A.dependency(self.routes, imp, year)
        top = table.index[0]
        text = (
            f"**{imp} in {year}:** {table['share_pct'].iloc[0]:.0f}% of its imports here come from **{top}**. "
            f"HHI = {hhi:.2f} → **{A.hhi_label(hhi)}**.\n\n"
            f"Note: shares are only among the {len(table)} suppliers in this dataset, not all of {imp}'s trade partners."
        )
        return Answer(text=text, table=table, chart=table, chart_type="bar")

    def _h_anomalies(self, q, ents):
        df = self.anomalies
        if ents.countries:
            mask = df["importer"].isin(ents.countries) | df["supplier"].isin(ents.countries)
            df = df[mask]
        year = self._year(ents)
        if year:
            df = df[df["year"] == year]
        flagged = df[df["is_anomaly"]]
        if flagged.empty:
            text = "No route-years were flagged as unusual for that selection."
            show = df.head(5)
        else:
            lines = [
                f"- **{r.route} {r.year}**: {A.fmt_pct(r.growth_pct)} vs previous year ({r.main_reason})"
                for r in flagged.itertuples()
            ]
            text = (
                f"The Isolation Forest flagged **{len(flagged)}** unusual route-year(s):\n" + "\n".join(lines)
                + "\n\nThese are statistical flags, not proof of an error; they're worth checking."
            )
            show = flagged
        table = show.set_index(["route", "year"])[["growth_pct", "anomaly_score", "main_reason"]].round(2)
        return Answer(text=text, table=table)

    def _h_data_quality(self, q, ents):
        r = self.data.report
        text = (
            f"**Data quality summary**\n\n"
            f"- Raw rows: {r['raw_rows']:,} (years {r['years'][0]}–{r['years'][-1]}, flow: {', '.join(r['trade_flows'])})\n"
            f"- Summary rows used: {r['summary_rows']} → {r['routes']} routes\n"
            f"- Germany 2020–2023 also has {r['product_line_rows']:,} product-line rows. "
            f"They add up exactly to Germany's total rows, so summing everything counts Germany twice.\n"
            f"- Naive total: {A.fmt_value(r['naive_sum'])} → cleaned total: {A.fmt_value(r['clean_sum'])} "
            f"(removed {A.fmt_value(r['double_counted'])} of double counting)\n"
            f"- Invalid or empty values dropped: {r['invalid_rows_dropped']}"
        )
        table = pd.DataFrame(
            {"value": [r["naive_sum"], r["clean_sum"], r["double_counted"]]},
            index=["simple column sum", "cleaned sum", "double counted"],
        )
        return Answer(text=text, table=_money_table(table))

    def _h_help(self, q, ents):
        examples = "\n".join(f"- {e}" for e in EXAMPLES)
        text = (
            "Hi! I answer questions about the supply chain data "
            f"({', '.join(self.data.importers)} importing from {', '.join(self.data.suppliers)}, "
            f"{min(self.data.years)}–{self.latest_year}).\n\nYou can ask things like:\n" + examples
        )
        return Answer(text=text)
