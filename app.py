import pandas as pd
import streamlit as st

from supplychain import analytics as A
from supplychain import llm, nlp
from supplychain.assistant import EXAMPLES, SupplyChainAssistant
from supplychain.forecast import METHOD_LABELS, METHODS

st.set_page_config(page_title="Global Supply Chain AI", page_icon="🌍", layout="wide")


@st.cache_resource(show_spinner="Loading data and training models...")
def get_assistant() -> SupplyChainAssistant:
    return SupplyChainAssistant()


bot = get_assistant()
data = bot.data
routes = data.routes

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_answer(msg: dict):
    st.markdown(msg["content"])
    chart = msg.get("chart")
    if chart is not None and not chart.empty:
        if msg.get("chart_type") == "line":
            st.line_chart(chart)
        else:
            st.bar_chart(chart)
    table = msg.get("table")
    if table is not None and not table.empty:
        with st.expander("Show data"):
            st.dataframe(table)
    if msg.get("intent"):
        extra = " · rephrased by LLM" if msg.get("llm_used") else ""
        st.caption(f"intent: {msg['intent']} ({msg['confidence']:.0%}){extra}")


def run_question(question: str):
    ans = bot.ask(question, use_llm=st.session_state.get("use_llm", False))
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "content": ans.text,
        "table": ans.table,
        "chart": ans.chart,
        "chart_type": ans.chart_type,
        "intent": ans.intent,
        "confidence": ans.confidence,
        "llm_used": ans.llm_used,
    })


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🌍 Supply Chain AI")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.toggle(
        "Rephrase answers with LLM",
        key="use_llm",
        disabled=not llm.available(),
        help="Needs OPENAI_API_KEY. Numbers always come from the data, the LLM only rewords them.",
    )
    if not llm.available():
        st.caption("LLM off (no OPENAI_API_KEY). Everything else runs locally.")

    st.divider()
    st.subheader("Try asking")
    for ex in EXAMPLES[:6]:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            run_question(ex)
            st.rerun()

    st.divider()
    st.subheader("📜 Recent questions")
    user_qs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if not user_qs:
        st.caption("No questions yet.")
    for q in user_qs[-8:][::-1]:
        st.write("💬 " + q)


# ---------------- main ----------------
st.title("🌍 Global Supply Chain AI Assistant")
st.caption(
    f"{', '.join(data.importers)} importing from {', '.join(data.suppliers)} · "
    f"{min(data.years)}–{max(data.years)} · values in US$ (UN Comtrade)"
)

tab_chat, tab_dash, tab_fc, tab_risk, tab_quality, tab_models = st.tabs(
    ["💬 Assistant", "📊 Dashboard", "📈 Forecast", "⚠️ Risk & anomalies", "🧹 Data quality", "🧠 Models"]
)

with tab_chat:
    if not st.session_state.messages:
        st.info("Ask a question below, or click an example in the sidebar.")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                show_answer(msg)

question = st.chat_input("Ask a question about your supply-chain data...")
if question:
    run_question(question)
    st.rerun()


with tab_dash:
    year_options = data.years
    c1, c2 = st.columns([2, 1])
    picked_importers = c1.multiselect("Importers", data.importers, default=data.importers)
    picked_year = c2.selectbox("Year for rankings", year_options[::-1])

    view = routes[routes["importer"].isin(picked_importers)]
    latest = A.total_value(view, year=max(data.years))
    prev = A.total_value(view, year=max(data.years) - 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total (all years)", A.fmt_value(A.total_value(view)))
    m2.metric(f"{max(data.years)}", A.fmt_value(latest), A.fmt_pct((latest / prev - 1) * 100) if prev else None)
    m3.metric("Routes", view["route"].nunique())
    m4.metric("Double counting removed", A.fmt_value(data.report["double_counted"]))

    st.subheader("Trade value per route")
    st.line_chart(A.route_pivot(view))

    left, right = st.columns(2)
    with left:
        st.subheader(f"Suppliers in {picked_year}")
        st.bar_chart(A.rank(view, "supplier", year=picked_year))
    with right:
        st.subheader(f"Growth in {picked_year}")
        if picked_year == min(data.years):
            st.caption("No previous year to compare with.")
        else:
            g = A.growth(view, picked_year).set_index("route")["growth_pct"]
            st.bar_chart(g)


with tab_fc:
    fc_model = bot.forecaster
    route_list = routes[["supplier", "importer"]].drop_duplicates().sort_values(["importer", "supplier"])
    labels = [f"{s} → {i}" for s, i in route_list.values]

    c1, c2, c3 = st.columns([2, 2, 1])
    picked = c1.selectbox("Route", labels, index=labels.index("China → Germany") if "China → Germany" in labels else 0)
    method = c2.selectbox(
        "Model",
        METHODS,
        index=METHODS.index(fc_model.best_method),
        format_func=lambda m: METHOD_LABELS[m] + (" ✓ best in backtest" if m == fc_model.best_method else ""),
    )
    horizon = c3.slider("Years ahead", 1, 3, 2)

    sup, imp = picked.split(" → ")
    fc = fc_model.forecast(imp, sup, horizon=horizon, method=method)
    chart = pd.concat([fc.history.rename("actual"), fc.table.set_index("year")[["forecast", "low", "high"]]], axis=1)
    chart.loc[fc.history.index[-1], ["forecast", "low", "high"]] = fc.history.iloc[-1]
    st.line_chart(chart)

    nxt = fc.table.iloc[0]
    st.markdown(
        f"**{int(nxt['year'])}:** {A.fmt_value(nxt['forecast'])} "
        f"(range {A.fmt_value(nxt['low'])} – {A.fmt_value(nxt['high'])}, "
        f"about ±{fc.interval_pct:.0f}% based on {fc_model.coverage:.0%} of backtest errors)"
    )

    st.subheader("Backtest: how well each model predicted 2022 and 2023")
    st.dataframe(fc_model.summary())
    st.caption(
        "MAPE = mean absolute percentage error over all routes. With only 5 years per route, "
        "simple models are hard to beat, so the app picks whichever model actually scored best "
        "instead of the most complex one."
    )


with tab_risk:
    st.subheader("Supplier concentration (HHI)")
    ov = A.dependency_overview(routes)
    hhi_chart = ov.pivot(index="year", columns="importer", values="hhi")
    st.line_chart(hhi_chart)
    st.caption("0 = spread across many suppliers, 1 = a single supplier. Above 0.25 is highly concentrated. "
               "Shares are only among the suppliers in this dataset.")
    st.dataframe(ov[ov["year"] == max(data.years)].set_index("importer"))

    st.subheader("Unusual route-years (Isolation Forest)")
    an = bot.anomalies
    show = an[["route", "year", "growth_pct", "anomaly_score", "is_anomaly", "main_reason"]].round(2)
    st.dataframe(show.style.apply(
        lambda r: ["background-color: rgba(255, 99, 71, 0.18)" if r["is_anomaly"] else "" for _ in r], axis=1
    ))

    st.subheader("Germany: how many product lines carry each route")
    pc = A.product_concentration(data.product_lines)
    st.dataframe(pc.set_index(["route", "year"]))
    st.caption("lines_for_80pct = how few product lines make up 80% of the route's value.")


with tab_quality:
    r = data.report
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw rows", f"{r['raw_rows']:,}")
    q2.metric("Summary rows used", r["summary_rows"])
    q3.metric("Product-line rows", f"{r['product_line_rows']:,}")
    q4.metric("Routes", r["routes"])

    st.markdown(
        f"""
**What was wrong:** for Germany in 2020–2023 the file has one *total* row per supplier and year,
**plus** many product-line rows that add up to exactly that total. Summing the whole column
counts Germany twice.

**What the cleaning does:** in every importer-supplier-year group, if the other rows add up to the
biggest row (within 1%), the biggest row is kept as the total and the rest are stored separately as
product lines. Groups with only one row are already totals.

| | value |
|---|---|
| Naive sum of the column | {A.fmt_value(r['naive_sum'])} |
| Cleaned total | {A.fmt_value(r['clean_sum'])} |
| Double counted | {A.fmt_value(r['double_counted'])} |
"""
    )
    st.subheader("Cleaned route table")
    st.dataframe(routes.pivot_table(index="route", columns="year", values="value").map(A.fmt_value))


with tab_models:
    st.subheader("Intent classifier")
    met = nlp.load_metrics()
    if met:
        k1, k2, k3 = st.columns(3)
        k1.metric("CV accuracy", f"{met['accuracy']:.1%}")
        k2.metric("Macro F1", f"{met['macro_f1']:.2f}")
        k3.metric("Training examples", met["n_examples"])
        st.bar_chart(pd.Series(met["per_intent_f1"], name="F1"))
    st.markdown(
        "TF-IDF (word 1–2 grams + character 2–5 grams) → logistic regression, trained on "
        "`data/intents.csv`. Country names and years are replaced by placeholder tokens before "
        "training, so the model learns the question pattern, not the country. Entities are pulled out "
        "with rules plus fuzzy matching (handles typos like *Germny*)."
    )
    st.subheader("Test the classifier")
    probe = st.text_input("Type any question", "how much did germny buy from korea in 2022?")
    if probe:
        p = nlp.predict_intent(bot.model, probe)
        e = nlp.extract_entities(probe)
        st.write(f"**Intent:** {p.intent} ({p.confidence:.0%}) · alternatives: "
                 + ", ".join(f"{i} ({c:.0%})" for i, c in p.alternatives))
        st.json(e.to_dict())
