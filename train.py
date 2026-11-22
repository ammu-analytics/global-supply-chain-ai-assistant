"""Train and evaluate all models, and print a short report.

    python train.py

Saves the intent classifier to models/ and writes models/report.json.
"""

import json
from pathlib import Path

from supplychain import nlp
from supplychain.anomaly import detect
from supplychain.data import load_data
from supplychain.forecast import Forecaster


def old_keyword_rules(q: str) -> str:
    """A simple if/elif keyword chatbot, mapped to the intent names.

    Used as a baseline so I can show the ML model is actually better.
    """
    q = q.lower().strip()
    if any(k in q for k in ("total trade", "overall trade", "total value")):
        return "total_trade"
    if "supplier" in q and any(k in q for k in ("biggest", "largest", "highest", "most", "top", "smallest", "lowest", "least")):
        return "rank_suppliers"
    if "importer" in q and any(k in q for k in ("biggest", "largest", "highest", "most", "top", "smallest", "lowest", "least")):
        return "rank_importers"
    if any(k in q for k in ("supplier totals", "supplier total", "supplier trade", "show suppliers")):
        return "rank_suppliers"
    if any(k in q for k in ("importer totals", "importer total", "importer trade", "show importers")):
        return "rank_importers"
    if any(k in q for k in ("trade by year", "trade value by year", "yearly trade", "trade each year", "trade every year")):
        return "trend"
    if "germany" in q and "china" in q:
        return "route_value"
    if any(s in q for s in ("china", "japan", "netherlands", "rep. of korea")):
        return "total_trade"
    return "unknown"


def main():
    df = nlp.load_training_data()

    print("1) Intent classifier (TF-IDF + logistic regression)")
    nlp.train(save=True)
    ml = nlp.load_metrics()
    base_acc = (df["text"].map(old_keyword_rules) == df["intent"]).mean()
    print(f"   examples: {ml['n_examples']}, intents: {ml['n_intents']}")
    print(f"   5-fold CV accuracy: {ml['accuracy']:.1%}  (macro F1 {ml['macro_f1']:.2f})")
    print(f"   keyword-rule baseline on the same questions: {base_acc:.1%}")

    data = load_data()
    print("\n2) Forecasting backtest (predict 2022 and 2023 from earlier years)")
    fc = Forecaster(data.routes)
    print(fc.summary().to_string())
    print(f"   -> chosen model: {fc.best_method}")

    print("\n3) Anomaly detection (Isolation Forest)")
    an = detect(data.routes)
    flagged = an[an["is_anomaly"]]
    for r in flagged.itertuples():
        print(f"   {r.route:26} {r.year}  {r.growth_pct:+6.1f}%  {r.main_reason}")

    report = {
        "intent_model": ml,
        "keyword_baseline_accuracy": round(float(base_acc), 3),
        "forecast_backtest": fc.summary().reset_index().rename(columns={"index": "model"}).to_dict(orient="records"),
        "forecast_model": fc.best_method,
        "anomalies": flagged[["route", "year", "growth_pct", "main_reason"]].round(2).to_dict(orient="records"),
        "data_quality": {k: v for k, v in data.report.items()},
    }
    out = Path(__file__).parent / "models" / "report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved {out.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
