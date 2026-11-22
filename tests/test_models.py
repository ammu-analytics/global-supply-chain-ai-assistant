import json

import pytest

from supplychain import anomaly, nlp
from supplychain.forecast import METHODS, Forecaster


@pytest.mark.parametrize("text,expected", [
    ("How much does Germany import from China?", ["Germany", "China"]),
    ("trade from south korea to the usa", ["Rep. of Korea", "USA"]),
    ("dutch exports to germny", ["Netherlands", "Germany"]),  # typo
    ("tell us the total", []),  # "us" here is not the USA
    ("what did the US buy from Japan", ["USA", "Japan"]),
])
def test_country_extraction(text, expected):
    assert nlp.extract_entities(text).countries == expected


def test_years_and_top_n():
    e = nlp.extract_entities("top 3 smallest suppliers in 2021 and 2022")
    assert e.years == [2021, 2022]
    assert e.top_n == 3
    assert e.ascending


def test_normalize_hides_entities():
    assert nlp.normalize("Germany imports from China in 2023") == "xcountryx imports from xcountryx in xyearx"


def test_intent_model_quality():
    metrics = nlp.evaluate(nlp.load_training_data())
    assert metrics["accuracy"] >= 0.75
    assert metrics["macro_f1"] >= 0.75


@pytest.mark.parametrize("q,intent", [
    ("who is the biggest supplier", "rank_suppliers"),
    ("forecast germany china for 2024", "forecast"),
    ("any anomalies?", "anomalies"),
    ("how dependent is the usa on china", "dependency"),
    ("show trade by year", "trend"),
])
def test_intent_predictions(bot, q, intent):
    assert nlp.predict_intent(bot.model, q).intent == intent


def test_backtest_covers_all_methods(data):
    f = Forecaster(data.routes)
    assert set(f.bt["method"]) == set(METHODS)
    assert f.best_method in METHODS
    assert (f.bt["ape_pct"] >= 0).all()


def test_forecast_shape(data):
    f = Forecaster(data.routes)
    fc = f.forecast("USA", "Japan", horizon=3)
    assert list(fc.table["year"]) == [2024, 2025, 2026]
    assert (fc.table["low"] <= fc.table["forecast"]).all()
    assert (fc.table["forecast"] <= fc.table["high"]).all()
    # range should widen further out
    width = fc.table["high"] - fc.table["low"]
    assert width.is_monotonic_increasing


def test_anomalies(data):
    an = anomaly.detect(data.routes)
    assert len(an) == 44  # 11 routes x 4 years with a previous year
    assert 1 <= an["is_anomaly"].sum() <= 10
    assert an["anomaly_score"].is_monotonic_decreasing


def test_saved_metrics_are_json():
    nlp.train(save=True)
    assert json.loads(nlp.METRICS_PATH.read_text())["n_intents"] == 11
