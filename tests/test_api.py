import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["routes"] == 11


def test_ask():
    r = client.post("/ask", json={"question": "who is the biggest supplier?"})
    assert r.status_code == 200
    assert r.json()["intent"] == "rank_suppliers"


def test_ask_validation():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_forecast():
    r = client.get("/forecast", params={"importer": "germany", "supplier": "china", "years": 2})
    assert r.status_code == 200
    assert len(r.json()["forecast"]) == 2


def test_forecast_unknown_country():
    assert client.get("/forecast", params={"importer": "mars", "supplier": "china"}).status_code == 404


def test_anomalies_and_quality():
    assert client.get("/anomalies").status_code == 200
    assert client.get("/quality").json()["routes"] == 11
