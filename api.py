"""REST API for the assistant, so other apps (or a JS frontend) can use it.

    uvicorn api:app --reload
    -> docs at http://127.0.0.1:8000/docs
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from supplychain import __version__
from supplychain.assistant import SupplyChainAssistant
from supplychain.forecast import METHODS

app = FastAPI(
    title="Supply Chain AI API",
    version=__version__,
    description="Ask questions about the trade data, get forecasts and anomaly flags.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a demo; restrict this before deploying for real
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def bot() -> SupplyChainAssistant:
    return SupplyChainAssistant()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=300, examples=["How much does Germany import from China?"])
    use_llm: bool = False


@app.get("/health")
def health():
    b = bot()
    return {"status": "ok", "version": __version__, "routes": int(b.routes["route"].nunique()), "years": b.data.years}


@app.post("/ask")
def ask(req: AskRequest):
    return bot().ask(req.question, use_llm=req.use_llm).to_dict()


@app.get("/routes")
def routes(importer: str | None = None, supplier: str | None = None):
    df = bot().routes
    if importer:
        df = df[df["importer"].str.lower() == importer.lower()]
    if supplier:
        df = df[df["supplier"].str.lower() == supplier.lower()]
    return df.to_dict(orient="records")


@app.get("/forecast")
def forecast(
    importer: str,
    supplier: str,
    years: int = Query(2, ge=1, le=3),
    method: str | None = Query(None, description=f"one of {METHODS}; default = best in backtest"),
):
    b = bot()
    if method and method not in METHODS:
        raise HTTPException(400, f"method must be one of {METHODS}")
    names = {c.lower(): c for c in b.data.countries}
    imp, sup = names.get(importer.lower()), names.get(supplier.lower())
    if not imp or not sup:
        raise HTTPException(404, f"unknown country; available: {b.data.countries}")
    try:
        fc = b.forecaster.forecast(imp, sup, horizon=years, method=method)
    except ValueError as err:
        raise HTTPException(404, str(err))
    return {
        "route": fc.route,
        "method": fc.method,
        "backtest_mape_pct": fc.backtest_mape,
        "history": {int(k): float(v) for k, v in fc.history.items()},
        "forecast": fc.table.to_dict(orient="records"),
    }


@app.get("/anomalies")
def anomalies(only_flagged: bool = True):
    df = bot().anomalies
    if only_flagged:
        df = df[df["is_anomaly"]]
    return df.round(4).to_dict(orient="records")


@app.get("/quality")
def quality():
    return bot().data.report
