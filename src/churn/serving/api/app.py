"""On-demand scoring API. The batch job is the main path; this covers single lookups."""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from churn import __version__, config
from churn.decision.policy import customer_value, decide, expected_value, risk_band
from churn.models.bundle import load_bundle
from churn.serving.api.schemas import Customer, Prediction

PREDICTION_LOG = Path(os.getenv("PREDICTION_LOG", config.ROOT / "logs" / "api_predictions.jsonl"))
MAX_BATCH = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("churn-api")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (config.CHAMPION_DIR / "model.joblib").exists():
        raise RuntimeError(f"No champion found in {config.CHAMPION_DIR}. Run `churn init` first.")
    STATE.update(load_bundle(config.CHAMPION_DIR))
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    logger.info("loaded model version=%s", STATE["metadata"]["version"])
    yield
    STATE.clear()


app = FastAPI(title="Churn Prediction API", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    logger.info(
        "%s %s status=%s latency_ms=%.1f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - start) * 1000,
        request.state.request_id,
    )
    return response


def _score(customers: list[Customer], request_id: str) -> list[Prediction]:
    df = pd.DataFrame([c.model_dump() for c in customers])
    proba = STATE["model"].predict_proba(df[config.FEATURES])[:, 1]
    value = customer_value(df)
    policy = STATE["policy"]
    contact = decide(proba, value, policy)
    ev = expected_value(proba, value)
    version = STATE["metadata"]["version"]

    predictions = [
        Prediction(
            customer_id=c.customer_id,
            churn_probability=round(float(p), 4),
            contact=bool(k),
            expected_value=round(float(e), 2),
            risk_band=risk_band(float(p)),
            decision_rule=policy["rule"],
            model_version=version,
        )
        for c, p, k, e in zip(customers, proba, contact, ev, strict=True)
    ]

    now = datetime.now(UTC).isoformat(timespec="seconds")
    with open(PREDICTION_LOG, "a") as f:
        for c, pred in zip(customers, predictions, strict=True):
            f.write(
                json.dumps({"ts": now, "request_id": request_id, "input": c.model_dump()} | pred.model_dump()) + "\n"
            )
    return predictions


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/model")
def model_info():
    meta = STATE["metadata"]
    return {
        "version": meta["version"],
        "model": meta["model"],
        "run_id": meta["run_id"],
        "trigger": meta["trigger"],
        "trained_on_cohorts": meta["trained_on_cohorts"],
        "decision_policy": STATE["policy"],
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request):
    try:
        return _score([customer], request.state.request_id)[0]
    except Exception as exc:
        logger.exception("inference error request_id=%s", request.state.request_id)
        raise HTTPException(status_code=500, detail=f"inference error: {exc}") from exc


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer], request: Request):
    if not customers:
        raise HTTPException(status_code=400, detail="empty list")
    if len(customers) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"max {MAX_BATCH} customers per request, use the batch job")
    return _score(customers, request.state.request_id)
