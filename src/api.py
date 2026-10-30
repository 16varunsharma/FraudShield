"""FastAPI service: /health, /score, /explain.

Run:  uvicorn src.api:app --reload
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import PolicyRetriever, explain

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
app = FastAPI(title="FraudShield API", version="1.0")


class Transaction(BaseModel):
    features: dict[str, float] = Field(..., description="Time, V1..V28, Amount")


@lru_cache(maxsize=1)
def _state():
    if not (MODELS_DIR / "model.joblib").exists():
        raise HTTPException(503, "Model not trained yet. Run: python -m src.train")
    model = joblib.load(MODELS_DIR / "model.joblib")
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    return model, meta


@lru_cache(maxsize=1)
def _retriever():
    return PolicyRetriever()


def _drivers(model, meta, row: pd.DataFrame, k: int = 5):
    """Top-k feature contributions: SHAP for tree models, coef*scaled value for the linear pipeline."""
    feats = meta["features"]
    if meta["best_model"] == "xgboost":
        import shap
        contrib = shap.TreeExplainer(model).shap_values(row)[0]
    else:
        scaler, lr = model.steps[0][1], model.steps[1][1]
        contrib = scaler.transform(row)[0] * lr.coef_[0]
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"feature": feats[i], "impact": float(contrib[i])} for i in order]


def _score(tx: Transaction):
    model, meta = _state()
    missing = [f for f in meta["features"] if f not in tx.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing[:5]}...")
    row = pd.DataFrame([[tx.features[f] for f in meta["features"]]], columns=meta["features"])
    prob = float(model.predict_proba(row)[0, 1])
    return model, meta, row, prob, prob >= meta["threshold"]


@app.get("/health")
def health():
    meta = json.loads((MODELS_DIR / "meta.json").read_text()) if (MODELS_DIR / "meta.json").exists() else None
    return {"status": "ok", "model_loaded": meta is not None,
            "best_model": meta and meta["best_model"], "data_source": meta and meta["data_source"]}


@app.post("/score")
def score(tx: Transaction):
    model, meta, row, prob, flagged = _score(tx)
    return {"fraud_probability": prob, "flagged": flagged, "threshold": meta["threshold"],
            "top_drivers": _drivers(model, meta, row)}


@app.post("/explain")
def explain_tx(tx: Transaction):
    model, meta, row, prob, flagged = _score(tx)
    drivers = _drivers(model, meta, row)
    out = explain(_retriever(), prob, flagged, drivers, float(tx.features.get("Amount", 0)))
    return {"fraud_probability": prob, "flagged": flagged, "top_drivers": drivers, **out}
