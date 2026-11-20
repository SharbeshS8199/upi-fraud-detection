"""Real-time scoring API.   uvicorn app.api:app --port 8000

POST /score with a transaction -> fraud probability, decision (allow / review / block) and the top
reasons in plain English.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud import __author__  # noqa: E402

MODEL_PATH = os.environ.get("FRAUD_MODEL", str(ROOT / "models" / "fraud_model.joblib"))
model = joblib.load(MODEL_PATH)
app = FastAPI(title="UPI fraud scoring", version="0.1.0",
              contact={"name": __author__, "url": "https://github.com/SharbeshS8199"})


class Transaction(BaseModel):
    amount: float = Field(gt=0, examples=[18500])
    hour: int = Field(ge=0, le=23, examples=[2])
    is_weekend: int = Field(0, ge=0, le=1)
    txn_type: Literal["P2P", "P2M"] = "P2P"
    merchant_category: str = "none"
    payer_bank: str = "SBI"
    channel: Literal["app_qr", "app_vpa", "app_contact", "collect"] = "app_vpa"
    amount_to_avg_30d: float = Field(1.0, ge=0, examples=[14.2])
    txns_last_1h: int = Field(0, ge=0)
    txns_last_24h: int = Field(2, ge=0)
    is_new_payee: int = Field(0, ge=0, le=1, examples=[1])
    payee_account_age_days: int = Field(700, ge=0, examples=[12])
    payee_fraud_reports: int = Field(0, ge=0)
    is_new_device: int = Field(0, ge=0, le=1, examples=[1])
    device_age_days: int = Field(300, ge=0, examples=[0])
    sim_swap_last_48h: int = Field(0, ge=0, le=1, examples=[1])
    failed_pin_24h: int = Field(0, ge=0)
    distance_from_home_km: float = Field(5.0, ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "model": model.name, "thresholds": {"review": model.t_review, "block": model.t_block},
            "author": __author__}


@app.post("/score")
def score(txn: Transaction):
    t0 = time.perf_counter()
    d = txn.model_dump()
    d["is_collect_request"] = int(d["channel"] == "collect")
    res = model.score(d)
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return res
