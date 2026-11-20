"""Data helpers for the dashboard (pure pandas/numpy, unit-tested)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(reports: str | Path = "reports") -> dict | None:
    p = Path(reports) / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_scores(reports: str | Path = "reports") -> pd.DataFrame | None:
    p = Path(reports) / "test_scores.csv.gz"
    return pd.read_csv(p, parse_dates=["timestamp"]) if p.exists() else None


def comparison_table(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results["results"])
    df["variant"] = df["model"].str.replace(r" \(.*\)", "", regex=True) + " · " + df["imbalance"]
    return df


def threshold_curve(y: np.ndarray, p: np.ndarray, amount: np.ndarray, review_cost: float, n: int = 200) -> pd.DataFrame:
    """Precision, recall, alerts and net ₹ savings for a grid of thresholds (quantiles of the scores,
    so the grid is dense where the decisions actually change)."""
    qs = np.unique(np.quantile(p, np.linspace(0.90, 0.99999, n)))
    rows = []
    total_fraud_amt = float(amount[y == 1].sum()) or 1.0
    for t in qs:
        flag = p >= t
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        caught = float(amount[flag & (y == 1)].sum())
        rows.append({"threshold": float(t), "precision": tp / max(1, tp + fp), "recall": tp / max(1, int(y.sum())),
                     "alerts": int(flag.sum()), "alerts_per_1k": 1000 * flag.mean(),
                     "fraud_amount_caught_pct": caught / total_fraud_amt,
                     "net_savings": caught - review_cost * int(flag.sum())})
    return pd.DataFrame(rows)


def at_threshold(y, p, amount, t: float, review_cost: float) -> dict:
    flag = p >= t
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    caught = float(amount[flag & (y == 1)].sum())
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, int(y.sum())), "alerts": int(flag.sum()),
            "missed": int(((~flag) & (y == 1)).sum()), "net_savings": caught - review_cost * int(flag.sum()),
            "caught_amount": caught}


def hourly_volume(scores: pd.DataFrame, p_col: str, t_review: float, t_block: float) -> pd.DataFrame:
    d = scores.copy()
    d["decision"] = np.where(d[p_col] >= t_block, "block", np.where(d[p_col] >= t_review, "review", "allow"))
    d["day"] = d["timestamp"].dt.floor("D")
    g = d[d["decision"] != "allow"].groupby(["day", "decision"]).size().reset_index(name="transactions")
    return g
