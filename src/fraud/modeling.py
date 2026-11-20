"""Features, models, imbalance strategies, cost-based thresholds and explanations."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud.data import CATEGORICAL, NUMERIC, TARGET

HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SHAP = importlib.util.find_spec("shap") is not None
SKEWED = {"amount", "amount_to_avg_30d", "payee_account_age_days", "device_age_days", "distance_from_home_km", "Amount"}


def feature_spec(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if all(c in df.columns for c in NUMERIC):
        return list(NUMERIC), list(CATEGORICAL)
    num = [c for c in df.columns if c.startswith("V")] + ["Amount"]      # Kaggle credit-card dataset
    return num, []


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    skew = [c for c in num if c in SKEWED]
    rest = [c for c in num if c not in SKEWED]
    parts = [("skewed", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                  ("scale", StandardScaler())]), skew),
             ("num", StandardScaler(), rest)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(parts, sparse_threshold=0.0)


# --------------------------------------------------------------------------- imbalance

def random_oversample(X: np.ndarray, y: np.ndarray, ratio: float, rng: np.random.Generator):
    pos = np.flatnonzero(y == 1)
    target = int(ratio * (y == 0).sum())
    extra = rng.choice(pos, max(0, target - len(pos)), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    return X[idx], y[idx]


def smote(X: np.ndarray, y: np.ndarray, ratio: float, rng: np.random.Generator, k: int = 5):
    """SMOTE (Chawla et al. 2002): new minority points on segments to minority neighbours.
    Uses imbalanced-learn when installed; this is the same algorithm in ~10 lines."""
    if importlib.util.find_spec("imblearn") is not None:
        from imblearn.over_sampling import SMOTE

        return SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=int(rng.integers(1e9))).fit_resample(X, y)
    P = X[y == 1]
    need = max(0, int(ratio * (y == 0).sum()) - len(P))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(P))).fit(P)
    neigh = nn.kneighbors(P, return_distance=False)[:, 1:]
    base = rng.integers(0, len(P), need)
    mate = neigh[base, rng.integers(0, neigh.shape[1], need)]
    synth = P[base] + rng.random((need, 1)) * (P[mate] - P[base])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(need, dtype=y.dtype)])


# --------------------------------------------------------------------------- models

def make_estimator(kind: str, class_weight: bool, pos_weight: float, seed: int = 0):
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced" if class_weight else None)
    if kind == "gbdt":
        if HAS_XGB:
            from xgboost import XGBClassifier

            return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=2, eval_metric="aucpr",
                                 tree_method="hist", scale_pos_weight=pos_weight if class_weight else 1.0,
                                 random_state=seed, n_jobs=-1)
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                              l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                              class_weight="balanced" if class_weight else None, random_state=seed)
    if kind == "mlp":
        if HAS_TORCH:
            from fraud.torch_mlp import TorchMLP

            return TorchMLP(focal=class_weight, seed=seed)
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512, learning_rate_init=1e-3,
                             early_stopping=True, max_iter=60, random_state=seed)
    raise ValueError(kind)


def estimator_name(kind: str) -> str:
    return {"logreg": "Logistic regression",
            "gbdt": "XGBoost" if HAS_XGB else "Gradient boosting (sklearn HistGB)",
            "mlp": "Neural net (PyTorch MLP)" if HAS_TORCH else "Neural net (sklearn MLP)"}[kind]


# --------------------------------------------------------------------------- thresholds

def cost_threshold(y: np.ndarray, p: np.ndarray, amount: np.ndarray, review_cost: float = 30.0) -> tuple[float, float]:
    """Threshold maximising net savings = fraud amount caught - review_cost x flagged transactions."""
    order = np.argsort(-p)
    caught = np.cumsum(amount[order] * y[order])
    flagged = np.arange(1, len(p) + 1)
    net = caught - review_cost * flagged
    best = int(np.argmax(net))
    return float(p[order][best]), float(net[best])


def precision_threshold(y: np.ndarray, p: np.ndarray, min_precision: float = 0.9) -> float:
    """Lowest threshold whose precision >= min_precision (used for auto-block)."""
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    prec = tp / np.arange(1, len(p) + 1)
    ok = np.flatnonzero(prec >= min_precision)
    ok = ok[ok >= 4] if len(ok[ok >= 4]) else ok        # don't trust precision of the first 1-4 alerts
    return float(p[order][ok[-1]]) if len(ok) else 1.01


# --------------------------------------------------------------------------- the deployable model

REASONS = {
    "sim_swap_last_48h": "SIM card was swapped in the last 48 hours",
    "is_new_device": "payment from a new device",
    "device_age_days": "device first seen very recently",
    "failed_pin_24h": "several wrong UPI PIN attempts today",
    "is_collect_request": "it is a collect (payment request) approval",
    "channel": "unusual payment channel",
    "is_new_payee": "first payment to this payee",
    "payee_account_age_days": "payee account was opened very recently",
    "payee_fraud_reports": "payee has prior fraud reports",
    "amount": "unusually large amount",
    "amount_to_avg_30d": "amount is far above this user's usual spend",
    "txns_last_1h": "many transactions in the last hour",
    "txns_last_24h": "many transactions in the last 24 hours",
    "hour": "unusual time of day",
    "distance_from_home_km": "far from the user's usual location",
    "txn_type": "transaction type", "merchant_category": "merchant category", "payer_bank": "payer bank",
    "is_weekend": "weekend timing",
}


@dataclass
class FraudModel:
    pipeline: Pipeline
    num: list[str]
    cat: list[str]
    t_review: float
    t_block: float
    reference: dict = field(default_factory=dict)      # typical (median/mode) legit values, for explanations
    name: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return self.num + self.cat

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[self.features])[:, 1]

    def decide(self, p: float) -> str:
        return "block" if p >= self.t_block else "review" if p >= self.t_review else "allow"

    def explain(self, row: pd.DataFrame, top: int = 3) -> list[dict]:
        """Per-feature contribution in log-odds: how much the score drops if this feature took its
        typical legitimate value. Model-agnostic, works on raw features (so reasons read naturally).
        `fraud.explain.shap_explain` gives the SHAP equivalent for tree models."""
        logit = lambda q: np.log(np.clip(q, 1e-9, 1 - 1e-9) / np.clip(1 - q, 1e-9, 1))
        # one batched call: row 0 is the original, row i+1 has feature i set to its typical value
        batch = pd.concat([row] * (len(self.features) + 1), ignore_index=True)
        batch[self.num] = batch[self.num].astype(float)
        for i, f in enumerate(self.features, start=1):
            batch.loc[i, f] = self.reference[f]
        lo = logit(self.predict_proba(batch))
        out = []
        for i, f in enumerate(self.features, start=1):
            v = row[f].iloc[0]
            out.append({"feature": f, "value": v.item() if hasattr(v, "item") else v,
                        "contribution": round(float(lo[0] - lo[i]), 3), "reason": REASONS.get(f, f)})
        out.sort(key=lambda d: -d["contribution"])
        return [d for d in out[:top] if d["contribution"] > 0.05]

    def score(self, txn: dict) -> dict:
        row = pd.DataFrame([txn])
        for f in self.features:
            if f not in row:
                row[f] = self.reference[f]
        p = float(self.predict_proba(row)[0])
        d = self.decide(p)
        return {"fraud_probability": round(p, 5), "decision": d,
                "reasons": self.explain(row) if d != "allow" else []}


def reference_values(df: pd.DataFrame, num: list[str], cat: list[str]) -> dict:
    legit = df[df[TARGET] == 0]
    ref = {c: float(legit[c].median()) for c in num}
    ref.update({c: legit[c].mode().iloc[0] for c in cat})
    return ref
