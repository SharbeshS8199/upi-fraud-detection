import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from fraud.data import TARGET, synthetic_upi, time_split
from fraud.modeling import (FraudModel, cost_threshold, feature_spec, make_estimator, make_preprocessor,
                            precision_threshold, random_oversample, reference_values, smote)


@pytest.fixture(scope="module")
def data():
    return synthetic_upi(40_000, fraud_rate=0.01, seed=1)


@pytest.fixture(scope="module")
def model(data):
    num, cat = feature_spec(data)
    train, val, _ = time_split(data)
    pre = make_preprocessor(num, cat).fit(train[num + cat])
    clf = make_estimator("gbdt", class_weight=False, pos_weight=1.0).fit(pre.transform(train[num + cat]), train[TARGET])
    p_val = clf.predict_proba(pre.transform(val[num + cat]))[:, 1]
    t_rev, _ = cost_threshold(val[TARGET].to_numpy(), p_val, val["amount"].to_numpy())
    return FraudModel(Pipeline([("pre", pre), ("clf", clf)]), num, cat, t_rev,
                      max(t_rev, precision_threshold(val[TARGET].to_numpy(), p_val)), reference_values(train, num, cat))


def test_generator_rate_order_and_split(data):
    assert abs(data[TARGET].mean() - 0.01) < 1e-3
    assert data["timestamp"].is_monotonic_increasing
    tr, va, te = time_split(data)
    assert len(tr) + len(va) + len(te) == len(data)
    assert tr["timestamp"].max() <= va["timestamp"].min() <= te["timestamp"].min()   # no look-ahead


def test_resampling():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 4))
    y = (np.arange(1000) < 20).astype(int)
    Xo, yo = random_oversample(X, y, 0.25, rng)
    assert yo.sum() == int(0.25 * 980) and len(yo) == 980 + yo.sum()
    Xs, ys = smote(X, y, 0.25, rng)
    assert ys.sum() == int(0.25 * 980)
    new = Xs[len(X):]
    P = X[y == 1]
    assert (new >= P.min(0) - 1e-9).all() and (new <= P.max(0) + 1e-9).all()   # synthetic points stay inside minority hull box


def test_cost_threshold_picks_profitable_cutoff():
    y = np.array([1, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    amt = np.array([1000, 50, 500, 50, 50, 50])
    t, net = cost_threshold(y, p, amt, review_cost=30)
    assert t == 0.7 and net == 1500 - 90                      # flag the top 3: catch both frauds
    t2, _ = cost_threshold(y, p, amt, review_cost=600)
    assert t2 == 0.9                                          # reviews expensive -> only the surest
    assert precision_threshold(np.array([1, 1, 1, 1, 1, 0, 0]), np.linspace(1, 0.4, 7), 0.9) >= 0.5


def test_model_scores_scam_pattern_high_with_reasons(model, data):
    risky = dict(amount=25000, hour=2, is_weekend=0, txn_type="P2P", merchant_category="none", payer_bank="SBI",
                 channel="collect", is_collect_request=1, amount_to_avg_30d=20, txns_last_1h=4, txns_last_24h=12,
                 is_new_payee=1, payee_account_age_days=7, payee_fraud_reports=2, is_new_device=1,
                 device_age_days=0, sim_swap_last_48h=1, failed_pin_24h=3, distance_from_home_km=150)
    normal = {**model.reference}
    r, n = model.score(risky), model.score(normal)
    assert r["fraud_probability"] > 0.5 > n["fraud_probability"]
    assert r["decision"] in ("review", "block") and n["decision"] == "allow"
    assert r["reasons"] and all(x["contribution"] > 0 and x["reason"] for x in r["reasons"])
    assert n["reasons"] == []


def test_model_beats_random_on_holdout(model, data):
    from sklearn.metrics import average_precision_score

    _, _, test = time_split(data)
    ap = average_precision_score(test[TARGET], model.predict_proba(test))
    assert ap > 3 * test[TARGET].mean()
