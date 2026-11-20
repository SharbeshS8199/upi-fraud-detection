"""Data sources.

1. `synthetic_upi()` — a realistic synthetic UPI transaction stream (users with spending habits,
   devices, payees, velocity, SIM swaps, mule accounts). Real UPI data is never public, so this is
   what the repo runs on out of the box. Fraud patterns follow publicly described UPI scams:
   account takeover after a SIM swap, "collect request" scams, mule accounts, rapid-fire draining.
2. `load_creditcard()` — the public Kaggle ULB credit-card dataset (284,807 rows, 0.17% fraud,
   PCA features V1..V28), for numbers comparable with published work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["txn_type", "merchant_category", "payer_bank", "channel"]
NUMERIC = ["amount", "hour", "is_weekend", "amount_to_avg_30d", "txns_last_1h", "txns_last_24h",
           "is_new_payee", "payee_account_age_days", "payee_fraud_reports", "is_new_device",
           "device_age_days", "sim_swap_last_48h", "failed_pin_24h", "distance_from_home_km",
           "is_collect_request"]
TARGET = "is_fraud"


def synthetic_upi(n: int = 200_000, fraud_rate: float = 0.003, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_users = max(1000, n // 40)
    user_avg = rng.lognormal(mean=6.3, sigma=0.9, size=n_users)          # typical ticket ~₹550
    user = rng.integers(0, n_users, n)
    avg = user_avg[user]

    ts = np.sort(rng.uniform(0, 90 * 24 * 3600, n))                       # 90 days, ordered in time
    hour = ((ts / 3600) % 24).astype(int)
    hour = np.where(rng.random(n) < 0.85, np.clip(rng.normal(14, 4, n), 6, 23).astype(int), hour)
    dow = ((ts // 86400) % 7).astype(int)

    df = pd.DataFrame({
        "txn_id": [f"T{i:08d}" for i in range(n)],
        "timestamp": pd.Timestamp("2026-07-01") + pd.to_timedelta(ts, unit="s"),
        "user_id": user,
        "txn_type": rng.choice(["P2P", "P2M"], n, p=[0.42, 0.58]),
        "merchant_category": rng.choice(["grocery", "food", "fuel", "utilities", "ecommerce", "travel", "gaming",
                                         "none"], n, p=[.2, .16, .08, .1, .16, .05, .03, .22]),
        "payer_bank": rng.choice(["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "Other"], n,
                                 p=[.3, .2, .16, .1, .07, .07, .1]),
        "channel": rng.choice(["app_qr", "app_vpa", "app_contact", "collect"], n, p=[.45, .25, .25, .05]),
        "amount": np.round(avg * rng.lognormal(0, 0.6, n), 2),
        "hour": hour,
        "is_weekend": (dow >= 5).astype(int),
        "txns_last_1h": rng.poisson(0.4, n),
        "txns_last_24h": rng.poisson(3, n),
        "is_new_payee": (rng.random(n) < 0.18).astype(int),
        "payee_account_age_days": np.clip(rng.lognormal(6.5, 1.0, n), 1, 5000).astype(int),
        "payee_fraud_reports": rng.poisson(0.02, n),
        "is_new_device": (rng.random(n) < 0.02).astype(int),
        "device_age_days": np.clip(rng.lognormal(5.5, 1.0, n), 0, 3000).astype(int),
        "sim_swap_last_48h": (rng.random(n) < 0.002).astype(int),
        "failed_pin_24h": rng.poisson(0.05, n),
        "distance_from_home_km": np.round(rng.exponential(6, n), 1),
    })
    # ---- inject fraud. Each scam leaves only *some* of its signals, and amounts are often ordinary,
    # so classes overlap the way real fraud data does.
    n_fraud = int(n * fraud_rate)
    idx = rng.choice(n, n_fraud, replace=False)
    pattern = rng.choice(["ato_sim_swap", "collect_scam", "mule_payee", "rapid_drain"], n_fraud, p=[.25, .3, .25, .2])
    f = df.loc[idx].copy()
    mult = np.where(rng.random(n_fraud) < 0.45, rng.uniform(0.7, 2.5, n_fraud), rng.uniform(2, 12, n_fraud))
    some = lambda k, prob: rng.random(k) < prob
    for p_name in np.unique(pattern):
        mask = pattern == p_name
        m = f.index[mask]
        k = len(m)
        if p_name == "ato_sim_swap":            # account takeover after SIM swap
            f.loc[m, "sim_swap_last_48h"] = some(k, 0.45).astype(int)
            f.loc[m, "is_new_device"] = some(k, 0.55).astype(int)
            f.loc[m, "device_age_days"] = np.where(some(k, 0.5), rng.integers(0, 5, k), f.loc[m, "device_age_days"])
            f.loc[m, "failed_pin_24h"] = rng.poisson(0.8, k)
            f.loc[m, "hour"] = np.where(some(k, 0.4), rng.choice([0, 1, 2, 3, 4, 23], k), f.loc[m, "hour"])
        elif p_name == "collect_scam":          # victim approves a fake "collect" request
            is_c = some(k, 0.7)
            f.loc[m, "channel"] = np.where(is_c, "collect", f.loc[m, "channel"])
            f.loc[m, "is_new_payee"] = some(k, 0.6).astype(int)
            f.loc[m, "txn_type"] = np.where(some(k, 0.8), "P2P", f.loc[m, "txn_type"])
        elif p_name == "mule_payee":            # money sent to a recently opened mule account
            f.loc[m, "payee_account_age_days"] = np.where(some(k, 0.6), rng.integers(1, 60, k),
                                                          f.loc[m, "payee_account_age_days"])
            f.loc[m, "payee_fraud_reports"] = rng.poisson(0.5, k)
            f.loc[m, "is_new_payee"] = some(k, 0.7).astype(int)
        else:                                   # rapid-fire draining
            f.loc[m, "txns_last_1h"] = rng.poisson(3, k) + some(k, 0.6) * 2
            f.loc[m, "txns_last_24h"] = rng.poisson(8, k) + 2
            f.loc[m, "distance_from_home_km"] = np.where(some(k, 0.5), np.round(rng.exponential(120, k), 1),
                                                         f.loc[m, "distance_from_home_km"])
        f.loc[m, "amount"] = np.round(f.loc[m, "amount"].to_numpy() * mult[mask], 2)
    df.loc[idx] = f
    df["is_collect_request"] = (df["channel"] == "collect").astype(int)
    df[TARGET] = 0
    df.loc[idx, TARGET] = 1

    # ---- benign look-alikes: genuine users who trip the same signals
    legit = np.setdiff1d(np.arange(n), idx)
    def pick(frac):
        return rng.choice(legit, int(frac * n), replace=False)
    new_phone = pick(0.004)                                   # changed phone / re-issued SIM
    df.loc[new_phone, ["is_new_device", "device_age_days"]] = [1, 0]
    df.loc[new_phone[: len(new_phone) // 3], "sim_swap_last_48h"] = 1
    new_shop = pick(0.006)                                    # paying a new small merchant
    df.loc[new_shop, "is_new_payee"] = 1
    df.loc[new_shop, "payee_account_age_days"] = rng.integers(5, 90, len(new_shop))
    big_buy = pick(0.006)                                     # rent, fees, gadgets
    df.loc[big_buy, "amount"] = df.loc[big_buy, "amount"] * rng.uniform(3, 15, len(big_buy))
    night = pick(0.01)
    df.loc[night, "hour"] = rng.choice([0, 1, 2, 23], len(night))
    travel = pick(0.004)
    df.loc[travel, "distance_from_home_km"] = np.round(rng.exponential(200, len(travel)), 1)
    busy = pick(0.004)
    df.loc[busy, "txns_last_1h"] = rng.poisson(4, len(busy))

    df["amount_to_avg_30d"] = np.round(df["amount"] / avg, 3)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_creditcard(path: str | Path = "data/creditcard.csv") -> pd.DataFrame:
    """Kaggle 'mlg-ulb/creditcardfraud'. Renames Class -> is_fraud and orders by Time."""
    df = pd.read_csv(path).rename(columns={"Class": TARGET})
    df["timestamp"] = pd.Timestamp("2013-09-01") + pd.to_timedelta(df["Time"], unit="s")
    return df.sort_values("timestamp").reset_index(drop=True)


def time_split(df: pd.DataFrame, val: float = 0.15, test: float = 0.15) -> tuple[pd.DataFrame, ...]:
    """Chronological split: train on the past, validate on the next period, test on the latest.
    Random splits leak future behaviour into training and inflate fraud metrics."""
    n = len(df)
    a, b = int(n * (1 - val - test)), int(n * (1 - test))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
