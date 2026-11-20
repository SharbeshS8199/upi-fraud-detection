"""Replay the test period as a live stream through the scorer and measure latency and outcomes.

    python -m fraud.simulate --n 2000
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np

from fraud.data import TARGET, synthetic_upi, time_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fraud_model.joblib")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    model = joblib.load(args.model)
    _, _, test = time_split(synthetic_upi())
    stream = test.head(args.n)
    lat, decisions = [], {"allow": 0, "review": 0, "block": 0}
    caught = frauds = 0
    for _, row in stream.iterrows():
        txn = {k: (v.item() if hasattr(v, "item") else v) for k, v in row[model.features].items()}
        t0 = time.perf_counter()
        res = model.score(txn)
        lat.append((time.perf_counter() - t0) * 1000)
        decisions[res["decision"]] += 1
        frauds += int(row[TARGET])
        caught += int(row[TARGET] and res["decision"] != "allow")
    lat = np.array(lat)
    print(f"{len(stream)} transactions | p50 {np.median(lat):.1f} ms | p99 {np.percentile(lat, 99):.1f} ms | "
          f"~{1000 / lat.mean():.0f} txn/s single-threaded")
    print(f"decisions {decisions} | frauds {frauds}, stopped or reviewed {caught}")


if __name__ == "__main__":
    main()
