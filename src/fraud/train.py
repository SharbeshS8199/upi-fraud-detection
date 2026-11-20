"""Run the model comparison, pick the best, tune thresholds on validation, save model + report.

    python -m fraud.train                         # synthetic UPI data (default, runs anywhere)
    python -m fraud.train --data creditcard       # Kaggle credit-card dataset at data/creditcard.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

from fraud.data import TARGET, load_creditcard, synthetic_upi, time_split
from fraud.modeling import (FraudModel, cost_threshold, estimator_name, feature_spec, make_estimator,
                            make_preprocessor, precision_threshold, random_oversample, reference_values, smote)

EXPERIMENTS = [
    ("logreg", "class_weight"),
    ("gbdt", "none"),
    ("gbdt", "class_weight"),
    ("gbdt", "oversample"),
    ("gbdt", "smote"),
    ("mlp", "oversample"),
    ("mlp", "smote"),
]


def fit_one(kind, strategy, Xtr, ytr, seed=0):
    rng = np.random.default_rng(seed)
    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    if strategy == "oversample":
        Xtr, ytr = random_oversample(Xtr, ytr, 0.1, rng)
    elif strategy == "smote":
        Xtr, ytr = smote(Xtr, ytr, 0.1, rng)
    est = make_estimator(kind, class_weight=strategy == "class_weight", pos_weight=pos_weight, seed=seed)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    return est, time.perf_counter() - t0


def evaluate(y, p, amount, t_review, review_cost):
    flag = p >= t_review
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    caught = float(amount[flag & (y == 1)].sum())
    total = float(amount[y == 1].sum())
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, int((y == 1).sum())), 4),
        "flag_rate": round(float(flag.mean()), 5),
        "fraud_amount_caught_pct": round(caught / max(total, 1e-9), 4),
        "net_savings": round(caught - review_cost * int(flag.sum()), 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["upi", "creditcard"], default="upi")
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--review-cost", type=float, default=30.0, help="₹ cost of manually reviewing one alert")
    ap.add_argument("--out", default="models")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    df = synthetic_upi(args.rows) if args.data == "upi" else load_creditcard()
    amount_col = "amount" if "amount" in df else "Amount"
    num, cat = feature_spec(df)
    train, val, test = time_split(df)
    print(f"rows: train {len(train)} / val {len(val)} / test {len(test)}; fraud rate {df[TARGET].mean():.4%}")

    pre = make_preprocessor(num, cat).fit(train[num + cat])
    Xtr, Xva, Xte = (pre.transform(d[num + cat]) for d in (train, val, test))
    ytr, yva, yte = (d[TARGET].to_numpy() for d in (train, val, test))
    amt_va, amt_te = val[amount_col].to_numpy(), test[amount_col].to_numpy()

    results, fitted, curves = [], {}, {}
    for kind, strategy in EXPERIMENTS:
        est, secs = fit_one(kind, strategy, Xtr, ytr)
        p_va, p_te = est.predict_proba(Xva)[:, 1], est.predict_proba(Xte)[:, 1]
        t_rev, _ = cost_threshold(yva, p_va, amt_va, args.review_cost)
        row = {"model": estimator_name(kind), "kind": kind, "imbalance": strategy, "train_sec": round(secs, 1),
               "val_pr_auc": round(float(average_precision_score(yva, p_va)), 4),
               **{f"test_{k}": v for k, v in evaluate(yte, p_te, amt_te, t_rev, args.review_cost).items()}}
        results.append(row)
        fitted[(kind, strategy)] = (est, p_va, p_te)
        curves[f"{row['model']} / {strategy}"] = precision_recall_curve(yte, p_te)[:2]
        print(f"{row['model']:38s} {strategy:12s} val PR-AUC {row['val_pr_auc']:.3f}  "
              f"test PR-AUC {row['test_pr_auc']:.3f}  recall {row['test_recall']:.3f}  precision {row['test_precision']:.3f}")

    best = max(results, key=lambda r: r["val_pr_auc"])          # selected on validation, not test
    est, p_va, p_te = fitted[(best["kind"], best["imbalance"])]
    t_review, _ = cost_threshold(yva, p_va, amt_va, args.review_cost)
    t_block = max(precision_threshold(yva, p_va, 0.9), t_review)
    model = FraudModel(Pipeline([("pre", pre), ("clf", est)]), num, cat, t_review, t_block,
                       reference_values(train, num, cat), name=f"{best['model']} + {best['imbalance']}",
                       metrics={k: v for k, v in best.items() if k.startswith("test_")})

    out, rep = Path(args.out), Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "fraud_model.joblib")

    # decisions on test with both thresholds
    dec = np.where(p_te >= t_block, "block", np.where(p_te >= t_review, "review", "allow"))
    tiers = {d: {"n": int((dec == d).sum()), "fraud": int(((dec == d) & (yte == 1)).sum())} for d in ("block", "review", "allow")}

    # explanations for a few caught frauds + global importance from local attributions
    caught_idx = np.flatnonzero((dec != "allow") & (yte == 1))[:150]
    agg: dict[str, float] = {}
    examples = []
    for j, i in enumerate(caught_idx):
        row = test.iloc[[i]]
        reasons = model.explain(row, top=len(model.features))
        for r in reasons:
            agg[r["feature"]] = agg.get(r["feature"], 0.0) + r["contribution"] / len(caught_idx)
        if j < 3:
            examples.append({"txn_id": row["txn_id"].iloc[0] if "txn_id" in row else int(i),
                             "p": round(float(p_te[i]), 4), "reasons": [r["reason"] for r in reasons[:3]]})
    importance = sorted(agg.items(), key=lambda kv: -kv[1])[:10]

    summary = {"data": args.data, "rows": len(df), "fraud_rate": round(float(df[TARGET].mean()), 5),
               "split": {"train": len(train), "val": len(val), "test": len(test)},
               "review_cost": args.review_cost, "results": results, "selected": model.name,
               "thresholds": {"review": round(t_review, 5), "block": round(t_block, 5)}, "tiers_on_test": tiers,
               "importance": importance, "examples": examples}
    (rep / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(summary, rep)
    plot(curves, importance, rep)
    print(f"\nselected: {model.name}  thresholds review={t_review:.4f} block={t_block:.4f}")
    print(f"saved {out / 'fraud_model.joblib'} and {rep / 'REPORT.md'}")


def write_report(s: dict, rep: Path) -> None:
    L = ["# Fraud detection results", "",
         f"Data: **{s['data']}**, {s['rows']:,} transactions, fraud rate {s['fraud_rate']:.3%}. "
         f"Chronological split train/val/test = {s['split']['train']:,}/{s['split']['val']:,}/{s['split']['test']:,}. "
         f"Review threshold chosen on validation to maximise net savings (₹{s['review_cost']:.0f} per manual review).", "",
         "| Model | Imbalance handling | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Precision | Recall | Fraud ₹ caught | Alerts per 1k txns | Train s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["results"]:
        L.append(f"| {r['model']} | {r['imbalance']} | {r['val_pr_auc']:.3f} | {r['test_pr_auc']:.3f} | {r['test_roc_auc']:.3f} | "
                 f"{r['test_precision']:.3f} | {r['test_recall']:.3f} | {r['test_fraud_amount_caught_pct']:.1%} | "
                 f"{r['test_flag_rate'] * 1000:.1f} | {r['train_sec']} |")
    t = s["tiers_on_test"]
    L += ["", f"**Selected (best validation PR-AUC):** {s['selected']}", "",
          "Decision tiers on the test period:", "",
          "| Decision | Transactions | of which fraud |", "|---|---|---|"]
    L += [f"| {d} | {t[d]['n']:,} | {t[d]['fraud']:,} |" for d in ("block", "review", "allow")]
    L += ["", "## What drives the alerts", "", "Mean log-odds contribution on caught frauds:", "",
          "| Feature | Contribution |", "|---|---|"] + [f"| {f} | {v:.2f} |" for f, v in s["importance"]]
    L += ["", "## Example explanations", ""]
    L += [f"- `{e['txn_id']}` (p={e['p']}): " + "; ".join(e["reasons"]) for e in s["examples"]]
    L += ["", "![PR curves](pr_curves.png)", "", "![Importance](importance.png)", ""]
    (rep / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def plot(curves: dict, importance: list, rep: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (prec, rec) in curves.items():
        ax.plot(rec, prec, label=name, lw=1.5)
    ax.set_xlabel("Recall"), ax.set_ylabel("Precision"), ax.set_title("Precision–recall on the test period")
    ax.legend(fontsize=7), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(rep / "pr_curves.png", dpi=130), plt.close(fig)
    if importance:
        fig, ax = plt.subplots(figsize=(7, 4))
        names, vals = zip(*importance[::-1])
        ax.barh(names, vals, color="#2a6f97")
        ax.set_xlabel("mean log-odds contribution on caught frauds"), ax.set_title("Top fraud signals")
        fig.tight_layout(), fig.savefig(rep / "importance.png", dpi=130), plt.close(fig)


if __name__ == "__main__":
    main()
