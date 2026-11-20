"""SHAP explanations (when `shap` is installed): global beeswarm + per-transaction values.

    python -m fraud.explain --n 2000

The API uses FraudModel.explain (a model-agnostic, raw-feature attribution) so it works without
shap; this script produces the SHAP plots for the README / interviews and lets you check the two
methods agree on the top drivers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from fraud.data import TARGET, synthetic_upi, time_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fraud_model.joblib")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    model = joblib.load(args.model)
    _, _, test = time_split(synthetic_upi())
    sample = test.sample(min(args.n, len(test)), random_state=0)
    sample = sample.sort_values(TARGET, ascending=False)            # keep all frauds in the sample
    pre, clf = model.pipeline.named_steps["pre"], model.pipeline.named_steps["clf"]
    X = pre.transform(sample[model.features])
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    try:
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(X)
    except Exception:                                                # non-tree model
        background = shap.sample(X, 200, random_state=0)
        explainer = shap.KernelExplainer(lambda z: clf.predict_proba(z)[:, 1], background)
        values = explainer.shap_values(X[:300])
        X = X[:300]
    values = values[1] if isinstance(values, list) else values
    shap.summary_plot(values, X, feature_names=names, show=False, max_display=12)
    Path(args.out).mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(args.out) / "shap_summary.png", dpi=130)
    top = np.argsort(-np.abs(values).mean(0))[:10]
    print("Top SHAP features:", [names[i] for i in top])


if __name__ == "__main__":
    main()
