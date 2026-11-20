"""Fraud monitoring dashboard.   streamlit run dashboard/app.py"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "dashboard")]
import ui  # noqa: E402
from fraud.dashboard_data import (at_threshold, comparison_table, hourly_volume, load_results,  # noqa: E402
                                  load_scores, threshold_curve)

ui.page("UPI Fraud Monitor", "Imbalanced-data fraud detection · cost-aware thresholds · explainable alerts", "🛡️")
REPORTS, MODEL = ROOT / "reports", ROOT / "models" / "fraud_model.joblib"


@st.cache_resource
def load_model():
    import joblib

    return joblib.load(MODEL) if MODEL.exists() else None


results, scores, model = load_results(REPORTS), load_scores(REPORTS), load_model()
if results is None or scores is None or model is None:
    ui.empty_state("No trained model yet",
                   "Training compares 7 model / imbalance-handling combinations on 200k synthetic UPI transactions "
                   "and takes about a minute on a laptop.", "python -m fraud.train")
    if st.button("Train now", type="primary"):
        with st.spinner("Training 7 models…"):
            subprocess.run([sys.executable, "-m", "fraud.train"], cwd=ROOT, check=True,
                           env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
        st.cache_resource.clear()
        st.rerun()
    ui.footer()
    st.stop()

cmp = comparison_table(results)
best = cmp.loc[cmp["val_pr_auc"].idxmax()]
y, amt, p = scores["is_fraud"].to_numpy(), scores["amount"].to_numpy(), scores["p::selected"].to_numpy()
tiers = results["tiers_on_test"]
blocked_prec = tiers["block"]["fraud"] / max(1, tiers["block"]["n"])
ui.kpis([
    ("Selected model", best["variant"].split(" · ")[0].replace("Gradient boosting", "Grad. boosting"), best["imbalance"]),
    ("Test PR-AUC", f"{best['test_pr_auc']:.3f}", f"fraud rate {results['fraud_rate']:.2%}"),
    ("Fraud ₹ caught", f"{best['test_fraud_amount_caught_pct']:.0%}", "review + block tiers"),
    ("Auto-block precision", f"{blocked_prec:.0%}", f"{tiers['block']['fraud']}/{tiers['block']['n']} blocks were fraud"),
    ("Alerts", f"{best['test_flag_rate'] * 1000:.1f} / 1k", "transactions sent to review"),
])
st.write("")
t_overview, t_threshold, t_feed, t_score, t_explain = st.tabs(
    ["📊 Model comparison", "🎚️ Threshold explorer", "📡 Live feed", "🔍 Score a transaction", "💡 Why alerts fire"])

# ---------------------------------------------------------------- model comparison
with t_overview:
    c1, c2 = st.columns([3, 2])
    fig = px.bar(cmp.sort_values("test_pr_auc"), x="test_pr_auc", y="variant", orientation="h", color="model",
                 text="test_pr_auc", title="Test PR-AUC (higher is better; random = fraud rate)")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    c1.plotly_chart(ui.style_fig(fig, 380), use_container_width=True)
    fig2 = px.scatter(cmp, x="test_flag_rate", y="test_recall", color="model", symbol="imbalance", size="test_pr_auc",
                      hover_name="variant", title="Recall vs alert rate at each model's cost-optimal threshold",
                      labels={"test_flag_rate": "share of transactions flagged", "test_recall": "fraud recall"})
    c2.plotly_chart(ui.style_fig(fig2, 380), use_container_width=True)
    pr_rows = []
    from sklearn.metrics import precision_recall_curve

    for col in [c for c in scores.columns if c.startswith("p::") and c != "p::selected"]:
        prec, rec, _ = precision_recall_curve(y, scores[col])
        keep = np.linspace(0, len(prec) - 1, min(len(prec), 300)).astype(int)
        pr_rows.append(pd.DataFrame({"recall": rec[keep], "precision": prec[keep], "variant": col[3:]}))
    pr = pd.concat(pr_rows)
    fig3 = px.line(pr, x="recall", y="precision", color="variant", title="Precision–recall curves (test period)")
    st.plotly_chart(ui.style_fig(fig3, 420), use_container_width=True)
    st.dataframe(cmp[["model", "imbalance", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_precision", "test_recall",
                      "test_fraud_amount_caught_pct", "train_sec"]].sort_values("val_pr_auc", ascending=False),
                 hide_index=True, use_container_width=True)
    st.caption("Model selected on **validation** PR-AUC, never on test. ROC-AUC barely separates the models; "
               "PR-AUC differs by more than 2×, which is why it is the headline metric under heavy imbalance.")

# ---------------------------------------------------------------- threshold explorer
with t_threshold:
    c1, c2 = st.columns([1, 3])
    cost = c1.number_input("Cost of one manual review (₹)", min_value=1.0, max_value=1000.0, value=float(results["review_cost"]))
    curve = threshold_curve(y, p, amt, cost)
    best_i = int(curve["net_savings"].idxmax())
    pct = c1.slider("Review threshold (percentile of scores)", 90.0, 99.99, 99.0, 0.01,
                    help="Flag every transaction whose score is above this percentile")
    t = float(np.quantile(p, pct / 100))
    m = at_threshold(y, p, amt, t, cost)
    c1.markdown(f"**Threshold** `{t:.4f}`  \n**Cost-optimal** `{curve.loc[best_i, 'threshold']:.4f}`")
    with c2:
        ui.kpis([("Precision", f"{m['precision']:.1%}", "alerts that are fraud"),
                 ("Recall", f"{m['recall']:.1%}", f"{m['missed']} frauds missed"),
                 ("Alerts", f"{m['alerts']:,}", f"of {len(y):,} transactions"),
                 ("Net savings", f"₹{m['net_savings']:,.0f}", "caught − review cost")])
        long = curve.melt(id_vars="threshold", value_vars=["precision", "recall", "fraud_amount_caught_pct"],
                          var_name="metric", value_name="value")
        f1 = px.line(long, x="threshold", y="value", color="metric", log_x=True, title="Trade-off as the threshold moves")
        f1.add_vline(x=t, line_dash="dash", line_color=ui.MUTED)
        st.plotly_chart(ui.style_fig(f1, 300), use_container_width=True)
        f2 = px.area(curve, x="threshold", y="net_savings", log_x=True, title="Net ₹ savings (pick the peak)")
        f2.add_vline(x=t, line_dash="dash", line_color=ui.MUTED)
        st.plotly_chart(ui.style_fig(f2, 260), use_container_width=True)

# ---------------------------------------------------------------- live feed
with t_feed:
    c1, c2, c3 = st.columns([2, 2, 1])
    n = c1.slider("Transactions to replay", 200, 5000, 1000, 100)
    show = c2.multiselect("Show decisions", ["block", "review", "allow"], default=["block", "review"])
    feed = scores.tail(n).copy()
    feed["p"] = feed["p::selected"]
    feed["decision"] = np.where(feed["p"] >= model.t_block, "block", np.where(feed["p"] >= model.t_review, "review", "allow"))
    counts = feed["decision"].value_counts()
    c3.metric("Flagged", int(counts.get("block", 0) + counts.get("review", 0)), f"of {n}")
    vol = hourly_volume(scores, "p::selected", model.t_review, model.t_block)
    fv = px.bar(vol, x="day", y="transactions", color="decision", title="Alerts per day (test period)",
                color_discrete_map={"block": ui.BAD, "review": ui.WARN})
    st.plotly_chart(ui.style_fig(fv, 260), use_container_width=True)
    view = feed[feed["decision"].isin(show)].sort_values("timestamp", ascending=False)
    view["actually fraud"] = view["is_fraud"].map({1: "yes", 0: "no"})
    st.dataframe(view[[c for c in ["timestamp", "txn_id", "amount", "p", "decision", "actually fraud"] if c in view]],
                 hide_index=True, use_container_width=True, height=360,
                 column_config={"p": st.column_config.ProgressColumn("fraud score", min_value=0, max_value=1, format="%.3f"),
                                "amount": st.column_config.NumberColumn("amount (₹)", format="%.0f")})

# ---------------------------------------------------------------- score one
with t_score:
    with st.form("score"):
        a, b, c = st.columns(3)
        amount = a.number_input("Amount (₹)", min_value=1.0, value=18500.0)
        ratio = a.number_input("Amount ÷ user's 30-day average", min_value=0.0, value=14.0)
        hour = a.slider("Hour of day", 0, 23, 2)
        channel = b.selectbox("Channel", ["collect", "app_vpa", "app_qr", "app_contact"])
        new_payee = b.checkbox("First payment to this payee", True)
        payee_age = b.number_input("Payee account age (days)", min_value=0, value=12)
        reports_ = b.number_input("Fraud reports against payee", min_value=0, value=1)
        new_dev = c.checkbox("New device", True)
        sim = c.checkbox("SIM swapped in last 48 h", True)
        pins = c.number_input("Wrong PIN attempts today", min_value=0, value=2)
        tx1h = c.number_input("Transactions in last hour", min_value=0, value=3)
        go_ = st.form_submit_button("Score transaction", type="primary")
    txn = {**model.reference, "amount": amount, "amount_to_avg_30d": ratio, "hour": hour, "channel": channel,
           "is_collect_request": int(channel == "collect"), "is_new_payee": int(new_payee),
           "payee_account_age_days": payee_age, "payee_fraud_reports": reports_, "is_new_device": int(new_dev),
           "device_age_days": 0 if new_dev else 400, "sim_swap_last_48h": int(sim), "failed_pin_24h": pins,
           "txns_last_1h": tx1h, "txn_type": "P2P"}
    res = model.score(txn)
    colour = {"block": ui.BAD, "review": ui.WARN, "allow": ui.GOOD}[res["decision"]]
    st.markdown(f"### {ui.pill(res['decision'].upper(), colour)} &nbsp; fraud score **{res['fraud_probability']:.3f}**",
                unsafe_allow_html=True)
    if res["reasons"]:
        rdf = pd.DataFrame(res["reasons"])
        fr = px.bar(rdf.iloc[::-1], x="contribution", y="reason", orientation="h",
                    title="Top reasons (log-odds contribution)", color_discrete_sequence=[ui.BAD])
        st.plotly_chart(ui.style_fig(fr, 240), use_container_width=True)
    else:
        st.caption("Allowed: no strong risk signals.")

# ---------------------------------------------------------------- explanations
with t_explain:
    imp = pd.DataFrame(results["importance"], columns=["feature", "contribution"])
    fi = px.bar(imp.iloc[::-1], x="contribution", y="feature", orientation="h",
                title="What drives alerts: mean log-odds contribution on caught frauds")
    st.plotly_chart(ui.style_fig(fi, 380), use_container_width=True)
    for e in results["examples"]:
        st.markdown(f"- `{e['txn_id']}` score {e['p']}: " + "; ".join(e["reasons"]))
    if (REPORTS / "shap_summary.png").exists():
        st.image(str(REPORTS / "shap_summary.png"), caption="SHAP summary (python -m fraud.explain)")
    else:
        st.caption("Install `.[full]` and run `python -m fraud.explain` to add a SHAP summary plot here.")

ui.footer("Synthetic UPI data: patterns modelled on publicly reported scams")
