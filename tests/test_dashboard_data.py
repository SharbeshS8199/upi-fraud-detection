import numpy as np

from fraud.dashboard_data import at_threshold, threshold_curve


def test_threshold_curve_is_monotone_and_consistent():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.01).astype(int)
    p = np.clip(rng.normal(0.1, 0.05, 5000) + 0.5 * y, 0, 1)
    amt = rng.uniform(100, 5000, 5000)
    c = threshold_curve(y, p, amt, review_cost=30)
    assert c["threshold"].is_monotonic_increasing
    assert c["alerts"].is_monotonic_decreasing and c["recall"].is_monotonic_decreasing
    t = float(c.loc[c["net_savings"].idxmax(), "threshold"])
    m = at_threshold(y, p, amt, t, 30)
    assert abs(m["net_savings"] - c["net_savings"].max()) < 1e-6
    assert m["missed"] + round(m["recall"] * y.sum()) == y.sum()
