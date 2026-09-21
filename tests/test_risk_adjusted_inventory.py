import numpy as np
import pandas as pd

from src.risk_adjusted_inventory import (
    allocate_sigma_to_stores,
    base_safety_stock,
    risk_multiplier,
    simulate_policy,
    compare_policies,
)


def test_allocate_sigma_splits_by_store_share():
    panel = pd.DataFrame({
        "store_nbr": [1, 1, 2, 2],
        "family": ["A", "A", "A", "A"],
        "unit_sales": [100, 100, 300, 300],  # store 2 has 3x store 1's volume
    })
    family_sigma = pd.DataFrame({"family": ["A"], "sigma_daily": [10.0]})
    result = allocate_sigma_to_stores(panel, family_sigma)

    s1 = result[result.store_nbr == 1]["sigma_store"].iloc[0]
    s2 = result[result.store_nbr == 2]["sigma_store"].iloc[0]
    assert s2 > s1  # more volume -> more allocated uncertainty
    np.testing.assert_allclose(s1 ** 2 + s2 ** 2, 10.0 ** 2, rtol=1e-6)


def test_risk_multiplier_bounds():
    assert risk_multiplier(0) == 1.0
    assert risk_multiplier(100, uplift_cap=0.5) == 1.5
    assert risk_multiplier(50, uplift_cap=0.5) == 1.25


def test_risk_adjusted_policy_holds_more_stock_when_risk_is_high():
    dates = np.arange(60)
    rng = np.random.RandomState(1)
    actual = np.maximum(0, 100 + rng.normal(0, 15, 60))
    forecast = np.full(60, 100.0)
    sigma = 15.0

    low_risk = np.zeros(60)
    high_risk = np.full(60, 100.0)

    low = simulate_policy(dates, actual, forecast, sigma, risk_scores=low_risk)
    high = simulate_policy(dates, actual, forecast, sigma, risk_scores=high_risk)

    assert high["avg_safety_stock"] > low["avg_safety_stock"]


def test_compare_policies_runs_end_to_end():
    dates = pd.date_range("2020-01-01", periods=100)
    rng = np.random.RandomState(2)
    df = pd.DataFrame({
        "date": dates,
        "unit_sales": np.maximum(0, 200 + rng.normal(0, 20, 100)),
        "expected_sales": 200.0,
        "composite_risk_score": rng.uniform(0, 100, 100),
    })
    result = compare_policies(df, sigma=20.0)
    assert result["n_days"] == 100
    assert "total_cost_delta" in result
