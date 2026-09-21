import numpy as np
import pandas as pd

from src.risk_scoring import (
    add_changepoint_indicator,
    compute_oil_volatility,
    compute_composite_risk_score,
    latest_risk_ranking,
)


def _toy_panel(n=120):
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "store_nbr": 1,
            "family": "TEST",
            "date": dates,
            "oil_price": 90.0,
        }
    )


def test_add_changepoint_indicator_flags_only_matching_rows():
    panel = _toy_panel(10)
    cps = pd.DataFrame(
        {"store_nbr": [1], "family": ["TEST"], "date": [panel["date"].iloc[5]]}
    )
    result = add_changepoint_indicator(panel, cps)
    assert result["is_changepoint"].sum() == 1
    assert result.loc[5, "is_changepoint"] == True


def test_compute_oil_volatility_higher_for_swingier_series():
    stable = _toy_panel(120)
    stable["oil_price"] = 90.0

    volatile = _toy_panel(120)
    rng = np.random.RandomState(0)
    volatile["oil_price"] = 90 + rng.normal(0, 5, 120).cumsum() * 0.1

    stable_vol = compute_oil_volatility(stable, window_days=60, min_periods=10)
    volatile_vol = compute_oil_volatility(volatile, window_days=60, min_periods=10)

    assert stable_vol["oil_volatility"].iloc[-1] == 0
    assert volatile_vol["oil_volatility"].iloc[-1] > 0


def test_composite_score_ranks_higher_risk_series_first():
    dates = pd.date_range("2020-01-01", periods=5)
    df = pd.DataFrame(
        {
            "store_nbr": [1, 1, 2, 2, 2],
            "family": ["A", "A", "B", "B", "B"],
            "date": [dates[0], dates[1], dates[0], dates[1], dates[2]],
            "shock_rate": [0.0, 0.0, 0.5, 0.5, 0.5],
            "avg_abs_z": [0.2, 0.2, 3.0, 3.0, 3.0],
            "changepoint_rate": [0.0, 0.0, 0.3, 0.3, 0.3],
            "oil_volatility": [0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )
    scored = compute_composite_risk_score(df)
    ranking = latest_risk_ranking(scored)

    assert ranking.iloc[0]["store_nbr"] == 2
    assert ranking.iloc[0]["family"] == "B"
    assert ranking.iloc[0]["composite_risk_score"] > ranking.iloc[1]["composite_risk_score"]
