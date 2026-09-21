import numpy as np
import pandas as pd

from src.disruption_detection import add_seasonal_baseline, flag_shocks


def test_add_seasonal_baseline_tracks_weekday_pattern():
    dates = pd.date_range("2020-01-06", periods=70)  # starts on a Monday
    df = pd.DataFrame(
        {
            "store_nbr": 1,
            "family": "TEST",
            "date": dates,
            "unit_sales": [10 + d.dayofweek for d in dates],  # constant per weekday
        }
    )
    result = add_seasonal_baseline(df, window_weeks=4)
    mondays = result[result["date"].dt.dayofweek == 0]
    assert mondays["expected_sales"].iloc[-1] == 10


def test_flag_shocks_detects_unexplained_spike():
    dates = pd.date_range("2020-01-01", periods=100)
    df = pd.DataFrame(
        {
            "store_nbr": 1,
            "family": "TEST",
            "date": dates,
            "residual": np.random.RandomState(0).normal(0, 1, 100),
            "n_items_on_promo": 0,
            "is_holiday": False,
        }
    )
    df.loc[90, "residual"] = 50
    result = flag_shocks(df, z_window=60, z_thresh=3.0, min_periods=20)
    assert result.loc[90, "is_disruption_shock"] == True


def test_flag_shocks_ignores_promo_explained_spike():
    dates = pd.date_range("2020-01-01", periods=100)
    df = pd.DataFrame(
        {
            "store_nbr": 1,
            "family": "TEST",
            "date": dates,
            "residual": np.random.RandomState(1).normal(0, 1, 100),
            "n_items_on_promo": 0,
            "is_holiday": False,
        }
    )
    df.loc[90, "residual"] = 50
    df.loc[90, "n_items_on_promo"] = 5
    result = flag_shocks(df, z_window=60, z_thresh=3.0, min_periods=20)
    assert result.loc[90, "is_disruption_shock"] == False


def test_flag_shocks_stable_series_has_no_shocks():
    dates = pd.date_range("2020-01-01", periods=100)
    df = pd.DataFrame(
        {
            "store_nbr": 1,
            "family": "TEST",
            "date": dates,
            "residual": np.random.RandomState(2).normal(0, 1, 100),
            "n_items_on_promo": 0,
            "is_holiday": False,
        }
    )
    result = flag_shocks(df, z_window=60, z_thresh=3.0, min_periods=20)
    assert result["is_disruption_shock"].sum() == 0
