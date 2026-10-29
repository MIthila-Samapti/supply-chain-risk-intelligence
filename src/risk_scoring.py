"""
Composite disruption risk scoring.

Combines the Phase 1 signals (unexplained shocks, change points) with
an external macro volatility proxy (oil price volatility) into one
0-100 risk score per store/category/day. Rolling, not a one-time
snapshot - shows how risk moves over time and feeds the risk-adjusted
inventory policy in the next phase.
"""

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "shock_rate": 0.35,
    "avg_abs_z": 0.25,
    "changepoint_rate": 0.25,
    "oil_volatility": 0.15,
}


def add_changepoint_indicator(panel, changepoints):
    # True on the exact date a change point was detected for that store/family
    cp_flags = changepoints.copy()
    cp_flags["date"] = pd.to_datetime(cp_flags["date"])
    cp_flags["is_changepoint"] = True
    cp_flags = cp_flags[["store_nbr", "family", "date", "is_changepoint"]]

    merged = panel.merge(cp_flags, on=["store_nbr", "family", "date"], how="left")
    merged["is_changepoint"] = merged["is_changepoint"].eq(True)
    return merged


def compute_oil_volatility(panel, window_days=90, min_periods=30):
    # oil price is one national daily series repeated across every
    # store/category row, so pull the distinct date/price series and
    # take a trailing rolling std of the daily pct change - standard
    # macro volatility proxy. returns one row per date.
    oil = panel[["date", "oil_price"]].drop_duplicates("date").sort_values("date")
    oil["oil_pct_change"] = oil["oil_price"].pct_change()
    oil["oil_volatility"] = (
        oil["oil_pct_change"].rolling(window_days, min_periods=min_periods).std()
    )
    return oil[["date", "oil_volatility"]]


def compute_rolling_features(shocks_df, changepoints, panel, window_days=90, min_periods=30):
    # feature table the composite score is built from: trailing shock
    # rate, trailing avg shock severity, trailing change-point rate (all
    # per store/category), plus oil volatility as a shared macro overlay
    df = add_changepoint_indicator(shocks_df, changepoints)

    grp = df.groupby(["store_nbr", "family"])
    df["shock_rate"] = grp["is_disruption_shock"].transform(
        lambda s: s.shift(1).rolling(window_days, min_periods=min_periods).mean()
    )
    df["avg_abs_z"] = grp["residual_z"].transform(
        lambda s: s.abs().shift(1).rolling(window_days, min_periods=min_periods).mean()
    )
    df["changepoint_rate"] = grp["is_changepoint"].transform(
        lambda s: s.shift(1).rolling(window_days, min_periods=min_periods).mean()
    )

    oil_vol = compute_oil_volatility(panel, window_days=window_days, min_periods=min_periods)
    df = df.merge(oil_vol, on="date", how="left")

    return df


def compute_composite_risk_score(df, weights=None):
    # percentile-ranks each risk feature across the dataset (holds up
    # better than min-max scaling against skewed stuff like shock
    # counts), combines with `weights` into one 0-100 score, keeps the
    # percentile columns around too for transparency
    weights = weights or DEFAULT_WEIGHTS
    df = df.copy()

    pct_cols = {}
    for feature in weights:
        pct_col = f"{feature}_pct"
        df[pct_col] = df[feature].rank(pct=True)
        pct_cols[feature] = pct_col

    df["composite_risk_score"] = sum(
        df[pct_cols[feature]].fillna(0) * weight for feature, weight in weights.items()
    ) * 100

    return df


def latest_risk_ranking(scored_df):
    # one row per store/category, most recent score, highest risk first
    # - the "so what" view of what's riskiest to stock right now
    latest = scored_df.sort_values("date").groupby(["store_nbr", "family"]).tail(1)
    cols = ["store_nbr", "family", "date", "shock_rate", "avg_abs_z", "changepoint_rate",
            "oil_volatility", "composite_risk_score"]
    return latest[cols].sort_values("composite_risk_score", ascending=False).reset_index(drop=True)
