"""
Composite disruption risk scoring.

Combines the Phase 1 disruption signals (unexplained shocks, change
points) with an external macro volatility proxy (oil price volatility)
into a single 0-100 risk score per store/category/day. This is a
continuous, rolling score — not a one-time snapshot — so it can show how
risk moves over time and feed the risk-adjusted inventory policy in the
next phase.
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
    """
    Adds a boolean `is_changepoint` column to `panel`: True on the exact
    date a change point was detected for that store/family.
    """
    cp_flags = changepoints.copy()
    cp_flags["date"] = pd.to_datetime(cp_flags["date"])
    cp_flags["is_changepoint"] = True
    cp_flags = cp_flags[["store_nbr", "family", "date", "is_changepoint"]]

    merged = panel.merge(cp_flags, on=["store_nbr", "family", "date"], how="left")
    merged["is_changepoint"] = merged["is_changepoint"].eq(True)
    return merged


def compute_oil_volatility(panel, window_days=90, min_periods=30):
    """
    Oil price is a single national daily series (repeated across every
    store/category row), so this pulls the distinct date/oil_price
    series and computes a trailing rolling standard deviation of the
    daily percent change — a standard macro volatility proxy.

    Returns a DataFrame with one row per date: `date`, `oil_volatility`.
    """
    oil = panel[["date", "oil_price"]].drop_duplicates("date").sort_values("date")
    oil["oil_pct_change"] = oil["oil_price"].pct_change()
    oil["oil_volatility"] = (
        oil["oil_pct_change"].rolling(window_days, min_periods=min_periods).std()
    )
    return oil[["date", "oil_volatility"]]


def compute_rolling_features(shocks_df, changepoints, panel, window_days=90, min_periods=30):
    """
    Builds the rolling feature table the composite score is computed
    from: trailing shock rate, trailing average shock severity, trailing
    change-point rate (all per store/category), plus oil price
    volatility (a macro overlay shared across every store/category on a
    given date).
    """
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
    """
    Percentile-ranks each of the four risk features across the full
    dataset (robust to skewed distributions like shock counts, unlike
    min-max scaling), combines them with `weights` into one 0-100
    composite risk score, and adds the percentile columns alongside it
    for transparency.
    """
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
    """
    Returns one row per store/category — its most recent composite risk
    score — sorted highest risk first. This is the "so what" view: which
    store/category combinations are the highest-risk to stock right now.
    """
    latest = scored_df.sort_values("date").groupby(["store_nbr", "family"]).tail(1)
    cols = ["store_nbr", "family", "date", "shock_rate", "avg_abs_z", "changepoint_rate",
            "oil_volatility", "composite_risk_score"]
    return latest[cols].sort_values("composite_risk_score", ascending=False).reset_index(drop=True)
