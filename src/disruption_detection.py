"""
Disruption detection for the supply chain risk intelligence project.

Flags demand shocks in the daily sales panel that aren't explained by
known promotions or holidays, using two complementary signals:

  1. A rolling seasonal baseline + z-score on the residual (fast, easy to
     explain, runs on every row).
  2. Change-point detection (ruptures) on each store/family's residual
     series, to catch sustained shifts a single-day z-score would miss.

Both feed into the composite risk score built in the next phase.
"""

import numpy as np
import pandas as pd
import ruptures as rpt


def load_panel(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return df


def add_seasonal_baseline(df, window_weeks=8, min_periods=3):
    """
    Adds an `expected_sales` column: the trailing median of unit_sales for
    the same store/family/day-of-week, over the last `window_weeks`
    occurrences (not counting the current day). This is a rolling version
    of the seasonal-naive baseline from the demand-forecasting project —
    it updates week by week instead of being fixed at one cutoff date.
    """
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek

    df["expected_sales"] = (
        df.groupby(["store_nbr", "family", "dow"])["unit_sales"]
        .transform(lambda s: s.shift(1).rolling(window_weeks, min_periods=min_periods).median())
    )
    return df


def compute_residuals(df):
    df = df.copy()
    df["residual"] = df["unit_sales"] - df["expected_sales"]
    return df


def flag_shocks(df, z_window=90, z_thresh=3.0, min_periods=20):
    """
    Flags unexplained demand shocks: residuals more than `z_thresh` rolling
    standard deviations from zero, on days with no promotion and no
    holiday. Shocks that coincide with a known driver (promo/holiday)
    are not flagged as disruptions — they already have an explanation.
    """
    df = df.copy()
    grp = df.groupby(["store_nbr", "family"])["residual"]
    roll_mean = grp.transform(lambda s: s.shift(1).rolling(z_window, min_periods=min_periods).mean())
    roll_std = grp.transform(lambda s: s.shift(1).rolling(z_window, min_periods=min_periods).std())

    df["residual_z"] = (df["residual"] - roll_mean) / roll_std.replace(0, np.nan)

    known_driver = (df["n_items_on_promo"] > 0) | (df["is_holiday"] == True)
    df["is_disruption_shock"] = (
        (df["residual_z"].abs() >= z_thresh) & (~known_driver) & df["residual_z"].notna()
    )
    return df


def detect_changepoints(df, model="l2", penalty_mult=5.0, min_size=21, min_obs=60):
    """
    Runs PELT change-point detection on each store/family's residual
    series, using an adaptive BIC-style penalty (penalty_mult * log(n) *
    variance) rather than one fixed threshold — necessary because
    different store/family series have very different demand scales
    (e.g. PRODUCE vs. CLEANING), so a single fixed penalty would be far
    too sensitive for some series and not sensitive enough for others.
    Returns one row per detected change point (a date where the
    underlying demand pattern shifted, not explained by a single-day
    shock).
    """
    records = []
    for (store, family), g in df.groupby(["store_nbr", "family"]):
        g = g.dropna(subset=["residual"]).sort_values("date")
        if len(g) < min_obs:
            continue
        signal = g["residual"].to_numpy().reshape(-1, 1)
        variance = float(np.var(signal))
        if variance == 0:
            continue
        penalty = penalty_mult * np.log(len(signal)) * variance
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
        bkps = algo.predict(pen=penalty)
        for idx in bkps[:-1]:  # ruptures' last breakpoint is len(signal), not a real one
            records.append(
                {"store_nbr": store, "family": family, "date": g["date"].iloc[idx]}
            )
    return pd.DataFrame(records, columns=["store_nbr", "family", "date"])


def summarize_disruption_risk(shocks_df, changepoints_df, window_days=90):
    """
    Aggregates the shock and change-point signals into a per
    store/family risk summary over the most recent `window_days` of data.
    This is the feature table the Phase 2 composite risk score builds on.
    """
    cutoff = shocks_df["date"].max() - pd.Timedelta(days=window_days)
    recent = shocks_df[shocks_df["date"] >= cutoff]

    shock_count = recent.groupby(["store_nbr", "family"])["is_disruption_shock"].sum()
    avg_abs_z = recent.groupby(["store_nbr", "family"])["residual_z"].apply(
        lambda s: s.abs().mean()
    )

    cp_recent = changepoints_df[changepoints_df["date"] >= cutoff]
    cp_count = cp_recent.groupby(["store_nbr", "family"]).size()

    summary = (
        pd.concat(
            [shock_count.rename("shock_count"), avg_abs_z.rename("avg_abs_z"), cp_count.rename("changepoint_count")],
            axis=1,
        )
        .fillna(0)
        .reset_index()
    )
    return summary
