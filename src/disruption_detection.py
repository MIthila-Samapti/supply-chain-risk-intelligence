"""
Disruption detection for the supply chain risk intelligence project.

Flags demand shocks in the daily sales panel that aren't explained by
known promos or holidays, with two signals: a rolling seasonal
baseline + z-score on the residual (fast, easy to explain), and
change-point detection (ruptures) per store/family to catch sustained
shifts a single-day z-score would miss. Both feed the composite risk
score in the next phase.
"""

import numpy as np
import pandas as pd
import ruptures as rpt


def load_panel(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return df


def add_seasonal_baseline(df, window_weeks=8, min_periods=3):
    # trailing median of unit_sales for the same store/family/day-of-week
    # over the last `window_weeks` occurrences (excluding the current
    # day). rolling version of the seasonal-naive baseline from the
    # demand-forecasting project - updates week by week instead of
    # being pinned to one cutoff date
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
    # flags residuals more than z_thresh rolling std devs from zero, on
    # days with no promo and no holiday. shocks that line up with a
    # known driver aren't flagged - they're already explained
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
    # PELT change-point detection per store/family, with an adaptive
    # BIC-style penalty (penalty_mult * log(n) * variance) instead of
    # one fixed threshold - PRODUCE and CLEANING are on totally
    # different demand scales, so a fixed penalty would be way too
    # sensitive for one and not sensitive enough for the other.
    # returns one row per detected change point.
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
    # rolls the shock + change-point signals up into a per store/family
    # summary over the last window_days - feeds the Phase 2 composite score
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
