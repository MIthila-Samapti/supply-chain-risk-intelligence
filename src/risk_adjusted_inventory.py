"""
Risk-adjusted safety stock and reorder points.

Same newsvendor math as the demand-forecasting project's inventory.py,
but the safety stock isn't fixed anymore -- it scales up and down with
the composite risk score from Phase 2, so a category currently flagged
as higher-risk carries more buffer, and calms back down once its risk
score drops.

Forecast error (sigma) in the original project was only computed at the
family level (summed across all 10 stores), not per store. To get a
per-store sigma without redoing that backtest, I allocate the family
sigma across stores by each store's share of family demand, assuming
store-level errors are roughly independent so variance scales with
volume share (store_sigma ~= family_sigma * sqrt(store_share)). That's
an approximation, not a measured number -- noted here and in the
notebook.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
RISK_UPLIFT_CAP = 0.5  # a risk score of 100 doubles... no, adds up to 50% more safety stock

HOLDING_COST_PER_UNIT_PER_DAY = 0.05
STOCKOUT_COST_PER_UNIT = 2.00


def allocate_sigma_to_stores(panel, family_sigma):
    """
    family_sigma: DataFrame with columns family, sigma_daily (family-level
    forecast error std, from the demand-forecasting project's backtest).

    Returns a DataFrame with store_nbr, family, sigma_store.
    """
    totals = (
        panel.groupby(["store_nbr", "family"])["unit_sales"].sum().rename("store_total").reset_index()
    )
    family_totals = totals.groupby("family")["store_total"].transform("sum")
    totals["store_share"] = totals["store_total"] / family_totals

    merged = totals.merge(family_sigma, on="family", how="left")
    merged["sigma_store"] = merged["sigma_daily"] * np.sqrt(merged["store_share"])
    return merged[["store_nbr", "family", "store_share", "sigma_store"]]


def base_safety_stock(sigma, lead_time=LEAD_TIME_DAYS, service_level=SERVICE_LEVEL):
    z = norm.ppf(service_level)
    return max(0.0, z * sigma * np.sqrt(lead_time))


def risk_multiplier(risk_score, uplift_cap=RISK_UPLIFT_CAP):
    """risk_score expected 0-100. 1.0 at score 0, 1+uplift_cap at score 100."""
    risk_score = np.nan_to_num(risk_score, nan=0.0)
    return 1.0 + uplift_cap * (risk_score / 100.0)


def simulate_policy(dates, actual, forecast, sigma, risk_scores=None, lead_time=LEAD_TIME_DAYS,
                     service_level=SERVICE_LEVEL, uplift_cap=RISK_UPLIFT_CAP):
    """
    Daily-review, order-up-to-reorder-point simulation, same mechanics as
    the demand-forecasting project's simulate_policy. The difference:
    if `risk_scores` is given, the safety stock and reorder point are
    recalculated every day from that day's risk score instead of being
    fixed for the whole run. Pass risk_scores=None (or an all-zero array)
    to get the risk-blind baseline for comparison.
    """
    n = len(actual)
    base_ss = base_safety_stock(sigma, lead_time, service_level)

    if risk_scores is None:
        mult = np.ones(n)
    else:
        mult = risk_multiplier(np.asarray(risk_scores), uplift_cap)

    ss_series = base_ss * mult
    rop_series = forecast * lead_time + ss_series  # forecast here is that day's expected demand, not a fixed avg

    on_hand = rop_series[0]
    pipeline = {}
    stockout_units = 0.0
    total_demand = 0.0
    on_hand_history = []
    ss_history = []

    for day in range(n):
        if day in pipeline:
            on_hand += pipeline.pop(day)

        demand = actual[day]
        total_demand += demand
        shortfall = max(0.0, demand - on_hand)
        stockout_units += shortfall
        on_hand = max(0.0, on_hand - demand)

        rop = rop_series[day]
        on_order = sum(pipeline.values())
        inventory_position = on_hand + on_order
        if inventory_position <= rop:
            order_qty = max(0.0, rop - inventory_position)
            arrival_day = day + lead_time
            pipeline[arrival_day] = pipeline.get(arrival_day, 0.0) + order_qty

        on_hand_history.append(on_hand)
        ss_history.append(ss_series[day])

    avg_on_hand = float(np.mean(on_hand_history))
    fill_rate = 1 - (stockout_units / total_demand if total_demand > 0 else 0.0)
    holding_cost = avg_on_hand * HOLDING_COST_PER_UNIT_PER_DAY * n
    stockout_cost = stockout_units * STOCKOUT_COST_PER_UNIT

    return {
        "avg_safety_stock": float(np.mean(ss_history)),
        "avg_on_hand": avg_on_hand,
        "total_demand": total_demand,
        "stockout_units": stockout_units,
        "fill_rate": fill_rate,
        "holding_cost": holding_cost,
        "stockout_cost": stockout_cost,
        "total_cost": holding_cost + stockout_cost,
    }


def compare_policies(df, sigma, lead_time=LEAD_TIME_DAYS, service_level=SERVICE_LEVEL, uplift_cap=RISK_UPLIFT_CAP):
    """
    df needs columns: date, unit_sales (actual), expected_sales (forecast),
    composite_risk_score. Runs the risk-blind and risk-adjusted policies
    over the same series and returns both results plus the delta.
    """
    d = df.dropna(subset=["expected_sales", "composite_risk_score"]).sort_values("date")
    dates = d["date"].values
    actual = d["unit_sales"].values
    forecast = d["expected_sales"].values
    risk = d["composite_risk_score"].values

    blind = simulate_policy(dates, actual, forecast, sigma, risk_scores=None,
                             lead_time=lead_time, service_level=service_level)
    adjusted = simulate_policy(dates, actual, forecast, sigma, risk_scores=risk,
                                lead_time=lead_time, service_level=service_level, uplift_cap=uplift_cap)

    return {
        "n_days": len(d),
        "risk_blind": blind,
        "risk_adjusted": adjusted,
        "stockout_cost_delta": adjusted["stockout_cost"] - blind["stockout_cost"],
        "holding_cost_delta": adjusted["holding_cost"] - blind["holding_cost"],
        "total_cost_delta": adjusted["total_cost"] - blind["total_cost"],
    }
