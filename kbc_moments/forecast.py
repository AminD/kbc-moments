"""Monte-Carlo forecast of current + savings account, month by month.

Fixed flows (salary, rent, subscriptions, recurring transfers) are replayed from the
latest month; everyday spending is resampled from the customer's own last 90 days.
Output: probability of overdraft per month and month savings run out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import VARIABLE_CATS

FIXED_CATS = {"salary", "rent", "mortgage", "telecom", "subscriptions"}


def monthly_profile(df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    df = df[df["date"] <= as_of]
    last_month = (as_of.to_period("M") - 1).to_timestamp()
    lm = df[(df["date"] >= last_month) & (df["date"] < last_month + pd.offsets.MonthBegin(1))]
    fixed = lm[(lm["account"] == "current") & lm["category"].isin(FIXED_CATS)]["amount"].sum()
    transfer = lm[(lm["account"] == "savings")]["amount"].sum()  # + to savings, - from savings
    recent = df[(df["date"] > as_of - pd.Timedelta(days=90)) & df["category"].isin(VARIABLE_CATS)]
    daily = recent.groupby(recent["date"].dt.normalize())["amount"].sum()
    daily = daily.reindex(pd.date_range(as_of - pd.Timedelta(days=89), as_of, freq="D"), fill_value=0.0)
    return {"fixed": float(fixed), "transfer_to_savings": float(transfer), "daily_variable": daily.to_numpy()}


def simulate(df, balances: dict, as_of, months=12, n=2000, seed=0, extra_saving=0.0) -> dict:
    """Simulate `n` futures. extra_saving: planned monthly amount moved to savings."""
    rng = np.random.default_rng(seed)
    p = monthly_profile(df, as_of)
    cur = np.full(n, balances["current"])
    sav = np.full(n, balances["savings"])
    over = np.zeros((months, n), bool)
    cur_paths, sav_paths, surplus = [], [], []
    for m in range(months):
        var = rng.choice(p["daily_variable"], size=(n, 30)).sum(axis=1)
        s = p["fixed"] + var  # monthly surplus before transfers
        surplus.append(s)
        t = p["transfer_to_savings"] + extra_saving
        if t < 0:  # customer pulls money from savings while there is some
            pulled = np.minimum(-t, sav)
            sav -= pulled
            cur += s + pulled
        else:
            cur += s - t
            sav += t
        over[m] = cur < 0
        cur_paths.append(cur.copy())
        sav_paths.append(sav.copy())
    cum_over = np.maximum.accumulate(over, axis=0).mean(axis=1)
    sav_arr = np.array(sav_paths)
    months_idx = pd.date_range(as_of + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    empty = (sav_arr < 50).mean(axis=1)
    return {
        "months": months_idx,
        "p_overdraft_by": cum_over,  # P(at least one overdraft by month m)
        "p_savings_empty": empty,
        "current_p10": np.percentile(cur_paths, 10, axis=1),
        "current_p50": np.percentile(cur_paths, 50, axis=1),
        "current_p90": np.percentile(cur_paths, 90, axis=1),
        "savings_p50": np.percentile(sav_arr, 50, axis=1),
        "surplus": np.array(surplus),  # (months, n)
    }


def first_month_above(res, key, threshold=0.5):
    idx = np.argmax(res[key] >= threshold)
    return None if res[key][idx] < threshold else (idx + 1, res["months"][idx])
