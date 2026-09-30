"""Synthetic data only: demo customers (current + savings account) and a peer population.

No real customer data is used anywhere in this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

START = pd.Timestamp("2026-01-01")
TODAY = pd.Timestamp("2026-09-30")
COLUMNS = ["date", "account", "merchant", "category", "amount"]


def _month_starts():
    return pd.date_range(START, TODAY, freq="MS")


def _daily_spending(rng, rows, level, start=START, end=TODAY):
    for day in pd.date_range(start, end, freq="D"):
        if rng.random() < 0.55:
            rows.append((day, "current", "Colruyt", "groceries", -round(rng.gamma(2.0, 12) * level, 2)))
        if rng.random() < 0.30:
            rows.append((day, "current", "Café / restaurant", "restaurants", -round(rng.gamma(2.0, 9) * level, 2)))
        if rng.random() < 0.25:
            rows.append((day, "current", "De Lijn / NMBS", "transport", -round(rng.uniform(2, 12), 2)))
        if rng.random() < 0.10:
            rows.append((day, "current", "Online shop", "shopping", -round(rng.gamma(2.0, 20) * level, 2)))


def _transfer(rows, day, amount, to_savings=True):
    """Pair of rows for a transfer between current and savings account."""
    if to_savings:
        rows.append((day, "current", "Transfer to savings", "transfer_to_savings", -amount))
        rows.append((day, "savings", "Transfer from current", "transfer_to_savings", amount))
    else:
        rows.append((day, "savings", "Transfer to current", "transfer_from_savings", -amount))
        rows.append((day, "current", "Transfer from savings", "transfer_from_savings", amount))


def make_lina(seed=7):
    """Lina, 27. Rent goes 750 -> 980 in July; since then she dips into her savings every month."""
    rng = np.random.default_rng(seed)
    rows = []
    july = pd.Timestamp("2026-07-01")
    _daily_spending(rng, rows, level=0.95, end=july - pd.Timedelta(days=1))
    _daily_spending(rng, rows, level=1.15, start=july)  # new neighbourhood, pricier
    for m in _month_starts():
        rent = 750.0 if m < july else 980.0
        rows.append((m, "current", "Rent - Immo Leuven", "rent", -rent))
        rows.append((m + pd.Timedelta(days=2), "current", "Telenet", "telecom", -45.0))
        rows.append((m + pd.Timedelta(days=5), "current", "Car loan", "subscriptions", -180.0))
        rows.append((m + pd.Timedelta(days=24), "current", "Salary - Accenture Belgium", "salary", 1980.0))
        if m < july:
            _transfer(rows, m + pd.Timedelta(days=25), 150.0, to_savings=True)
        else:
            _transfer(rows, m + pd.Timedelta(days=10), 300.0, to_savings=False)
    return _to_df(rows)


def make_yanis(seed=11):
    """Yanis, 23. First job since March; from June he suddenly puts 250 EUR aside every month."""
    rng = np.random.default_rng(seed)
    rows = []
    _daily_spending(rng, rows, level=0.8)
    for m in _month_starts():
        rows.append((m, "current", "Rent - Kot Gent", "rent", -520.0))
        rows.append((m + pd.Timedelta(days=2), "current", "Proximus", "telecom", -30.0))
        salary = 900.0 if m < pd.Timestamp("2026-03-01") else 1850.0
        rows.append((m + pd.Timedelta(days=24), "current", "Salary", "salary", salary))
        if m >= pd.Timestamp("2026-06-01"):
            _transfer(rows, m + pd.Timedelta(days=26), 250.0, to_savings=True)
    return _to_df(rows)


def make_marc(seed=13):
    """Marc, 52. Stable, good income, no pension saving at all."""
    rng = np.random.default_rng(seed)
    rows = []
    _daily_spending(rng, rows, level=1.6)
    for m in _month_starts():
        rows.append((m, "current", "Mortgage KBC", "mortgage", -1100.0))
        rows.append((m + pd.Timedelta(days=24), "current", "Salary - Barco", "salary", 3900.0))
        _transfer(rows, m + pd.Timedelta(days=26), 400.0, to_savings=True)
    return _to_df(rows)


def _to_df(rows):
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("date").reset_index(drop=True)


# Balances on START. Profile fields are what the bank already knows.
CUSTOMERS = {
    "C-0001": {"name": "Lina", "age": 27, "builder": make_lina,
               "start": {"current": 350.0, "savings": 1300.0}},
    "C-0002": {"name": "Yanis", "age": 23, "builder": make_yanis,
               "start": {"current": 300.0, "savings": 400.0}},
    "C-0003": {"name": "Marc", "age": 52, "builder": make_marc,
               "start": {"current": 2500.0, "savings": 21000.0}},
}


def get_transactions(customer_id: str) -> pd.DataFrame:
    return CUSTOMERS[customer_id]["builder"]()


def balances(df: pd.DataFrame, customer_id: str, as_of=None) -> dict:
    as_of = TODAY if as_of is None else as_of
    start = CUSTOMERS[customer_id]["start"]
    d = df[df["date"] <= as_of]
    return {acc: round(start[acc] + d.loc[d["account"] == acc, "amount"].sum(), 2)
            for acc in ("current", "savings")}


# ---------------- Peer population (for "customers like you") ----------------
def make_population(n=20_000, seed=42) -> pd.DataFrame:
    """Synthetic customers with profile + what they did in the following 3 years.

    Event probabilities are ASSUMPTIONS (to be calibrated on real, consented data).
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 75, n)
    income = np.clip(rng.normal(1500 + 45 * np.minimum(age, 55), 700), 700, 9000)
    housing = np.clip(income * rng.uniform(0.25, 0.45, n), 300, 2500)
    savings = np.clip(rng.lognormal(np.log(income * 3), 0.9), 0, 400_000)

    def sig(x):
        return 1 / (1 + np.exp(-x))

    p_house = sig(-1.2 - 0.004 * (age - 31) ** 2 + 0.0004 * (income - 2500) + 0.00003 * (savings - 15000))
    p_car = sig(-1.6 + 0.0002 * (income - 2500))
    p_abroad = sig(-2.2 - 0.25 * (age - 23)) * 0.8
    p_pension = sig(-2.0 + 0.09 * (age - 35) + 0.0003 * (income - 2500))
    return pd.DataFrame({
        "age": age, "income": income.round(), "housing": housing.round(), "savings": savings.round(),
        "bought_home_3y": rng.random(n) < p_house,
        "bought_car_3y": rng.random(n) < p_car,
        "study_or_move_abroad_3y": rng.random(n) < p_abroad,
        "started_pension_saving_3y": rng.random(n) < p_pension,
        "consent": rng.random(n) < 0.85,
    })
