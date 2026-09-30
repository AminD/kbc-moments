"""UNDERSTAND - detect behaviour changes on current + savings accounts.

Each signal compares a recent window to the customer's own baseline, and returns
the evidence (so the customer can see *why* KBC reacts).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

VARIABLE_CATS = {"groceries", "restaurants", "transport", "shopping"}


@dataclass
class Signal:
    key: str
    title: str
    strength: float  # 0..1
    evidence: list[str] = field(default_factory=list)


def _monthly(df, mask):
    s = df[mask].set_index("date")["amount"].resample("MS").sum()
    full = pd.date_range(df["date"].min().to_period("M").to_timestamp(), df["date"].max(), freq="MS")
    return s.reindex(full, fill_value=0.0)


def detect(df: pd.DataFrame, as_of: pd.Timestamp, age: int) -> list[Signal]:
    df = df[df["date"] <= as_of]
    out: list[Signal] = []

    # 1. Rent increase
    rent = df[df["category"] == "rent"].sort_values("date")
    if len(rent) >= 2:
        last_amt = rent["amount"].iloc[-1]
        before = rent[rent["amount"] != last_amt]
        if len(before):
            old, last = -before["amount"].iloc[-1], -last_amt
            if last > old * 1.10:
                since = rent[(rent["date"] > before["date"].iloc[-1])]["date"].iloc[0]
                out.append(Signal("rent_increase", "Rent went up", min(1.0, (last - old) / old * 3),
                                  [f"Rent {old:.0f} EUR -> {last:.0f} EUR since {since:%B %Y} (+{(last-old)/old:.0%})"]))

    # 2. Savings: drawdown or new saving pattern
    sav = _monthly(df, df["account"] == "savings")
    sav = sav[sav.index < as_of.to_period("M").to_timestamp()]  # complete months only
    if len(sav) >= 5:
        recent, base = sav.iloc[-3:], sav.iloc[:-3]
        if recent.mean() < 0 and base.mean() >= 0:
            out.append(Signal("savings_drawdown", "Dipping into savings",
                              min(1.0, -recent.mean() / 300),
                              [f"Savings: {base.mean():+.0f} EUR/month before, {recent.mean():+.0f} EUR/month over the last 3 months"]))
        elif recent.mean() > base.mean() + 100 and (recent > 0).all():
            out.append(Signal("savings_build", "New saving habit",
                              min(1.0, (recent.mean() - base.mean()) / 300),
                              [f"Savings: {base.mean():+.0f} EUR/month before, {recent.mean():+.0f} EUR/month over the last 3 months"]))

    # 3. Everyday spending shift
    var = -_monthly(df, df["category"].isin(VARIABLE_CATS))
    var = var[var.index < as_of.to_period("M").to_timestamp()]
    if len(var) >= 5:
        recent, base = var.iloc[-2:].mean(), var.iloc[:-2].mean()
        if recent > base * 1.2:
            out.append(Signal("spending_up", "Spending pattern changed", min(1.0, (recent / base - 1) * 2),
                              [f"Everyday spending {base:.0f} -> {recent:.0f} EUR/month"]))

    # 4. Life stage: pension gap
    if age >= 40 and not (df["category"] == "pension_saving").any():
        out.append(Signal("no_pension", "No pension saving yet", 0.6,
                          [f"Age {age}, no pension saving product detected"]))
    return out
