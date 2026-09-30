"""ADAPT - turn signals into ONE well-timed, conversational question + a concrete plan.

Inferred intent (signals, peers) decides WHEN to ask. Declared intent (the customer's
answer) decides WHAT the plan is. We then keep watching and re-check when patterns move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .forecast import first_month_above, simulate

MAX_GOAL = 500_000
MAX_MONTHS = 360


@dataclass
class Nudge:
    trigger: str
    question: str
    options: list[str]
    context: str


def choose_nudge(signals, sim, peers) -> Nudge | None:
    """Pick the single most relevant conversation opener (never more than one)."""
    keys = {s.key for s in signals}
    od = first_month_above(sim, "p_overdraft_by", 0.5)
    empty = first_month_above(sim, "p_savings_empty", 0.5)
    if {"rent_increase", "savings_drawdown"} & keys and (od or empty):
        parts = []
        if empty:
            parts.append(f"your savings will likely run out in about {empty[0]} months ({empty[1]:%B %Y})")
        if od:
            parts.append(f"you'd risk an overdraft from around {od[1]:%B %Y}")
        return Nudge(
            "overdraft_risk",
            "Since your rent went up, you've been topping up from your savings every month. At this pace, "
            + " and ".join(parts) + ". Want me to build a plan so that doesn't happen?",
            ["Yes, show me a plan", "I'm moving soon anyway", "Not now"],
            "Triggered by: rent increase + savings drawdown + 2,000 simulated futures of your accounts.")
    if "savings_build" in keys:
        top = max(peers.items(), key=lambda kv: kv[1])
        return Nudge(
            "saving_goal",
            "You've started putting money aside every month - nice! Are you saving towards something? "
            "Tell us and we'll help you get there.",
            ["A home", "Studies / a master abroad", "A car", "Just a safety buffer", "Rather not say"],
            f"Triggered by: new saving habit. People with a profile like yours most often: {top[0]} ({top[1]:.0%}).")
    if "no_pension" in keys:
        return Nudge(
            "pension",
            f"Most people with a profile like yours ({peers.get('started pension saving', 0):.0%}) start preparing "
            "their pension around now - and you get a tax reduction for it. Want to see what 30 minutes of setup "
            "could mean for your retirement?",
            ["Show me", "I already have it elsewhere", "Not now"],
            "Triggered by: life stage + no pension saving product detected + peer behaviour.")
    return None


def plan_for_goal(goal: str, amount: float, months: int, already_saved: float, df, balances, as_of) -> dict:
    """Monthly amount to reach the goal + probability the customer can actually afford it."""
    goal = (goal or "your goal").strip()[:60]
    amount = float(min(max(amount, 0), MAX_GOAL))
    months = int(min(max(months, 1), MAX_MONTHS))
    already = float(min(max(already_saved, 0), amount))
    monthly = math.ceil((amount - already) / months / 5) * 5  # round up to 5 EUR
    sim = simulate(df, balances, as_of, months=min(months, 24), extra_saving=monthly)
    base = simulate(df, balances, as_of, months=min(months, 24))
    afford = float((base["surplus"] >= monthly).mean())
    return {
        "goal": goal, "amount": amount, "months": months, "monthly": monthly,
        "p_affordable_month": afford,
        "p_overdraft_with_plan": float(sim["p_overdraft_by"][-1]),
        "message": f"To afford {goal} ({amount:,.0f} EUR in {months} months), set aside {monthly} EUR/month. "
                   f"Based on your last 3 months, that fits your budget in {afford:.0%} of months. "
                   "I can set up an automatic transfer on salary day.",
    }


def plan_overdraft(df, balances, as_of) -> dict:
    """For the overdraft case: how much to cut per month so the risk drops below 10 %."""
    options = []
    for cut in range(0, 501, 20):
        res = simulate(_shift(df, cut), balances, as_of, months=12)
        p = float(res["p_overdraft_by"][-1])
        options.append((cut, p))
        if p < 0.10:
            break
    return {"options": options, "cut": options[-1][0], "p_after": options[-1][1],
            "p_now": options[0][1]}


def _shift(df, monthly_cut):
    """Reduce everyday spending by `monthly_cut` EUR/month (proportionally)."""
    if monthly_cut == 0:
        return df
    df = df.copy()
    from .signals import VARIABLE_CATS
    m = df["category"].isin(VARIABLE_CATS)
    recent = df[m & (df["date"] > df["date"].max() - np.timedelta64(90, "D"))]["amount"].sum() / 3
    factor = max(0.0, 1 - monthly_cut / max(-recent, 1))
    df.loc[m, "amount"] *= factor
    return df


def needs_recheck(planned_monthly: float, actual_last_months: list[float]) -> bool:
    """Re-check the declared intent when the customer is off-plan 2 months in a row."""
    return len(actual_last_months) >= 2 and all(a < 0.5 * planned_monthly for a in actual_last_months[-2:])
