"""KBC Moments - Streamlit demo.  Run:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from kbc_moments import forecast, intent, peers, signals, synth
from kbc_moments.auth import authenticate, require_role

st.set_page_config(page_title="KBC Moments", page_icon="💬", layout="wide")
KBC_CUSTOMERS = 2_300_000
LOGO = Path(__file__).parent / "assets" / "kbc_logo.png"  # add the official logo file here (not included)
BLUE, TEAL, RED = "#1E64C8", "#0A9E8C", "#C62828"  # chart colours (validated for colour-blind readers)


def header(title: str):
    if LOGO.exists():
        c1, c2 = st.columns([1, 8], vertical_alignment="center")
        c1.image(str(LOGO), width=90)
        c2.title(title)
    else:
        st.title(title)
    st.caption("Hackathon proof of concept for KBC - not an official KBC application. Synthetic data only.")


# ---------------------------------------------------------------- auth
def login_screen():
    header("KBC Moments")
    st.caption("Understand what each customer needs - and respond at exactly the right moment.")
    try:
        users = {k: dict(v) for k, v in st.secrets["users"].items()}
    except (KeyError, FileNotFoundError):
        st.error("No demo users configured. Run `python setup_secrets.py` first.")
        st.stop()
    with st.form("login"):
        u = st.text_input("Username (lina, yanis, marc or advisor)")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            user = authenticate(users, u, p)
            if user:
                st.session_state.clear()
                st.session_state["user"] = user
                st.rerun()
            st.error("Invalid credentials.")
    st.stop()


user = st.session_state.get("user")
if not user:
    login_screen()

with st.sidebar:
    st.write(f"Logged in as **{user['username']}** ({user['role']})")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()


@st.cache_data(show_spinner=False)
def load(customer_id: str):
    return synth.get_transactions(customer_id)


def customer_context(cid: str, as_of: pd.Timestamp):
    c = synth.CUSTOMERS[cid]
    df = load(cid)
    bal = synth.balances(df, cid, as_of)
    sig = signals.detect(df, as_of, c["age"])
    sim = forecast.simulate(df, bal, as_of)
    hist = df[df["date"] <= as_of]
    profile = {"age": c["age"],
               "income": float(hist.loc[hist["category"] == "salary", "amount"].iloc[-1]),
               "housing": float(-hist.loc[hist["category"].isin(["rent", "mortgage"]), "amount"].iloc[-1]),
               "savings": bal["savings"]}
    pi = peers.peer_insights(profile)
    nudge = intent.choose_nudge(sig, sim, pi)
    return df, bal, sig, sim, pi, nudge


def kate(text: str):
    with st.chat_message("assistant", avatar="💬"):
        st.markdown(f"**Kate** · {text}")


def me(text: str):
    with st.chat_message("user"):
        st.text(text)  # st.text: user input is never rendered as HTML/markdown


# ---------------------------------------------------------------- customer view
def customer_view():
    cid = user["customer_id"]  # from the authenticated session only (no IDOR)
    c = synth.CUSTOMERS[cid]
    header(f"Hi {c['name']} 👋")

    dates = [pd.Timestamp(d) for d in ("2026-06-30", "2026-07-31", "2026-08-31", "2026-09-30")]
    as_of = st.select_slider("Demo time machine - replay the account up to:", options=dates,
                             value=dates[-1], format_func=lambda d: d.strftime("%d %b %Y"))
    df, bal, sig, sim, pi, nudge = customer_context(cid, as_of)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current account", f"€ {bal['current']:,.0f}")
    m2.metric("Savings account", f"€ {bal['savings']:,.0f}")
    m3.metric("Overdraft risk (12 months)", f"{sim['p_overdraft_by'][-1]:.0%}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Kate")
        conversation(cid, df, bal, as_of, sig, pi, nudge)
    with right:
        st.subheader("Why am I seeing this?")
        if sig:
            for s in sig:
                st.markdown(f"**{s.title}** · strength {s.strength:.0%}")
                for e in s.evidence:
                    st.caption(e)
        else:
            st.caption("No behaviour change detected - Kate stays quiet.")
        st.markdown("**What customers similar to you did in the next 3 years**")
        st.caption("Similar = close in age, income, housing cost and savings. "
                   "Demo: computed on 20,000 simulated customers, not real KBC data.")
        for label, p in sorted(pi.items(), key=lambda kv: -kv[1]):
            if p >= 0.03:
                st.progress(min(p, 1.0), text=f"{p:.0%} {label}")
        if nudge:
            st.caption(nudge.context)

    st.subheader("Your money over the next 12 months")
    st.altair_chart(future_chart(sim), use_container_width=True)
    worst = sim["current_p10"].min()
    st.caption(
        "We imagined 2,000 possible versions of your next year, based on your own habits. "
        "The **blue line** is the most likely path of your current account; the **light blue band** shows "
        "where it ends up in 8 out of 10 cases (a bit better or a bit worse). The **green line** is your savings. "
        + ("Below the red line means overdraft." if worst < 0 else ""))


def future_chart(sim):
    months = sim["months"]
    band = pd.DataFrame({"month": months, "low": sim["current_p10"], "high": sim["current_p90"]})
    lines = pd.concat([
        pd.DataFrame({"month": months, "amount": sim["current_p50"], "what": "Current account (most likely)"}),
        pd.DataFrame({"month": months, "amount": sim["savings_p50"], "what": "Savings account (most likely)"}),
    ])
    x = alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y", grid=False))
    area = alt.Chart(band).mark_area(color=BLUE, opacity=0.15).encode(
        x=x, y=alt.Y("low:Q", title="€"), y2="high:Q",
        tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                 alt.Tooltip("low:Q", title="If things go worse (€)", format=",.0f"),
                 alt.Tooltip("high:Q", title="If things go better (€)", format=",.0f")])
    colour = alt.Color("what:N", title=None, legend=alt.Legend(orient="top", labelLimit=0),
                       scale=alt.Scale(domain=["Current account (most likely)", "Savings account (most likely)"],
                                       range=[BLUE, TEAL]))
    line = alt.Chart(lines).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40)).encode(
        x=x, y="amount:Q", color=colour,
        tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"), alt.Tooltip("what:N", title=""),
                 alt.Tooltip("amount:Q", title="Amount (€)", format=",.0f")])
    zero = alt.Chart(pd.DataFrame({"y": [0], "label": ["Overdraft below this line"]}))
    rule = zero.mark_rule(color=RED, strokeDash=[4, 4], strokeWidth=1.5).encode(y="y:Q")
    text = zero.mark_text(color=RED, align="left", dx=4, dy=-6, fontSize=11).encode(
        y="y:Q", text="label:N", x=alt.value(0))
    return (area + line + rule + text).properties(height=320)


def conversation(cid, df, bal, as_of, sig, pi, nudge):
    key = f"conv_{cid}_{as_of:%Y%m%d}"
    state = st.session_state.setdefault(key, {"answer": None})
    if not nudge:
        kate("Everything looks on track. I'll only get in touch when something actually changes.")
        return
    kate(nudge.question)
    if state["answer"] is None:
        cols = st.columns(len(nudge.options))
        for col, opt in zip(cols, nudge.options):
            if col.button(opt, key=f"{key}_{opt}"):
                state["answer"] = opt
                st.rerun()
        return
    me(state["answer"])
    ans = state["answer"]

    if ans in ("Not now", "Rather not say"):
        kate("No problem - I won't ask again unless your situation changes.")
    elif ans in ("I'm moving soon anyway", "I already have it elsewhere"):
        kate("Thanks, noted as your declared intent. It overrides what I inferred - "
             "I'll check in again only if your pattern moves.")
    elif nudge.trigger == "overdraft_risk":
        with st.spinner("Simulating..."):
            plan = intent.plan_overdraft(df, bal, as_of)
        kate(f"Here's the smallest change that works: spend **€{plan['cut']} less per month** on everyday "
             f"expenses. Your overdraft risk drops from **{plan['p_now']:.0%} to {plan['p_after']:.0%}**. "
             "I can also set a soft budget alert and review your insurance to free up margin. Shall I?")
        opts = pd.DataFrame(plan["options"], columns=["cut", "risk"])
        opts["label"] = "−€" + opts["cut"].astype(str) + "/month"
        bars = alt.Chart(opts).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("label:N", sort=None, title="Spend less each month"),
            y=alt.Y("risk:Q", title="Chance of overdraft within 12 months", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("label:N", title="Change"), alt.Tooltip("risk:Q", title="Overdraft risk", format=".0%")])
        st.altair_chart(bars.properties(height=240), use_container_width=True)
    elif nudge.trigger == "saving_goal":
        saving_plan(key, state, ans, df, bal, as_of)
    elif nudge.trigger == "pension":
        monthly, years = 87.5, 67 - synth.CUSTOMERS[cid]["age"]
        fv = sum(monthly * 12 * (1.03 ** (years - y)) for y in range(years))
        kate(f"With €{monthly:.0f}/month in pension saving (the yearly tax-advantaged maximum, assumption: "
             f"€1,050), you'd build about **€{fv:,.0f}** by 67 (3 %/year assumed) and get around "
             f"**€{1050 * 0.30:,.0f} back in taxes every year**. Want to start it now in 2 clicks?")

    if st.button("↩ Reset this conversation", key=f"{key}_reset"):
        st.session_state[key] = {"answer": None}
        st.rerun()


def saving_plan(key, state, goal_choice, df, bal, as_of):
    defaults = {"A home": ("a deposit for a home", 20000, 36),
                "Studies / a master abroad": ("your master in Madrid", 6000, 24),
                "A car": ("a car", 12000, 24),
                "Just a safety buffer": ("a safety buffer", 5000, 18)}
    label, amount, months = defaults.get(goal_choice, ("your goal", 5000, 24))
    kate("Great! Just two details and I'll make you a plan.")
    with st.form(f"{key}_goal"):
        goal = st.text_input("What exactly?", value=label, max_chars=60)
        amount = st.number_input("How much do you need (€)?", 100, intent.MAX_GOAL, amount, step=500)
        months = st.number_input("In how many months?", 1, intent.MAX_MONTHS, months)
        submitted = st.form_submit_button("Make my plan")
    if submitted:
        state["plan"] = intent.plan_for_goal(goal, amount, months, bal["savings"], df, bal, as_of)
    plan = state.get("plan")
    if not plan:
        return
    me(f"{plan['goal']} - €{plan['amount']:,.0f} in {plan['months']} months")
    kate(plan["message"])
    if st.button("✅ Set up the automatic transfer", key=f"{key}_go"):
        state["active"] = True
    if state.get("active"):
        st.success(f"Plan active: €{plan['monthly']}/month on salary day → {plan['goal']}.")
        st.caption("Kate keeps watching. If you're off-plan 2 months in a row, she re-checks your intent.")
        if st.button("⏩ Demo: 2 months later, transfers stopped", key=f"{key}_ff"):
            state["recheck"] = intent.needs_recheck(plan["monthly"], [0.0, 0.0])
        if state.get("recheck"):
            kate(f"Hey - no transfers towards {plan['goal']} for 2 months. Still planning it? "
                 "I can stretch the plan, lower the amount, or pause it. Your call.")


# ---------------------------------------------------------------- advisor view
def advisor_view():
    if not require_role(user, "advisor"):
        st.error("Access denied.")
        st.stop()
    header("KBC Moments - advisor & scale view")

    st.subheader("Customers Kate would talk to today")
    rows = []
    for cid, c in synth.CUSTOMERS.items():
        _, bal, sig, sim, pi, nudge = customer_context(cid, synth.TODAY)
        rows.append({"Customer": f"{cid} ({c['name']})", "Signals": ", ".join(s.title for s in sig) or "-",
                     "Conversation": nudge.trigger if nudge else "none (stay quiet)",
                     "Overdraft risk 12m": f"{sim['p_overdraft_by'][-1]:.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader(f"At scale: {KBC_CUSTOMERS:,} customers")
    st.caption("Monthly trigger rates and answer rates below are ASSUMPTIONS - replace with calibrated values.")
    c1, c2, c3, c4 = st.columns(4)
    r_over = c1.slider("Overdraft-risk trigger / month", 0.0, 5.0, 1.2, 0.1, format="%.1f%%") / 100
    r_goal = c2.slider("New saving habit / month", 0.0, 5.0, 0.8, 0.1, format="%.1f%%") / 100
    r_pens = c3.slider("Pension gap reached / month", 0.0, 5.0, 0.3, 0.1, format="%.1f%%") / 100
    answer = c4.slider("Answer rate (in-context question)", 0, 100, 35, format="%d%%") / 100
    survey = 0.05
    triggers = KBC_CUSTOMERS * (r_over + r_goal + r_pens)
    k1, k2, k3 = st.columns(3)
    k1.metric("Well-timed conversations / month", f"{triggers:,.0f}")
    k2.metric("Declared intents collected / month", f"{triggers * answer:,.0f}")
    k3.metric("vs. a classic survey (5 % answer)", f"{triggers * survey:,.0f}",
              delta=f"x{answer / survey:.0f}")
    st.markdown(
        "**How it scales** - one pipeline for every customer: "
        "`transactions` → **signals** (behaviour change vs own baseline) → **inferred intent** "
        "(signals + peers) → **one question at the right moment** → **declared intent** → **plan** "
        "→ **monitor & re-check**. New life moments or products = new signal/plan rules, not a new app. "
        "The same conversation can run in KBC Mobile (Kate), by e-mail or through an advisor.")


if user["role"] == "advisor":
    advisor_view()
elif user["role"] == "customer":
    customer_view()
else:
    st.error("Unknown role.")
