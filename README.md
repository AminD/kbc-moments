# KBC Moments

**Tectonic Hackathon - KBC challenge:** *How can KBC understand what each customer needs and respond at exactly the right moment?*

People ignore surveys. So the way - and the moment - we ask decides whether it works.
KBC Moments watches for **behaviour changes** on the current and savings accounts, asks **one conversational question at the right moment**, turns the answer into a **concrete plan**, and **re-checks** when the pattern moves again.

## The idea

| Step | What happens | Example |
|---|---|---|
| **Understand - behaviour** | Compare each customer to their *own* baseline (current + savings account) | Rent 750 → 980 €, and since then Lina pulls 300 €/month from savings |
| **Understand - inferred intent** | Signals + what customers with the same profile did next | 47 % of people like Marc start pension saving around 52 |
| **Adapt - predict** | Monte-Carlo: 2,000 simulated futures of both accounts | "Savings run out in ~5 months, overdraft risk from ~Jun 2027 (72 %)" |
| **Adapt - ask at the right moment** | One question, triggered by a signal, never a survey | New saving habit → *"Are you saving towards something? Tell us and we'll help you get there."* |
| **Adapt - declared intent → plan** | The answer overrides the guess and becomes a plan | *"To afford your master in Madrid, save 195 €/month"* |
| **Monitor & re-check** | Off-plan 2 months in a row → ask again | *"Still planning Madrid? I can stretch or pause the plan."* |
| **Scale** | One pipeline for 2.3M customers; new moments = new rules; any channel (Kate, e-mail, advisor) | Advisor/scale view with assumption sliders |

We can never guess that someone wants to study abroad in 4 years - that's why **inferred and declared intent are combined**: signals decide *when* to ask, the customer decides *what* the plan is.

## Demo personas (100 % synthetic data)

- **Lina, 27** - rent increase + dipping into savings → overdraft prediction + smallest fix (−120 €/month → risk 72 % → 7 %).
- **Yanis, 23** - first job, new saving habit → "saving towards something?" → master in Madrid plan → re-check.
- **Marc, 52** - stable, no pension saving → peer-based nudge (pension saving + tax benefit).
- **advisor** - which customers Kate would talk to today + projection to 2.3M customers.

Use the **time machine** slider: in June Kate stays quiet; the conversation only starts once behaviour changes.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python setup_secrets.py         # choose a demo password -> .streamlit/secrets.toml (git-ignored)
streamlit run app.py
```

Log in as `lina`, `yanis`, `marc` or `advisor` with the password you chose.

## Structure

```
app.py                  Streamlit UI (customer view + advisor/scale view)
setup_secrets.py        creates hashed demo credentials (never committed)
kbc_moments/
  synth.py              synthetic customers (current + savings) and peer population
  signals.py            behaviour-change detection with evidence ("why am I seeing this?")
  forecast.py           Monte-Carlo forecast of both accounts
  peers.py              "customers like you" - nearest-neighbour peer behaviour
  intent.py             nudge selection, goal plan, overdraft plan, re-check rule
  auth.py               PBKDF2 password hashing, login throttling, role checks
```

## Security

- No secrets in the repo: credentials are salted PBKDF2 hashes in a git-ignored file.
- **No IDOR:** the customer id comes from the authenticated session, never from user input or the URL.
- Advisor view requires the `advisor` role. The advisor list shows pseudonymous ids.
- User input (declared goals) is length-limited, range-checked and rendered as plain text.
- Brute-force throttling on login (5 attempts / 5 minutes per username).

## Limitations / unfinished

- All data is synthetic; peer-behaviour probabilities and scale rates are **assumptions**, to be calibrated on real, consented data.
- Monte-Carlo assumes fixed flows stay fixed and days are independent; no rare events.
- Declared intents and plans live in the session only (no database).
- Pension figures (1,050 €/year, 30 % tax reduction, 3 % return) are illustrative assumptions.
