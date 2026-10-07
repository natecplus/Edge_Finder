"""Today's slate: one card per game with Bet / Lean / Pass and the reasons why.

Run:  streamlit run app/streamlit_app.py
"""
from datetime import timedelta

import pandas as pd
import streamlit as st

from common import STATUS_TEXT, fmt_ml, picks_for, setup, verdict_badge

from edge import tracking
from edge.timeutil import et_date, et_display, et_today
from edge.train import active_row

league = setup("Today")

day = st.sidebar.date_input("Date", et_today(), min_value=et_today() - timedelta(days=60),
                            max_value=et_today() + timedelta(days=7))
only_bets = st.sidebar.toggle("Show Bets and Leans only", False)

st.title(f"{league.upper()} slate · {day:%a %b %d}")

model = active_row(league)
status = tracking.league_status(league, model["test_log_loss"] if model else None)
label, help_text = STATUS_TEXT[status]
cols = st.columns(3)
cols[0].metric("League status", label, help=help_text)
if model:
    cols[1].metric("Model", model["kind"], help=f"Version {model['version']}, trained {model['trained_at']}")
    cols[2].metric("Bet threshold", f"{model['bet_threshold']:.0%} edge",
                   help="Chosen on the validation season. Edge = model win % minus the market's fair win %.")
else:
    st.info("No trained model yet. Run `python -m scripts.train_model --league %s`." % league)

picks = picks_for(league)
picks = picks[picks.start_time.map(lambda s: et_date(s) == day)]
if only_bets:
    picks = picks[picks.verdict.isin(["Bet", "Lean"])]
if picks.empty:
    st.write("No games with picks for this date. (The scheduler makes picks on game days; "
             "check the Settings page for data health.)")
    st.stop()

order = {"Bet": 0, "Lean": 1, "Pass": 2}
picks = picks.assign(_o=picks.verdict.map(order)).sort_values(["_o", "start_time"])

counts = picks.verdict.value_counts()
st.caption(" · ".join(f"{counts.get(v, 0)} {v}" for v in ("Bet", "Lean", "Pass")))

for p in picks.itertuples():
    with st.container(border=True):
        left, mid, right = st.columns([3, 3, 1.3])
        left.subheader(f"{p.away_team} @ {p.home_team}")
        line = et_display(p.start_time)
        if pd.notna(p.home_score) and pd.notna(p.away_score) and "FINAL" in str(p.status):
            line += f" · Final {int(p.away_score)}-{int(p.home_score)}"
        if isinstance(p.locked_at, str):
            line += " · 🔒 locked"
        left.markdown(f'<span class="muted">{line}</span>', unsafe_allow_html=True)

        if isinstance(p.side, str):
            mid.metric(f"Model win % for {p.pick_team}", f"{p.model_prob:.0%}",
                       f"{p.edge * 100:+.1f} pts vs market ({p.fair_prob:.0%})")
            mid.markdown(f'<span class="muted">Best price {fmt_ml(p.odds_taken)} at {p.book} · '
                         f'EV ${p.ev:+.2f} per $100</span>', unsafe_allow_html=True)
        else:
            mid.metric("Home win %", f"{p.home_prob:.0%}", "no odds")
        right.markdown(verdict_badge(p.verdict), unsafe_allow_html=True)

        reasons = [r for r in (p.reasons or "").split("|") if r]
        main = [r for r in reasons if not r.startswith("Model factor:")]
        factors = [r.replace("Model factor: ", "") for r in reasons if r.startswith("Model factor:")]
        for r in main[:4]:
            st.markdown(f'<div class="reason">• {r}</div>', unsafe_allow_html=True)
        if factors:
            with st.expander("What drives the model's number"):
                for f in factors:
                    st.markdown(f'<div class="factor">{f}</div>', unsafe_allow_html=True)
                st.caption("Points = how much each factor moves the win probability compared with an "
                           "average game. The edge comes from comparing the total with the market.")

st.caption("Model estimates, not guarantees. Bet only where it's legal for you, within a budget.")
