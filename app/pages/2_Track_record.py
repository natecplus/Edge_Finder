"""The page that proves (or disproves) the tool works."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from common import STATUS_TEXT, setup

from edge import tracking
from edge.train import active_row

league = setup("Track record")
st.title(f"{league.upper()} track record")

model = active_row(league)
s = tracking.summary(league)
status = tracking.league_status(league, model["test_log_loss"] if model else None)

c = st.columns(5)
c[0].metric("Status", STATUS_TEXT[status][0], help=STATUS_TEXT[status][1])
c[1].metric("Graded Bets", s["bets"], help="Flat $100 per Bet. Leans and Passes don't count toward ROI.")
c[2].metric("ROI", "n/a" if s["roi"] is None else f"{s['roi']:+.1%}",
            help="Profit / amount wagered.")
c[3].metric("Avg CLV", "n/a" if s["avg_clv"] is None else f"{s['avg_clv'] * 100:+.2f} pts",
            help="Closing line value: did the market move toward our pick after we locked it? "
                 "Consistently positive CLV is the best early sign of a real edge.")
c[4].metric("Beat the close", "n/a" if s["beat_close_rate"] is None else f"{s['beat_close_rate']:.0%}")
if s["bets"] and s["bets"] < 100:
    st.info(f"Only {s['bets']} graded bets: ROI this early is mostly luck. Watch CLV and calibration instead.")

g = tracking.graded(league)
bets = g[g.verdict == "Bet"].copy()
if not bets.empty:
    bets["start_time"] = pd.to_datetime(bets.start_time)
    bets["cumulative_profit"] = bets.profit.cumsum()
    st.subheader("Cumulative profit ($100 per Bet)")
    st.altair_chart(alt.Chart(bets).mark_line().encode(
        x=alt.X("start_time:T", title=None), y=alt.Y("cumulative_profit:Q", title="Profit ($)"),
        tooltip=["game_key", "pick_team", "odds_taken", "won", "profit", "cumulative_profit"]),
        width="stretch")

cal = tracking.calibration(league)
if not cal.empty:
    st.subheader("Calibration: when the model says X%, does it happen X% of the time?")
    diag = pd.DataFrame({"predicted": [0, 1], "actual": [0, 1]})
    pts = alt.Chart(cal).mark_circle().encode(
        x=alt.X("predicted:Q", title="Model said (home win %)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("actual:Q", title="Actually happened", scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("n:Q", title="Games"), tooltip=["predicted", "actual", "n"])
    line = alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="predicted", y="actual")
    st.altair_chart(line + pts, width="stretch")
    st.caption("Dots on the dashed line = well calibrated. Live picks only.")

if model:
    st.subheader("Active model: how it scored before going live")
    r = json.loads(model["report"])
    rows = [("Calibrated " + model["kind"], r["test"]["log_loss"]),
            ("Elo baseline", r["test_elo_baseline"]["log_loss"])]
    if r.get("market_log_loss") is not None:
        rows.append(("Closing market", r["market_log_loss"]))
    st.dataframe(pd.DataFrame(rows, columns=["Predictor", f"Log loss, {r['test_season']} season (lower = better)"]),
                 hide_index=True, width="stretch")
    bt = r["backtest_test"]
    st.write(f"Backtest on the {r['test_season']} season at a {r['bet_threshold']:.0%} edge threshold: "
             f"**{bt['bets']} bets, ROI {bt['roi']:+.1%}** at closing prices.")
    st.caption("Beating the Elo baseline is expected. Getting close to the market is the hard part; "
               "a model clearly worse than the market will mostly find 'edges' that are its own errors.")

if not g.empty:
    st.subheader("Recent graded picks")
    show = g.sort_values("start_time", ascending=False).head(50)[
        ["start_time", "away_team", "home_team", "verdict", "pick_team", "odds_taken", "edge", "won", "profit", "clv"]]
    st.dataframe(show, hide_index=True, width="stretch")
