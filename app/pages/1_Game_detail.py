"""Everything behind one pick: ratings, injuries, odds movement, rules fired."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from common import fmt_ml, picks_for, setup, verdict_badge

from edge import db
from edge.config import league_config
from edge.features import build_features
from edge.odds_math import fair_probs
from edge.timeutil import et_display

league = setup("Game detail")
st.title("Game detail")

picks = picks_for(league).sort_values("start_time", ascending=False)
if picks.empty:
    st.write("No picks yet.")
    st.stop()

labels = {r.game_key: f"{r.away_team} @ {r.home_team} · {et_display(r.start_time)} · {r.verdict}"
          for r in picks.itertuples()}
key = st.selectbox("Game", list(labels), format_func=labels.get)
p = picks[picks.game_key == key].iloc[0]

st.markdown(f"### {p.away_team} @ {p.home_team} &nbsp; {verdict_badge(p.verdict)}", unsafe_allow_html=True)
st.caption(f"{et_display(p.start_time)} · {p.season_type} · model {p.model_version}")

c = st.columns(4)
c[0].metric("Model P(home wins)", f"{p.home_prob:.1%}")
if isinstance(p.side, str):
    c[1].metric(f"Pick: {p.pick_team}", f"{p.model_prob:.1%}", f"{p.edge * 100:+.1f} pts edge")
    c[2].metric("Market fair %", f"{p.fair_prob:.1%}")
    c[3].metric("Price", f"{fmt_ml(p.odds_taken)} ({p.book})", f"EV ${p.ev:+.2f}/$100")

st.subheader("Why")
for r in (p.reasons or "").split("|"):
    if r:
        st.write("• " + r.replace("Model factor: ", "Model factor · "))
fired = json.loads(p.rules_fired or "[]")
st.caption("Rules fired: " + (", ".join(fired) if fired else "none"))


@st.cache_data(ttl=300)
def team_ratings(lg: str) -> pd.DataFrame:
    games = db.games_df(lg)
    return build_features(games, league_config(lg)).set_index("game_key")[
        ["home_elo", "away_elo", "form_home", "form_away", "rest_home", "rest_away",
         "gp_home", "gp_away", "miss_home", "miss_away"]]


feats = team_ratings(league)
if key in feats.index:
    f = feats.loc[key]
    st.subheader("Matchup")
    st.dataframe(pd.DataFrame({
        "": ["Elo rating (before game)", "Avg margin, recent games", "Days of rest", "Games played"],
        p.away_team: [f"{f.away_elo:.0f}", f"{f.form_away:+.1f}" if pd.notna(f.form_away) else "-",
                      f"{f.rest_away:.0f}", int(f.gp_away)],
        p.home_team: [f"{f.home_elo:.0f}", f"{f.form_home:+.1f}" if pd.notna(f.form_home) else "-",
                      f"{f.rest_home:.0f}", int(f.gp_home)],
    }), hide_index=True, width="stretch")

inj = db.read_sql("SELECT team, player, status, position, captured_at FROM injuries WHERE game_key = :k "
                  "ORDER BY captured_at DESC", k=key)
st.subheader("Injury report")
if inj.empty:
    st.write("No injuries reported.")
else:
    latest = inj[inj.captured_at == inj.captured_at.max()]
    st.dataframe(latest.drop(columns="captured_at"), hide_index=True, width="stretch")
    st.caption(f"As of {latest.captured_at.iloc[0]} UTC")

odds = db.read_sql("SELECT book, home_ml, away_ml, kind, captured_at FROM odds_snapshots WHERE game_key = :k "
                   "ORDER BY captured_at", k=key)
st.subheader("Odds movement")
if odds.empty:
    st.write("No odds captured.")
else:
    odds["home_fair_%"] = [fair_probs(h, a)[0] * 100 for h, a in zip(odds.home_ml, odds.away_ml)]
    odds["captured_at"] = pd.to_datetime(odds.captured_at)
    chart = alt.Chart(odds).mark_line(point=True).encode(
        x=alt.X("captured_at:T", title="Captured (UTC)"),
        y=alt.Y("home_fair_%:Q", title=f"Market fair % for {p.home_team}", scale=alt.Scale(zero=False)),
        color=alt.Color("book:N", title="Book"),
        tooltip=["book", "kind", "home_ml", "away_ml", alt.Tooltip("home_fair_%:Q", format=".1f")])
    st.altair_chart(chart, width="stretch")
    with st.expander("All snapshots"):
        st.dataframe(odds, hide_index=True, width="stretch")
