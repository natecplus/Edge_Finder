"""Everything behind one pick: ratings, injuries, odds movement, rules fired,
and a form to enter a price by hand (handy for tennis, where free odds are thin)."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from common import fmt_ml, is_tennis, matchup, picks_for, setup, verdict_badge

from edge import db
from edge.config import league_config
from edge.features import build_features
from edge.odds_math import fair_probs
from edge.sources.base import OddsSnapshot
from edge.tennis_features import rank_text
from edge.timeutil import et_display, parse_utc, utcnow

league = setup("Game detail")
tennis = is_tennis(league)
st.title("Match detail" if tennis else "Game detail")

picks = picks_for(league).sort_values("start_time", ascending=False)
if picks.empty:
    st.write("No picks yet.")
    st.stop()

labels = {r.game_key: f"{matchup(league, r.home_team, r.away_team)} · {et_display(r.start_time)} · {r.verdict}"
          for r in picks.itertuples()}
key = st.selectbox("Match" if tennis else "Game", list(labels), format_func=labels.get)
p = picks[picks.game_key == key].iloc[0]

st.markdown(f"### {matchup(league, p.home_team, p.away_team)} &nbsp; {verdict_badge(p.verdict)}",
            unsafe_allow_html=True)
where = f"{p.tournament} · {p.round} · {p.surface} · " if tennis and isinstance(p.tournament, str) else ""
st.caption(f"{where}{et_display(p.start_time)} · {p.season_type} · model {p.model_version}")

c = st.columns(4)
c[0].metric(f"Model P({p.home_team} wins)" if tennis else "Model P(home wins)", f"{p.home_prob:.1%}")
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

# ------------------------------------------------------------ manual price entry
started = parse_utc(p.start_time) <= utcnow()
locked = isinstance(p.locked_at, str)
with st.expander("Enter odds by hand", expanded=tennis and not isinstance(p.side, str) and not started):
    if started or locked:
        st.write("This pick is locked (the match has started or is about to).")
    else:
        with st.form("manual_odds"):
            st.caption("American odds from your sportsbook, e.g. -150 or +130. The pick is recalculated right away.")
            f1, f2, f3 = st.columns(3)
            h_ml = f1.number_input(f"{p.home_team}", value=-110, step=5)
            a_ml = f2.number_input(f"{p.away_team}", value=-110, step=5)
            book = f3.text_input("Sportsbook", "Manual")
            if st.form_submit_button("Save price and recalculate", type="primary"):
                if abs(h_ml) < 100 or abs(a_ml) < 100:
                    st.error("American odds are at least 100 in size (e.g. -110, +120).")
                else:
                    db.insert_odds([OddsSnapshot(key, book or "Manual", int(h_ml), int(a_ml), "live", "manual")])
                    from edge.predict import predict_league
                    predict_league(league, keys=[key])
                    st.cache_data.clear()
                    st.rerun()


# ------------------------------------------------------------ matchup
@st.cache_data(ttl=300)
def ratings(lg: str) -> pd.DataFrame:
    return build_features(db.games_df(lg), league_config(lg)).set_index("game_key")


feats = ratings(league)
if key in feats.index:
    f = feats.loc[key]
    st.subheader("Matchup")
    if tennis:
        table = {
            "": ["Overall Elo", f"{f.surface} Elo", "Ranking", "Won, last 10 matches",
                 "Matches in last 7 days", "Days since last match", "Matches this season"],
            p.home_team: [f"{f.home_elo:.0f}", f"{f.home_surf_elo:.0f}", rank_text(f.rank_home),
                          "-" if pd.isna(f.form_home) else f"{f.form_home:.0%}", int(f.fat_home),
                          f"{f.rest_home:.0f}", int(f.gp_home)],
            p.away_team: [f"{f.away_elo:.0f}", f"{f.away_surf_elo:.0f}", rank_text(f.rank_away),
                          "-" if pd.isna(f.form_away) else f"{f.form_away:.0%}", int(f.fat_away),
                          f"{f.rest_away:.0f}", int(f.gp_away)],
        }
    else:
        table = {
            "": ["Elo rating (before game)", "Avg margin, recent games", "Days of rest", "Games played"],
            p.away_team: [f"{f.away_elo:.0f}", f"{f.form_away:+.1f}" if pd.notna(f.form_away) else "-",
                          f"{f.rest_away:.0f}", int(f.gp_away)],
            p.home_team: [f"{f.home_elo:.0f}", f"{f.form_home:+.1f}" if pd.notna(f.form_home) else "-",
                          f"{f.rest_home:.0f}", int(f.gp_home)],
        }
    st.dataframe(pd.DataFrame({k: [str(v) for v in vals] for k, vals in table.items()}),
                 hide_index=True, width="stretch")

if not tennis:
    inj = db.read_sql("SELECT team, player, status, position, captured_at FROM injuries WHERE game_key = :k "
                      "ORDER BY captured_at DESC", k=key)
    st.subheader("Injury report")
    if inj.empty:
        st.write("No injuries reported.")
    else:
        latest = inj[inj.captured_at == inj.captured_at.max()]
        st.dataframe(latest.drop(columns="captured_at"), hide_index=True, width="stretch")
        st.caption(f"As of {latest.captured_at.iloc[0]} UTC")

odds = db.read_sql("SELECT book, home_ml, away_ml, kind, source, captured_at FROM odds_snapshots "
                   "WHERE game_key = :k ORDER BY captured_at", k=key)
st.subheader("Odds movement")
if odds.empty:
    st.write("No odds captured yet.")
else:
    odds["home_fair_%"] = [fair_probs(h, a)[0] * 100 for h, a in zip(odds.home_ml, odds.away_ml)]
    odds["captured_at"] = pd.to_datetime(odds.captured_at)
    chart = alt.Chart(odds).mark_line(point=True).encode(
        x=alt.X("captured_at:T", title="Captured (UTC)"),
        y=alt.Y("home_fair_%:Q", title=f"Market fair % for {p.home_team}", scale=alt.Scale(zero=False)),
        color=alt.Color("book:N", title="Book"),
        tooltip=["book", "kind", "source", "home_ml", "away_ml", alt.Tooltip("home_fair_%:Q", format=".1f")])
    st.altair_chart(chart, width="stretch")
    with st.expander("All snapshots"):
        st.dataframe(odds, hide_index=True, width="stretch")
