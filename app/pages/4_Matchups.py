"""Player matchups for the next 7 days: who has the edge against the defense he faces."""
import pandas as pd
import streamlit as st

from common import matchup, picks_for, setup, verdict_badge

from edge import matchups
from edge.config import DATA_DIR
from edge.timeutil import et_display

league = setup("Matchups")
st.title(f"{league.upper()} player matchups · next 7 days")

if not matchups.supported(league):
    st.info("Player matchup reports cover the NFL and NBA. Pick one of those in the sidebar.")
    st.stop()


@st.cache_resource(ttl=3600, show_spinner="Crunching play-by-play...")
def get_context(lg: str, data_dir: str):
    return matchups.context(lg)


ctx = get_context(league, str(DATA_DIR))
games = matchups.upcoming_games(league, 7)
if games.empty:
    st.write("No upcoming games stored yet. The nightly job loads the next 7 days; to do it now, "
             "use **Settings → Load next 7 days**.")
    st.stop()

with st.expander("How to read this", expanded=False):
    st.markdown(
        "- **Baseline** = the player's usual output (his recent average volume x efficiency).\n"
        "- **Matchup** = the baseline adjusted for this defense. **Edge** is the difference in %.\n"
        "- NFL: receivers are scored on scheme fit (his man vs zone splits x how often this defense plays "
        "each) and how many yards per target the defense allows to his position; RBs on run defense; "
        "QBs on pass defense, scheme and pressure.\n"
        "- NBA: points/rebounds/assists the defense allows to his position (guard, forward, center) "
        "vs league average over its last 20 games.\n"
        "- Small samples are pulled toward average, so extreme numbers need real evidence. This is a "
        "scouting report, not a guarantee; `python -m scripts.evaluate_matchups` shows how well it "
        "has tracked real results.")


def fmt_table(df: pd.DataFrame, with_game: bool = False) -> pd.DataFrame:
    out = pd.DataFrame({
        "Player": df.player, "Pos": df.position, "Team": df.team, "Opp": df.opponent,
        "Stat": df.metric, "Baseline": df.baseline.round(1), "Matchup": df.projection.round(1),
        "Edge": (df.advantage * 100).round(1).map(lambda v: f"{v:+.1f}%"),
        "Grade": df.grade if "grade" in df else "",
    })
    if with_game:
        out.insert(0, "Game", df.game_key.map(labels))
    return out


labels = {r.game_key: f"{matchup(league, r.home_team, r.away_team)} · {et_display(r.start_time)}"
          for r in games.itertuples()}

best = matchups.best_matchups(league, list(games.game_key), ctx, top=12)
st.subheader("Best matchups this week")
if best.empty:
    st.write(matchups.report(league, games.game_key.iloc[0], ctx)["data_note"])
else:
    if "grade" not in best:
        from edge.matchups_nfl import _grade
        best["grade"] = best.advantage.map(_grade)
    st.dataframe(fmt_table(best, with_game=True), hide_index=True, width="stretch")

st.divider()
key = st.selectbox("Game", list(labels), format_func=labels.get)
rep = matchups.report(league, key, ctx)
st.caption(rep["data_note"])

pick = picks_for(league)
pick = pick[pick.game_key == key]
if not pick.empty:
    p = pick.iloc[0]
    st.markdown(f"Moneyline: {verdict_badge(p.verdict)} &nbsp; model gives **{p.home_team}** "
                f"{p.home_prob:.0%} to win", unsafe_allow_html=True)

if not rep["sides"]:
    st.stop()

tabs = st.tabs([f"{s['offense']} offense vs {s['defense']} defense" for s in rep["sides"]])
for tab, side in zip(tabs, rep["sides"]):
    with tab:
        st.markdown(f"**{side['defense']} defense**")
        for line in side["defense_profile"]:
            st.write("• " + line)
        t = side["players"]
        if t is None or t.empty:
            st.write("No players with enough recent data.")
            continue
        if "grade" not in t:
            from edge.matchups_nfl import _grade
            t = t.assign(grade=t.advantage.map(_grade))
        st.dataframe(fmt_table(t), hide_index=True, width="stretch")
        for r in t.head(6).itertuples():
            with st.expander(f"{r.player} · {r.metric} {r.advantage * 100:+.0f}% ({r.grade})"):
                for reason in r.reasons:
                    st.write("• " + reason)
                if league == "nba":
                    st.caption(f"Projection: {r.pts_proj:.1f} pts, {r.reb_proj:.1f} reb, {r.ast_proj:.1f} ast, "
                               f"{r.fg3m_proj:.1f} 3PM in ~{r.minutes:.0f} min")

st.caption("Matchup edges describe the opponent, not certainty. Bet only where legal, within a budget.")
