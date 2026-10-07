"""Shared bits for every page: imports, sidebar, styling, data helpers.
The app only ever READS the database; the scheduler does all the fetching."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `import edge` works

import pandas as pd            # noqa: E402
import streamlit as st         # noqa: E402

from edge import db            # noqa: E402
from edge.config import DATA_DIR, all_leagues   # noqa: E402

VERDICT_COLORS = {"Bet": "#1a7f37", "Lean": "#b58900", "Pass": "#6e7781"}
STATUS_TEXT = {
    "active": ("Active", "Enough graded bets; results are being tracked."),
    "learning": ("Still learning", "Fewer than 50 graded bets. Treat results as unproven."),
    "pass_only": ("Pass only", "Tracked ROI and CLV are negative. No bets until it improves."),
    "drift": ("Drift warning", "Recent accuracy is worse than in testing."),
}

CSS = """
<style>
.verdict {display:inline-block;padding:4px 14px;border-radius:999px;color:white;
          font-weight:700;font-size:1.05rem;letter-spacing:.02em}
.muted {color:#6e7781;font-size:.9rem}
.reason {margin:2px 0;font-size:.95rem}
.factor {margin:2px 0;font-size:.88rem;color:#57606a}
</style>
"""


def setup(title: str):
    st.set_page_config(page_title=f"Edge Finder · {title}", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    db.init()
    if "demo" in str(DATA_DIR).lower():
        st.warning("DEMO MODE: simulated games and odds, not real sports data.", icon="🧪")
    leagues = all_leagues()
    if "league" not in st.session_state:
        st.session_state.league = leagues[0]
    st.sidebar.selectbox("League", leagues, key="league", format_func=str.upper)
    st.sidebar.caption("Model estimates, not guarantees. Bet only where legal, within a budget.")
    return st.session_state.league


def verdict_badge(v: str) -> str:
    return f'<span class="verdict" style="background:{VERDICT_COLORS.get(v, "#6e7781")}">{v}</span>'


def fmt_ml(ml) -> str:
    if ml is None or pd.isna(ml):
        return "n/a"
    ml = int(ml)
    return f"+{ml}" if ml > 0 else str(ml)


@st.cache_data(ttl=60)
def picks_for(league: str) -> pd.DataFrame:
    return db.read_sql(
        "SELECT p.*, g.start_time, g.home_team, g.away_team, g.season_type, g.status, "
        "g.home_score, g.away_score FROM picks p JOIN games g ON g.game_key = p.game_key "
        "WHERE p.league = :lg ORDER BY g.start_time", lg=league)
