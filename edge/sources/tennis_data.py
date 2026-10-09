"""tennis-data.co.uk adapter: ATP results with closing odds, one Excel file per year.

Columns used: Date, Tournament, Series, Surface, Round, Best of, Winner, Loser,
WRank, LRank, Wsets, Lsets, Comment, and the odds pairs PSW/PSL (Pinnacle),
B365W/B365L (Bet365), AvgW/AvgL (market average). Odds are decimal; we convert
to American. The current year's file is updated during the season, so the
weekly retrain re-reads it.
"""
import io
from datetime import datetime, timezone

import pandas as pd

from edge.config import RAW_DIR, league_config
from edge.http import get
from edge.players import ordered
from edge.timeutil import to_iso
from .base import Game, OddsSnapshot
from .espn_tennis import TennisMatch

BOOKS = {"Pinnacle": ("PSW", "PSL"), "Bet365": ("B365W", "B365L"), "Average": ("AvgW", "AvgL")}


def decimal_to_american(d) -> int | None:
    try:
        d = float(d)
    except (TypeError, ValueError):
        return None
    if not d or d <= 1.0 or pd.isna(d):
        return None
    return round((d - 1) * 100) if d >= 2.0 else round(-100 / (d - 1))


def _rank(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def parse_history(df: pd.DataFrame, league: str = "atp") -> list[TennisMatch]:
    out = []
    for r in df.to_dict("records"):
        winner, loser = str(r.get("Winner") or "").strip(), str(r.get("Loser") or "").strip()
        comment = str(r.get("Comment") or "Completed").strip()
        if not winner or not loser or winner == "nan" or comment.lower().startswith("walkover"):
            continue                                   # no match was played
        when = pd.Timestamp(r["Date"])
        start = to_iso(datetime(when.year, when.month, when.day, 12, tzinfo=timezone.utc))
        home, away = ordered(winner, loser)            # alphabetical, never winner-first
        w_sets = int(r["Wsets"]) if pd.notna(r.get("Wsets")) else 1
        l_sets = int(r["Lsets"]) if pd.notna(r.get("Lsets")) else 0
        if w_sets <= l_sets:                           # retirement: winner still gets the edge
            w_sets = l_sets + 1
        sets = {winner: w_sets, loser: l_sets}
        ranks = {winner: _rank(r.get("WRank")), loser: _rank(r.get("LRank"))}
        game = Game(league, None, start, when.year, "regular", home, away,
                    sets[home], sets[away], "STATUS_FINAL")
        meta = {"game_key": game.game_key, "tournament": str(r.get("Tournament") or ""),
                "series": str(r.get("Series") or ""), "surface": str(r.get("Surface") or ""),
                "round": str(r.get("Round") or ""),
                "best_of": int(r["Best of"]) if pd.notna(r.get("Best of")) else 3,
                "home_rank": ranks[home], "away_rank": ranks[away], "comment": comment}
        odds = []
        for book, (wcol, lcol) in BOOKS.items():
            w, l = decimal_to_american(r.get(wcol)), decimal_to_american(r.get(lcol))
            if w and l:
                price = {winner: w, loser: l}
                odds.append(OddsSnapshot(game.game_key, book, price[home], price[away], "close", "tennis-data"))
        out.append(TennisMatch(game, meta, odds))
    return out


class TennisDataHistory:
    name = "tennis-data"

    def __init__(self, use_cache: bool = False):
        self.use_cache = use_cache

    def season(self, league: str, year: int) -> list[TennisMatch]:
        url = league_config(league)["history_url"].format(year=year)
        cache = RAW_DIR / "tennis-data" / f"{year}.xlsx"
        if self.use_cache and cache.exists():
            content = cache.read_bytes()
        else:
            content = get(url, delay=2.0, timeout=60).content
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(content)
        df = pd.read_excel(io.BytesIO(content))
        return parse_history(df, league)
