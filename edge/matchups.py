"""One entry point for player matchup reports, whatever the sport.

    report(league, game_key)      -> {"sides": [...], "data_note": str}
    best_matchups(league, keys)   -> DataFrame of the strongest edges across games
    upcoming_games(league, days)  -> the next week's schedule

Each side = one offense vs the defense it faces:
    {"offense", "defense", "defense_profile": [str], "players": DataFrame with
     player, position, metric, baseline, projection, advantage, grade, reasons}
"""
from datetime import timedelta

import pandas as pd

from edge import db
from edge.config import league_config
from edge.timeutil import et_date, et_today

SUPPORTED = {"nfl", "nba"}
MIN_BASELINE = {"nfl": 20.0, "nba": 8.0}     # hide fringe players from "best of the week"


def supported(league: str) -> bool:
    return league in SUPPORTED and league_config(league).get("sport", "team") != "tennis"


def context(league: str):
    """Expensive shared data, built once per refresh (the app caches it)."""
    if league == "nfl":
        from edge.matchups_nfl import NflContext, default_seasons
        return NflContext(seasons=default_seasons())
    if league == "nba":
        from edge.matchups_nba import NbaContext
        return NbaContext()
    return None


def report(league: str, game_key: str, ctx=None) -> dict:
    ctx = context(league) if ctx is None else ctx
    if league == "nfl":
        from edge.matchups_nfl import game_report
        return game_report(league, game_key, ctx)
    if league == "nba":
        from edge.matchups_nba import game_report
        return game_report(league, game_key, ctx)
    return {"game_key": game_key, "sides": [], "data_note": "Matchup reports cover the NFL and NBA."}


def upcoming_games(league: str, days: int = 7) -> pd.DataFrame:
    g = db.games_df(league)
    if g.empty:
        return g
    start, end = et_today(), et_today() + timedelta(days=days)
    dates = g.start_time.map(et_date)
    return g[(dates >= start) & (dates <= end) & g.home_score.isna()].sort_values("start_time")


def best_matchups(league: str, game_keys: list[str], ctx=None, top: int = 10) -> pd.DataFrame:
    ctx = context(league) if ctx is None else ctx
    frames = []
    for key in game_keys:
        for side in report(league, key, ctx)["sides"]:
            t = side["players"]
            if t is not None and not t.empty:
                frames.append(t.assign(game_key=key))
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    allp = allp[allp.baseline >= MIN_BASELINE.get(league, 0)]
    return allp.sort_values("advantage", ascending=False).head(top)
