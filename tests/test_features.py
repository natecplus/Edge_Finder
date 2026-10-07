"""The leakage test: features for a game must be identical whether or not the
future exists. If someone 'improves' a feature with data from after tip-off,
this fails."""
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from edge.config import league_config
from edge.features import build_features

TEAMS = ["A", "B", "C", "D", "E", "F"]


def make_games(n_days=120, seed=1):
    rng = random.Random(seed)
    rows, t0 = [], datetime(2024, 10, 20, 23, tzinfo=timezone.utc)
    for d in range(n_days):
        teams = rng.sample(TEAMS, 4)
        season = 2025 if d < 60 else 2026
        for j in (0, 2):
            start = (t0 + timedelta(days=d, hours=j)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append({"game_key": f"g{d}_{j}", "league": "nba", "season": season,
                         "season_type": "regular", "start_time": start,
                         "home_team": teams[j], "away_team": teams[j + 1],
                         "home_score": rng.randint(90, 130), "away_score": rng.randint(90, 130),
                         "status": "STATUS_FINAL"})
    return pd.DataFrame(rows)


COLS = ["elo_diff", "form_diff", "season_margin_diff", "rest_diff", "b2b_home", "b2b_away",
        "gp_home", "gp_away"]


def test_no_leakage():
    cfg = league_config("nba")
    games = make_games()
    full = build_features(games, cfg).set_index("game_key")
    rng = random.Random(3)
    for key in rng.sample(list(games.game_key), 25):
        start = games.loc[games.game_key == key, "start_time"].item()
        # the world as it looked just before this game: no later games, its own score unknown
        past = games[games.start_time <= start].copy()
        past.loc[past.game_key == key, ["home_score", "away_score"]] = np.nan
        past.loc[past.game_key == key, "status"] = "STATUS_SCHEDULED"
        partial = build_features(past, cfg).set_index("game_key")
        for col in COLS:
            assert full.loc[key, col] == partial.loc[key, col] or (
                abs(full.loc[key, col] - partial.loc[key, col]) < 1e-9), (key, col)


def test_label_and_rest():
    cfg = league_config("nba")
    df = build_features(make_games(10), cfg)
    assert set(df.home_win.dropna().unique()) <= {0.0, 1.0}
    assert (df.rest_home >= 0).all() and (df.rest_home <= 7).all()
