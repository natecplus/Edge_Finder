"""Elo ratings: the baseline model and the strongest single ML feature.

Every team starts at 1500. After each game the winner takes points from the
loser; upsets move more points. We record each team's rating BEFORE every game,
so using it as a feature never leaks the result.
"""
import pandas as pd


def expected(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def run_elo(games: pd.DataFrame, k: float = 20, home_adv: float = 70,
            keep: float = 2 / 3) -> tuple[pd.DataFrame, dict]:
    """games: rows with game_key, season, season_type, start_time, home_team,
    away_team, home_score, away_score (scores NaN for unplayed games).

    Returns (pregame, ratings):
      pregame  -> DataFrame indexed by game_key with home_elo, away_elo (before the game)
      ratings  -> current rating per team after all finished games
    Preseason games are skipped: teams rest starters, so results mean little.
    """
    ratings: dict[str, float] = {}
    season = None
    rows = []
    g = games[games.season_type != "preseason"].sort_values("start_time")
    for row in g.itertuples(index=False):
        if row.season != season:
            # New season: rosters change, so pull everyone 1/3 of the way back to 1500.
            ratings = {t: 1500 + keep * (r - 1500) for t, r in ratings.items()}
            season = row.season
        rh = ratings.get(row.home_team, 1500.0)
        ra = ratings.get(row.away_team, 1500.0)
        rows.append((row.game_key, rh, ra))
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue                       # unplayed: record pregame rating, no update
        p_home = expected(rh + home_adv, ra)
        result = 1.0 if row.home_score > row.away_score else 0.0 if row.home_score < row.away_score else 0.5
        # Margin-of-victory multiplier (FiveThirtyEight-style), damped for favorites.
        margin = abs(row.home_score - row.away_score)
        diff = (rh + home_adv - ra) * (1 if result == 1 else -1)
        mult = ((margin + 3) ** 0.8) / (7.5 + 0.006 * diff) if result != 0.5 else 1.0
        delta = k * mult * (result - p_home)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
    pregame = pd.DataFrame(rows, columns=["game_key", "home_elo", "away_elo"]).set_index("game_key")
    return pregame, ratings
