import pandas as pd
from pytest import approx

from edge.elo import expected, run_elo


def _games(rows):
    return pd.DataFrame(rows, columns=["game_key", "season", "season_type", "start_time",
                                       "home_team", "away_team", "home_score", "away_score"])


def test_expected_symmetry():
    assert expected(1500, 1500) == approx(0.5)
    assert expected(1600, 1500) + expected(1500, 1600) == approx(1.0)


def test_winner_gains_and_pregame_is_before_result():
    g = _games([
        ("g1", 2025, "regular", "2025-01-01T00:00:00Z", "A", "B", 110, 100),
        ("g2", 2025, "regular", "2025-01-03T00:00:00Z", "A", "B", None, None),
    ])
    pregame, ratings = run_elo(g)
    assert pregame.loc["g1", "home_elo"] == 1500          # nobody had played yet
    assert ratings["A"] > 1500 > ratings["B"]
    assert pregame.loc["g2", "home_elo"] == ratings["A"]  # unplayed game sees the latest rating
    assert ratings["A"] + ratings["B"] == approx(3000)    # points are only transferred


def test_preseason_ignored():
    g = _games([("p1", 2025, "preseason", "2025-01-01T00:00:00Z", "A", "B", 150, 80)])
    _, ratings = run_elo(g)
    assert ratings == {}


def test_new_season_regresses_to_mean():
    g = _games([
        ("g1", 2024, "regular", "2024-01-01T00:00:00Z", "A", "B", 130, 90),
        ("g2", 2025, "regular", "2025-01-01T00:00:00Z", "A", "B", None, None),
    ])
    _, after_2024 = run_elo(g.iloc[:1])
    end_2024 = after_2024["A"]
    pregame, _ = run_elo(g)
    assert pregame.loc["g2", "home_elo"] == approx(1500 + (end_2024 - 1500) * 2 / 3)
