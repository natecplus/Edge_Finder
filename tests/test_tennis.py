"""Tennis: name matching, parsers, no result leakage, de-duplication."""
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import approx

from edge import db
from edge.config import RAW_DIR, league_config
from edge.players import PlayerIndex, display_name, ordered, player_key
from edge.sources.espn_tennis import parse_scoreboard
from edge.sources.tennis_data import TennisDataHistory, decimal_to_american, parse_history
from edge.tennis_features import TENNIS_FEATURES, build_tennis_features
from edge.tennis_ingest import Existing, _write

FIX = Path(__file__).parent / "fixtures"


def test_name_matching():
    assert player_key("Alex de Minaur") == player_key("De Minaur A.")
    assert player_key("Zhizhen Zhang") == player_key("Zhang Zh.")
    assert player_key("Félix Auger-Aliassime") == player_key("Auger-Aliassime F.")
    assert display_name("Alex de Minaur") == "De Minaur A."
    idx = PlayerIndex(["Etcheverry T.", "Zhang Zh.", "Alcaraz C."])
    assert idx.resolve("Tomás Martín Etcheverry") == "Etcheverry T."     # loose surname match
    assert idx.resolve("Zhizhen Zhang") == "Zhang Zh."
    assert idx.resolve("New Guy") == "Guy N."
    assert ordered("Sinner J.", "Alcaraz C.") == ("Alcaraz C.", "Sinner J.")


def history_frame():
    return pd.DataFrame([
        {"Date": pd.Timestamp(2025, 10, 5), "Tournament": "Shanghai Masters", "Series": "Masters 1000",
         "Surface": "Hard", "Round": "2nd Round", "Best of": 3, "Winner": "Sinner J.", "Loser": "Alcaraz C.",
         "WRank": 2, "LRank": 1, "Wsets": 2, "Lsets": 1, "Comment": "Completed",
         "PSW": 2.5, "PSL": 1.6, "B365W": 2.4, "B365L": 1.57, "AvgW": None, "AvgL": None},
        {"Date": pd.Timestamp(2025, 10, 6), "Tournament": "Shanghai Masters", "Series": "Masters 1000",
         "Surface": "Hard", "Round": "3rd Round", "Best of": 3, "Winner": "Zverev A.", "Loser": "Fritz T.",
         "WRank": 3, "LRank": 4, "Wsets": None, "Lsets": None, "Comment": "Walkover",
         "PSW": None, "PSL": None},
    ])


def test_tennis_data_parser():
    assert decimal_to_american(2.5) == 150 and decimal_to_american(1.5) == -200
    matches = parse_history(history_frame())
    assert len(matches) == 1                                   # walkover dropped
    m = matches[0]
    # alphabetical order: the winner (Sinner) is NOT first, so slot order can't leak the result
    assert (m.game.home_team, m.game.away_team) == ("Alcaraz C.", "Sinner J.")
    assert (m.game.home_score, m.game.away_score) == (1, 2)
    assert (m.meta["home_rank"], m.meta["away_rank"]) == (1, 2)
    pin = [o for o in m.odds if o.book == "Pinnacle"][0]
    assert (pin.home_ml, pin.away_ml) == (decimal_to_american(1.6), 150)   # prices follow the players
    assert {o.book for o in m.odds} == {"Pinnacle", "Bet365"}


def test_history_reads_excel(tmp_path):
    path = RAW_DIR / "tennis-data" / "2025.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    history_frame().to_excel(path, index=False)
    matches = TennisDataHistory(use_cache=True).season("atp", 2025)
    assert len(matches) == 1 and matches[0].meta["surface"] == "Hard"


def test_espn_tennis_parser():
    data = json.loads((FIX / "sample_espn_tennis.json").read_text(encoding="utf-8"))
    matches = parse_scoreboard(data)
    assert len(matches) == 2                                   # doubles + TBD skipped
    done, upcoming = matches
    assert (done.game.home_team, done.game.away_team) == ("De Minaur A.", "Sinner J.")
    assert (done.game.home_score, done.game.away_score) == (2, 1) and done.game.is_final
    assert done.meta["tournament"] == "Rolex Shanghai Masters" and done.meta["round"] == "Quarterfinal"
    assert (upcoming.game.home_team, upcoming.game.away_team) == ("Alcaraz C.", "Zhang Z.")
    o = upcoming.odds[0]
    assert (o.home_ml, o.away_ml) == (-650, 450)               # ESPN's home/away mapped to our order


def synthetic_games(n_days=200, seed=4):
    rng = random.Random(seed)
    players = [f"P{i:02d} X." for i in range(20)]
    rows, meta = [], []
    t0 = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    for d in range(n_days):
        for j in range(4):
            a, b = ordered(*rng.sample(players, 2))
            key = f"g{d}_{j}"
            start = (t0 + timedelta(days=d, hours=j)).strftime("%Y-%m-%dT%H:%M:%SZ")
            hs = rng.choice([0, 1, 2])
            rows.append({"game_key": key, "league": "atp", "season": 2024 + d // 120, "season_type": "regular",
                         "start_time": start, "home_team": a, "away_team": b,
                         "home_score": hs, "away_score": 2 if hs < 2 else rng.choice([0, 1]), "status": "STATUS_FINAL"})
            meta.append({"game_key": key, "tournament": "T", "surface": rng.choice(["Hard", "Clay"]),
                         "round": "R1", "series": "ATP250", "home_rank": rng.randint(1, 100),
                         "away_rank": rng.randint(1, 100), "comment": "Completed"})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def test_tennis_features_no_leakage():
    cfg = league_config("atp")
    games, meta = synthetic_games()
    full = build_tennis_features(games, cfg, meta).set_index("game_key")
    rng = random.Random(9)
    for key in rng.sample(list(games.game_key), 25):
        start = games.loc[games.game_key == key, "start_time"].item()
        past = games[games.start_time <= start].copy()
        past.loc[past.game_key == key, ["home_score", "away_score"]] = np.nan
        part = build_tennis_features(past, cfg, meta).set_index("game_key")
        for col in TENNIS_FEATURES:
            assert full.loc[key, col] == approx(part.loc[key, col]), (key, col)
    assert set(full.home_win.dropna().unique()) <= {0.0, 1.0}


def test_write_dedupes_and_never_merges_future_into_past():
    db.init()
    matches = parse_history(history_frame())
    first = _write("atp", matches)
    again = _write("atp", parse_history(history_frame()))
    assert first["new"] == 1 and again["new"] == 0
    # same two players scheduled a day later: a NEW match, not the finished one
    data = json.loads((FIX / "sample_espn_tennis.json").read_text(encoding="utf-8"))
    m = parse_scoreboard(data)[1]
    from dataclasses import replace
    g = replace(m.game, home_team="Alcaraz C.", away_team="Sinner J.", start_time="2025-10-06T08:00:00Z")
    m.game = g
    m.meta["game_key"] = g.game_key
    assert Existing("atp").find(g.home_team, g.away_team, g.start_time, final=False) is None
    assert _write("atp", [m])["new"] == 1


def test_refresh_tennis_day_resolves_names_and_surface(monkeypatch):
    from datetime import date
    from edge.tennis_ingest import refresh_tennis_day
    db.init()
    _write("atp", parse_history(history_frame()))          # history knows "Alcaraz C."
    db.upsert_tennis_meta([{"game_key": "x", "tournament": "Shanghai Masters", "surface": "Hard"}])
    data = json.loads((FIX / "sample_espn_tennis.json").read_text(encoding="utf-8"))
    monkeypatch.setattr("edge.sources.espn_tennis.get_json", lambda *a, **k: data)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    out = refresh_tennis_day("atp", date(2026, 10, 9))
    assert out["games"] == 2
    meta = db.tennis_meta_df("atp")
    row = meta[meta.game_key == "atp:2026-10-09:Zhang Z.@Alcaraz C."]
    assert row.surface.item() == "Hard" and row.tournament.item() == "Rolex Shanghai Masters"
    odds = db.read_sql("SELECT * FROM odds_snapshots WHERE game_key = 'atp:2026-10-09:Zhang Z.@Alcaraz C.'")
    assert (odds.home_ml.item(), odds.away_ml.item()) == (-650, 450)
