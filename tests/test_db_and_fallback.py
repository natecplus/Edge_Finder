from datetime import date

import pytest

from edge import db
from edge.grade import grade_pick
from edge.sources import AllSourcesFailed, fetch_with_fallback
from edge.sources.base import Game, OddsSnapshot


def game(**kw):
    base = dict(league="nba", source_id="1", start_time="2026-01-10T00:30:00Z", season=2026,
                season_type="regular", home_team="BOS", away_team="NY", home_score=None,
                away_score=None, status="STATUS_SCHEDULED")
    base.update(kw)
    return Game(**base)


def test_upsert_is_idempotent_and_merges_backup_rows():
    db.init()
    db.upsert_games([game()])
    db.upsert_games([game()])
    # same game from a backup source: no ESPN id, now final
    db.upsert_games([game(source_id=None, home_score=101, away_score=99, status="STATUS_FINAL")])
    rows = db.read_sql("SELECT * FROM games WHERE game_key = 'nba:2026-01-09:NY@BOS'")
    assert len(rows) == 1
    assert rows.source_id[0] == "1" and rows.home_score[0] == 101


def test_open_close_odds_stored_once_live_odds_accumulate():
    db.init()
    k = game().game_key
    db.insert_odds([OddsSnapshot(k, "DK", -150, 130, "close", "espn")])
    db.insert_odds([OddsSnapshot(k, "DK", -155, 135, "close", "espn")])
    db.insert_odds([OddsSnapshot(k, "DK", -150, 130, "live", "espn")] * 2)
    o = db.read_sql("SELECT * FROM odds_snapshots WHERE game_key = :k", k=k)
    assert (o.kind == "close").sum() == 1 and o[o.kind == "close"].home_ml.item() == -155
    assert (o.kind == "live").sum() == 2


class Broken:
    name = "broken"

    def games_on(self, league, day):
        raise ConnectionError("site down")


class Works:
    name = "works"

    def games_on(self, league, day):
        return [game()]


def test_fallback_uses_backup_and_logs_failure():
    db.init()
    rows = fetch_with_fallback([Broken(), Works()], "games_on", "nba", date(2026, 1, 9), task="t")
    assert len(rows) == 1
    runs = db.read_sql("SELECT source, ok FROM source_runs WHERE task = 't'")
    assert list(zip(runs.source, runs.ok)) == [("broken", 0), ("works", 1)]


def test_all_sources_failed():
    db.init()
    with pytest.raises(AllSourcesFailed):
        fetch_with_fallback([Broken()], "games_on", "nba", date(2026, 1, 9))


def test_grade_pick():
    r = grade_pick("away", "Bet", 150, home_won=False, closing_ml=130)
    assert r["won"] == 1 and r["profit"] == 150
    assert r["clv"] > 0                         # we got +150, it closed +130: beat the close
    r = grade_pick("home", "Lean", -120, home_won=False, closing_ml=-130)
    assert r["won"] == 0 and r["profit"] == 0   # Leans don't count toward ROI
