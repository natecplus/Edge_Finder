"""End-to-end ingest with the network swapped for saved sample files."""
import json
from datetime import date
from pathlib import Path

from edge import db, ingest

FIX = Path(__file__).parent / "fixtures"


def fake_get_json(url, params=None, cache=None, use_cache=False, delay=0):
    if url.endswith("/scoreboard"):
        return json.loads((FIX / "sample_espn_scoreboard.json").read_text(encoding="utf-8"))
    if url.endswith("/summary"):
        return json.loads((FIX / "sample_espn_summary.json").read_text(encoding="utf-8"))
    raise AssertionError(url)


def test_refresh_day(monkeypatch):
    monkeypatch.setattr("edge.sources.espn.get_json", fake_get_json)
    db.init()
    out = ingest.refresh_day("nba", date(2026, 10, 6))
    assert out["games"] == 2
    keys = set(db.games_df("nba").game_key)
    assert {"nba:2026-10-06:BKN@CHA", "nba:2026-10-06:UTAH@GS"} <= keys
    minutes = db.minutes_df("nba")
    assert "Brandon Miller" in set(minutes.player)          # finished game -> box score stored
    odds = db.odds_df("nba")
    assert (odds.kind == "close").any()
