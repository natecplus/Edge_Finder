"""Parser tests against saved sample pages (offline, deterministic).

The sample_*.json/html files mirror each site's format. When a site changes,
save a fresh real copy into tests/fixtures, see which test breaks, and fix
that one adapter file."""
import json
from pathlib import Path

from edge.sources.bref import parse_schedule_table
from edge.sources.espn import EspnDetail, EspnSchedule, parse_ml
from edge.sources.sbr import parse_odds_page

FIX = Path(__file__).parent / "fixtures"


def scoreboard_games():
    data = json.loads((FIX / "sample_espn_scoreboard.json").read_text(encoding="utf-8"))
    return [EspnSchedule()._parse("nba", ev) for ev in data["events"]]


def test_espn_scoreboard():
    final, upcoming = scoreboard_games()
    assert (final.home_team, final.away_team) == ("CHA", "BKN")
    assert final.season == 2027 and final.season_type == "preseason"
    assert (final.home_score, final.away_score) == (112, 105) and final.is_final
    assert final.start_time == "2026-10-06T23:00:00Z"
    assert final.game_key == "nba:2026-10-06:BKN@CHA"
    # 02:30Z on the 7th is 10:30pm ET on the 6th: the game_key uses the Eastern date
    assert upcoming.game_key == "nba:2026-10-06:UTAH@GS"
    assert upcoming.home_score is None and not upcoming.is_final


def test_espn_summary_odds_injuries_minutes():
    final = scoreboard_games()[0]
    data = json.loads((FIX / "sample_espn_summary.json").read_text(encoding="utf-8"))
    d = EspnDetail().parse(final, data)
    kinds = sorted((o.book, o.kind) for o in d.odds)
    assert ("DraftKings", "close") in kinds and ("DraftKings", "open") in kinds
    assert all(o.book != "Broken Book" for o in d.odds)          # "OFF" lines are skipped
    espn_bet = [o for o in d.odds if o.book == "ESPN BET"][0]
    assert (espn_bet.home_ml, espn_bet.away_ml) == (-145, 125)
    assert [i.player for i in d.injuries if i.status == "Out"] == ["LaMelo Ball"]
    mins = {m.player: m.minutes for m in d.minutes}
    assert mins["Brandon Miller"] == 34 and mins["Miles Bridges"] == 31.5 and mins["LaMelo Ball"] == 0


def test_parse_ml():
    assert parse_ml("+130") == 130 and parse_ml(-150) == -150
    assert parse_ml("EVEN") == 100 and parse_ml("OFF") is None and parse_ml(None) is None


def test_sbr_odds():
    snaps = parse_odds_page((FIX / "sample_sbr_odds.html").read_text(encoding="utf-8"), "nba")
    assert {s.game_key for s in snaps} == {"nba:2026-10-06:BKN@CHA"}   # team names normalized
    live = {(s.book, s.home_ml, s.away_ml) for s in snaps if s.kind == "live"}
    assert ("Fanduel", -148, 126) in live and ("Betmgm", -150, 125) in live
    assert any(s.kind == "open" for s in snaps)


def test_bref_schedule():
    games = parse_schedule_table((FIX / "sample_bref_schedule.html").read_text(encoding="utf-8"), 2026)
    assert len(games) == 3
    reg, po, upcoming = games
    assert (reg.home_team, reg.away_team, reg.season_type) == ("NY", "BOS", "regular")
    assert reg.start_time == "2026-04-14T23:30:00Z"            # 7:30pm EDT
    assert po.season_type == "postseason" and po.home_team == "DEN" and po.home_score == 101
    assert upcoming.home_score is None and upcoming.game_key == "nba:2026-04-18:UTAH@PHX"
