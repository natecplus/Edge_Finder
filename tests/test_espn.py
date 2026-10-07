import json
from pathlib import Path

from edge.sources.espn import EspnSchedule

FIXTURE = Path(__file__).parent / "fixtures" / "espn_nba_scoreboard.json"


def load_games():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [EspnSchedule()._parse("nba", ev) for ev in data["events"]]


def test_parses_some_games():
    assert len(load_games()) > 0


def test_team_abbreviations():
    for g in load_games():
        assert g.home_team.isupper() and 2 <= len(g.home_team) <= 4
        assert g.away_team.isupper() and 2 <= len(g.away_team) <= 4
        assert g.home_team != g.away_team


def test_season_type():
    for g in load_games():
        assert g.season_type == "preseason"   # fixture is from early October
        assert g.season == 2027               # ESPN labels by end year


def test_finished_games_have_scores():
    for g in load_games():
        if g.status == "STATUS_FINAL":
            assert g.home_score is not None and g.away_score is not None