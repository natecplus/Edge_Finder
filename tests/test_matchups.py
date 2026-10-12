"""Matchup engines: NFL coverage/position fit and NBA defense vs position."""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from edge.matchups_nba import defense_table, player_table, pos_group
from edge.matchups_nfl import NflContext
from edge.sources.espn import EspnDetail, EspnSchedule
from edge.synthetic_nfl import generate

FIX = Path(__file__).parent / "fixtures"
DATA = generate([2025, 2026], weeks_last=4, weeks_current=3, seed=3)


def test_nfl_side_report():
    ctx = NflContext(data=DATA)
    assert ctx.season == 2026 and ctx.charted_seasons == [2025, 2026]
    side = ctx.side("DAL", "LAR")                      # nflverse "LA" is normalized to "LAR"
    t = side["players"]
    assert not t.empty and set(t.metric) <= {"Receiving yards", "Rush yards", "Pass yards"}
    assert np.isfinite(t.advantage).all() and (t.projection > 0).all()
    assert all(isinstance(r, list) and r for r in t.reasons)
    assert any("man" in line.lower() for line in side["defense_profile"])
    assert list(t.advantage) == sorted(t.advantage, reverse=True)


def test_nfl_without_coverage_charting_still_works():
    data = {**DATA, "participation": pd.DataFrame()}
    ctx = NflContext(data=data)
    assert ctx.charted_seasons == []
    side = ctx.side("KC", "BUF")
    assert not side["players"].empty
    assert "no coverage charting" in ctx.note()


def test_nfl_shrinkage_keeps_small_samples_near_average():
    ctx = NflContext(data=DATA)
    rec = ctx.players[ctx.players.kind == "receiver"]
    # with ~30 pseudo-targets of prior, splits can't stray far from the player's overall
    assert ((rec.ypt_man - rec.ypt).abs() < rec.ypt * 0.6).all()


def nba_stats(seed=0):
    rng = np.random.default_rng(seed)
    teams = ["BOS", "NY", "MIA", "LAL"]
    rows = []
    for d in range(40):
        a, b = teams[d % 4], teams[(d + 1) % 4]
        key, start = f"g{d}", f"2026-01-{1 + d % 28:02d}T00:{d:02d}:00Z"
        for team, opp in ((a, b), (b, a)):
            for i, pos in enumerate(["PG", "SG", "SF", "PF", "C"]):
                boost = 1.3 if (opp == "MIA" and pos in ("PG", "SG")) else 1.0   # MIA can't guard guards
                rows.append({"game_key": key, "team": team, "player": f"{team}{i}", "position": pos,
                             "minutes": 32.0, "pts": 16 * boost * rng.uniform(0.8, 1.2), "reb": 6.0, "ast": 4.0,
                             "fg3m": 2.0, "start_time": start, "home_team": a, "away_team": b,
                             "season": 2026, "season_type": "regular"})
    return pd.DataFrame(rows)


def test_nba_defense_vs_position():
    assert pos_group("PG") == "G" and pos_group("F-C") == "C" and pos_group("SF") == "F"
    d = defense_table(nba_stats(), "2027-01-01T00:00:00Z")
    mia_g = d[(d.defense == "MIA") & (d.group == "G")].iloc[0]
    assert mia_g.pts_factor > 1.08 and mia_g.pts_rank == 4           # worst of 4 vs guards
    others = d[(d.defense != "MIA") & (d.group == "G")]
    assert (others.pts_factor < mia_g.pts_factor).all()
    p = player_table(nba_stats(), "2027-01-01T00:00:00Z")
    assert set(p.group) == {"G", "F", "C"} and (p.proj_min > 30).all()


def test_espn_player_lines():
    games = [EspnSchedule()._parse("nba", ev) for ev in
             json.loads((FIX / "sample_espn_scoreboard.json").read_text(encoding="utf-8"))["events"]]
    data = json.loads((FIX / "sample_espn_summary.json").read_text(encoding="utf-8"))
    lines = {ln.player: ln for ln in EspnDetail().player_lines(games[0], data)}
    assert lines["Brandon Miller"].pts == 24 and lines["Brandon Miller"].minutes == 34
    assert lines["LaMelo Ball"].minutes == 0


def test_refresh_upcoming(monkeypatch):
    from edge import db, ingest
    db.init()
    data = json.loads((FIX / "sample_espn_scoreboard.json").read_text(encoding="utf-8"))
    monkeypatch.setattr("edge.sources.espn.get_json", lambda *a, **k: data)
    out = ingest.refresh_upcoming("nba", days=2, start=date(2026, 10, 5))
    assert out == {"2026-10-06": 2, "2026-10-07": 2}
