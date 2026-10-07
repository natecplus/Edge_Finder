from edge.config import league_config
from edge.rules import Context, decide

CFG = dict(league_config("nba"))


def ctx(**overrides):
    game = {"season_type": "regular", "gp_home": 20, "gp_away": 20, "home_team": "BOS",
            "away_team": "NY", "b2b_home": 0, "b2b_away": 0}
    game.update(overrides.pop("game", {}))
    base = dict(game=game, cfg=CFG, side="home", fair_home_now=0.5, fair_home_open=0.5,
                odds_age_hours=0.2, league_status="active")
    base.update(overrides)
    return Context(**base)


def test_bet_lean_pass_thresholds():
    assert decide(ctx(), 0.06).level == "Bet"
    assert decide(ctx(), 0.03).level == "Lean"
    assert decide(ctx(), 0.01).level == "Pass"


def test_preseason_is_always_pass():
    v = decide(ctx(game={"season_type": "preseason"}), 0.20)
    assert v.level == "Pass" and "preseason" in v.fired


def test_small_sample_caps_at_lean():
    v = decide(ctx(game={"gp_home": 2}), 0.10)
    assert v.level == "Lean"


def test_two_soft_flags_downgrade_bet():
    c = ctx(game={"b2b_home": 1}, fair_home_open=0.60, fair_home_now=0.50)   # rest + line move
    assert decide(c, 0.06).level == "Lean"


def test_stale_and_missing_odds_stop():
    assert decide(ctx(odds_age_hours=10), 0.10).level == "Pass"
    assert decide(ctx(fair_home_now=None), None).level == "Pass"


def test_qb_out_on_pick_side_is_hard_stop():
    v = decide(ctx(availability={"home": {"qb_out": ["Joe QB"]}}), 0.10)
    assert v.level == "Pass" and "qb_out" in v.fired


def test_pass_only_league():
    assert decide(ctx(league_status="pass_only"), 0.10).level == "Pass"


def test_unproven_model_caps_at_lean():
    assert decide(ctx(model_proven=False), 0.10).level == "Lean"
