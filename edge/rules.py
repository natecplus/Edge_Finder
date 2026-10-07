"""Bet / Lean / Pass rules engine.

The model says how likely each team is to win. Rules decide whether that
number is trustworthy enough to act on. A rule can only make a verdict MORE
cautious, never less:

  hard stop  -> always Pass (e.g. preseason)
  soft flag  -> counts against Bet; 2+ soft flags turn a Bet into a Lean

Adding a rule = writing one small function and adding it to RULES.
"""
from dataclasses import dataclass, field


@dataclass
class Context:
    """Everything a rule might need about one game."""
    game: dict                     # the feature row for this game (season_type, gp_home, ...)
    cfg: dict                      # league config (+ settings overrides)
    side: str | None               # "home" / "away" / None
    availability: dict = field(default_factory=dict)   # from features.upcoming_availability
    fair_home_open: float | None = None
    fair_home_now: float | None = None
    odds_age_hours: float | None = None
    league_status: str = "active"  # active | learning | pass_only | drift
    model_proven: bool = True      # False if no edge threshold was profitable in validation


@dataclass
class Verdict:
    level: str = "Pass"
    reasons: list[str] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)
    hard_stop: bool = False
    soft_flags: int = 0
    cap_lean: bool = False         # some flags allow at most a Lean

    def stop(self, rule: str, reason: str):
        self.hard_stop = True
        self.fired.append(rule)
        self.reasons.append(reason)

    def flag(self, rule: str, reason: str):
        self.soft_flags += 1
        self.fired.append(rule)
        self.reasons.append(reason)


# ------------------------------------------------------------------ rules
def preseason(ctx: Context, v: Verdict):
    if ctx.game.get("season_type") not in ("regular", "postseason"):
        v.stop("preseason", "Preseason / exhibition: starters rest, results are noise. No bet")


def no_odds(ctx: Context, v: Verdict):
    if ctx.fair_home_now is None:
        v.stop("no_odds", "No current odds from any source")


def stale_odds(ctx: Context, v: Verdict):
    limit = ctx.cfg.get("stale_odds_hours", 3)
    if ctx.odds_age_hours is not None and ctx.odds_age_hours > limit:
        v.stop("stale_odds", f"Odds are {ctx.odds_age_hours:.0f}h old (data refresh may be failing)")


def small_sample(ctx: Context, v: Verdict):
    gp = min(ctx.game.get("gp_home", 0), ctx.game.get("gp_away", 0))
    if ctx.game.get("season_type") == "regular" and gp < ctx.cfg.get("min_games_for_bet", 5):
        v.flag("small_sample", f"Early season: only {gp} games played, ratings still settling")
        v.cap_lean = True


def key_players_out(ctx: Context, v: Verdict):
    share = ctx.cfg.get("key_minutes_share", 0.25)
    for side in ("home", "away"):
        a = ctx.availability.get(side) or {}
        team = ctx.game.get(f"{side}_team")
        if a.get("top_player_out"):
            v.flag("top_player_out", f"{team}: top player {a['top_player_out']} is out")
        elif a.get("miss", 0) >= share:
            v.flag("key_players_out", f"{team} missing {a['miss']:.0%} of usual minutes")
        if a.get("qb_out"):
            names = ", ".join(a["qb_out"])
            if side == ctx.side:
                # The NFL model has no injury feature, but the market does: never
                # bet ON a team whose QB is out based on a model that can't see it.
                v.stop("qb_out", f"{team}: QB listed out ({names}); model can't price this. No bet")
            else:
                v.flag("qb_out", f"{team}: QB listed out ({names})")


def rest_disadvantage(ctx: Context, v: Verdict):
    # The model already prices rest in; this just makes it visible when it
    # works AGAINST the pick.
    if ctx.side and ctx.game.get(f"b2b_{ctx.side}"):
        v.flag("rest", f"{ctx.game.get(f'{ctx.side}_team')} on a back-to-back / short week")


def late_season(ctx: Context, v: Verdict):
    gp = max(ctx.game.get("gp_home", 0), ctx.game.get("gp_away", 0))
    if ctx.game.get("season_type") == "regular" and gp >= ctx.cfg.get("late_season_games", 75):
        v.flag("late_season", "Late season: teams that clinched or are eliminated may rest players")


def line_move(ctx: Context, v: Verdict):
    if ctx.fair_home_open is None or ctx.fair_home_now is None or ctx.side is None:
        return
    move = ctx.fair_home_now - ctx.fair_home_open          # + = market moved toward home
    toward_pick = move if ctx.side == "home" else -move
    if abs(move) >= ctx.cfg.get("line_move_points", 0.05) and toward_pick < 0:
        v.flag("line_move", f"Market moved {abs(move) * 100:.0f} pts AGAINST the pick since open "
                            "(it may know something)")


def league_trust(ctx: Context, v: Verdict):
    if ctx.league_status == "pass_only":
        v.stop("league_pass_only", "League on Pass-only: tracked results are negative")
    elif ctx.league_status == "drift":
        v.flag("model_drift", "Model has been less accurate than usual lately")
    elif ctx.league_status == "learning":
        v.reasons.append("Still learning: fewer than 50 graded bets in this league")


def unproven_model(ctx: Context, v: Verdict):
    if not ctx.model_proven:
        v.cap_lean = True
        v.fired.append("unproven_model")
        v.reasons.append("Model has not beaten the closing line in testing: max verdict is Lean")


RULES = [preseason, no_odds, stale_odds, small_sample, key_players_out,
         rest_disadvantage, late_season, line_move, league_trust, unproven_model]


def decide(ctx: Context, edge: float | None) -> Verdict:
    v = Verdict()
    for rule in RULES:
        rule(ctx, v)
    bet_at = ctx.cfg.get("bet_edge", 0.04)
    lean_at = ctx.cfg.get("lean_edge", 0.02)
    if v.hard_stop or edge is None or edge < lean_at:
        v.level = "Pass"
    elif edge >= bet_at and v.soft_flags <= 1 and not v.cap_lean:
        v.level = "Bet"
    else:
        v.level = "Lean"
    return v
