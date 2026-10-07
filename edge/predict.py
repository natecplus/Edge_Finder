"""Make today's picks: model probability -> edge vs market -> rules -> verdict.

Runs every 30 minutes on game days. A pick keeps updating until 60 minutes
before the game, then it's locked: the locked price is what gets graded.
"""
import json
import logging
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from edge import db, tracking
from edge.config import league_config
from edge.explain import reasons_for
from edge.features import build_features, fill_upcoming_availability
from edge.market import current_lines, opening_fair
from edge.odds_math import expected_value
from edge.rules import Context, decide
from edge.timeutil import ET, et_date, parse_utc, to_iso, utcnow
from edge.train import load_active

log = logging.getLogger(__name__)


def effective_config(league: str) -> dict:
    """League YAML plus any overrides saved from the Settings page."""
    cfg = dict(league_config(league))
    cfg.update(db.get_setting(f"{league}.overrides", {}) or {})
    return cfg


def predict_league(league: str, day: date | None = None, now: datetime | None = None,
                   keys: list[str] | None = None, lock_all: bool = False,
                   features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score games and upsert rows into `picks`.

    By default: today's (Eastern) games that haven't started.
    keys / now / lock_all are used by the demo to replay past days."""
    now = now or utcnow()
    cfg = effective_config(league)
    if not cfg.get("enabled", True):
        return pd.DataFrame()
    bundle = load_active(league)
    if bundle is None:
        log.warning("No trained model for %s yet. Run: python -m scripts.train_model --league %s", league, league)
        return pd.DataFrame()
    if "bet_edge" not in (db.get_setting(f"{league}.overrides", {}) or {}):
        cfg["bet_edge"] = bundle.bet_threshold          # tuned on the validation season
        cfg["lean_edge"] = min(cfg.get("lean_edge", 0.02), bundle.bet_threshold / 2)

    games = db.games_df(league)
    if games.empty:
        return pd.DataFrame()
    now_iso = to_iso(now)
    minutes = db.minutes_df(league) if cfg.get("use_minutes") else pd.DataFrame(
        columns=["game_key", "team", "player", "minutes"])
    injuries = db.injuries_df(league)
    injuries = injuries[injuries.captured_at <= now_iso]
    odds = db.odds_df(league)
    odds = odds[odds.captured_at <= now_iso]

    # `features` lets a caller reuse one feature table for many calls (the demo replay)
    df = build_features(games, cfg, minutes=minutes) if features is None else features.copy()
    availability = fill_upcoming_availability(df, cfg, minutes, injuries, keys=keys)

    if keys is not None:
        rows = df[df.game_key.isin(keys)].copy()
    else:
        day = day or now.astimezone(ET).date()
        rows = df[(df.start_time > now_iso) & df.start_time.map(lambda s: et_date(s) == day)].copy()
    if rows.empty:
        return rows

    odds = odds[odds.game_key.isin(rows.game_key)]
    lines = current_lines(odds).set_index("game_key")
    opens = opening_fair(odds)
    p_home = bundle.predict(rows)
    expected_ll = (bundle.report.get("test") or {}).get("log_loss")
    status = tracking.league_status(league, expected_ll)

    picks, sides = [], []
    for (idx, g), p in zip(rows.iterrows(), p_home):
        line = lines.loc[g.game_key] if g.game_key in lines.index else None
        fair_home = float(line.fair_home) if line is not None else None
        side = edge = odds_taken = book = ev = model_p = fair_p = None
        if fair_home is not None:
            edge_home = p - fair_home
            side = "home" if edge_home > 0 else "away"
            model_p = p if side == "home" else 1 - p
            fair_p = fair_home if side == "home" else 1 - fair_home
            edge = model_p - fair_p
            odds_taken = int(line[f"best_{side}_ml"])
            book = line[f"best_{side}_book"]
            ev = expected_value(model_p, odds_taken)
        age = None
        if line is not None:
            age = (now - parse_utc(line.last_seen)).total_seconds() / 3600
        ctx = Context(game=g.to_dict(), cfg=cfg, side=side,
                      availability=availability.get(g.game_key, {}),
                      fair_home_open=float(opens[g.game_key]) if g.game_key in opens.index else None,
                      fair_home_now=fair_home, odds_age_hours=age, league_status=status,
                      model_proven=bundle.report.get("validation_profitable", True))
        verdict = decide(ctx, edge)
        sides.append(side)
        picks.append({
            "game_key": g.game_key, "league": league, "home_prob": float(p),
            "side": side, "pick_team": g[f"{side}_team"] if side else None,
            "model_prob": model_p, "fair_prob": fair_p, "edge": edge, "ev": ev,
            "odds_taken": odds_taken, "book": book, "verdict": verdict.level,
            "rule_reasons": verdict.reasons, "rules_fired": json.dumps(verdict.fired),
            "model_version": bundle.version, "start_time": g.start_time,
        })

    model_reasons = reasons_for(bundle, rows, p_home, sides)
    out = []
    for pick, mr in zip(picks, model_reasons):
        head = []
        if pick["side"]:
            head = [f"Model {pick['model_prob']:.0%} vs market {pick['fair_prob']:.0%} for "
                    f"{pick['pick_team']} ({pick['edge'] * 100:+.1f} pts)"]
        # rule reasons first (they decide the verdict), then the model's main factors
        pick["reasons"] = "|".join(head + pick.pop("rule_reasons") + [f"Model factor: {m}" for m in mr])
        out.append(pick)
    _upsert_picks(out, now, cfg.get("lock_minutes_before_start", 60), lock_all)
    return pd.DataFrame(out)


def _upsert_picks(picks: list[dict], now: datetime, lock_minutes: int, lock_all: bool) -> None:
    now_iso = to_iso(now)
    with db.get_engine().begin() as conn:
        locked = {r[0] for r in conn.execute(
            select(db.picks.c.game_key).where(db.picks.c.locked_at.is_not(None)))}
        for p in picks:
            if p["game_key"] in locked:
                continue                      # a locked pick never changes
            start = parse_utc(p.pop("start_time"))
            lock = lock_all or (start - now) <= timedelta(minutes=lock_minutes)
            values = {**p, "updated_at": now_iso,
                      "locked_at": to_iso(min(now, start - timedelta(minutes=lock_minutes))) if lock else None}
            stmt = insert(db.picks).values(**values)
            conn.execute(stmt.on_conflict_do_update(
                index_elements=["game_key"],
                set_={k: stmt.excluded[k] for k in values if k != "game_key"}))
