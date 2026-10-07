"""Keeping score honestly: ROI, closing line value, calibration, and the
per-league trust status that can switch a league to Pass-only."""
import numpy as np
import pandas as pd

from edge import db
from edge.config import league_config


def graded(league: str) -> pd.DataFrame:
    return db.read_sql(
        "SELECT p.*, r.home_won, r.won, r.profit, r.clv, r.closing_ml, r.graded_at, g.start_time, "
        "g.home_team, g.away_team, g.home_score, g.away_score "
        "FROM picks p JOIN results r ON r.pick_id = p.id JOIN games g ON g.game_key = p.game_key "
        "WHERE p.league = :lg ORDER BY g.start_time", lg=league)


def summary(league: str) -> dict:
    df = graded(league)
    bets = df[df.verdict == "Bet"]
    n = len(bets)
    out = {
        "graded_picks": len(df),
        "bets": n,
        "wins": int(bets.won.sum()) if n else 0,
        "profit": float(bets.profit.sum()) if n else 0.0,
        "roi": float(bets.profit.sum() / (100 * n)) if n else None,
        "avg_clv": float(bets.clv.mean()) if n else None,
        "beat_close_rate": float((bets.clv > 0).mean()) if n else None,
        "live_log_loss_last50": None,
    }
    recent = df.dropna(subset=["home_prob", "home_won"]).tail(50)
    if len(recent) >= 50:
        p = recent.home_prob.clip(1e-6, 1 - 1e-6)
        y = recent.home_won
        out["live_log_loss_last50"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return out


def league_status(league: str, expected_log_loss: float | None = None) -> str:
    """active | learning | pass_only | drift"""
    cfg = league_config(league)
    s = summary(league)
    if s["bets"] < cfg.get("min_graded_bets", 50):
        status = "learning"
    elif s["roi"] < cfg.get("pass_only_roi", -0.05) and (s["avg_clv"] or 0) < 0:
        return "pass_only"
    else:
        status = "active"
    if (expected_log_loss is not None and s["live_log_loss_last50"] is not None
            and s["live_log_loss_last50"] > expected_log_loss + 0.03):
        return "drift"
    return status


def calibration(league: str, bins: int = 10) -> pd.DataFrame:
    df = graded(league).dropna(subset=["home_prob", "home_won"])
    if df.empty:
        return pd.DataFrame(columns=["predicted", "actual", "n"])
    df["bin"] = pd.cut(df.home_prob, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(predicted=("home_prob", "mean"),
                                             actual=("home_won", "mean"), n=("home_won", "size"))
    return t.reset_index(drop=True)
