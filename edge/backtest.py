"""Backtest: pretend we bet every game where the edge cleared a threshold,
at the closing price, $100 flat per bet. Closing prices are a conservative
choice: they're usually the hardest prices to beat."""
import numpy as np
import pandas as pd

from edge.odds_math import fair_probs, profit_if_win

THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]


def simulate(df: pd.DataFrame, p_home, threshold: float, stake: float = 100) -> tuple[pd.DataFrame, dict]:
    """df needs: game_key, start_time, home_win, close_home_ml, close_away_ml."""
    rows = []
    for g, p in zip(df.itertuples(index=False), p_home):
        if pd.isna(g.close_home_ml) or pd.isna(g.close_away_ml) or pd.isna(g.home_win):
            continue
        fh, fa = fair_probs(int(g.close_home_ml), int(g.close_away_ml))
        for side, mp, fp, ml, won in (("home", p, fh, g.close_home_ml, g.home_win == 1),
                                       ("away", 1 - p, fa, g.close_away_ml, g.home_win == 0)):
            e = mp - fp
            if e >= threshold:
                rows.append({"game_key": g.game_key, "start_time": g.start_time, "side": side,
                             "edge": e, "odds": int(ml), "won": bool(won),
                             "profit": profit_if_win(int(ml), stake) if won else -stake})
    bets = pd.DataFrame(rows, columns=["game_key", "start_time", "side", "edge", "odds", "won", "profit"])
    n = len(bets)
    summary = {
        "threshold": threshold,
        "bets": n,
        "win_rate": float(bets.won.mean()) if n else 0.0,
        "profit": float(bets.profit.sum()) if n else 0.0,
        "roi": float(bets.profit.sum() / (stake * n)) if n else 0.0,
        "avg_edge": float(bets.edge.mean()) if n else 0.0,
    }
    return bets, summary


def choose_threshold(valid: pd.DataFrame, p_valid, min_bets: int = 30, default: float = 0.04) -> tuple[float, list]:
    """Pick the threshold on the VALIDATION season only. Choosing it on the test
    season would make the test result meaningless."""
    table = [simulate(valid, p_valid, t)[1] for t in THRESHOLDS]
    ok = [s for s in table if s["bets"] >= min_bets]
    if not ok:
        return default, table
    best = max(ok, key=lambda s: s["roi"])
    return best["threshold"], table


def market_log_loss(df: pd.DataFrame) -> float | None:
    """How good are the closing odds themselves as probabilities? This is the
    real benchmark: a model that can't get close to this has no edge."""
    d = df.dropna(subset=["close_fair_home", "home_win"])
    if d.empty:
        return None
    p = d.close_fair_home.clip(1e-6, 1 - 1e-6)
    y = d.home_win
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
