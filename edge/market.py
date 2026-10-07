"""Read the betting market out of the odds_snapshots table.

- closing_lines: the last price before each game (what sharp bettors measure against)
- current_lines: today's latest prices, best price per side, and when we last saw them
"""
import pandas as pd

from edge.odds_math import fair_probs, implied_prob


def _with_fair(df: pd.DataFrame) -> pd.DataFrame:
    fair = [fair_probs(h, a) for h, a in zip(df.home_ml, df.away_ml)]
    df = df.copy()
    df["fair_home"] = [f[0] for f in fair]
    return df


def closing_lines(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One consensus closing line per game.

    Prefers explicit 'close' rows (ESPN gives these for finished games);
    otherwise uses each book's last 'live' snapshot captured before start.
    Consensus = median vig-free home probability across books, plus the
    median prices (used for ROI)."""
    cols = ["game_key", "close_home_ml", "close_away_ml", "close_fair_home", "n_books"]
    if odds.empty:
        return pd.DataFrame(columns=cols)
    starts = games.set_index("game_key").start_time
    o = odds.dropna(subset=["home_ml", "away_ml"]).copy()
    o["start_time"] = o.game_key.map(starts)

    close = o[o.kind == "close"]
    live = o[(o.kind == "live") & (o.captured_at <= o.start_time)]
    live = live.sort_values("captured_at").groupby(["game_key", "book"]).tail(1)
    live = live[~live.game_key.isin(close.game_key)]
    picked = _with_fair(pd.concat([close, live], ignore_index=True))
    if picked.empty:
        return pd.DataFrame(columns=cols)
    agg = picked.groupby("game_key").agg(
        close_home_ml=("home_ml", "median"), close_away_ml=("away_ml", "median"),
        close_fair_home=("fair_home", "median"), n_books=("book", "nunique")).reset_index()
    agg["close_home_ml"] = agg.close_home_ml.round().astype(int)
    agg["close_away_ml"] = agg.close_away_ml.round().astype(int)
    return agg


def opening_fair(odds: pd.DataFrame) -> pd.Series:
    """Consensus vig-free home probability at open (for the line-move rule)."""
    o = odds[odds.kind == "open"].dropna(subset=["home_ml", "away_ml"])
    if o.empty:
        # fall back to the first live snapshot we captured
        o = odds[odds.kind == "live"].sort_values("captured_at").groupby(["game_key", "book"]).head(1)
    if o.empty:
        return pd.Series(dtype=float)
    return _with_fair(o).groupby("game_key").fair_home.median()


def current_lines(odds: pd.DataFrame) -> pd.DataFrame:
    """Latest snapshot per book per game, summarized:
    fair_home (median across books), best_home_ml / best_away_ml (+ which book),
    last_seen (most recent capture)."""
    cols = ["game_key", "fair_home", "best_home_ml", "best_home_book",
            "best_away_ml", "best_away_book", "n_books", "last_seen"]
    o = odds[odds.kind == "live"].dropna(subset=["home_ml", "away_ml"])
    if o.empty:
        return pd.DataFrame(columns=cols)
    latest = _with_fair(o.sort_values("captured_at").groupby(["game_key", "book"]).tail(1))
    rows = []
    for key, g in latest.groupby("game_key"):
        # best price = lowest implied probability = biggest payout for that side
        bh = g.loc[g.home_ml.map(implied_prob).idxmin()]
        ba = g.loc[g.away_ml.map(implied_prob).idxmin()]
        rows.append({
            "game_key": key, "fair_home": g.fair_home.median(),
            "best_home_ml": int(bh.home_ml), "best_home_book": bh.book,
            "best_away_ml": int(ba.away_ml), "best_away_book": ba.book,
            "n_books": g.book.nunique(), "last_seen": g.captured_at.max(),
        })
    return pd.DataFrame(rows, columns=cols)
