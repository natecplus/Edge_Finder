"""Grade locked picks once their games are final.

profit: flat $100 on every Bet (Leans and Passes are graded for calibration
        but don't count toward ROI).
CLV:    closing line value = implied probability at the close minus implied
        probability of the price we locked. Positive = the market moved toward
        our pick after we made it, the best early sign of a real edge.
"""
import pandas as pd
from sqlalchemy.dialects.sqlite import insert

from edge import db
from edge.market import closing_lines
from edge.odds_math import implied_prob, profit_if_win
from edge.timeutil import utcnow_iso


def grade_pick(side: str | None, verdict: str, odds_taken: int | None, home_won: bool,
               closing_ml: int | None, stake: float = 100) -> dict:
    won = None if side is None else (home_won if side == "home" else not home_won)
    profit = 0.0
    if verdict == "Bet" and won is not None and odds_taken:
        profit = profit_if_win(odds_taken, stake) if won else -stake
    clv = None
    if closing_ml and odds_taken:
        clv = implied_prob(closing_ml) - implied_prob(odds_taken)
    return {"won": None if won is None else int(won), "profit": profit, "clv": clv}


def grade_league(league: str) -> int:
    todo = db.read_sql(
        "SELECT p.id, p.game_key, p.side, p.verdict, p.odds_taken, g.home_score, g.away_score "
        "FROM picks p JOIN games g ON g.game_key = p.game_key "
        "LEFT JOIN results r ON r.pick_id = p.id "
        "WHERE p.league = :lg AND p.locked_at IS NOT NULL AND r.id IS NULL "
        "AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL AND g.status LIKE '%FINAL%'",
        lg=league)
    if todo.empty:
        return 0
    close = closing_lines(db.odds_df(league), db.games_df(league)).set_index("game_key")
    now = utcnow_iso()
    n = 0
    with db.get_engine().begin() as conn:
        for p in todo.itertuples(index=False):
            if p.home_score == p.away_score:
                continue                       # push / tie: no result
            home_won = p.home_score > p.away_score
            closing_ml = None
            if p.game_key in close.index and p.side:
                closing_ml = int(close.loc[p.game_key, f"close_{p.side}_ml"])
            odds_taken = None if pd.isna(p.odds_taken) else int(p.odds_taken)
            r = grade_pick(p.side, p.verdict, odds_taken, home_won, closing_ml)
            stmt = insert(db.results).values(pick_id=int(p.id), game_key=p.game_key,
                                             home_won=int(home_won), closing_ml=closing_ml,
                                             graded_at=now, **r)
            conn.execute(stmt.on_conflict_do_nothing(index_elements=["pick_id"]))
            n += 1
    return n
