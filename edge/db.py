"""The database: your own copy of everything, so no source can take it away.

SQLite file at data/edge.db (one file, nothing to install). SQLAlchemy Core
keeps the SQL portable, so moving to Postgres later is a one-line URL change.
"""
import json
from dataclasses import asdict

import pandas as pd
from sqlalchemy import (Column, Float, Integer, MetaData, String, Table, Text,
                        UniqueConstraint, create_engine, event, select, text)
from sqlalchemy.dialects.sqlite import insert

from edge.config import DB_PATH, ensure_dirs
from edge.timeutil import utcnow_iso

meta = MetaData()

games = Table(
    "games", meta,
    Column("id", Integer, primary_key=True),
    Column("game_key", String, nullable=False, unique=True),   # nba:2026-10-06:BKN@CHA
    Column("league", String, nullable=False),
    Column("source_id", String),                                # ESPN event id
    Column("season", Integer),
    Column("season_type", String),
    Column("start_time", String),                               # UTC ISO
    Column("home_team", String), Column("away_team", String),
    Column("home_score", Integer), Column("away_score", Integer),
    Column("status", String),
    Column("updated_at", String),
)

odds_snapshots = Table(
    "odds_snapshots", meta,
    Column("id", Integer, primary_key=True),
    Column("game_key", String, nullable=False, index=True),
    Column("book", String), Column("home_ml", Integer), Column("away_ml", Integer),
    Column("kind", String),          # live | open | close
    Column("source", String),
    Column("captured_at", String),
    # open/close rows are stored once per book; live rows (NULL key) pile up as a time series
    Column("dedupe_key", String, unique=True),
)

injuries = Table(
    "injuries", meta,
    Column("id", Integer, primary_key=True),
    Column("game_key", String, nullable=False, index=True),
    Column("team", String), Column("player", String), Column("status", String),
    Column("position", String), Column("source", String),
    Column("captured_at", String),
)

player_minutes = Table(
    "player_minutes", meta,
    Column("id", Integer, primary_key=True),
    Column("game_key", String, nullable=False, index=True),
    Column("team", String), Column("player", String), Column("minutes", Float),
    UniqueConstraint("game_key", "team", "player"),
)

picks = Table(
    "picks", meta,
    Column("id", Integer, primary_key=True),
    Column("game_key", String, nullable=False, unique=True),
    Column("league", String),
    Column("home_prob", Float),        # model's calibrated P(home wins)
    Column("side", String),            # "home" | "away" | None (no positive edge)
    Column("pick_team", String),
    Column("model_prob", Float),       # model P(pick side wins)
    Column("fair_prob", Float),        # market P(pick side wins), vig removed
    Column("edge", Float),
    Column("ev", Float),               # expected profit per $100
    Column("odds_taken", Integer),     # best price available for the pick side
    Column("book", String),
    Column("verdict", String),         # Bet | Lean | Pass
    Column("reasons", Text),           # "reason one|reason two"
    Column("rules_fired", Text),       # JSON list of rule names
    Column("model_version", String),
    Column("updated_at", String),
    Column("locked_at", String),       # once set, the pick never changes
)

results = Table(
    "results", meta,
    Column("id", Integer, primary_key=True),
    Column("pick_id", Integer, nullable=False, unique=True),
    Column("game_key", String),
    Column("home_won", Integer),
    Column("won", Integer),            # did the picked side win (NULL if no side)
    Column("profit", Float),           # flat $100 stake, Bets only (0 otherwise)
    Column("closing_ml", Integer),
    Column("clv", Float),              # closing implied prob - taken implied prob
    Column("graded_at", String),
)

model_versions = Table(
    "model_versions", meta,
    Column("version", String, primary_key=True),
    Column("league", String),
    Column("kind", String),
    Column("trained_at", String),
    Column("data_cutoff", String),
    Column("features", Text),          # JSON list
    Column("valid_log_loss", Float),
    Column("test_log_loss", Float),
    Column("test_brier", Float),
    Column("market_log_loss", Float),
    Column("bet_threshold", Float),
    Column("backtest_roi", Float),
    Column("backtest_bets", Integer),
    Column("n_train", Integer),
    Column("path", String),
    Column("is_active", Integer, default=0),
    Column("report", Text),            # full JSON report
)

source_runs = Table(
    "source_runs", meta,
    Column("id", Integer, primary_key=True),
    Column("source", String), Column("task", String),
    Column("ran_at", String), Column("ok", Integer),
    Column("rows", Integer), Column("error", Text),
)

settings = Table(
    "settings", meta,
    Column("key", String, primary_key=True),
    Column("value", Text),
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        ensure_dirs()
        _engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")    # app can read while jobs write
            cur.execute("PRAGMA busy_timeout=10000")
            cur.close()
    return _engine


def init() -> None:
    meta.create_all(get_engine())


# ---------------------------------------------------------------- writes
def upsert_games(rows) -> int:
    rows = list(rows)
    now = utcnow_iso()
    with get_engine().begin() as conn:
        for g in rows:
            values = asdict(g)
            values["game_key"] = g.game_key
            values["updated_at"] = now
            stmt = insert(games).values(**values)
            ex = stmt.excluded
            stmt = stmt.on_conflict_do_update(
                index_elements=["game_key"],
                set_={
                    "start_time": ex.start_time,
                    "status": ex.status,
                    "home_score": ex.home_score,
                    "away_score": ex.away_score,
                    "updated_at": ex.updated_at,
                    # keep the ESPN id / season info if a backup source has none
                    "source_id": text("COALESCE(excluded.source_id, games.source_id)"),
                    "season_type": text(
                        "CASE WHEN games.season_type = 'preseason' THEN games.season_type "
                        "ELSE excluded.season_type END"),
                },
            )
            conn.execute(stmt)
    return len(rows)


def insert_odds(snapshots, captured_at: str | None = None) -> int:
    captured_at = captured_at or utcnow_iso()
    n = 0
    with get_engine().begin() as conn:
        for s in snapshots:
            values = asdict(s)
            values["captured_at"] = captured_at
            values["dedupe_key"] = (f"{s.game_key}|{s.book}|{s.kind}|{s.source}"
                                    if s.kind in ("open", "close") else None)
            stmt = insert(odds_snapshots).values(**values)
            if values["dedupe_key"]:
                stmt = stmt.on_conflict_do_update(
                    index_elements=["dedupe_key"],
                    set_={"home_ml": stmt.excluded.home_ml, "away_ml": stmt.excluded.away_ml},
                )
            conn.execute(stmt)
            n += 1
    return n


def insert_injuries(reports, captured_at: str | None = None) -> int:
    captured_at = captured_at or utcnow_iso()
    rows = [{**asdict(r), "captured_at": captured_at} for r in reports]
    if rows:
        with get_engine().begin() as conn:
            conn.execute(injuries.insert(), rows)
    return len(rows)


def upsert_minutes(rows) -> int:
    rows = list(rows)
    with get_engine().begin() as conn:
        for r in rows:
            stmt = insert(player_minutes).values(**asdict(r))
            stmt = stmt.on_conflict_do_update(
                index_elements=["game_key", "team", "player"],
                set_={"minutes": stmt.excluded.minutes})
            conn.execute(stmt)
    return len(rows)


def log_source_run(source: str, task: str, ok: bool, rows: int = 0, error: str = "") -> None:
    with get_engine().begin() as conn:
        conn.execute(source_runs.insert().values(
            source=source, task=task, ran_at=utcnow_iso(), ok=int(ok), rows=rows, error=error[:500]))


def get_setting(key: str, default=None):
    with get_engine().connect() as conn:
        row = conn.execute(select(settings.c.value).where(settings.c.key == key)).first()
    return json.loads(row[0]) if row else default


def set_setting(key: str, value) -> None:
    with get_engine().begin() as conn:
        stmt = insert(settings).values(key=key, value=json.dumps(value))
        conn.execute(stmt.on_conflict_do_update(index_elements=["key"], set_={"value": stmt.excluded.value}))


# ---------------------------------------------------------------- reads
def read_sql(sql: str, **params) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def games_df(league: str) -> pd.DataFrame:
    return read_sql("SELECT * FROM games WHERE league = :lg ORDER BY start_time", lg=league)


def odds_df(league: str) -> pd.DataFrame:
    return read_sql(
        "SELECT o.* FROM odds_snapshots o JOIN games g ON g.game_key = o.game_key "
        "WHERE g.league = :lg", lg=league)


def injuries_df(league: str) -> pd.DataFrame:
    return read_sql(
        "SELECT i.* FROM injuries i JOIN games g ON g.game_key = i.game_key "
        "WHERE g.league = :lg", lg=league)


def minutes_df(league: str) -> pd.DataFrame:
    return read_sql(
        "SELECT m.* FROM player_minutes m JOIN games g ON g.game_key = m.game_key "
        "WHERE g.league = :lg", lg=league)
