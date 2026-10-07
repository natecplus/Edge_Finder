# edge/db.py
from dataclasses import asdict
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, UniqueConstraint
from sqlalchemy.dialects.sqlite import insert

engine = create_engine("sqlite:///data/edge.db")
meta = MetaData()

games = Table("games", meta,
    Column("id", Integer, primary_key=True),
    Column("league", String), Column("source_id", String),
    Column("season", Integer), Column("season_type", String),
    Column("start_time", String),
    Column("home_team", String), Column("away_team", String),
    Column("home_score", Integer), Column("away_score", Integer),
    Column("status", String),
    UniqueConstraint("league", "source_id"),
)

def init():
    meta.create_all(engine)

def upsert_games(rows):
    with engine.begin() as conn:
        for g in rows:
            stmt = insert(games).values(**asdict(g))
            stmt = stmt.on_conflict_do_update(
                index_elements=["league", "source_id"],
                set_={c: stmt.excluded[c] for c in ("start_time", "home_score", "away_score", "status")},
            )
            conn.execute(stmt)