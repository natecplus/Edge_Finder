"""Fetch today's games (and their odds + injuries) into the database.

    python -m scripts.fetch_today            # NBA
    python -m scripts.fetch_today nfl
"""
import sys

from edge import db
from edge.ingest import refresh_day
from edge.timeutil import et_today

league = sys.argv[1] if len(sys.argv) > 1 else "nba"
db.init()                                  # creates data/edge.db + tables if missing
counts = refresh_day(league, et_today())
print(f"{league.upper()} {et_today()}: {counts}")
