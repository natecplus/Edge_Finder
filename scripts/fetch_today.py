from datetime import date

from edge import db
from edge.sources.espn import EspnSchedule

db.init()                                  # creates data/edge.db + tables if missing
games = EspnSchedule().games_on("nba", date.today())
db.upsert_games(games)
print(f"Saved {len(games)} games for {date.today()}")