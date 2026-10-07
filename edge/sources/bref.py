"""Basketball-Reference adapter: the HTML backup for NBA schedules and results.

Plain HTML tables, parsed with pandas.read_html. It doesn't list preseason games
or odds, so it can only back up the schedule. Sports-Reference blocks fast
scrapers (roughly 20 requests/minute), hence the long delay.
"""
import io
from datetime import date, datetime

import pandas as pd

from edge.config import RAW_DIR
from edge.http import get_text
from edge.teams import normalize
from edge.timeutil import ET, et_date, to_iso
from .base import Game, ScheduleSource

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
DELAY = 6.0


def nba_season_label(day: date) -> int:
    """Season label by END year, matching ESPN (Oct 2026 -> 2027)."""
    return day.year + 1 if day.month >= 8 else day.year


def _start_utc(day_str: str, time_str) -> str:
    d = datetime.strptime(day_str, "%a, %b %d, %Y")
    t = str(time_str).strip().lower() if isinstance(time_str, str) else ""
    hour, minute = 19, 0                       # unknown time -> assume 7pm ET
    if t and t[-1] in "ap" and ":" in t:
        h, m = t[:-1].split(":")
        hour = int(h) % 12 + (12 if t[-1] == "p" else 0)
        minute = int(m)
    local = d.replace(hour=hour, minute=minute, tzinfo=ET)
    return to_iso(local)


def parse_schedule_table(html: str, season: int) -> list[Game]:
    table = pd.read_html(io.StringIO(html), attrs={"id": "schedule"})[0]
    games, postseason = [], False
    for row in table.itertuples(index=False):
        values = list(row)
        day_str = str(values[0])
        if day_str.strip().lower() == "playoffs":
            postseason = True
            continue
        if day_str in ("Date", "nan"):
            continue
        try:
            start = _start_utc(day_str, values[1])
        except ValueError:
            continue
        away, away_pts, home, home_pts = values[2], values[3], values[4], values[5]
        done = pd.notna(away_pts) and pd.notna(home_pts)
        games.append(Game(
            league="nba", source_id=None, start_time=start, season=season,
            season_type="postseason" if postseason else "regular",
            home_team=normalize("nba", str(home)), away_team=normalize("nba", str(away)),
            home_score=int(home_pts) if done else None,
            away_score=int(away_pts) if done else None,
            status="STATUS_FINAL" if done else "STATUS_SCHEDULED",
        ))
    return games


class BrefSchedule(ScheduleSource):
    name = "bref"

    def __init__(self, use_cache: bool = False):
        self.use_cache = use_cache

    def games_on(self, league: str, day: date) -> list[Game]:
        if league != "nba":
            return []          # this backup only covers the NBA
        season = nba_season_label(day)
        month = MONTHS[day.month - 1]
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games-{month}.html"
        html = get_text(url, cache=RAW_DIR / "bref" / f"NBA_{season}_{month}.html",
                        use_cache=self.use_cache, delay=DELAY)
        return [g for g in parse_schedule_table(html, season) if et_date(g.start_time) == day]
