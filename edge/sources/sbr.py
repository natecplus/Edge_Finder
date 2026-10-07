"""SportsBookReview adapter: the backup for moneyline odds.

SBR's odds page is a Next.js site: the odds table is embedded as JSON inside a
<script id="__NEXT_DATA__"> tag. We pull that tag out with BeautifulSoup and
walk the JSON for "gameRows". If SBR changes its layout, the fixture test in
tests/test_sbr.py tells you, and this file is the only one to fix.
"""
import json
from datetime import date

from bs4 import BeautifulSoup

from edge.config import RAW_DIR
from edge.http import get_text
from edge.teams import normalize
from .base import OddsSnapshot, OddsSource, make_game_key
from .espn import parse_ml

SPORT_PATHS = {"nba": "nba-basketball", "nfl": "nfl-football"}


def _find_all(obj, key: str):
    """Yield every value stored under `key` anywhere in nested JSON."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _find_all(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from _find_all(item, key)


def _team(league: str, team: dict) -> str:
    for field in ("shortName", "abbreviation", "fullName", "name"):
        if team.get(field):
            return normalize(league, str(team[field]))
    return ""


def parse_odds_page(html: str, league: str) -> list[OddsSnapshot]:
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag is None or not tag.string:
        raise ValueError("SBR page has no __NEXT_DATA__ block; the layout probably changed")
    data = json.loads(tag.string)

    out = []
    for rows in _find_all(data, "gameRows"):
        for row in rows or []:
            gv = row.get("gameView") or {}
            start = gv.get("startDate")
            home = _team(league, gv.get("homeTeam") or {})
            away = _team(league, gv.get("awayTeam") or {})
            if not (start and home and away):
                continue
            key = make_game_key(league, start, away, home)
            for view in row.get("oddsViews") or []:
                if not view:
                    continue
                book = str(view.get("sportsbook") or "unknown").title()
                for field, kind in (("currentLine", "live"), ("openingLine", "open")):
                    line = view.get(field) or {}
                    h, a = parse_ml(line.get("homeOdds")), parse_ml(line.get("awayOdds"))
                    if h and a:
                        out.append(OddsSnapshot(key, book, h, a, kind, "sbr"))
    return out


class SbrOdds(OddsSource):
    name = "sbr"

    def __init__(self, use_cache: bool = False):
        self.use_cache = use_cache

    def odds_on(self, league: str, day: date) -> list[OddsSnapshot]:
        path = SPORT_PATHS.get(league)
        if path is None:
            return []
        url = f"https://www.sportsbookreview.com/betting-odds/{path}/money-line/full-game/"
        html = get_text(url, params={"date": day.isoformat()},
                        cache=RAW_DIR / "sbr" / league / f"{day.isoformat()}.html",
                        use_cache=self.use_cache, delay=5.0)
        return parse_odds_page(html, league)
