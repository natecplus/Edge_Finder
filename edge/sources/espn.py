# edge/sources/espn.py
import json, pathlib, time
import requests
from .base import Game, ScheduleSource

PATHS = {"nba": "basketball/nba", "nfl": "football/nfl"}
SEASON_TYPES = {1: "preseason", 2: "regular", 3: "postseason"}
RAW = pathlib.Path("data/raw/espn")
HEADERS = {"User-Agent": "moneyline-edge personal project"}

def _score(team, done):
    return int(team["score"]) if done else None

class EspnSchedule(ScheduleSource):
    name = "espn"

    def games_on(self, league, day):
        url = f"https://site.api.espn.com/apis/site/v2/sports/{PATHS[league]}/scoreboard"
        r = requests.get(url, params={"dates": day.strftime("%Y%m%d")}, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / f"{league}_{day}.json").write_text(json.dumps(data))  # keep raw copy
        time.sleep(3)  # be polite
        return [self._parse(league, ev) for ev in data.get("events", [])]

    def _parse(self, league, ev):
        comp = ev["competitions"][0]
        side = {c["homeAway"]: c for c in comp["competitors"]}
        done = comp["status"]["type"]["completed"]
        return Game(
            league=league,
            source_id=ev["id"],
            start_time=ev["date"],
            season=ev["season"]["year"],
            season_type=SEASON_TYPES.get(ev["season"]["type"], "other"),
            home_team=side["home"]["team"]["abbreviation"],
            away_team=side["away"]["team"]["abbreviation"],
            home_score=_score(side["home"], done),
            away_score=_score(side["away"], done),
            status=comp["status"]["type"]["name"],
        )