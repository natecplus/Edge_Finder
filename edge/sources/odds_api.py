"""Optional live tennis odds from The Odds API (the-odds-api.com, free tier).

Only used when the ODDS_API_KEY environment variable is set. Tennis is the one
sport where free scraped odds are thin, so this is the practical backup; the
app still works without it (enter odds by hand on the Game detail page).

Each tournament is its own "sport" (e.g. tennis_atp_shanghai_masters), so we
list active sports once, then fetch head-to-head odds for the tennis ones.
"""
import os

from edge.config import league_config
from edge.http import get
from edge.players import ordered
from edge.timeutil import parse_utc, to_iso
from .base import Game, OddsSnapshot

BASE = "https://api.the-odds-api.com/v4"


def parse_events(events: list, league: str, resolve) -> list[tuple[Game, list[OddsSnapshot]]]:
    """events: the API's JSON list. resolve: maps a full name to the stored player name."""
    out = []
    for ev in events or []:
        p1, p2 = resolve(ev.get("home_team", "")), resolve(ev.get("away_team", ""))
        if not p1 or not p2:
            continue
        home, away = ordered(p1, p2)
        start = to_iso(parse_utc(ev["commence_time"]))
        game = Game(league, None, start, parse_utc(start).year, "regular", home, away,
                    None, None, "STATUS_SCHEDULED")
        snaps = []
        for bm in ev.get("bookmakers") or []:
            for market in bm.get("markets") or []:
                if market.get("key") != "h2h":
                    continue
                price = {resolve(o["name"]): int(o["price"]) for o in market.get("outcomes") or []
                         if isinstance(o.get("price"), (int, float))}
                if price.get(home) and price.get(away):
                    snaps.append(OddsSnapshot(game.game_key, bm.get("title", "unknown"),
                                              price[home], price[away], "live", "odds-api"))
        out.append((game, snaps))
    return out


class OddsApiTennis:
    name = "odds-api"

    def __init__(self, key: str | None = None):
        self.key = key or os.environ.get("ODDS_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def fetch(self, league: str, resolve):
        prefix = league_config(league).get("odds_api_prefix", "tennis_atp")
        sports = get(f"{BASE}/sports", params={"apiKey": self.key}, delay=1.0).json()
        keys = [s["key"] for s in sports if s.get("active") and str(s.get("key", "")).startswith(prefix)]
        out = []
        for k in keys:
            events = get(f"{BASE}/sports/{k}/odds", delay=1.0, params={
                "apiKey": self.key, "regions": "us,eu", "markets": "h2h", "oddsFormat": "american"}).json()
            out.extend(parse_events(events, league, resolve))
        return out
