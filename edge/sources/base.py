"""The 'plug shape' every data source must produce.

Adapters (espn.py, bref.py, sbr.py) turn each site's own format into these
dataclasses. Nothing outside edge/sources/ ever sees raw ESPN JSON or HTML.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from edge.timeutil import et_date


def make_game_key(league: str, start_time: str, away_team: str, home_team: str) -> str:
    """Source-independent id for a game: 'nba:2026-10-06:BKN@CHA'.
    ESPN and Basketball-Reference rows for the same game get the same key,
    so a backup source updates the same row instead of creating a duplicate."""
    return f"{league}:{et_date(start_time).isoformat()}:{away_team}@{home_team}"


@dataclass
class Game:
    league: str
    source_id: str | None    # ESPN event id (None if the game came from a backup source)
    start_time: str          # ISO 8601, always UTC
    season: int              # ESPN labels NBA seasons by END year: 2026-27 season = 2027
    season_type: str         # "preseason" | "regular" | "postseason"
    home_team: str           # ESPN abbreviation, e.g. "CHA"
    away_team: str
    home_score: int | None   # None until the game is finished
    away_score: int | None
    status: str              # e.g. STATUS_SCHEDULED, STATUS_IN_PROGRESS, STATUS_FINAL

    @property
    def game_key(self) -> str:
        return make_game_key(self.league, self.start_time, self.away_team, self.home_team)

    @property
    def is_final(self) -> bool:
        return self.home_score is not None and self.away_score is not None and "FINAL" in self.status.upper()


@dataclass
class OddsSnapshot:
    game_key: str
    book: str                # e.g. "DraftKings", or "consensus"
    home_ml: int             # American odds, e.g. -150
    away_ml: int             # e.g. +130
    kind: str = "live"       # "live" (captured now), "open", or "close"
    source: str = ""


@dataclass
class InjuryReport:
    game_key: str
    team: str
    player: str
    status: str              # "Out", "Doubtful", "Questionable", "Day-To-Day", ...
    position: str = ""
    source: str = ""


@dataclass
class PlayerMinutes:
    """Who actually played and how much (from a finished game's box score).
    Used to measure how much of a team's usual rotation is missing."""
    game_key: str
    team: str
    player: str
    minutes: float


@dataclass
class GameDetail:
    """Everything one game-summary request gives us."""
    odds: list[OddsSnapshot] = field(default_factory=list)
    injuries: list[InjuryReport] = field(default_factory=list)
    minutes: list[PlayerMinutes] = field(default_factory=list)


class ScheduleSource(ABC):
    """Contract: any schedule source must return a list of Game objects for a day."""
    name: str

    @abstractmethod
    def games_on(self, league: str, day: date) -> list[Game]:
        ...


class DetailSource(ABC):
    """Contract: odds / injuries / box-score minutes for one game."""
    name: str

    @abstractmethod
    def detail(self, game: Game) -> GameDetail:
        ...


class OddsSource(ABC):
    """Contract for odds-only sources (the backup when ESPN odds fail)."""
    name: str

    @abstractmethod
    def odds_on(self, league: str, day: date) -> list[OddsSnapshot]:
        ...
