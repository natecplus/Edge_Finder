from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


@dataclass
class Game:
    league: str
    source_id: str
    start_time: str          # ISO 8601, always UTC
    season: int              # ESPN labels by end year: 2026-27 season = 2027
    season_type: str         # "preseason" | "regular" | "postseason"
    home_team: str           # short code, e.g. "CHA"
    away_team: str
    home_score: int | None   # None until the game is finished
    away_score: int | None
    status: str


class ScheduleSource(ABC):
    name: str

    @abstractmethod
    def games_on(self, league: str, day: date) -> list[Game]:
        ...