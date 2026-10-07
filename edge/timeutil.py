"""Time helpers. Rule: store UTC, convert to Eastern only for dates and display."""
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo   # on Windows this needs the `tzdata` package (in requirements)

ET = ZoneInfo("America/New_York")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> datetime:
    """Parse ESPN-style times like '2026-10-06T23:00Z' or '2026-10-06T23:00:00Z'."""
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def et_date(start_time_utc: str) -> date:
    """The calendar date of the game in Eastern time (a 10pm ET tip is 02:00Z next day)."""
    return parse_utc(start_time_utc).astimezone(ET).date()


def et_today() -> date:
    return utcnow().astimezone(ET).date()


def et_display(start_time_utc: str) -> str:
    return parse_utc(start_time_utc).astimezone(ET).strftime("%a %b %d, %I:%M %p ET").replace(" 0", " ")
