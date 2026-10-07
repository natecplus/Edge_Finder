"""One team = one code everywhere.

Each source names teams differently ("GSW", "GS", "Golden State Warriors",
"Warriors"). Every adapter runs names through `normalize()` so the database
only ever stores ESPN's abbreviation.
"""

NBA = {
    "ATL": ["Atlanta Hawks", "Hawks", "Atlanta"],
    "BOS": ["Boston Celtics", "Celtics", "Boston"],
    "BKN": ["Brooklyn Nets", "Nets", "Brooklyn", "BRK", "BKN"],
    "CHA": ["Charlotte Hornets", "Hornets", "Charlotte", "CHO"],
    "CHI": ["Chicago Bulls", "Bulls", "Chicago"],
    "CLE": ["Cleveland Cavaliers", "Cavaliers", "Cavs", "Cleveland"],
    "DAL": ["Dallas Mavericks", "Mavericks", "Mavs", "Dallas"],
    "DEN": ["Denver Nuggets", "Nuggets", "Denver"],
    "DET": ["Detroit Pistons", "Pistons", "Detroit"],
    "GS": ["Golden State Warriors", "Warriors", "Golden State", "GSW"],
    "HOU": ["Houston Rockets", "Rockets", "Houston"],
    "IND": ["Indiana Pacers", "Pacers", "Indiana"],
    "LAC": ["Los Angeles Clippers", "LA Clippers", "Clippers"],
    "LAL": ["Los Angeles Lakers", "LA Lakers", "Lakers"],
    "MEM": ["Memphis Grizzlies", "Grizzlies", "Memphis"],
    "MIA": ["Miami Heat", "Heat", "Miami"],
    "MIL": ["Milwaukee Bucks", "Bucks", "Milwaukee"],
    "MIN": ["Minnesota Timberwolves", "Timberwolves", "Wolves", "Minnesota"],
    "NO": ["New Orleans Pelicans", "Pelicans", "New Orleans", "NOP"],
    "NY": ["New York Knicks", "Knicks", "New York", "NYK"],
    "OKC": ["Oklahoma City Thunder", "Thunder", "Oklahoma City"],
    "ORL": ["Orlando Magic", "Magic", "Orlando"],
    "PHI": ["Philadelphia 76ers", "76ers", "Sixers", "Philadelphia"],
    "PHX": ["Phoenix Suns", "Suns", "Phoenix", "PHO"],
    "POR": ["Portland Trail Blazers", "Trail Blazers", "Blazers", "Portland"],
    "SAC": ["Sacramento Kings", "Kings", "Sacramento"],
    "SA": ["San Antonio Spurs", "Spurs", "San Antonio", "SAS"],
    "TOR": ["Toronto Raptors", "Raptors", "Toronto"],
    "UTAH": ["Utah Jazz", "Jazz", "Utah", "UTA"],
    "WSH": ["Washington Wizards", "Wizards", "Washington", "WAS"],
}

NFL = {
    "ARI": ["Arizona Cardinals", "Cardinals", "Arizona"],
    "ATL": ["Atlanta Falcons", "Falcons", "Atlanta"],
    "BAL": ["Baltimore Ravens", "Ravens", "Baltimore"],
    "BUF": ["Buffalo Bills", "Bills", "Buffalo"],
    "CAR": ["Carolina Panthers", "Panthers", "Carolina"],
    "CHI": ["Chicago Bears", "Bears", "Chicago"],
    "CIN": ["Cincinnati Bengals", "Bengals", "Cincinnati"],
    "CLE": ["Cleveland Browns", "Browns", "Cleveland"],
    "DAL": ["Dallas Cowboys", "Cowboys", "Dallas"],
    "DEN": ["Denver Broncos", "Broncos", "Denver"],
    "DET": ["Detroit Lions", "Lions", "Detroit"],
    "GB": ["Green Bay Packers", "Packers", "Green Bay", "GNB"],
    "HOU": ["Houston Texans", "Texans", "Houston"],
    "IND": ["Indianapolis Colts", "Colts", "Indianapolis"],
    "JAX": ["Jacksonville Jaguars", "Jaguars", "Jacksonville", "JAC"],
    "KC": ["Kansas City Chiefs", "Chiefs", "Kansas City", "KAN"],
    "LV": ["Las Vegas Raiders", "Raiders", "Las Vegas", "LVR"],
    "LAC": ["Los Angeles Chargers", "Chargers", "LA Chargers"],
    "LAR": ["Los Angeles Rams", "Rams", "LA Rams", "LA"],
    "MIA": ["Miami Dolphins", "Dolphins", "Miami"],
    "MIN": ["Minnesota Vikings", "Vikings", "Minnesota"],
    "NE": ["New England Patriots", "Patriots", "New England", "NWE"],
    "NO": ["New Orleans Saints", "Saints", "New Orleans", "NOR"],
    "NYG": ["New York Giants", "Giants"],
    "NYJ": ["New York Jets", "Jets"],
    "PHI": ["Philadelphia Eagles", "Eagles", "Philadelphia"],
    "PIT": ["Pittsburgh Steelers", "Steelers", "Pittsburgh"],
    "SF": ["San Francisco 49ers", "49ers", "San Francisco", "SFO"],
    "SEA": ["Seattle Seahawks", "Seahawks", "Seattle"],
    "TB": ["Tampa Bay Buccaneers", "Buccaneers", "Bucs", "Tampa Bay", "TAM"],
    "TEN": ["Tennessee Titans", "Titans", "Tennessee"],
    "WSH": ["Washington Commanders", "Commanders", "Washington", "WAS"],
}

_TABLES = {"nba": NBA, "nfl": NFL}
_LOOKUP = {
    league: {
        alias.lower(): code
        for code, aliases in table.items()
        for alias in [code, *aliases]
    }
    for league, table in _TABLES.items()
}


def normalize(league: str, name: str) -> str:
    """Return the ESPN abbreviation for any known name. Unknown names come back
    upper-cased (e.g. international preseason opponents) instead of crashing."""
    if not name:
        return ""
    key = name.strip().lower()
    return _LOOKUP.get(league, {}).get(key, name.strip().upper())


def is_known(league: str, code: str) -> bool:
    return code in _TABLES.get(league, {})


def full_name(league: str, code: str) -> str:
    aliases = _TABLES.get(league, {}).get(code)
    return aliases[0] if aliases else code
