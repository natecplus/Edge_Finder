"""Simulated data so you can run the whole app without scraping anything.

    python -m scripts.demo_data

Writes to data/demo/ (never touches your real data/edge.db), trains both
leagues, replays ~6 weeks of picks so the Track Record page has history, and
makes picks for today's simulated slate. Then view it with:

    PowerShell:  $env:EDGE_DATA_DIR = "data/demo"; streamlit run app/streamlit_app.py
    Mac/Linux:   EDGE_DATA_DIR=data/demo streamlit run app/streamlit_app.py

The simulation has hidden "true" team strengths, injuries, rest effects and a
noisy but well-informed betting market. Because the market here knows almost
as much as the model, don't expect big profits: that's realistic.
"""
import math
import os
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "data" / "demo"
os.environ["EDGE_DATA_DIR"] = str(DEMO_DIR)      # must be set before importing edge
sys.path.insert(0, str(ROOT))

from edge import db, grade, predict, train                        # noqa: E402
from edge.odds_math import american_from_prob                     # noqa: E402
from edge.sources.base import Game, InjuryReport, OddsSnapshot, PlayerMinutes  # noqa: E402
from edge.teams import NBA, NFL                                   # noqa: E402
from edge.timeutil import ET, et_date, to_iso, utcnow             # noqa: E402

rng = random.Random(7)
BOOKS = ["DraftKings", "FanDuel", "BetMGM"]
VIG = 0.045
BASE_MINUTES = [36, 34, 32, 30, 28, 24, 20, 16, 12, 8]            # sums to 240
FIRST = ["Jalen", "Marcus", "Tyrese", "Devin", "Anthony", "Cam", "Jordan", "Malik", "Darius",
         "Kevin", "Isaiah", "Andre", "Trey", "Miles", "Josh", "Derrick", "Zion", "Evan"]
LAST = ["Brooks", "Carter", "Hayes", "Mitchell", "Porter", "Reed", "Sanders", "Turner", "Walker",
        "Young", "Bryant", "Coleman", "Ellis", "Foster", "Grant", "Harris", "Jenkins", "Lewis"]


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def logit(p):
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def at_et(day, hour, minute=0):
    return to_iso(datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET))


class Team:
    def __init__(self, code, league):
        self.code = code
        self.strength = rng.gauss(0, 0.65)
        self.players = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in BASE_MINUTES]
        self.players = [f"{n} ({code})" for n in self.players]   # unique names per team
        self.injured_for = [0] * len(BASE_MINUTES)
        self.qb = f"{rng.choice(FIRST)} {rng.choice(LAST)} ({code})"
        self.qb_out = 0
        self.last_game = None

    def new_season(self):
        self.strength = 0.6 * self.strength + 0.52 * rng.gauss(0, 1)

    def tick_injuries(self):
        for i in range(len(self.injured_for)):
            if self.injured_for[i] > 0:
                self.injured_for[i] -= 1
            elif rng.random() < 0.035:
                self.injured_for[i] = 1 + int(rng.expovariate(1 / 4))
        if self.qb_out > 0:
            self.qb_out -= 1
        elif rng.random() < 0.02:
            self.qb_out = 1 + int(rng.expovariate(1 / 3))

    def injury_points(self, league):
        if league == "nfl":
            return 4.0 if self.qb_out else 0.0
        return sum(BASE_MINUTES[i] / 240 * 1.3 * 14 for i, d in enumerate(self.injured_for) if d > 0)


class Sim:
    def __init__(self, league):
        self.league = league
        codes = list((NBA if league == "nba" else NFL).keys())
        self.teams = {c: Team(c, league) for c in codes}
        self.games, self.odds, self.minutes, self.injuries = [], [], [], []
        self.sd = 12.0 if league == "nba" else 13.5
        self.home_pts = 2.5 if league == "nba" else 1.8

    def rest_penalty(self, team, day):
        if team.last_game is None:
            return 0.0
        rest = (day - team.last_game).days
        if self.league == "nba":
            return 1.5 if rest <= 1 else 0.0
        return 1.0 if rest <= 5 else 0.0

    def play(self, home, away, start, season, season_type, day, final=True):
        h, a = self.teams[home], self.teams[away]
        for t in (h, a):
            t.tick_injuries()
            t.strength += rng.gauss(0, 0.02)
        mu = (6 * (h.strength - a.strength) + self.home_pts
              - self.rest_penalty(h, day) + self.rest_penalty(a, day)
              - h.injury_points(self.league) + a.injury_points(self.league))
        if season_type == "preseason":
            mu *= 0.3
        p_true = phi(mu / self.sd)
        g = Game(self.league, f"demo{len(self.games)}", start, season, season_type, home, away,
                 None, None, "STATUS_SCHEDULED")
        key = g.game_key
        if final:
            margin = round(rng.gauss(mu, self.sd)) or rng.choice([-1, 1])
            base = 110 if self.league == "nba" else 22
            g.home_score, g.away_score = base + max(margin, 0) + rng.randint(0, 8), base + max(-margin, 0) + rng.randint(0, 8)
            g.status = "STATUS_FINAL"
            h.last_game = a.last_game = day
            if self.league == "nba":
                for t in (h, a):
                    avail = [i for i, d in enumerate(t.injured_for) if d == 0]
                    scale = 240 / sum(BASE_MINUTES[i] for i in avail)
                    for i, name in enumerate(t.players):
                        mins = BASE_MINUTES[i] * scale + rng.gauss(0, 2) if i in avail else 0.0
                        self.minutes.append(PlayerMinutes(key, t.code, name, max(mins, 0.0)))
        if final:
            # pregame injury report, 3 hours before the game
            report_time = to_iso(datetime.fromisoformat(start.replace("Z", "+00:00")) - timedelta(hours=3))
            for t in (h, a):
                outs = ([(t.players[i], "") for i, d in enumerate(t.injured_for) if d > 0]
                        if self.league == "nba" else ([(t.qb, "QB")] if t.qb_out else []))
                for name, pos in outs:
                    self.injuries.append((InjuryReport(key, t.code, name, "Out", pos, "demo"), report_time))
        self.games.append(g)
        self._market(key, p_true, start, final)
        return g, p_true

    def _market(self, key, p_true, start, final):
        z_open = logit(p_true) + rng.gauss(0, 0.25)
        z_close = logit(p_true) + rng.gauss(0, 0.12)
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        for book in BOOKS:
            for kind, z, when in (("open", z_open, start_dt - timedelta(hours=20)),
                                  ("live", z_close, start_dt - timedelta(hours=2)),
                                  ("close", z_close, start_dt - timedelta(minutes=5))):
                if kind == "close" and not final:
                    continue
                f = sigmoid(z + rng.gauss(0, 0.03))
                snap = OddsSnapshot(key, book, american_from_prob(f * (1 + VIG)),
                                    american_from_prob((1 - f) * (1 + VIG)), kind, "demo")
                self.odds.append((snap, to_iso(when)))


def nba_season(sim, label, first_day, days):
    for t in sim.teams.values():
        t.new_season()
        t.last_game = None
    codes = list(sim.teams)
    for d in range(days):
        day = first_day + timedelta(days=d)
        teams = rng.sample(codes, 16)
        for j in range(0, 16, 2):
            sim.play(teams[j], teams[j + 1], at_et(day, rng.choice([19, 19, 20, 22])), label, "regular", day)


def nfl_season(sim, label, first_sunday, weeks):
    for t in sim.teams.values():
        t.new_season()
        t.last_game = None
    codes = list(sim.teams)
    for w in range(weeks):
        sunday = first_sunday + timedelta(weeks=w)
        rng.shuffle(codes)
        for j in range(0, 32, 2):
            if j == 0:
                day, hour = sunday - timedelta(days=3), 20      # Thursday night
            elif j == 2:
                day, hour = sunday + timedelta(days=1), 20      # Monday night
            else:
                day, hour = sunday, rng.choice([13, 13, 16])
            sim.play(codes[j], codes[j + 1], at_et(day, hour, 0 if hour != 16 else 25), label, "regular", day)


def todays_slate(sim, today, now):
    """Unplayed games today, with injury reports and fresh odds."""
    codes = list(sim.teams)
    if sim.league == "nba":
        rng.shuffle(codes)
        pairs = [codes[i:i + 2] for i in range(0, 14, 2)]
        hours = [19, 19, 19, 20, 20, 22, 22]
        out = [sim.play(h, a, at_et(today, hr, 30 if hr == 19 else 0), CURRENT["nba"], "regular", today, final=False)
               for (h, a), hr in zip(pairs, hours)]
        # one preseason exhibition so the preseason rule shows up
        out.append(sim.play(codes[14], codes[15], at_et(today, 21), CURRENT["nba"], "preseason", today, final=False))
    else:
        rng.shuffle(codes)
        out = [sim.play(codes[i], codes[i + 1], at_et(today, hr), CURRENT["nfl"], "regular", today, final=False)
               for i, hr in zip(range(0, 10, 2), [20, 20, 21, 21, 22])]
    captured = to_iso(now - timedelta(hours=1))
    for g, p_true in out:
        for code in (g.home_team, g.away_team):
            t = sim.teams[code]
            for i, d in enumerate(t.injured_for if sim.league == "nba" else []):
                if d > 0:
                    sim.injuries.append((InjuryReport(g.game_key, code, t.players[i], "Out", "", "demo"), captured))
            if sim.league == "nfl" and t.qb_out:
                sim.injuries.append((InjuryReport(g.game_key, code, t.qb, "Out", "QB", "demo"), captured))
        # fresh live odds 10 minutes ago, so nothing looks stale
        z = logit(p_true) + rng.gauss(0, 0.12)
        for book in BOOKS:
            f = sigmoid(z + rng.gauss(0, 0.03))
            sim.odds.append((OddsSnapshot(g.game_key, book, american_from_prob(f * (1 + VIG)),
                                          american_from_prob((1 - f) * (1 + VIG)), "live", "demo"),
                             to_iso(now - timedelta(minutes=10))))


def save(sim):
    db.upsert_games(sim.games)
    by_time = {}
    for snap, when in sim.odds:
        by_time.setdefault(when, []).append(snap)
    for when, snaps in by_time.items():
        db.insert_odds(snaps, captured_at=when)
    db.upsert_minutes(sim.minutes)
    inj = {}
    for rep, when in sim.injuries:
        inj.setdefault(when, []).append(rep)
    for when, reps in inj.items():
        db.insert_injuries(reps, captured_at=when)
    db.log_source_run("demo", f"schedule:{sim.league}", ok=True, rows=len(sim.games))


def replay_picks(league, since):
    """Make + lock + grade picks for recent finished days, one day at a time."""
    games = db.games_df(league)
    past = games[(games.start_time >= to_iso(since)) & games.home_score.notna()]
    from edge.features import build_features
    cfg = predict.effective_config(league)
    feats = build_features(games, cfg, minutes=db.minutes_df(league) if cfg.get("use_minutes") else None)
    days = sorted({et_date(s) for s in past.start_time})
    for day in days:
        todays = past[past.start_time.map(lambda s: et_date(s) == day)]
        # one pass per start time, 61 minutes before it (exactly when a real pick locks)
        for start, slot in todays.groupby("start_time"):
            t = datetime.fromisoformat(start.replace("Z", "+00:00"))
            predict.predict_league(league, now=t - timedelta(minutes=61),
                                   keys=list(slot.game_key), lock_all=True, features=feats)
        grade.grade_league(league)
    print(f"  replayed {len(days)} days of {league.upper()} picks")


CURRENT = {}

SURFACES = ["Hard", "Clay", "Grass"]
SHANGHAI = "Rolex Shanghai Masters (demo)"


def tennis_demo(now, today):
    """96 simulated players with surface-specific skill, ~4.8 seasons of matches,
    and today's 'Shanghai' slate (two matches without odds, to try manual entry)."""
    from edge.players import ordered
    from edge.sources.espn_tennis import TennisMatch
    from edge.tennis_ingest import _write

    surnames = sorted({f"{rng.choice(LAST)}{rng.choice(['', 'son', 'ov', 'ez', 'ini', 'er'])}" for _ in range(400)})
    players = [f"{n} {chr(65 + i % 26)}." for i, n in enumerate(rng.sample(surnames, 96))]
    base = {p: rng.gauss(0, 0.55) for p in players}
    surf = {(p, s): base[p] + rng.gauss(0, 0.25) for p in players for s in SURFACES}

    def rank_of(day_idx):
        noisy = sorted(players, key=lambda p: -(base[p] + rng.gauss(0, 0.15)))
        return {p: i + 1 for i, p in enumerate(noisy)}

    matches, odds = [], []
    start_day = datetime(today.year - 4, 1, 3).date()
    n_days = (today - start_day).days
    ranks = rank_of(0)
    for d in range(n_days + 1):
        day = start_day + timedelta(days=d)
        if day.month == 12 or rng.random() < 0.25:
            continue                               # off-season / rest days
        if d % 28 == 0:
            ranks = rank_of(d)
            for p in players:
                base[p] += rng.gauss(0, 0.05)
        doy = day.timetuple().tm_yday
        s = "Clay" if 95 <= doy <= 160 else "Grass" if 161 <= doy <= 190 else "Hard"
        days_ago = (today - day).days
        tournament = SHANGHAI if days_ago <= 4 else f"ATP {s} Event {doy // 7}"
        if days_ago == 0:
            continue
        for a, b in [rng.sample(players, 2) for _ in range(10)]:
            p_true = 1 / (1 + math.exp(-1.3 * (surf[(a, s)] - surf[(b, s)])))
            winner, loser = (a, b) if rng.random() < p_true else (b, a)
            home, away = ordered(a, b)
            start = to_iso(datetime(day.year, day.month, day.day, 8 + rng.randint(0, 8), tzinfo=ET))
            sets = {winner: 2, loser: rng.choice([0, 1])}
            g = Game("atp", None, start, day.year, "regular", home, away, sets[home], sets[away], "STATUS_FINAL")
            p_home = p_true if home == a else 1 - p_true
            meta = {"game_key": g.game_key, "tournament": tournament, "series": "ATP", "surface": s,
                    "round": "Round of 32", "best_of": 3, "home_rank": ranks[home], "away_rank": ranks[away],
                    "comment": "Completed"}
            matches.append(TennisMatch(g, meta, []))
            odds.extend(_tennis_odds(g, p_home, start, final=True))

    # today's Shanghai slate
    slate = rng.sample(players, 16)
    for i in range(8):
        a, b = slate[2 * i], slate[2 * i + 1]
        home, away = ordered(a, b)
        start = to_iso(now + timedelta(minutes=75 + 12 * i))
        p_true = 1 / (1 + math.exp(-1.3 * (surf[(home, "Hard")] - surf[(away, "Hard")])))
        g = Game("atp", f"demo-t{i}", start, today.year, "regular", home, away, None, None, "STATUS_SCHEDULED")
        meta = {"game_key": g.game_key, "tournament": SHANGHAI, "series": "Masters 1000", "surface": "Hard",
                "round": "Quarterfinal" if i < 4 else "Round of 16", "best_of": 3, "comment": "Completed"}
        if i < 6:                                  # last two: no odds yet -> try manual entry
            odds.extend(_tennis_odds(g, p_true, start, final=False, live_at=now - timedelta(minutes=15)))
        matches.append(TennisMatch(g, meta, []))

    print(f"  {_write('atp', matches)}")
    by_time = {}
    for snap, when in odds:
        by_time.setdefault(when, []).append(snap)
    for when, snaps in by_time.items():
        db.insert_odds(snaps, captured_at=when)
    db.log_source_run("demo", "schedule:atp", ok=True, rows=len(matches))


def _tennis_odds(g, p_home, start, final, live_at=None):
    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    z = logit(min(max(p_home, 0.02), 0.98))
    out = []
    for book in BOOKS:
        snaps = [("open", z + rng.gauss(0, 0.3), t0 - timedelta(hours=18))]
        zc = z + rng.gauss(0, 0.15)
        snaps.append(("live", zc, live_at or (t0 - timedelta(hours=2))))
        if final:
            snaps.append(("close", zc, t0 - timedelta(minutes=5)))
        for kind, zz, when in snaps:
            f = sigmoid(zz + rng.gauss(0, 0.03))
            o = OddsSnapshot(g.game_key, book, american_from_prob(f * (1 + VIG)),
                             american_from_prob((1 - f) * (1 + VIG)), kind, "demo")
            out.append((o, to_iso(when)))
    return out


def main():
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    db.init()
    now = utcnow()
    today = now.astimezone(ET).date()

    print("Simulating NBA: 5 past seasons + current season ...")
    nba = Sim("nba")
    this_label = today.year + 1 if today.month >= 8 else today.year
    for label in range(this_label - 5, this_label):
        nba_season(nba, label, datetime(label - 1, 10, 22).date(), 165)
    CURRENT["nba"] = this_label
    # demo pretends the current season tipped off 45 days ago
    nba_season(nba, this_label, today - timedelta(days=45), 45)
    todays_slate(nba, today, now)
    save(nba)

    print("Simulating NFL: 5 past seasons + current season ...")
    nfl = Sim("nfl")
    this_nfl = today.year if today.month >= 3 else today.year - 1
    for label in range(this_nfl - 5, this_nfl):
        first_sun = datetime(label, 9, 7).date()
        first_sun += timedelta(days=(6 - first_sun.weekday()) % 7)
        nfl_season(nfl, label, first_sun, 18)
    CURRENT["nfl"] = this_nfl
    nfl_season(nfl, this_nfl, today - timedelta(days=(today.weekday() + 1) % 7 + 35), 5)
    todays_slate(nfl, today, now)
    save(nfl)

    print("Simulating ATP tennis: ~5 seasons + today's Shanghai slate ...")
    tennis_demo(now, today)

    for league in ("nba", "nfl", "atp"):
        print(f"\nTraining {league.upper()} ...")
        result = train.train_league(league)
        r = result["report"]
        print(f"  best={r['best_kind']}  test log loss {r['test']['log_loss']:.4f} "
              f"(Elo {r['test_elo_baseline']['log_loss']:.4f}, market {r['market_log_loss']:.4f})")
        replay_picks(league, now - timedelta(days=42 if league != "atp" else 21))
        picks = predict.predict_league(league, now=now)
        if not picks.empty:
            print(f"  today: {picks.verdict.value_counts().to_dict()}")

    print(f"\nDemo database ready at {DEMO_DIR / 'edge.db'}")
    print('View it:  $env:EDGE_DATA_DIR = "data/demo"; streamlit run app/streamlit_app.py')


if __name__ == "__main__":
    main()
