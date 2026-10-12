"""Synthetic nflverse-shaped data (play-by-play, coverage charting, rosters).

Used by the demo and the tests, so the matchup engine can be exercised without
downloading anything. Defenses have hidden traits (man rate, weakness vs WRs /
TEs / RBs, run defense, pressure); receivers are better vs man or vs zone.
"""
import random

import numpy as np
import pandas as pd

NFLVERSE_TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
                  "HOU", "IND", "JAX", "KC", "LV", "LAC", "LA", "MIA", "MIN", "NE", "NO", "NYG",
                  "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"]
ROUTES = ["HITCH/CURL", "IN/DIG", "SLANT", "GO", "POST", "QUICK OUT", "CORNER", "SCREEN", "SHALLOW CROSS/DRAG"]
COVERS = {"man": ["COVER_1", "COVER_0", "2_MAN"], "zone": ["COVER_3", "COVER_2", "COVER_4", "COVER_6"]}


def generate(seasons, weeks_last=17, weeks_current=6, seed=11):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    first = ["Jalen", "Marcus", "Tyler", "Devin", "Amon", "Cam", "Puka", "Malik", "Trey", "Zay", "Josh", "Kyle"]
    last = ["Brooks", "Carter", "Hayes", "Mitchell", "Porter", "Reed", "Sanders", "Turner", "Walker", "Young"]
    defense = {t: {"man": rng.uniform(0.2, 0.55), "wr": rng.gauss(1, 0.12), "te": rng.gauss(1, 0.15),
                   "rb": rng.gauss(1, 0.12), "run": rng.gauss(1, 0.1), "press": rng.uniform(0.22, 0.4),
                   "route": {r: rng.gauss(1, 0.15) for r in ROUTES}} for t in NFLVERSE_TEAMS}
    players, roster = {}, []
    pid = 0
    for t in NFLVERSE_TEAMS:
        for pos, n in (("WR", 3), ("TE", 1), ("RB", 2), ("QB", 1)):
            for i in range(n):
                pid += 1
                gid = f"00-00{pid:05d}"
                name = f"{rng.choice(first)[0]}.{rng.choice(last)}{pid % 7 or ''}"
                players[gid] = {"team": t, "pos": pos, "name": name, "skill": rng.gauss(1, 0.15),
                                "man": rng.gauss(1, 0.12), "zone": rng.gauss(1, 0.12),
                                "share": [0.26, 0.2, 0.13][i] if pos == "WR" else 0.17 if pos == "TE" else 0.06,
                                "routes": rng.sample(ROUTES, 3)}
                for s in seasons:
                    roster.append({"gsis_id": gid, "full_name": name, "position": pos, "team": t, "season": s})
    pbp, part = [], []
    for s in seasons:
        weeks = weeks_current if s == max(seasons) else weeks_last
        for w in range(1, weeks + 1):
            teams = NFLVERSE_TEAMS[:]
            rng.shuffle(teams)
            for a, b in zip(teams[::2], teams[1::2]):
                game_id = f"{s}_{w:02d}_{a}_{b}"
                play_id = 0
                for off, dfn in ((a, b), (b, a)):
                    d = defense[dfn]
                    roster_off = [g for g, p in players.items() if p["team"] == off]
                    qb = next(g for g in roster_off if players[g]["pos"] == "QB")
                    rbs = [g for g in roster_off if players[g]["pos"] == "RB"]
                    rec = [g for g in roster_off if players[g]["pos"] in ("WR", "TE", "RB")]
                    weights = np.array([players[g]["share"] for g in rec])
                    for _ in range(35):                               # dropbacks
                        play_id += 1
                        cov = "man" if rng.random() < d["man"] else "zone"
                        pressured = rng.random() < d["press"]
                        target = rec[nrng.choice(len(rec), p=weights / weights.sum())]
                        p = players[target]
                        route = rng.choice(p["routes"])
                        posf = d["wr" if p["pos"] == "WR" else "te" if p["pos"] == "TE" else "rb"]
                        mean = 7.5 * p["skill"] * p[cov] * posf * d["route"][route] * (0.75 if pressured else 1.05)
                        caught = rng.random() < 0.65
                        yards = max(0.0, nrng.normal(mean / 0.65, 5)) if caught else 0.0
                        pbp.append({"game_id": game_id, "play_id": play_id, "season": s, "week": w,
                                    "season_type": "REG", "posteam": off, "defteam": dfn, "play_type": "pass",
                                    "pass": 1, "rush": 0, "qb_dropback": 1, "sack": 0, "two_point_attempt": 0,
                                    "passer_player_id": qb, "passer_player_name": players[qb]["name"],
                                    "receiver_player_id": target, "receiver_player_name": p["name"],
                                    "rusher_player_id": None, "rusher_player_name": None,
                                    "yards_gained": round(yards), "epa": (yards - 6) / 10,
                                    "complete_pass": int(caught), "air_yards": yards * 0.6})
                        part.append({"nflverse_game_id": game_id, "play_id": play_id,
                                     "defense_man_zone_type": "MAN_COVERAGE" if cov == "man" else "ZONE_COVERAGE",
                                     "defense_coverage_type": rng.choice(COVERS[cov]), "route": route,
                                     "was_pressure": pressured})
                    for _ in range(26):                               # runs
                        play_id += 1
                        r = rbs[0] if rng.random() < 0.7 else rbs[1]
                        y = round(nrng.normal(4.3 * players[r]["skill"] * d["run"], 5))
                        pbp.append({"game_id": game_id, "play_id": play_id, "season": s, "week": w,
                                    "season_type": "REG", "posteam": off, "defteam": dfn, "play_type": "run",
                                    "pass": 0, "rush": 1, "qb_dropback": 0, "sack": 0, "two_point_attempt": 0,
                                    "passer_player_id": None, "passer_player_name": None,
                                    "receiver_player_id": None, "receiver_player_name": None,
                                    "rusher_player_id": r, "rusher_player_name": players[r]["name"],
                                    "yards_gained": y, "epa": (y - 4) / 10, "complete_pass": 0, "air_yards": None})
    return {"pbp": pd.DataFrame(pbp), "participation": pd.DataFrame(part), "roster": pd.DataFrame(roster)}
