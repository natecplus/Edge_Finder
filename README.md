# Edge Finder

ML-powered moneyline picks. For every NBA and NFL game and ATP tennis match it estimates each team's win probability with a calibrated model, compares that with what the sportsbook odds imply, and says **Bet, Lean, or Pass** with plain-English reasons. Then it grades itself after every game.

> Model estimates, not guarantees. Personal project; bet only where it's legal for you, within a budget.

## How it works

```
Scheduler (nightly + every 30 min on game days)
   │
   ▼
Source adapters ──► ESPN JSON (primary) · Basketball-Reference HTML · SportsBookReview HTML (backups)
   │                 automatic failover, every attempt logged
   ▼
SQLite database ──► games · odds snapshots · injuries · box-score minutes · picks · results · models
   │
   ▼
Features (leakage-tested) ──► Elo, recent form, point differential, rest, missing rotation minutes
   │
   ▼
Model ──► Elo baseline vs logistic regression vs LightGBM, season-based validation, Platt calibration
   │
   ▼
Edge = model win % − market's vig-free win %
   │
   ▼
Rules engine ──► preseason, key players out, small sample, stale odds, line moves, league trust …
   │
   ▼
Streamlit app ──► Today's slate · Game detail · Track record (ROI, CLV, calibration) · Settings / data health
```

## Quick start (Windows PowerShell, inside the project folder)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pytest
```

**See it working right away with simulated data** (no scraping, about 3 minutes):

```powershell
python -m scripts.demo_data
$env:EDGE_DATA_DIR = "data/demo"; streamlit run app/streamlit_app.py
```

Close the app with Ctrl+C. Run `Remove-Item Env:EDGE_DATA_DIR` (or open a new terminal) to go back to your real database.

**Real data:**

```powershell
python -m scripts.backfill --league nba --seasons 2022 2023 2024 2025 2026   # hours; resumable
python -m scripts.train_model --league nba
python -m scripts.fetch_today nba
python -m edge.jobs --once gameday --league nba        # make today's picks
streamlit run app/streamlit_app.py
python -m edge.jobs                                     # leave running: the full schedule
```

Season labels follow ESPN: NBA uses the season's **end** year (2025-26 = 2026), NFL its **start** year.

**Tennis (ATP):**

```powershell
python -m scripts.backfill --league atp --seasons 2022 2023 2024 2025 2026   # about a minute
python -m scripts.train_model --league atp
python -m edge.jobs --once gameday --league atp
streamlit run app/streamlit_app.py      # pick ATP in the sidebar; Tournament defaults to Shanghai
```

History and closing odds come from tennis-data.co.uk (one Excel file per year); today's matches from ESPN. Free live tennis odds are thin, so either type a price on the **Game detail** page (the pick recalculates instantly) or set a free [The Odds API](https://the-odds-api.com) key: `$env:ODDS_API_KEY = "your-key"` before starting the scheduler.

## Project layout

| Path | What it does |
| --- | --- |
| `edge/sources/` | One adapter per site, all returning the same dataclasses (`base.py`). Failover in `__init__.py`. |
| `edge/db.py` | Tables and read/write helpers (SQLAlchemy Core, SQLite). |
| `edge/ingest.py` | Pulls a day of games, odds, injuries and box scores into the database. |
| `edge/elo.py`, `edge/features.py` | Ratings and the per-game feature table. |
| `edge/train.py` | Season split, model comparison, calibration, versioning, promotion. |
| `edge/backtest.py`, `edge/market.py`, `edge/odds_math.py` | Betting math: fair odds, closing lines, ROI simulation. |
| `edge/rules.py`, `edge/predict.py`, `edge/explain.py` | Verdicts, today's picks, SHAP / coefficient reasons. |
| `edge/grade.py`, `edge/tracking.py` | Grading, ROI, CLV, calibration, league trust status. |
| `edge/jobs.py` | The scheduler. |
| `edge/players.py`, `edge/tennis_*.py`, `edge/sources/espn_tennis.py`, `tennis_data.py`, `odds_api.py` | Tennis: name matching across sources, surface Elo / ranking / form / fatigue features, ingest with cross-source de-duplication. |
| `config/leagues/*.yaml` | Everything league-specific. Adding a league = one YAML + one adapter path. |
| `app/` | Streamlit pages. |
| `scripts/` | Backfill, training report, demo data. |
| `tests/` | Odds math, Elo, leakage test, rules, parsers (saved fixtures), DB, failover, ingest. |

## Results

Fill this in from `python -m scripts.train_model --league nba` once you've backfilled real data:

| Predictor (test season) | Log loss |
| --- | --- |
| Elo baseline | … |
| Calibrated model | … |
| Closing market | … |

Backtest at the validation-chosen threshold: … bets, ROI …

## Design decisions worth knowing

- **Own the data.** Sources only feed the database; the app only reads the database. If every site disappeared tomorrow, history, models and the app would still work.
- **Odds are never a model feature.** A model that sees the line learns to copy it and can't disagree with the market. Odds are used only to measure the edge.
- **Season-based splits only.** Random splits leak the future. The threshold is chosen on the validation season, and the test season is scored once.
- **The market is the real benchmark.** Beating Elo is easy; getting close to the closing line is not. If no threshold is profitable on validation, every pick is capped at Lean.
- **Known train/serve gap:** for past games, availability comes from who actually played (box scores); for upcoming games, from the injury report. They differ only for late scratches.
- **Tennis players are stored alphabetically, never winner-first**, so which column a player is in can't leak the result.
- **Grading uses locked prices.** Picks lock 60 minutes before start; CLV compares that price with the close.

## What broke and how I fixed it

(Keep notes here as you go: scraper changes, leakage you caught, the NFL refactor. Interviewers love this section.)

## Deploy

```bash
docker compose up -d --build
```

Runs the scheduler and the app (port 8501) from one image, sharing `./data`. Restrict port 8501 to your own IP in the server firewall. Back up `data/edge.db` daily.
