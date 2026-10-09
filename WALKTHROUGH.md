# Walkthrough: understand every part before you show it to anyone

Read this once with the code open. Then you can explain the project in an interview without notes.

## 1. First run on your PC (do these in order)

1. Copy the files from the zip into your `Edge_Finder` folder. Let it replace `requirements.txt`, `README.md`, `scripts/fetch_today.py`, `tests/test_espn.py` and the empty files in `edge/`. Your `.gitignore` and your saved `tests/fixtures/espn_nba_scoreboard.json` are not in the zip, so they stay as they are.
2. With `(.venv)` showing:
   ```powershell
   pip install -r requirements.txt
   pip install -e .
   python -m pytest
   ```
   Expect everything to pass. Your own ESPN fixture is tested too.
3. Demo: `python -m scripts.demo_data`, then `$env:EDGE_DATA_DIR = "data/demo"; streamlit run app/streamlit_app.py`. Click through all four pages.
4. Real data, small first: open a new terminal (so EDGE_DATA_DIR is unset), then `python -m scripts.fetch_today nba`. Open `data/edge.db` in SQLite Viewer and look at `games`, `odds_snapshots`, `source_runs`.
5. **Check the parsers against real ESPN data.** I couldn't reach ESPN while building this, so the odds/injury/box-score parsers in `edge/sources/espn.py` follow ESPN's known layout and are tested on sample files I wrote. Save a real summary and compare:
   ```powershell
   Invoke-WebRequest -Uri "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=401901820" -OutFile "tests\fixtures\espn_nba_summary_real.json"
   ```
   Search it for `pickcenter`, `injuries` and `boxscore`. If a field name differs from what `espn.py` reads, fix that one spot. Same idea for SportsBookReview (`sbr.py`): it's the least certain parser, and the app works without it.
6. Backfill (start before bed): `python -m scripts.backfill --league nba --seasons 2022 2023 2024 2025 2026`. If it stops, run the same command again; it resumes from the cache.
7. `python -m scripts.train_model --league nba` and read the report (section 4 below explains it).
8. Leave `python -m edge.jobs` running on a game day, or deploy with Docker (README).

## 2. Reading order

1. `edge/sources/base.py` → the shapes. 2. `edge/sources/espn.py` → one adapter. 3. `edge/sources/__init__.py` → failover. 4. `edge/db.py` → tables. 5. `edge/ingest.py` → glue. 6. `edge/odds_math.py`, `edge/elo.py`. 7. `edge/features.py` (+ `tests/test_features.py`). 8. `edge/train.py`. 9. `edge/market.py`, `edge/backtest.py`. 10. `edge/rules.py`, `edge/predict.py`, `edge/explain.py`. 11. `edge/grade.py`, `edge/tracking.py`. 12. `edge/jobs.py`. 13. `app/`.

## 3. The one-minute pitch

"It's a sports betting analytics app. It scrapes schedules, odds and injury reports from three sources with automatic failover into its own database. A calibrated machine learning model (I compared an Elo baseline, logistic regression and LightGBM using season-based validation) estimates each team's win probability. The app compares that with the sportsbook's vig-free probability, runs a rules engine for things like preseason games and missing starters, and outputs Bet, Lean or Pass with SHAP-based explanations. It grades every pick against the closing line and can switch a league off if it isn't working. It runs unattended in Docker with weekly retraining."

## 4. Questions you should be able to answer

**Why not just predict who wins?** Favorites pay less. A pick only makes money if your probability is higher than the price implies. Edge = model % − market's fair %.

**How do you remove the vig?** Convert both prices to implied probabilities (they sum to more than 100%) and divide each by the total. `odds_math.fair_probs`.

**Why split by season instead of randomly?** A random split lets the model train on games played after the ones it's tested on, so it effectively sees the future and looks better than it is.

**What's leakage, and how do you prevent it?** Using information that wasn't available before the game. Every rolling stat uses `shift(1)`, Elo is recorded before each update, injury reports are only stored if captured before tip-off, and `tests/test_features.py` rebuilds features with the future deleted and checks they match.

**Why calibrate?** The edge is a difference of probabilities, so "60%" must really mean 60%. Logistic regression is usually close; boosted trees often aren't. Platt (sigmoid) scaling is fit on the validation season. The calibration table in the training report shows the result.

**Why log loss instead of accuracy?** Accuracy ignores confidence. Log loss punishes confident wrong answers, which is exactly what costs money.

**Why would LightGBM lose to logistic regression?** With ~1,200 games a season and a handful of features, there isn't enough data for trees to beat a simple linear model. That's a fine result to report: you tested it and kept what worked.

**What's the real benchmark?** The closing line. If your log loss is clearly worse than the market's, most "edges" are your model's mistakes. The training report prints both, and if no threshold is profitable on the validation season the app caps every pick at Lean.

**What's CLV and why track it?** Closing line value: did the price move toward your pick after you locked it? It shows skill much sooner than ROI, which needs hundreds of bets to separate from luck.

**How do you pick the Bet threshold?** It's tried on the validation season (2–15 points, with a minimum number of bets) and only then applied to the test season. Choosing it on the test season would make the test meaningless.

**What happens when a site changes its format?** The adapter raises, failover tries the backup, `source_runs` logs the failure, the Settings page turns red after 24 hours, and the fixture test pinpoints the field. Only one file needs fixing.

**Why is "key player out" a hard stop for the NFL but a soft flag for the NBA?** The NBA model has a "missing rotation minutes" feature, so it already prices injuries. The NFL model has no injury feature (ESPN doesn't give snap counts), so it can't see a missing QB, but the market can. Betting on that team would be betting on the model's blind spot.

**Known weakness?** For past games, availability comes from who actually played; for today's games, from the injury report. Late scratches make those differ slightly. It's documented, and fixing it means archiving pregame injury reports over a full season.

**How does tennis differ?** No home team, so the two players are stored alphabetically (storing the winner first would leak the result: a model would learn "player A wins"). Features are overall Elo, surface-specific Elo, ranking ratio, recent win rate, matches in the last 7 days and rest. Names differ between sources ("Alex de Minaur" vs "De Minaur A."), so `players.py` matches them on surname + first initial. The same match can come from ESPN and tennis-data.co.uk, so writes look for the same pair of players within 2 days and reuse that row, but never merge an upcoming match into a finished one.

**Why can you type odds by hand?** Free live tennis odds are scarce. Entering the price you see at your sportsbook gives the same edge calculation without depending on any odds feed.

## 5. Things to improve next (good "what would you do next" answers)

- Archive real pregame injury reports for a season, then retrain the availability feature on them.
- Add player ratings (e.g. from nba_api) so an absence is weighted by quality, not only minutes.
- Use nflverse data for NFL history and add a QB-quality feature.
- Line shopping across more books (needs a paid odds source).
- Post daily Bets to your Discord with a webhook from `jobs.nightly`.
