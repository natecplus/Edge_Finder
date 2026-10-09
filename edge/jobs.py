"""The scheduler: what runs when.

  nightly   05:00 ET  results for yesterday, grading, today's schedule + first picks
  game_day  every 30 min during the league's game window: odds, injuries, picks, locks
  retrain   Mondays 04:00 ET: rebuild features, retrain, promote if not worse

Run forever:          python -m edge.jobs
Run one job now:      python -m edge.jobs --once nightly --league nba
"""
import argparse
import logging
from datetime import timedelta

from apscheduler.schedulers.blocking import BlockingScheduler

from edge import db, grade, ingest, predict, train
from edge.config import all_leagues, league_config
from edge.timeutil import ET, et_today, utcnow

log = logging.getLogger("edge.jobs")


def _enabled_leagues() -> list[str]:
    out = []
    for lg in all_leagues():
        cfg = league_config(lg)
        overrides = db.get_setting(f"{lg}.overrides", {}) or {}
        if overrides.get("enabled", True) and et_today().month in cfg["active_months"]:
            out.append(lg)
    return out


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - one league failing must not stop the others
        log.exception("%s%s failed", fn.__name__, args)


def nightly(league: str | None = None) -> None:
    for lg in [league] if league else _enabled_leagues():
        today = et_today()
        yesterday = today - timedelta(days=1)
        log.info("[%s] nightly: results for %s", lg, yesterday)
        _safe(ingest.refresh_day, lg, yesterday)     # final scores, closing odds, minutes
        n = _safe(grade.grade_league, lg)
        log.info("[%s] graded %s picks", lg, n)
        _safe(ingest.refresh_day, lg, today)          # today's schedule, opening odds, injuries
        _safe(predict.predict_league, lg)


def game_day(league: str | None = None) -> None:
    now_et = utcnow().astimezone(ET)
    for lg in [league] if league else _enabled_leagues():
        start, end = league_config(lg).get("game_day_window", ["10:00", "23:59"])
        if not (start <= now_et.strftime("%H:%M") <= end) and league is None:
            continue
        log.info("[%s] game-day refresh", lg)
        _safe(ingest.refresh_day, lg, now_et.date())
        _safe(predict.predict_league, lg)


def retrain(league: str | None = None) -> None:
    for lg in [league] if league else all_leagues():
        log.info("[%s] weekly retrain", lg)
        if league_config(lg).get("sport") == "tennis":
            # pull the latest finished matches + closing odds before retraining
            from edge.tennis_ingest import load_history
            _safe(load_history, lg, [et_today().year], False)
        result = _safe(train.train_league, lg)
        if result:
            log.info("[%s] model %s promoted=%s", lg, result["version"], result["promoted"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", choices=["nightly", "gameday", "retrain"])
    parser.add_argument("--league")
    args = parser.parse_args()
    db.init()

    if args.once:
        {"nightly": nightly, "gameday": game_day, "retrain": retrain}[args.once](args.league)
        return

    sched = BlockingScheduler(timezone="America/New_York")
    sched.add_job(nightly, "cron", hour=5, minute=0, misfire_grace_time=3600)
    sched.add_job(game_day, "cron", minute="*/30", misfire_grace_time=600, max_instances=1)
    sched.add_job(retrain, "cron", day_of_week="mon", hour=4, minute=0, misfire_grace_time=3600)
    log.info("Scheduler started. Leagues: %s", ", ".join(all_leagues()))
    sched.start()


if __name__ == "__main__":
    main()
