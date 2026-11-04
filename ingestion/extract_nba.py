"""Extract season totals from stats.nba.com (via nba_api) into bronze.

stats.nba.com blocks most cloud/datacenter IPs, so this extractor is meant to
run on your own machine (or a self-hosted runner) and push to LAKE_ROOT,
which can be local or cloud storage.

    python -m ingestion.extract_nba --seasons 2024-25 2025-26
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashptstats,
    leaguehustlestatsplayer,
)

from config import API_SLEEP, SEASON_TYPE, SEASONS
from ingestion.lake import write_bronze

_TRACKING = {
    "drives": "Drives",
    "catch_shoot": "CatchShoot",
    "pullup": "PullUpShot",
    "post_touch": "PostTouch",
    "possessions": "Possessions",
    "passing": "Passing",
}


def _jobs() -> dict[str, tuple]:
    jobs = {
        "base": (leaguedashplayerstats.LeagueDashPlayerStats,
                 dict(measure_type_detailed_defense="Base", per_mode_detailed="Totals")),
        "advanced": (leaguedashplayerstats.LeagueDashPlayerStats,
                     dict(measure_type_detailed_defense="Advanced", per_mode_detailed="Totals")),
        "hustle": (leaguehustlestatsplayer.LeagueHustleStatsPlayer,
                   dict(per_mode_time="Totals")),
    }
    for table, measure in _TRACKING.items():
        jobs[table] = (leaguedashptstats.LeagueDashPtStats,
                       dict(pt_measure_type=measure, per_mode_simple="Totals",
                            player_or_team="Player"))
    return jobs


def _fetch(endpoint_cls, retries: int = 4, **params) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            df = endpoint_cls(**params, timeout=60).get_data_frames()[0]
            time.sleep(API_SLEEP)
            return df
        except Exception as exc:  # timeouts, throttling, connection resets
            wait = API_SLEEP * 4 * attempt
            print(f"    attempt {attempt} failed ({type(exc).__name__}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"{endpoint_cls.__name__} failed after {retries} attempts: {params}")


def extract_season(season: str, log=print) -> dict[str, int]:
    """Pull every bronze table for one season. Returns rows written per table."""
    counts = {}
    for table, (cls, params) in _jobs().items():
        df = _fetch(cls, season=season, season_type_all_star=SEASON_TYPE, **params)
        path = write_bronze(df, table, season, source="nba_api")
        counts[table] = len(df)
        log(f"{season} {table}: {len(df)} rows -> {path}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=SEASONS)
    args = parser.parse_args()
    for season in args.seasons:
        extract_season(season)


if __name__ == "__main__":
    main()
