"""Synthetic data that mimics the nba_api bronze tables, column for column.

Lets the whole pipeline (bronze -> dbt -> ML -> dashboard) run offline, in CI,
or before you've pulled real data. Players are fictional; teams are real
abbreviations so the dashboard looks familiar.

    python -m ingestion.extract_demo
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import SEASONS
from ingestion.lake import write_bronze

TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
         "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
         "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]

FIRST = ["Jalen", "Marcus", "Tyrese", "Darius", "Andre", "Malik", "Isaiah", "Caleb",
         "Jordan", "Devin", "Trey", "Kobe", "Elijah", "Xavier", "Miles", "Cam",
         "Bryce", "Damon", "Nico", "Theo", "Luka", "Omar", "Rashad", "Quentin",
         "Victor", "Amir", "Desmond", "Jaylen", "Kendrick", "Felix", "Grant", "Hassan"]
LAST = ["Holloway", "Pierce", "Okafor", "Delgado", "Whitfield", "Brennan", "Mbeki",
        "Sorensen", "Castillo", "Ashford", "Kowalski", "Nwosu", "Lindqvist", "Reyes",
        "Thornton", "Abara", "Fairbanks", "Hale", "Ivanov", "Jefferson", "Kimura",
        "Lacroix", "Mensah", "Novak", "Osei", "Pruitt", "Quarles", "Rasmussen",
        "Sutton", "Tavares", "Underwood", "Vance", "Whitaker", "Yilmaz", "Zeller"]

# Per-100-possession profiles:
# pts ts usg ftr 3pa cs drv pu spt ast astp pot tov oreb dreb post scr stl blk defl con
_COLS = ["pts", "ts", "usg", "ftr", "fg3a", "cs", "drv", "pu", "spt", "ast", "astp",
         "pot", "tov", "oreb", "dreb", "post", "scr", "stl", "blk", "defl", "con"]
_TEMPLATES = {
    "creator":   ([38, .59, .31, .35, 11, 2.0, 22, 14, 4.8, 9.0, .33, 17, 4.5, .03, .13, 1.0, .3, 1.6, .6, 3.0, 6], 0.08),
    "playmaker": ([24, .57, .22, .25, 9, 2.5, 20, 8, 5.5, 13, .38, 24, 4.0, .02, .11, .5, .2, 2.0, .4, 4.0, 6], 0.10),
    "shooter":   ([22, .61, .19, .15, 14, 9.0, 6, 3, 2.2, 4.0, .13, 7, 1.8, .03, .12, .3, .3, 1.4, .5, 3.3, 8], 0.18),
    "3andD":     ([16, .58, .15, .15, 10, 7.5, 4, 1.5, 1.9, 3.5, .10, 6, 1.6, .04, .14, .3, .4, 2.3, 1.0, 5.5, 13], 0.20),
    "slasher":   ([30, .57, .26, .40, 5, 2.0, 20, 7, 3.8, 5.5, .20, 10, 3.5, .06, .15, 2.0, .4, 1.6, .8, 3.5, 8], 0.12),
    "rim":       ([18, .66, .15, .40, .3, .2, 1, .2, 1.4, 2.5, .08, 4, 2.2, .13, .28, 3.0, 4.5, 1.3, 3.4, 3.0, 22], 0.13),
    "stretch":   ([22, .59, .20, .22, 7, 5.5, 3, 1.5, 1.9, 4.0, .13, 6.5, 2.2, .07, .22, 4.0, 2.0, 1.2, 1.8, 3.0, 16], 0.11),
    "post":      ([30, .60, .27, .38, 2, .8, 4, 2, 3.0, 8.0, .27, 14, 4.2, .10, .26, 12, 3.0, 1.5, 1.5, 3.2, 15], 0.08),
}


def _make_players(rng: np.random.Generator, n: int) -> pd.DataFrame:
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(FIRST)} {rng.choice(LAST)}")
    keys = list(_TEMPLATES)
    probs = np.array([_TEMPLATES[k][1] for k in keys]); probs /= probs.sum()
    n_seasons = len(SEASONS)
    rows = []
    for i, name in enumerate(sorted(names)):
        # most players span the whole window; some debut late or leave early
        start = 0 if rng.random() < 0.7 else int(rng.integers(1, n_seasons))
        end = n_seasons - 1 if rng.random() < 0.8 else int(rng.integers(start, n_seasons))
        length = end - start + 1
        arche = rng.choice(keys, p=probs)
        # ~15% of players change role over their career (for the evolution view)
        later = rng.choice(keys, p=probs) if rng.random() < 0.15 else arche
        rows.append(dict(
            player_id=1_630_000 + i, name=name, archetype=arche, later_archetype=later,
            start=start, end=start + length - 1, age0=int(rng.integers(20, 33)),
            team=rng.choice(TEAMS), skill=rng.lognormal(0, 0.15, size=len(_COLS)),
            minutes_scale=rng.lognormal(0, 0.55),
        ))
    return pd.DataFrame(rows)


def _season_profile(p, s_idx: int, rng: np.random.Generator) -> dict:
    span = max(p.end - p.start, 1)
    t = (s_idx - p.start) / span if p.archetype != p.later_archetype else 0.0
    base = (1 - t) * np.array(_TEMPLATES[p.archetype][0]) + t * np.array(_TEMPLATES[p.later_archetype][0])
    noise = rng.lognormal(0, 0.06, size=len(_COLS))
    vals = base * p.skill * noise
    prof = dict(zip(_COLS, vals))
    prof["ts"] = float(np.clip(prof["ts"], 0.45, 0.72))
    prof["usg"] = float(np.clip(prof["usg"], 0.09, 0.38))
    prof["astp"] = float(np.clip(prof["astp"], 0.03, 0.50))
    prof["oreb"] = float(np.clip(prof["oreb"], 0.005, 0.18))
    prof["dreb"] = float(np.clip(prof["dreb"], 0.06, 0.35))
    return prof


def build_season(players: pd.DataFrame, s_idx: int, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    active = players[(players.start <= s_idx) & (players.end >= s_idx)]
    recs = []
    for p in active.itertuples():
        pr = _season_profile(p, s_idx, rng)
        gp = int(np.clip(rng.normal(62, 14) * min(p.minutes_scale, 1.3), 5, 82))
        mpg = float(np.clip(rng.normal(22, 6) * p.minutes_scale, 4, 38))
        mins = round(gp * mpg, 1)
        poss = mins / 48 * rng.normal(99.5, 2.5)
        k = poss / 100
        fga = pr["pts"] / (2 * pr["ts"] * (1 + 0.44 * pr["ftr"]))
        fg3a = min(pr["fg3a"], 0.9 * fga)
        cs = min(pr["cs"], fga); pu = min(pr["pu"], fga)
        fta = pr["ftr"] * fga
        team = p.team if rng.random() > 0.12 else rng.choice(TEAMS)  # occasional trades
        recs.append(dict(
            PLAYER_ID=p.player_id, PLAYER_NAME=p.name, TEAM_ID=1_610_612_700 + TEAMS.index(team),
            TEAM_ABBREVIATION=team, AGE=p.age0 + s_idx, GP=gp, MIN=mins, POSS=round(poss),
            FGA=round(fga * k), FG3A=round(fg3a * k), FTA=round(fta * k), PTS=round(pr["pts"] * k),
            AST=round(pr["ast"] * k), TOV=round(pr["tov"] * k), STL=round(pr["stl"] * k),
            BLK=round(pr["blk"] * k), OREB=round(pr["oreb"] * 45 * k), DREB=round(pr["dreb"] * 45 * k),
            TS_PCT=round(pr["ts"], 3), USG_PCT=round(pr["usg"], 3), AST_PCT=round(pr["astp"], 3),
            OREB_PCT=round(pr["oreb"], 3), DREB_PCT=round(pr["dreb"], 3),
            DRIVES=round(pr["drv"] * k), CATCH_SHOOT_FGA=round(cs * k), PULL_UP_FGA=round(pu * k),
            POST_TOUCHES=round(pr["post"] * k), AVG_SEC_PER_TOUCH=round(pr["spt"], 2),
            TOUCHES=round(pr["pot"] * 3.5 * k + 40 * k), POTENTIAL_AST=round(pr["pot"] * k),
            PASSES_MADE=round((pr["pot"] * 2.5 + 25) * k),
            DEFLECTIONS=round(pr["defl"] * k), CONTESTED_SHOTS=round(pr["con"] * k),
            SCREEN_ASSISTS=round(pr["scr"] * k), LOOSE_BALLS_RECOVERED=round(rng.uniform(.3, 1.2) * k),
        ))
    df = pd.DataFrame(recs)
    id_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]
    df["FGM"] = (df.FGA * 0.47).round(); df["FG3M"] = (df.FG3A * 0.36).round()
    df["FTM"] = (df.FTA * 0.78).round(); df["REB"] = df.OREB + df.DREB
    return {
        "base": df[id_cols + ["AGE", "GP", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                              "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PTS"]],
        "advanced": df[id_cols + ["AGE", "GP", "MIN", "POSS", "TS_PCT", "USG_PCT", "AST_PCT",
                                  "OREB_PCT", "DREB_PCT"]],
        "hustle": df[id_cols + ["AGE", "MIN", "CONTESTED_SHOTS", "DEFLECTIONS", "SCREEN_ASSISTS",
                                "LOOSE_BALLS_RECOVERED"]].assign(G=df.GP),
        "drives": df[id_cols + ["GP", "MIN", "DRIVES"]],
        "catch_shoot": df[id_cols + ["GP", "MIN", "CATCH_SHOOT_FGA"]],
        "pullup": df[id_cols + ["GP", "MIN", "PULL_UP_FGA"]],
        "post_touch": df[id_cols + ["GP", "MIN", "POST_TOUCHES"]],
        "possessions": df[id_cols + ["GP", "MIN", "TOUCHES", "AVG_SEC_PER_TOUCH"]],
        "passing": df[id_cols + ["GP", "MIN", "PASSES_MADE", "POTENTIAL_AST"]],
    }


def extract_demo(seasons: list[str] | None = None, n_players: int = 560, seed: int = 7,
                 log=print) -> dict[str, int]:
    seasons = seasons or SEASONS
    rng = np.random.default_rng(seed)
    players = _make_players(rng, n_players)
    counts = {}
    for season in seasons:
        s_idx = SEASONS.index(season) if season in SEASONS else len(SEASONS) - 1
        for table, df in build_season(players, s_idx, rng).items():
            write_bronze(df, table, season, source="demo")
            counts[f"{season}/{table}"] = len(df)
        log(f"demo {season}: {counts[f'{season}/base']} players")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=SEASONS)
    args = parser.parse_args()
    extract_demo(args.seasons)


if __name__ == "__main__":
    main()
