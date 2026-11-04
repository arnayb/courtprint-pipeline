"""Central configuration for CourtPrint.

Storage locations come from environment variables so the same code runs
locally or in the cloud:

    LAKE_ROOT       where bronze files land.  Local default: ./data/lake
                    Cloud later:  gs://my-bucket/lake  or  s3://my-bucket/lake
    WAREHOUSE_PATH  DuckDB file.  Local default: ./data/warehouse.duckdb
                    Cloud later:  md:courtprint  (MotherDuck)
    COURTPRINT_SOURCE  "nba_api" (real data) or "demo" (synthetic, offline)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAKE_ROOT = os.environ.get("LAKE_ROOT", str(ROOT / "data" / "lake"))
WAREHOUSE_PATH = os.environ.get("WAREHOUSE_PATH", str(ROOT / "data" / "warehouse.duckdb"))
SOURCE = os.environ.get("COURTPRINT_SOURCE", "nba_api")
ARTIFACTS = ROOT / "artifacts"

# Bronze tables. Each is landed as Totals, one file per pull:
#   {LAKE_ROOT}/bronze/{table}/season={season}/ingested_at={ts}.parquet
BRONZE_TABLES = [
    "base", "advanced", "hustle",
    "drives", "catch_shoot", "pullup", "post_touch", "possessions", "passing",
]

# Seasons to pull and model. The model is fit on all seasons pooled together,
# so every player-season lives in ONE shared space (that's what makes the
# "player evolution" view possible).
SEASONS = ["2023-24", "2024-25", "2025-26"]
SEASON_TYPE = "Regular Season"

MIN_MINUTES = 500      # player-seasons below this are dropped
N_COMPONENTS = 10      # PCA components kept for similarity
N_CLUSTERS = 8         # number of archetypes
RANDOM_STATE = 42

# Seconds to wait between API calls. stats.nba.com rate-limits aggressively.
API_SLEEP = 1.5

# The in-season nightly schedule only re-pulls this season (incremental load).
CURRENT_SEASON = os.environ.get("COURTPRINT_CURRENT_SEASON", "2026-27")

# ---------------------------------------------------------------------------
# Features: column -> (display label, radar category)
# Category None = used in the model but not on the radar.
# Volume stats are per 100 possessions; rates/percentages are left as-is.
# ---------------------------------------------------------------------------
FEATURES = {
    # Scoring
    "pts_100":          ("Points /100", "Scoring"),
    "ts_pct":           ("True shooting %", "Scoring"),
    "usg_pct":          ("Usage %", "Scoring"),
    "fta_rate":         ("Free-throw rate", "Scoring"),
    # Shooting
    "fg3a_100":         ("3PA /100", "Shooting"),
    "fg3a_rate":        ("3PA rate", "Shooting"),
    "cs_fga_100":       ("Catch & shoot FGA /100", "Shooting"),
    # Creation
    "drives_100":       ("Drives /100", "Creation"),
    "pullup_fga_100":   ("Pull-up FGA /100", "Creation"),
    "sec_per_touch":    ("Seconds per touch", "Creation"),
    # Playmaking
    "ast_100":          ("Assists /100", "Playmaking"),
    "ast_pct":          ("Assist %", "Playmaking"),
    "pot_ast_100":      ("Potential assists /100", "Playmaking"),
    "tov_100":          ("Turnovers /100", None),
    # Paint / rebounding
    "oreb_pct":         ("Off. rebound %", "Paint"),
    "dreb_pct":         ("Def. rebound %", "Paint"),
    "post_touches_100": ("Post touches /100", "Paint"),
    "screen_ast_100":   ("Screen assists /100", "Paint"),
    # Defense
    "stl_100":          ("Steals /100", "Defense"),
    "blk_100":          ("Blocks /100", "Defense"),
    "deflections_100":  ("Deflections /100", "Defense"),
    "contests_100":     ("Contested shots /100", "Defense"),
}

RADAR_CATEGORIES = ["Scoring", "Shooting", "Creation", "Playmaking", "Paint", "Defense"]

# Archetype names are assigned automatically by matching each cluster's profile
# to these templates (category -> weight). Edit freely.
ARCHETYPE_TEMPLATES = {
    "Primary Creator":          {"Creation": 1.0, "Scoring": 1.0, "Playmaking": 0.5, "Paint": -0.5},
    "Lead Playmaker":           {"Playmaking": 1.2, "Creation": 0.5, "Scoring": -0.2},
    "Movement Shooter":         {"Shooting": 1.2, "Creation": -0.3, "Paint": -0.5},
    "3&D Wing":                 {"Shooting": 0.7, "Defense": 0.7, "Scoring": -0.5, "Playmaking": -0.3},
    "Slashing Scorer":          {"Creation": 0.8, "Scoring": 0.8, "Shooting": -0.5},
    "Rim Protector":            {"Paint": 1.0, "Defense": 1.0, "Shooting": -1.0, "Creation": -0.5},
    "Stretch Big":              {"Paint": 0.7, "Shooting": 0.7, "Creation": -0.7},
    "Post Hub":                 {"Paint": 1.0, "Scoring": 0.7, "Playmaking": 0.3, "Shooting": -0.7},
    "Point-of-Attack Defender": {"Defense": 1.2, "Playmaking": 0.3, "Scoring": -0.6, "Paint": -0.3},
    "Connector":                {"Playmaking": 0.5, "Shooting": 0.5, "Defense": 0.3, "Scoring": -0.5},
    "Low-Usage Role Player":    {c: -0.5 for c in RADAR_CATEGORIES},
}

# Optional manual overrides after you inspect clusters: {cluster_id: "Name"}
ARCHETYPE_OVERRIDES: dict[int, str] = {}
