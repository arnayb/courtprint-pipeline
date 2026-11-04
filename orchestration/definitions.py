"""Dagster definitions: the whole pipeline as a graph of assets.

    bronze/* (Python)  ->  silver + gold (dbt)  ->  ml/player_archetypes (Python)

Run the UI:      dagster dev -m orchestration.definitions
Run everything:  dagster asset materialize -m orchestration.definitions --select '*'
"""

import os
from pathlib import Path

from dagster import (AssetCheckResult, AssetExecutionContext, AssetKey, AssetSpec,
                     Config, DefaultScheduleStatus, Definitions, MaterializeResult,
                     MetadataValue, RunConfig, ScheduleDefinition, asset, asset_check,
                     define_asset_job, multi_asset)
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

import config as settings

# dbt and the extractors read these, so pin them before anything runs
os.environ.setdefault("LAKE_ROOT", settings.LAKE_ROOT)
os.environ.setdefault("WAREHOUSE_PATH", settings.WAREHOUSE_PATH)

DBT_DIR = Path(settings.ROOT) / "warehouse"
dbt_project = DbtProject(project_dir=DBT_DIR, profiles_dir=DBT_DIR)
dbt_project.prepare_if_dev()  # builds the dbt manifest when running `dagster dev`


# ---------------------------------------------------------------- bronze ----
class IngestConfig(Config):
    seasons: list[str] = settings.SEASONS
    source: str = settings.SOURCE  # "nba_api" or "demo"


@multi_asset(
    specs=[AssetSpec(AssetKey(["bronze", t]), group_name="bronze", kinds={"python", "parquet"},
                     description=f"Raw nba_api '{t}' table landed in the lake")
           for t in settings.BRONZE_TABLES],
)
def bronze_tables(context: AssetExecutionContext, config: IngestConfig):
    """Land one immutable Parquet file per table per season."""
    log = context.log.info
    totals: dict[str, int] = {t: 0 for t in settings.BRONZE_TABLES}
    if config.source == "demo":
        from ingestion.extract_demo import extract_demo
        for key, n in extract_demo(config.seasons, log=log).items():
            totals[key.split("/")[1]] += n
    else:
        from ingestion.extract_nba import extract_season
        for season in config.seasons:
            for table, n in extract_season(season, log=log).items():
                totals[table] += n
    for table, n in totals.items():
        yield MaterializeResult(
            asset_key=AssetKey(["bronze", table]),
            metadata={"rows_landed": n, "seasons": ", ".join(config.seasons), "source": config.source},
        )


# ------------------------------------------------------- silver + gold -----
@dbt_assets(manifest=dbt_project.manifest_path)
def warehouse_models(context: AssetExecutionContext, dbt: DbtCliResource):
    """dbt build: models + data tests (tests show up as Dagster asset checks)."""
    yield from dbt.cli(["build"], context=context).stream()


# ------------------------------------------------------------------- ml ----
@asset(
    key=AssetKey(["ml", "player_archetypes"]),
    deps=[AssetKey(["gold", "fct_player_season_features"])],
    group_name="ml",
    kinds={"scikitlearn", "duckdb"},
)
def player_archetypes(context: AssetExecutionContext) -> MaterializeResult:
    """PCA embeddings, archetype clusters, percentiles, and the serving export."""
    from ml.archetypes import run
    res = run(settings.WAREHOUSE_PATH, log=context.log.info)
    m = res["metrics"]
    counts = res["archetypes"].set_index("archetype")["n_player_seasons"]
    return MaterializeResult(metadata={
        **{k: MetadataValue.float(v) if isinstance(v, float) else v for k, v in m.items()},
        "archetypes": MetadataValue.md("\n".join(f"- **{k}**: {v}" for k, v in counts.items())),
    })


@asset_check(asset=player_archetypes, description="Compressed space must keep neighborhoods intact")
def embedding_quality(context) -> AssetCheckResult:
    import json
    m = json.loads((settings.ARTIFACTS / "metrics.json").read_text())
    ok = m["trustworthiness_k10"] >= 0.95 and m["explained_variance"] >= 0.80
    return AssetCheckResult(passed=ok, metadata={
        "trustworthiness_k10": m["trustworthiness_k10"],
        "explained_variance": m["explained_variance"],
    })


# ---------------------------------------------------------- schedules -----
full_refresh_job = define_asset_job("full_refresh", selection="*")

# During the season: re-pull only the current season every night, rebuild
# everything downstream. Old seasons are untouched (incremental load).
nightly_in_season = ScheduleDefinition(
    name="nightly_in_season",
    job=define_asset_job("nightly_refresh", selection="*"),
    cron_schedule="17 4 * 10-12,1-6 *",  # 4:17am, Oct through June
    execution_timezone="America/Los_Angeles",
    run_config=RunConfig(ops={"bronze_tables": IngestConfig(seasons=[settings.CURRENT_SEASON])}),
    default_status=DefaultScheduleStatus.STOPPED,  # turn on in the UI when ready
)

defs = Definitions(
    assets=[bronze_tables, warehouse_models, player_archetypes],
    asset_checks=[embedding_quality],
    jobs=[full_refresh_job],
    schedules=[nightly_in_season],
    resources={"dbt": DbtCliResource(project_dir=dbt_project, profiles_dir=str(DBT_DIR))},
)
