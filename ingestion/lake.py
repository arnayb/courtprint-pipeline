"""Bronze-layer writer.

Bronze is append-only and immutable: every pull writes a NEW file, nothing is
overwritten. Re-pulls are resolved downstream in the dbt staging models,
which keep only the latest pull per player-season. That makes ingestion safe
to retry and gives you a full history of what the API returned and when.

Layout (Hive-style partitions, readable by DuckDB, BigQuery, Spark, etc.):
    {LAKE_ROOT}/bronze/{table}/season={season}/ingested_at={ts}/data.parquet

LAKE_ROOT can be a local folder or a gs:// / s3:// URL (install gcsfs / s3fs).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import LAKE_ROOT


def _is_remote(root: str) -> bool:
    return "://" in root


def write_bronze(df: pd.DataFrame, table: str, season: str, source: str,
                 ingested_at: str | None = None) -> str:
    ts = ingested_at or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"{LAKE_ROOT}/bronze/{table}/season={season}/ingested_at={ts}/data.parquet"
    if not _is_remote(LAKE_ROOT):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["_source"] = source
    out.to_parquet(path, index=False)
    return path
