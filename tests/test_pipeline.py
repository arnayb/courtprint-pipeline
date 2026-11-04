"""End-to-end: demo bronze -> dbt build (with data tests) -> ML, in a temp dir."""
import os
import subprocess
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd, env, cwd=ROOT):
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return res.stdout


def test_end_to_end(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT),
           "LAKE_ROOT": str(tmp_path / "lake"),
           "WAREHOUSE_PATH": str(tmp_path / "wh.duckdb")}
    py = sys.executable

    _run([py, "-m", "ingestion.extract_demo"], env)
    # pull the latest season a second time: silver must dedupe it
    _run([py, "-m", "ingestion.extract_demo", "--seasons", "2025-26"], env)
    out = _run(["dbt", "build", "--profiles-dir", ".", "--target-path",
                str(tmp_path / "target")], env, cwd=ROOT / "warehouse")
    assert "ERROR=0" in out

    con = duckdb.connect(env["WAREHOUSE_PATH"], read_only=True)
    dupes = con.execute("select count(*) - count(distinct (player_id, season)) "
                    "from silver.stg_base").fetchone()[0]
    assert dupes == 0
    con.close()

    # ML writes into the warehouse; point serving export at tmp via cwd-independent path
    _run([py, "-c", "from config import WAREHOUSE_PATH; from ml.archetypes import fit; "
                    "import duckdb; df = duckdb.connect(WAREHOUSE_PATH, read_only=True)"
                    ".sql('select * from gold.fct_player_season_features').df(); "
                    "r = fit(df); m = r['metrics']; "
                    "assert m['trustworthiness_k10'] > 0.95, m; "
                    "assert r['player_seasons']['archetype'].nunique() >= 6; print(m)"], env)
