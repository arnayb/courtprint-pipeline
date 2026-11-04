"""Player embeddings + archetypes, fit on the gold features table.

Reads  gold.fct_player_season_features
Writes ml.player_seasons     PCA scores, 2D map coords, archetype, percentiles
       ml.archetypes         one row per archetype with its profile
       ml.pca_loadings       feature x component loadings
       ml.explained_variance
       ml.model_runs         append-only history of every fit + its metrics
       ml.player_profiles    view joining gold features + ML outputs (serving)
Also exports the serving tables to data/serving/*.parquet for the dashboard.

All seasons are fit together so every player-season shares one space, which
is what makes the "player evolution" view meaningful.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import (ARCHETYPE_OVERRIDES, ARCHETYPE_TEMPLATES, ARTIFACTS, FEATURES,
                    LAKE_ROOT, N_CLUSTERS, N_COMPONENTS, RADAR_CATEGORIES, RANDOM_STATE,
                    ROOT)

SERVING_DIR = ROOT / "data" / "serving"


def _neighbor_preservation(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    """Share of each point's k nearest neighbors kept after compression."""
    ia = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(a, return_distance=False)[:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(b).kneighbors(b, return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib)]))


def _name_clusters(cat_profile: pd.DataFrame) -> dict[int, str]:
    """Match clusters to archetype templates one-to-one (Hungarian algorithm)."""
    names = list(ARCHETYPE_TEMPLATES)
    score = np.array([[sum(w * cat_profile.loc[c].get(cat, 0.0)
                           for cat, w in ARCHETYPE_TEMPLATES[n].items())
                       for n in names] for c in cat_profile.index])
    rows, cols = linear_sum_assignment(-score)
    mapping = {int(cat_profile.index[r]): names[c] for r, c in zip(rows, cols)}
    mapping.update(ARCHETYPE_OVERRIDES)
    return mapping


def fit(df: pd.DataFrame) -> dict:
    feats = [f for f in FEATURES if f in df.columns and df[f].notna().mean() >= 0.9]
    if len(feats) < 5:
        raise ValueError(f"Only {len(feats)} usable features; check the gold table.")

    X = df[feats].copy()
    X = X.fillna(df.groupby("season")[feats].transform("median")).fillna(X.median())

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=min(N_COMPONENTS, len(feats)), random_state=RANDOM_STATE).fit(Xs)
    Z = pca.transform(Xs)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=RANDOM_STATE).fit(Z)
    labels = km.labels_

    # Cluster profiles in standardized feature space, rolled up to radar categories
    zdf = pd.DataFrame(Xs, columns=feats)
    zdf["cluster"] = labels
    feat_profile = zdf.groupby("cluster").mean()
    cat_profile = pd.DataFrame({
        cat: feat_profile[[f for f in feats if FEATURES[f][1] == cat]].mean(axis=1)
        for cat in RADAR_CATEGORIES
        if any(FEATURES[f][1] == cat for f in feats)
    })
    names = _name_clusters(cat_profile)

    perplexity = int(min(30, max(5, (len(df) - 1) // 4)))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  random_state=RANDOM_STATE).fit_transform(Z)

    out = df[["player_id", "season"]].copy()
    for i in range(Z.shape[1]):
        out[f"pc{i + 1}"] = Z[:, i]
    out["map_x"], out["map_y"] = coords[:, 0], coords[:, 1]
    out["cluster"] = labels
    out["archetype"] = [names[int(c)] for c in labels]

    # Percentiles are within-season so eras don't bleed into each other
    for f in feats:
        out[f"pct_{f}"] = df.groupby("season")[f].rank(pct=True).values * 100
    for cat in RADAR_CATEGORIES:
        cols = [f"pct_{f}" for f in feats if FEATURES[f][1] == cat]
        if cols:
            out[f"cat_{cat.lower()}"] = out[cols].mean(axis=1)

    # Archetype summary table
    latest = df["season"].max()
    arche_rows = []
    for c in sorted(feat_profile.index):
        top = feat_profile.loc[c].sort_values(ascending=False).head(3)
        members = df[(labels == c) & (df["season"] == latest)].nlargest(3, "minutes")
        arche_rows.append({
            "cluster": int(c),
            "archetype": names[int(c)],
            "n_player_seasons": int((labels == c).sum()),
            "top_traits": ", ".join(FEATURES[f][0] for f in top.index),
            "examples": ", ".join(members["player_name"]),
            **{f"z_{cat.lower()}": float(cat_profile.loc[c, cat]) for cat in cat_profile.columns},
        })

    loadings = pd.DataFrame(pca.components_.T, index=feats,
                            columns=[f"pc{i + 1}" for i in range(Z.shape[1])])
    loadings = (loadings.reset_index(names="feature")
                .melt(id_vars="feature", var_name="component", value_name="loading"))

    evr = pca.explained_variance_ratio_
    metrics = {
        "n_player_seasons": int(len(df)),
        "n_features": len(feats),
        "n_components": int(Z.shape[1]),
        "explained_variance": float(evr.sum()),
        "trustworthiness_k10": float(trustworthiness(Xs, Z, n_neighbors=10)),
        "neighbor_preservation_k10": _neighbor_preservation(Xs, Z, k=10),
        "silhouette": float(silhouette_score(Z, labels)),
    }
    return dict(
        features=feats, scaler=scaler, pca=pca, kmeans=km, player_seasons=out,
        archetypes=pd.DataFrame(arche_rows), loadings=loadings,
        explained_variance=pd.DataFrame({"component": [f"pc{i + 1}" for i in range(len(evr))],
                                         "ratio": evr, "cumulative": np.cumsum(evr)}),
        metrics=metrics,
    )


def run(warehouse_path: str, log=print) -> dict:
    con = duckdb.connect(warehouse_path)
    try:
        df = con.sql("select * from gold.fct_player_season_features").df()
        res = fit(df)

        con.execute("create schema if not exists ml")
        for name in ["player_seasons", "archetypes", "pca_loadings", "explained_variance"]:
            frame = res["loadings"] if name == "pca_loadings" else res[name]
            con.register("_frame", frame)
            con.execute(f"create or replace table ml.{name} as select * from _frame")
            con.unregister("_frame")

        run_row = pd.DataFrame([{
            "run_at": datetime.now(timezone.utc),
            "seasons": ", ".join(sorted(df["season"].unique())),
            "features": ", ".join(res["features"]),
            **res["metrics"],
        }])
        con.register("_run", run_row)
        con.execute("create table if not exists ml.model_runs as select * from _run where false")
        con.execute("insert into ml.model_runs by name select * from _run")
        con.unregister("_run")

        con.execute("""
            create or replace view ml.player_profiles as
            select f.*, m.* exclude (player_id, season)
            from gold.fct_player_season_features f
            join ml.player_seasons m using (player_id, season)
        """)

        # Serving layer: plain Parquet the dashboard can read anywhere
        SERVING_DIR.mkdir(parents=True, exist_ok=True)
        lake = con.sql(f"""
            select regexp_extract(filename, '/bronze/([^/]+)/', 1) as bronze_table,
                   season, max(ingested_at) as latest_pull, count(distinct ingested_at) as pulls,
                   count(*) as rows_total, any_value(_source) as source
            from read_parquet('{LAKE_ROOT}/bronze/*/*/*/*.parquet',
                              hive_partitioning = true, union_by_name = true, filename = true)
            group by all order by 1, 2
        """).df()
        lake.to_parquet(SERVING_DIR / "lake_inventory.parquet", index=False)
        for t in ["player_profiles", "archetypes", "pca_loadings", "explained_variance", "model_runs"]:
            con.sql(f"select * from ml.{t}").df().to_parquet(SERVING_DIR / f"{t}.parquet", index=False)
    finally:
        con.close()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name in ["scaler", "pca", "kmeans"]:
        with open(ARTIFACTS / f"{name}.pkl", "wb") as fh:
            pickle.dump(res[name], fh)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(
        {**res["metrics"], "features": res["features"]}, indent=2))

    log(f"model fit: {json.dumps(res['metrics'])}")
    return res


if __name__ == "__main__":
    from config import WAREHOUSE_PATH
    run(WAREHOUSE_PATH)
