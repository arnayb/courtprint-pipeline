"""CourtPrint dashboard. Reads only the serving layer (data/serving/*.parquet).

    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import ARCHETYPE_TEMPLATES, FEATURES, RADAR_CATEGORIES  # noqa: E402

SERVING = ROOT / "data" / "serving"

# ---- design tokens (validated categorical palette, dark surface) ---------
SURFACE = "#0f1116"
INK, INK_2, INK_3 = "#f0efec", "#c3c2b7", "#8a8980"
GRID = "#262830"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BLUE_STEPS = ["#104281", "#1c5cab", "#3987e5", "#86b6ef"]  # ordinal: oldest dim -> newest bright
NEUTRAL = "#6b6a63"

st.set_page_config(page_title="CourtPrint", layout="wide")
st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
  h1, h2, h3 {{ letter-spacing: -0.02em; }}
  [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
  .pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem;
          border:1px solid {GRID}; color:{INK_2}; margin-right:6px; }}
  .swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }}
  .muted {{ color:{INK_3}; font-size:.85rem; }}
</style>""", unsafe_allow_html=True)


# ---- data -----------------------------------------------------------------
@st.cache_data
def load():
    need = ["player_profiles", "archetypes", "pca_loadings", "explained_variance",
            "model_runs", "lake_inventory"]
    missing = [n for n in need if not (SERVING / f"{n}.parquet").exists()]
    if missing:
        return None
    return {n: pd.read_parquet(SERVING / f"{n}.parquet") for n in need}


data = load()
if data is None:
    st.title("CourtPrint")
    st.warning("No serving data yet. Run the pipeline first:\n\n"
               "`make demo` (synthetic data) or `make pipeline` (real nba_api data).")
    st.stop()

df = data["player_profiles"]
arche = data["archetypes"]
pc_cols = sorted([c for c in df.columns if c.startswith("pc") and c[2:].isdigit()],
                 key=lambda c: int(c[2:]))
cat_cols = {c: f"cat_{c.lower()}" for c in RADAR_CATEGORIES if f"cat_{c.lower()}" in df.columns}
feats = [f for f in FEATURES if f in df.columns]

# Color follows the archetype (stable template order), never its rank
order = [a for a in ARCHETYPE_TEMPLATES if a in set(df["archetype"])]
order += sorted(set(df["archetype"]) - set(order))
COLOR = {a: SERIES[i] if i < len(SERIES) else NEUTRAL for i, a in enumerate(order)}

is_demo = (data["lake_inventory"]["source"] == "demo").any()
seasons = sorted(df["season"].unique())


def base_layout(fig, height=420, **kw):
    kw.setdefault("legend", dict(font=dict(color=INK_2)))
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_2, size=12), margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="#1c1e25", bordercolor=GRID, font_color=INK), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def swatch(a):
    return f'<span class="swatch" style="background:{COLOR[a]}"></span>{a}'


def label(name, season):
    return f"{name} ({season})"


# ---- header ---------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.title("CourtPrint")
    st.markdown('<span class="muted">NBA player fingerprints: tracking data -> per-100 features -> '
                'PCA embeddings -> archetypes & statistical comps</span>', unsafe_allow_html=True)
with right:
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)
if is_demo:
    st.info("Showing **synthetic demo data** (fictional players). Run `make pipeline` on your "
            "machine to load real stats from stats.nba.com.")

S = df[df["season"] == season].reset_index(drop=True)

tabs = st.tabs(["Scouting report", "Archetype map", "Team builds", "Player evolution",
                "Pipeline & model"])


# ---- 1. scouting report ---------------------------------------------------
def similar_players(target_row, pool: pd.DataFrame, k=10):
    X = pool[pc_cols].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(pool))).fit(X)
    d, idx = nn.kneighbors(target_row[pc_cols].to_numpy().reshape(1, -1))
    # scale: typical distance to a 10th-nearest neighbour in this pool
    ref = np.median(nn.kneighbors(X, n_neighbors=min(11, len(pool)))[0][:, -1])
    out = pool.iloc[idx[0]].copy()
    out["match"] = 100 * np.exp(-(d[0] / (ref * 1.5)) ** 2)
    same = (out["player_id"] == target_row["player_id"]) & (out["season"] == target_row["season"])
    return out[~same].head(k)


def radar(traces):
    fig = go.Figure()
    cats = list(cat_cols)
    for name, vals, color, dash, fill in traces:
        r = list(vals) + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=cats + [cats[0]], name=name, mode="lines+markers",
            line=dict(color=color, width=2, dash=dash), marker=dict(size=8, color=color),
            fill="toself" if fill else None, opacity=0.95,
            hovertemplate="%{theta}: %{r:.0f}th pct<extra>" + name + "</extra>"))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], gridcolor=GRID,
                        tickfont=dict(color=INK_3, size=10), linecolor=GRID),
        angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=INK, size=12))))
    return base_layout(fig, height=430, legend=dict(orientation="h", y=-0.08, font=dict(color=INK_2)))


with tabs[0]:
    names = S.sort_values("minutes", ascending=False)["player_name"].tolist()
    c1, c2 = st.columns([2, 1])
    player = c1.selectbox("Player", names, key="scout_player")
    scope = c2.radio("Find comps in", ["This season", "All seasons"], horizontal=True)
    row = S[S["player_name"] == player].iloc[0]

    st.markdown(f"### {row.player_name} &nbsp; "
                f'<span class="pill">{row.team}</span>'
                f'<span class="pill">{swatch(row.archetype)}</span>'
                f'<span class="pill">Age {row.age:.0f}</span>', unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("Games", f"{row.games:.0f}")
    m[1].metric("Minutes", f"{row.minutes:,.0f}")
    m[2].metric("Points /100", f"{row.pts_100:.1f}")
    m[3].metric("True shooting", f"{row.ts_pct:.1%}")
    m[4].metric("Usage", f"{row.usg_pct:.1%}")
    m[5].metric("Assists /100", f"{row.ast_100:.1f}")

    pool = S if scope == "This season" else df
    sims = similar_players(row, pool.reset_index(drop=True))

    left, right = st.columns([1.1, 1])
    with right:
        st.markdown("#### Most similar players")
        st.caption("Nearest neighbours in the 10-dimensional PCA space. "
                   "Similar *statistical profile*, not similar quality.")
        table = sims[["player_name", "season", "team", "archetype", "match"]].rename(columns={
            "player_name": "Player", "season": "Season", "team": "Team",
            "archetype": "Archetype", "match": "Match"})
        st.dataframe(table, hide_index=True, width="stretch", height=388,
                     column_config={"Match": st.column_config.ProgressColumn(
                         "Match", format="%.0f", min_value=0, max_value=100)})
    with left:
        st.markdown("#### Profile vs archetype")
        comp_opts = ["None"] + [label(r.player_name, r.season) for r in sims.itertuples()]
        comp = st.selectbox("Overlay a comp", comp_opts, key="scout_comp")
        avg = S[S["archetype"] == row.archetype][list(cat_cols.values())].mean()
        traces = [(f"{row.archetype} avg", avg.values, INK_3, "dot", False),
                  (row.player_name, row[list(cat_cols.values())].values.astype(float), SERIES[0], "solid", False)]
        if comp != "None":
            cr = sims.iloc[comp_opts.index(comp) - 1]
            traces.append((label(cr.player_name, cr.season),
                           cr[list(cat_cols.values())].values.astype(float), SERIES[1], "solid", False))
        st.plotly_chart(radar(traces), width="stretch")

    st.markdown("#### Percentiles (within season)")
    pct = pd.DataFrame({
        "feature": [FEATURES[f][0] for f in feats],
        "category": [FEATURES[f][1] or "Other" for f in feats],
        "pct": [row[f"pct_{f}"] for f in feats],
        "value": [row[f] for f in feats],
    })
    pct["category"] = pd.Categorical(pct["category"], RADAR_CATEGORIES + ["Other"], ordered=True)
    pct = pct.sort_values(["category", "pct"], ascending=[False, True])
    fig = go.Figure(go.Bar(
        x=pct["pct"], y=pct["feature"], orientation="h",
        marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2), cornerradius=4),
        customdata=np.stack([pct["value"], pct["category"].astype(str)], axis=1),
        text=[f"{p:.0f}" for p in pct["pct"]], textposition="outside",
        textfont=dict(color=INK_2),
        hovertemplate="%{y}<br>%{customdata[1]}<br>value %{customdata[0]:.3g}"
                      "<br>%{x:.0f}th percentile<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=INK_3, width=1, dash="dot"))
    base_layout(fig, height=max(380, 24 * len(pct)), showlegend=False)
    fig.update_xaxes(range=[0, 108], title="Percentile vs all qualified players this season")
    st.plotly_chart(fig, width="stretch")


# ---- 2. archetype map -----------------------------------------------------
with tabs[1]:
    st.markdown("#### Every qualified player, placed by statistical similarity")
    st.caption("2D t-SNE projection of the PCA embedding: nearby dots play similar statistical "
               "roles. Axes have no units. Hover for details; click legend items to isolate.")
    hl = st.selectbox("Highlight a player", ["None"] + sorted(S["player_name"]), key="map_hl")
    fig = go.Figure()
    for a in order:
        sub = S[S["archetype"] == a]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["map_x"], y=sub["map_y"], mode="markers", name=f"{a} ({len(sub)})",
            marker=dict(size=9, color=COLOR[a], line=dict(color=SURFACE, width=1.5)),
            customdata=np.stack([sub["player_name"], sub["team"], sub["pts_100"],
                                 sub["usg_pct"] * 100], axis=1),
            hovertemplate="<b>%{customdata[0]}</b> | %{customdata[1]}<br>" + a +
                          "<br>%{customdata[2]:.1f} pts/100 | %{customdata[3]:.0f}% usage<extra></extra>"))
    if hl != "None":
        r = S[S["player_name"] == hl].iloc[0]
        fig.add_trace(go.Scatter(x=[r.map_x], y=[r.map_y], mode="markers", showlegend=False,
                                 hoverinfo="skip",
                                 marker=dict(size=22, color="rgba(0,0,0,0)",
                                             line=dict(color=INK, width=2.5))))
        fig.add_annotation(x=r.map_x, y=r.map_y, text=hl, showarrow=True, arrowcolor=INK_2,
                           font=dict(color=INK, size=13), ay=-36, bgcolor="#1c1e25")
    base_layout(fig, height=620)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### What each archetype means")
    cols = st.columns(4)
    for i, a in enumerate(order):
        info = arche[arche["archetype"] == a].iloc[0]
        n_season = int((S["archetype"] == a).sum())
        with cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"**{swatch(a)}**", unsafe_allow_html=True)
                st.markdown(f'<span class="muted">Stands out in:</span> {info.top_traits}<br>'
                            f'<span class="muted">Examples:</span> {info.examples}<br>'
                            f'<span class="muted">{n_season} players this season</span>',
                            unsafe_allow_html=True)


# ---- 3. team builds -------------------------------------------------------
with tabs[2]:
    st.markdown("#### How each roster's minutes split across archetypes")
    st.caption("Share of each team's qualified-player minutes. Traded players count for "
               "their latest team.")
    mix = S.pivot_table(index="team", columns="archetype", values="minutes", aggfunc="sum").fillna(0)
    mix = mix.div(mix.sum(axis=1), axis=0) * 100
    sort_by = st.selectbox("Sort teams by share of", ["Team name"] + [a for a in order if a in mix.columns])
    mix = mix.sort_index(ascending=False) if sort_by == "Team name" else mix.sort_values(sort_by)
    fig = go.Figure()
    for a in order:
        if a not in mix.columns:
            continue
        fig.add_trace(go.Bar(
            y=mix.index, x=mix[a], name=a, orientation="h",
            marker=dict(color=COLOR[a], line=dict(color=SURFACE, width=2)),
            hovertemplate="%{y} | " + a + ": %{x:.0f}% of minutes<extra></extra>"))
    base_layout(fig, height=760, barmode="stack",
                legend=dict(orientation="h", y=1.04, x=0, traceorder="normal", font=dict(color=INK_2)))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, width="stretch")

    team = st.selectbox("Roster detail", sorted(S["team"].unique()))
    roster = S[S["team"] == team].sort_values("minutes", ascending=False)
    st.dataframe(roster[["player_name", "archetype", "minutes", "pts_100", "ast_100", "ts_pct", "usg_pct"]]
                 .rename(columns={"player_name": "Player", "archetype": "Archetype", "minutes": "Min",
                                  "pts_100": "Pts/100", "ast_100": "Ast/100", "ts_pct": "TS%",
                                  "usg_pct": "Usage"}),
                 hide_index=True, width="stretch",
                 column_config={"Min": st.column_config.NumberColumn(format="%.0f"),
                                "Pts/100": st.column_config.NumberColumn(format="%.1f"),
                                "Ast/100": st.column_config.NumberColumn(format="%.1f"),
                                "TS%": st.column_config.NumberColumn(format="%.3f"),
                                "Usage": st.column_config.NumberColumn(format="%.3f")})


# ---- 4. player evolution --------------------------------------------------
with tabs[3]:
    st.markdown("#### How a player's role changed over time")
    st.caption("Every season is embedded in the same space, so you can trace a career path. "
               "Players who switched archetypes are listed first.")
    multi = df.groupby("player_id").filter(lambda g: g["season"].nunique() > 1)
    if multi.empty:
        st.info("Load at least two seasons to use this view.")
    else:
        changed = multi.groupby("player_id")["archetype"].nunique()
        ranked = (multi.drop_duplicates("player_id", keep="last").set_index("player_id")
                  .assign(changed=changed).sort_values(["changed", "minutes"], ascending=False))
        opts = [f"{r.player_name}{' (role changed)' if r.changed > 1 else ''}" for r in ranked.itertuples()]
        pick = st.selectbox("Player", opts, key="evo_player")
        pid = ranked.index[opts.index(pick)]
        path = multi[multi["player_id"] == pid].sort_values("season")

        left, right = st.columns([1.3, 1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["map_x"], y=df["map_y"], mode="markers", hoverinfo="skip",
                                     marker=dict(size=6, color=[COLOR[a] for a in df["archetype"]],
                                                 opacity=0.18), showlegend=False))
            fig.add_trace(go.Scatter(x=path["map_x"], y=path["map_y"], mode="lines",
                                     line=dict(color=INK, width=2), hoverinfo="skip", showlegend=False))
            for i, r in enumerate(path.itertuples()):
                fig.add_trace(go.Scatter(
                    x=[r.map_x], y=[r.map_y], mode="markers+text", text=[r.season],
                    textposition="top center", textfont=dict(color=INK, size=12), showlegend=False,
                    marker=dict(size=14, color=COLOR[r.archetype], line=dict(color=INK, width=2)),
                    hovertemplate=f"<b>{r.season}</b> | {r.team}<br>{r.archetype}<extra></extra>"))
            base_layout(fig, height=520)
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            st.plotly_chart(fig, width="stretch")
        with right:
            steps = BLUE_STEPS[-len(path):]
            traces = [(r.season, np.array([getattr(r, c) for c in cat_cols.values()], dtype=float),
                       steps[min(i, len(steps) - 1)], "solid", False)
                      for i, r in enumerate(path.itertuples())]
            st.plotly_chart(radar(traces), width="stretch")
        st.dataframe(path[["season", "team", "archetype", "minutes", "pts_100", "usg_pct",
                           "ast_100", "fg3a_rate"]]
                     .rename(columns={"season": "Season", "team": "Team", "archetype": "Archetype",
                                      "minutes": "Min", "pts_100": "Pts/100", "usg_pct": "Usage",
                                      "ast_100": "Ast/100", "fg3a_rate": "3PA rate"}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ["Pts/100", "Usage", "Ast/100", "3PA rate"]})


# ---- 5. pipeline & model --------------------------------------------------
with tabs[4]:
    st.markdown("#### Pipeline")
    st.graphviz_chart(f"""
    digraph G {{
      rankdir=LR; bgcolor="transparent";
      node [shape=box style="rounded,filled" fillcolor="#1c1e25" color="{GRID}" fontcolor="{INK}"
            fontname="Helvetica" fontsize=11 margin="0.18,0.08"];
      edge [color="{INK_3}" arrowsize=0.7];
      api [label="stats.nba.com\\n(nba_api)"];
      bronze [label="Bronze\\nimmutable Parquet\\nseason / ingested_at"];
      silver [label="Silver (dbt)\\nlatest pull, typed\\n+ data tests"];
      gold [label="Gold (dbt)\\nper-100 features\\ndim_players"];
      ml [label="ML\\nPCA | k-means | t-SNE\\n+ quality check"];
      serve [label="Serving\\nParquet export"];
      dash [label="This dashboard"];
      api -> bronze -> silver -> gold -> ml -> serve -> dash;
      dagster [label="Dagster: orchestration, schedules, lineage" shape=note fillcolor="#15171d"
               fontcolor="{INK_2}"];
    }}""")

    inv = data["lake_inventory"]
    st.markdown("#### Lake freshness")
    fresh = inv.groupby("season").agg(tables=("bronze_table", "nunique"), pulls=("pulls", "max"),
                                      latest_pull=("latest_pull", "max"),
                                      source=("source", "first")).reset_index()
    fresh["latest_pull"] = pd.to_datetime(fresh["latest_pull"], format="%Y%m%dT%H%M%SZ", utc=True)
    st.dataframe(fresh.rename(columns={"season": "Season", "tables": "Bronze tables",
                                       "pulls": "Pulls (max per table)", "latest_pull": "Latest pull (UTC)",
                                       "source": "Source"}),
                 hide_index=True, width="stretch")

    runs = data["model_runs"].sort_values("run_at")
    last = runs.iloc[-1]
    st.markdown("#### Model diagnostics (latest run)")
    m = st.columns(5)
    m[0].metric("Player-seasons", f"{last.n_player_seasons:,}")
    m[1].metric("Variance explained", f"{last.explained_variance:.1%}",
                help=f"By {last.n_components} PCA components from {last.n_features} features")
    m[2].metric("Trustworthiness", f"{last.trustworthiness_k10:.3f}",
                help="Are neighbours in PCA space real neighbours in the full feature space? (k=10)")
    m[3].metric("Neighbour preservation", f"{last.neighbor_preservation_k10:.0%}",
                help="Share of each player's 10 nearest neighbours kept after compression")
    m[4].metric("Cluster silhouette", f"{last.silhouette:.2f}",
                help="Archetype separation, -1 to 1. Basketball roles overlap, so ~0.2-0.4 is normal")

    left, right = st.columns([1, 1.4])
    with left:
        ev = data["explained_variance"]
        fig = go.Figure(go.Bar(
            x=ev["component"].str.upper(), y=ev["ratio"] * 100,
            marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2), cornerradius=4),
            customdata=ev["cumulative"] * 100,
            hovertemplate="%{x}: %{y:.1f}% (cumulative %{customdata:.1f}%)<extra></extra>"))
        base_layout(fig, height=380, title=dict(text="Variance explained per component",
                                                font=dict(color=INK, size=13)))
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, width="stretch")
    with right:
        ld = data["pca_loadings"]
        ld = ld[ld["component"].isin(["pc1", "pc2", "pc3", "pc4", "pc5"])]
        mat = ld.pivot(index="feature", columns="component", values="loading").reindex(feats)
        fig = go.Figure(go.Heatmap(
            z=mat.values, x=[c.upper() for c in mat.columns], y=[FEATURES[f][0] for f in mat.index],
            zmid=0, zmin=-0.5, zmax=0.5,
            colorscale=[[0, "#d95926"], [0.5, "#2a2b30"], [1, "#3987e5"]],
            xgap=2, ygap=2, colorbar=dict(tickfont=dict(color=INK_2), outlinewidth=0),
            hovertemplate="%{y} on %{x}: %{z:.2f}<extra></extra>"))
        base_layout(fig, height=560, title=dict(text="What the first 5 components measure (loadings)",
                                                font=dict(color=INK, size=13)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Run history")
    st.dataframe(runs[["run_at", "seasons", "n_player_seasons", "explained_variance",
                       "trustworthiness_k10", "neighbor_preservation_k10", "silhouette"]]
                 .sort_values("run_at", ascending=False)
                 .rename(columns={"run_at": "Run at", "seasons": "Seasons", "n_player_seasons": "Player-seasons",
                                  "explained_variance": "Var. explained", "trustworthiness_k10": "Trustworthiness",
                                  "neighbor_preservation_k10": "Neighbour pres.", "silhouette": "Silhouette"}),
                 hide_index=True, width="stretch")
