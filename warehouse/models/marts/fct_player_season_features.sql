-- Gold: one row per qualified player-season, volume stats per 100 possessions.
-- This is the model's input and the main table the dashboard reads.
{% set per100 = {
    'pts': 'pts_100', 'fg3a': 'fg3a_100', 'ast': 'ast_100', 'tov': 'tov_100',
    'stl': 'stl_100', 'blk': 'blk_100', 'drives': 'drives_100',
    'catch_shoot_fga': 'cs_fga_100', 'pull_up_fga': 'pullup_fga_100',
    'potential_ast': 'pot_ast_100', 'post_touches': 'post_touches_100',
    'screen_assists': 'screen_ast_100', 'deflections': 'deflections_100',
    'contested_shots': 'contests_100'
} %}
select
    player_id,
    season,
    player_name,
    team,
    age,
    games,
    minutes,
    possessions,
    {% for raw, out in per100.items() -%}
    {{ raw }} / nullif(possessions, 0) * 100 as {{ out }},
    {% endfor -%}
    ts_pct,
    usg_pct,
    ast_pct,
    oreb_pct,
    dreb_pct,
    fg3a / nullif(fga, 0)  as fg3a_rate,
    fta  / nullif(fga, 0)  as fta_rate,
    avg_sec_per_touch      as sec_per_touch
from {{ ref('int_player_season_totals') }}
where minutes >= {{ var('min_minutes') }}
  and possessions > 0
