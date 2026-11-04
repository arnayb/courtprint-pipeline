-- Every source joined onto the box-score spine. Tracking/hustle tables are
-- left-joined: a missing tracking row becomes null, not a dropped player.
select
    b.*,
    a.possessions, a.ts_pct, a.usg_pct, a.ast_pct, a.oreb_pct, a.dreb_pct,
    h.deflections, h.contested_shots, h.screen_assists,
    d.drives,
    cs.catch_shoot_fga,
    pu.pull_up_fga,
    pt.post_touches,
    ps.touches, ps.avg_sec_per_touch,
    pa.passes_made, pa.potential_ast
from {{ ref('stg_base') }} b
join      {{ ref('stg_advanced') }}            a  using (player_id, season)
left join {{ ref('stg_hustle') }}              h  using (player_id, season)
left join {{ ref('stg_track_drives') }}        d  using (player_id, season)
left join {{ ref('stg_track_catch_shoot') }}   cs using (player_id, season)
left join {{ ref('stg_track_pullup') }}        pu using (player_id, season)
left join {{ ref('stg_track_post_touch') }}    pt using (player_id, season)
left join {{ ref('stg_track_possessions') }}   ps using (player_id, season)
left join {{ ref('stg_track_passing') }}       pa using (player_id, season)
