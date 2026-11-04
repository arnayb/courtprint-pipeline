-- One row per player: latest name/team and career span in the data.
select
    player_id,
    arg_max(player_name, season) as player_name,
    arg_max(team, season)        as latest_team,
    min(season)                  as first_season,
    max(season)                  as last_season,
    count(*)                     as seasons_in_data,
    sum(minutes)                 as total_minutes
from {{ ref('stg_base') }}
group by player_id
