with latest as ({{ latest_pull('catch_shoot') }})
select
    cast(PLAYER_ID as bigint) as player_id,
    season,
    cast(CATCH_SHOOT_FGA as double) as catch_shoot_fga
from latest
