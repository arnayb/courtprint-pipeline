with latest as ({{ latest_pull('hustle') }})
select
    cast(PLAYER_ID as bigint)          as player_id,
    season,
    cast(DEFLECTIONS as double)        as deflections,
    cast(CONTESTED_SHOTS as double)    as contested_shots,
    cast(SCREEN_ASSISTS as double)     as screen_assists
from latest
