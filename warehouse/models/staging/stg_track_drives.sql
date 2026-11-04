with latest as ({{ latest_pull('drives') }})
select
    cast(PLAYER_ID as bigint) as player_id,
    season,
    cast(DRIVES as double) as drives
from latest
