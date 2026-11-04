with latest as ({{ latest_pull('pullup') }})
select
    cast(PLAYER_ID as bigint) as player_id,
    season,
    cast(PULL_UP_FGA as double) as pull_up_fga
from latest
