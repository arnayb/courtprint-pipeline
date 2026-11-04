with latest as ({{ latest_pull('possessions') }})
select
    cast(PLAYER_ID as bigint) as player_id,
    season,
    cast(TOUCHES as double) as touches,
    cast(AVG_SEC_PER_TOUCH as double) as avg_sec_per_touch
from latest
