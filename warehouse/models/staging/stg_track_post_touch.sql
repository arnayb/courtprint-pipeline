with latest as ({{ latest_pull('post_touch') }})
select
    cast(PLAYER_ID as bigint) as player_id,
    season,
    cast(POST_TOUCHES as double) as post_touches
from latest
