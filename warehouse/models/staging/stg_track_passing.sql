with latest as ({{ latest_pull('passing') }})
select
    cast(PLAYER_ID as bigint) as player_id,
    season,
    cast(PASSES_MADE as double) as passes_made,
    cast(POTENTIAL_AST as double) as potential_ast
from latest
