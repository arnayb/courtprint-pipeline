with latest as ({{ latest_pull('advanced') }})
select
    cast(PLAYER_ID as bigint)   as player_id,
    season,
    cast(POSS as double)        as possessions,
    cast(TS_PCT as double)      as ts_pct,
    cast(USG_PCT as double)     as usg_pct,
    cast(AST_PCT as double)     as ast_pct,
    cast(OREB_PCT as double)    as oreb_pct,
    cast(DREB_PCT as double)    as dreb_pct
from latest
