with latest as ({{ latest_pull('base') }})
select
    cast(PLAYER_ID as bigint)        as player_id,
    season,
    PLAYER_NAME                      as player_name,
    cast(TEAM_ID as bigint)          as team_id,
    TEAM_ABBREVIATION                as team,
    cast(AGE as double)              as age,
    cast(GP as integer)              as games,
    cast(MIN as double)              as minutes,
    cast(FGA as double)              as fga,
    cast(FG3A as double)             as fg3a,
    cast(FTA as double)              as fta,
    cast(PTS as double)              as pts,
    cast(AST as double)              as ast,
    cast(TOV as double)              as tov,
    cast(STL as double)              as stl,
    cast(BLK as double)              as blk,
    cast(OREB as double)             as oreb,
    cast(DREB as double)             as dreb,
    ingested_at
from latest
