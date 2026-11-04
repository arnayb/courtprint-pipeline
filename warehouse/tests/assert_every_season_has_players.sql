-- Fails if any season in bronze produced fewer than 150 qualified players,
-- which usually means a broken or partial pull.
select season, count(*) as n
from {{ ref('fct_player_season_features') }}
group by season
having count(*) < 150
