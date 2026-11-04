{#
  Bronze is append-only, so a season pulled twice has two copies of every
  player. Keep only the most recent pull per player-season. This is what makes
  re-running ingestion idempotent.
#}
{% macro latest_pull(source_table) -%}
    select *
    from {{ source('bronze', source_table) }}
    qualify row_number() over (
        partition by PLAYER_ID, season
        order by ingested_at desc
    ) = 1
{%- endmacro %}
