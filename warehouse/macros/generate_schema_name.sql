{# Use the folder schema as-is (silver / gold) instead of dbt's "main_silver". #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name if custom_schema_name else target.schema }}
{%- endmacro %}
