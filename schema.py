import json
import logging
from collections import defaultdict

log = logging.getLogger(__name__)

# download tables.json from spider via hf_hub_download
# builds a db id to schema string dictionary
def load_spider_schemas():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="xlangai/spider",
            filename="tables.json",
            repo_type="dataset",
        )
    # fallback if download fails
    except Exception as e:
        log.warning(f"Could not download tables.json: {e}. Schema-aware prompting disabled.")
        return {}

    with open(path) as f:
        tables_data = json.load(f)

    lookup = {}
    for db in tables_data:
        lookup[db["db_id"]] = _format_schema(
            db["table_names_original"],
            db["column_names_original"],
        )
    log.info(f"Loaded schemas for {len(lookup)} databases")
    return lookup

# convert tables and colums list into concise table_name(col1,col2) format
def _format_schema(table_names, column_names_original):
    table_columns = defaultdict(list)
    for table_idx, col_name in column_names_original:
        if table_idx < 0:
            continue
        table_columns[table_idx].append(col_name)

    parts = []
    for i, name in enumerate(table_names):
        cols = ", ".join(table_columns.get(i, []))
        parts.append(f"{name}({cols})")
    return ", ".join(parts)

#trims long schemas
def truncate_schema(schema_str, max_length):
    if len(schema_str) <= max_length:
        return schema_str
    truncated = schema_str[:max_length]
    last_close = truncated.rfind(")")
    if last_close > 0:
        return truncated[:last_close + 1]
    return truncated
