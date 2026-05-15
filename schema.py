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
            db.get("column_types", []),
            db.get("foreign_keys", []),
        )
    log.info(f"Loaded schemas for {len(lookup)} databases")
    return lookup

# convert tables, columns, and foreign keys into concise
# "t1(c1:type, c2:type), t2(c3:type); FK: t1.c2=t2.c3" format
def _format_schema(table_names, column_names_original, column_types, foreign_keys):
    table_columns = defaultdict(list)
    for col_idx, (table_idx, col_name) in enumerate(column_names_original):
        if table_idx < 0:
            continue
        col_type = column_types[col_idx] if col_idx < len(column_types) else ""
        if col_type:
            table_columns[table_idx].append(f"{col_name}:{col_type}")
        else:
            table_columns[table_idx].append(col_name)

    parts = []
    for i, name in enumerate(table_names):
        cols = ", ".join(table_columns.get(i, []))
        parts.append(f"{name}({cols})")
    tables_str = ", ".join(parts)

    fk_parts = []
    for src_idx, dst_idx in foreign_keys:
        src_table_idx, src_col = column_names_original[src_idx]
        dst_table_idx, dst_col = column_names_original[dst_idx]
        if src_table_idx < 0 or dst_table_idx < 0:
            continue
        fk_parts.append(f"{table_names[src_table_idx]}.{src_col}={table_names[dst_table_idx]}.{dst_col}")
    if not fk_parts:
        return tables_str
    return f"{tables_str}; FK: {', '.join(fk_parts)}"

#trims long schemas, preferring to cut at a complete FK entry or table boundary
def truncate_schema(schema_str, max_length):
    if len(schema_str) <= max_length:
        return schema_str
    truncated = schema_str[:max_length]
    fk_marker_idx = truncated.find("; FK:")
    if fk_marker_idx > 0:
        # In the FK section: cut at the last complete FK entry
        last_comma = truncated.rfind(",")
        if last_comma > fk_marker_idx:
            return truncated[:last_comma]
    last_close = truncated.rfind(")")
    if last_close > 0:
        return truncated[:last_close + 1]
    return truncated
