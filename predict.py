import argparse
import logging
import os
import re
import sqlite3
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODEL_PATH, HF_MODEL_ID, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, NUM_BEAMS, PROMPT_TEMPLATE, MAX_SCHEMA_LENGTH, SPIDER_DB_DIR
from schema import load_spider_schemas, truncate_schema
import sqlglot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None

def get_model():
    global tokenizer, model
    if model is None:
        if os.path.exists(MODEL_PATH):
            source = MODEL_PATH
        else:
            log.info(f"Local model not found at '{MODEL_PATH}', downloading from HuggingFace: {HF_MODEL_ID}")
            source = HF_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSeq2SeqLM.from_pretrained(source)
        model = model.to(device)
        model.eval()
        log.info(f"Model loaded from {source} on {device}")
    return tokenizer, model

def predict(question, db_id="unknown", schema="unknown"):
    schema = truncate_schema(schema, MAX_SCHEMA_LENGTH)
    input_text = PROMPT_TEMPLATE.format(db_id=db_id, schema=schema, question=question)
    tokenizer, model = get_model()
    tokenized_input = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt")
    tokenized_outputs = model.generate(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        max_length=MAX_OUTPUT_LENGTH,
        num_beams=NUM_BEAMS,
    )
    return tokenizer.decode(tokenized_outputs[0], skip_special_tokens=True)

def execute_sql(sql, db_path):
    if not os.path.exists(db_path):
        return None
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        cursor = conn.cursor()
        cursor.execute(sql)
        return sorted(cursor.fetchall())
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def evaluate():
    dataset = load_dataset("spider")
    validation_split = dataset["validation"]

    schema_lookup = load_spider_schemas()

    has_dbs = os.path.isdir(SPIDER_DB_DIR)
    if not has_dbs:
        log.warning(f"Spider databases not found at '{SPIDER_DB_DIR}'. Execution accuracy will be skipped.")

    predictions = []
    samples = []
    db_ids = []

    for i, example in enumerate(validation_split):
        schema = schema_lookup.get(example["db_id"], "unknown")
        prediction = predict(example["question"], example["db_id"], schema=schema)
        sample = example["query"]
        predictions.append(normalize_sql(prediction))
        samples.append(normalize_sql(sample))
        db_ids.append(example["db_id"])

        if i % 100 == 0:
            log.info(f"Evaluating: {i}/{len(validation_split)}")

    exact_correct = sum(1 for p, s in zip(predictions, samples) if p == s)
    exact_accuracy = exact_correct / len(samples) * 100

    log.info(f"Exact Match Accuracy: {exact_accuracy:.2f}%")
    log.info(f"Exact Match Correct: {exact_correct}/{len(samples)}")

    exec_results = []
    if has_dbs:
        for pred, gold, db_id in zip(predictions, samples, db_ids):
            db_path = os.path.join(SPIDER_DB_DIR, db_id, f"{db_id}.sqlite")
            pred_result = execute_sql(pred, db_path)
            gold_result = execute_sql(gold, db_path)
            match = pred_result is not None and gold_result is not None and pred_result == gold_result
            exec_results.append(match)

        exec_correct = sum(exec_results)
        exec_accuracy = exec_correct / len(samples) * 100
        log.info(f"Execution Accuracy:  {exec_accuracy:.2f}%")
        log.info(f"Execution Correct:   {exec_correct}/{len(samples)}")

    print("\n--- Sample Predictions ---")
    for i in range(5):
        print(f"\nQuestion:  {validation_split[i]['question']}")
        print(f"Predicted: {predictions[i]}")
        print(f"Actual:    {samples[i]}")
        exact = predictions[i] == samples[i]
        exec_match = exec_results[i] if exec_results else None
        exact_mark = "✓" if exact else "✗"
        exec_mark = "✓" if exec_match else ("✗" if exec_match is not None else "-")
        print(f"Exact: {exact_mark}  Exec: {exec_mark}")

def normalize_sql(sql):
    sql = sql.strip().lower()
    sql = sql.replace('"', "'")
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
        sql = parsed.sql(dialect="sqlite")
    except Exception:
        pass  # fall back to raw string if parsing fails
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--db", type=str, default="unknown")
    parser.add_argument("--schema", type=str, default="unknown")
    args = parser.parse_args()

    if args.evaluate:
        evaluate()
    elif args.question:
        sql = predict(args.question, args.db, schema=args.schema)
        print(f"\nQuestion: {args.question}")
        print(f"SQL:      {sql}")
    else:
        print("Use --question or --evaluate")
