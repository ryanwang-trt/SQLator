import argparse
import logging
import os
import re
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import MODEL_PATH, HF_MODEL_ID, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, NUM_BEAMS, PROMPT_TEMPLATE, MAX_SCHEMA_LENGTH
from schema import load_spider_schemas, truncate_schema
import sqlglot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

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
        tokenizer = T5Tokenizer.from_pretrained(source)
        model = T5ForConditionalGeneration.from_pretrained(source)
        model.eval()
        log.info(f"Model loaded from {source}")
    return tokenizer, model

def predict(question, db_id="unknown", schema="unknown"):
    schema = truncate_schema(schema, MAX_SCHEMA_LENGTH)
    input_text = PROMPT_TEMPLATE.format(db_id=db_id, schema=schema, question=question)
    tokenizer, model = get_model()
    tokenized_input = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt")
    tokenized_outputs = model.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        max_length=MAX_OUTPUT_LENGTH,
        num_beams=NUM_BEAMS,
    )
    return tokenizer.decode(tokenized_outputs[0], skip_special_tokens=True)

def evaluate():
    dataset = load_dataset("spider")
    validation_split = dataset["validation"]

    schema_lookup = load_spider_schemas()

    predictions = []
    samples = []

    for i, example in enumerate(validation_split):
        schema = schema_lookup.get(example["db_id"], "unknown")
        prediction = predict(example["question"], example["db_id"], schema=schema)
        sample = example["query"]
        predictions.append(normalize_sql(prediction))
        samples.append(normalize_sql(sample))

        if i % 100 == 0:
            log.info(f"Evaluating: {i}/{len(validation_split)}")

    correct = sum(1 for p, s in zip(predictions, samples) if p == s)
    accuracy = correct / len(samples) * 100

    log.info(f"Exact Match Accuracy: {accuracy:.2f}%")
    log.info(f"Correct: {correct}/{len(samples)}")

    print("\n--- Sample Predictions ---")
    for i in range(5):
        print(f"\nQuestion:  {validation_split[i]['question']}")
        print(f"Predicted: {predictions[i]}")
        print(f"Actual:    {samples[i]}")
        print(f"Match:     {'✓' if predictions[i] == samples[i] else '✗'}")

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
