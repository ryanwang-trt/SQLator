import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/t5-sql")
BASE_MODEL = os.getenv("BASE_MODEL", "Salesforce/codet5p-220m")

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 6
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.1
NUM_BEAMS = 5
MAX_SCHEMA_LENGTH = 400

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "ryanwang-trt/t5-sql")

PROMPT_TEMPLATE = "translate English to SQL [database: {db_id} | tables: {schema}]: {question}"

SPIDER_DB_DIR = os.getenv("SPIDER_DB_DIR", "data/database")

MAX_QUESTION_LENGTH = 500
