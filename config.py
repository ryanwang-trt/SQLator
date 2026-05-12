import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/t5-sql")
BASE_MODEL = os.getenv("BASE_MODEL", "google/flan-t5-base")

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
NUM_BEAMS = 5
MAX_SCHEMA_LENGTH = 300

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "ryanwang-trt/t5-sql")

PROMPT_TEMPLATE = "translate English to SQL [database: {db_id} | tables: {schema}]: {question}"

SPIDER_DB_DIR = os.getenv("SPIDER_DB_DIR", "data/databases")

MAX_QUESTION_LENGTH = 500
