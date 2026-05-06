import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/t5-sql")
BASE_MODEL = os.getenv("BASE_MODEL", "t5-small")

MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
NUM_BEAMS = 5

PROMPT_TEMPLATE = "translate English to SQL [database: {db_id}]: {question}"

MAX_QUESTION_LENGTH = 500
