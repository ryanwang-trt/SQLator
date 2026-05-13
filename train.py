import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from config import (
    MODEL_PATH, BASE_MODEL, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, BATCH_SIZE,
    NUM_EPOCHS, LEARNING_RATE, PROMPT_TEMPLATE, ACCUMULATION_STEPS,
    MAX_SCHEMA_LENGTH, WARMUP_RATIO,
)
from schema import load_spider_schemas, truncate_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

dataset = load_dataset("spider")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = model.to(device)
# use_cache must be off for gradient checkpointing — otherwise checkpointing
# silently no-ops and the cache eats the memory we were trying to save.
model.config.use_cache = False
model.gradient_checkpointing_enable()
log.info(f"Model loaded on: {device}")

schema_lookup = load_spider_schemas()

def tokenize(example):
    schema = schema_lookup.get(example["db_id"], "unknown")
    schema = truncate_schema(schema, MAX_SCHEMA_LENGTH)
    input_text = PROMPT_TEMPLATE.format(db_id=example["db_id"], schema=schema, question=example["question"])
    target_text = example["query"]

    model_inputs = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(target_text, max_length=MAX_OUTPUT_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(tokenize)
tokenized["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True)
log.info(f"Number of batches: {len(train_loader)}")
log.info("Tokenization done!")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Linear warmup + decay across optimizer (not micro-batch) steps
steps_per_epoch = (len(train_loader) + ACCUMULATION_STEPS - 1) // ACCUMULATION_STEPS
total_steps = NUM_EPOCHS * steps_per_epoch
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
log.info(f"Scheduler: {warmup_steps} warmup / {total_steps} total optimizer steps")

# Switch model to training mode (enables dropout, etc.)
model.train()

# Size of the final (possibly partial) accumulation window per epoch — used to
# scale leftover gradients by the true micro-batch count instead of ACCUMULATION_STEPS.
num_batches = len(train_loader)
leftover = num_batches % ACCUMULATION_STEPS
leftover_start = num_batches - leftover if leftover else num_batches

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    # Zero out gradients at the start of each epoch
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        # Move batch tensors to GPU/CPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Replace pad tokens in labels with -100 so the loss function ignores them
        labels[labels == tokenizer.pad_token_id] = -100

        # Forward pass: model computes cross-entropy loss internally when labels are provided
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Divide by the accumulation-window size so the summed gradient equals a single
        # full-window update. The final partial window is divided by its actual size
        # (`leftover`) rather than ACCUMULATION_STEPS, otherwise that step underweights.
        divisor = leftover if step >= leftover_start else ACCUMULATION_STEPS
        loss = outputs.loss / divisor
        # Backward pass: compute gradients and accumulate them (not zeroed yet)
        loss.backward()

        # Track the original (un-scaled) loss for logging
        total_loss += loss.item() * divisor

        # Only update weights every ACCUMULATION_STEPS steps
        if (step + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()    # Apply accumulated gradients to update weights
            scheduler.step()
            optimizer.zero_grad()  # Reset gradients for the next accumulation window

        if step % 100 == 0:
            log.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {step}/{num_batches} | LR: {scheduler.get_last_lr()[0]:.2e} | Loss: {loss.item() * divisor:.4f}")

    # Handle leftover steps if the dataset size isn't evenly divisible by ACCUMULATION_STEPS
    if leftover:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    log.info(f"Epoch {epoch+1}/{NUM_EPOCHS} finished | Avg Loss: {avg_loss:.4f}")

log.info("Training complete!")
# Save the fine-tuned model and tokenizer so predict.py can load them later
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
log.info(f"Model saved to {MODEL_PATH}")
