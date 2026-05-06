import logging
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from config import (
    MODEL_PATH, BASE_MODEL, MAX_INPUT_LENGTH, BATCH_SIZE,
    NUM_EPOCHS, LEARNING_RATE, PROMPT_TEMPLATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

dataset = load_dataset("spider")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
model = model.to(device)
log.info(f"Model loaded on: {device}")

def tokenize(example):
    input_text = PROMPT_TEMPLATE.format(db_id=example["db_id"], question=example["question"])
    target_text = example["query"]

    model_inputs = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(target_text, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(tokenize)
tokenized["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True)
log.info(f"Number of batches: {len(train_loader)}")
log.info("Tokenization done!")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        labels[labels == tokenizer.pad_token_id] = -100

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            log.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    log.info(f"Epoch {epoch+1}/{NUM_EPOCHS} finished | Avg Loss: {avg_loss:.4f}")

log.info("Training complete!")
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
log.info(f"Model saved to {MODEL_PATH}")
