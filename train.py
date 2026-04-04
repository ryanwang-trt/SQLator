import torch

from datasets import load_dataset
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from torch.utils.data import DataLoader

#load dataset
dataset = load_dataset("spider")

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

#load the model and move to device
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = model.to(device)
print(f"Model loaded on: {device}")

#tokenize each example
def tokenize(example):
    input_text = "translate English to SQL [database: " + example["db_id"] + "]: " + example["question"]
    target_text = example["query"]

    # tokenize input and target seperately
    model_inputs = tokenizer(input_text, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(target_text, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(tokenize)

#keep only the columns the model needs
tokenized["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#create DataLoader
train_loader = DataLoader(tokenized["train"], batch_size=8, shuffle=True)

print(f"Number of batches: {len(train_loader)}")

print("Tokenization done!")
print("Sample input_ids:", tokenized["train"][0]["input_ids"][:10])
print("Sample labels:   ", tokenized["train"][0]["labels"][:10])

# --- Training Loop ---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

NUM_EPOCHS = 3

model.train() # switch to training mode 

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for step, batch in enumerate(train_loader): 
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # replace padding token id (0) with -100 so loss ignores padding
        labels[labels == tokenizer.pad_token_id] = -100

        optimizer.zero_grad() #clear old gradients
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) #model trying predict sql
        loss = outputs.loss
        loss.backward() #calculating weights
        optimizer.step() #update weights 

        total_loss += loss.item() #how wrong

        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} finished | Avg Loss: {avg_loss:.4f}")

print("Training complete!")

# --- Save the model ---
model.save_pretrained("models/t5-sql")
tokenizer.save_pretrained("models/t5-sql")
print("Model saved to models/t5-sql/")
