# Text → SQL

Fine-tuned T5-small on the Spider benchmark dataset to convert natural language questions into SQL queries. Built from scratch with PyTorch and HuggingFace Transformers.

![Demo](demo.gif)

---

## Results

| Metric | Score |
|--------|-------|
| Baseline (exact match) | 4.26% (44/1034) |
| + Schema-aware input | 4.45% (46/1034) |
| + Normalized evaluation | 8.80% (91/1034) |
| + Beam search (n=5) | 9.38% (97/1034) |

**Training Loss:** 1.14 → 0.56 → 0.41 across 3 epochs

> Many predictions are semantically correct but marked wrong due to cosmetic differences (spacing, quote style). The exact match metric is intentionally strict.

---

## Example Predictions

```
Question: how many employees are in each department
SQL:      SELECT count(*), dept_code FROM employees GROUP BY dept_code

Question: show the names of students with a grade greater than 90
SQL:      SELECT T1.name FROM student AS T1 JOIN grade AS T2 ON T1.grade = T2.grade WHERE T2.grade > 90

Question: find all customers who spent over 100
SQL:      SELECT DISTINCT customer_id FROM customers WHERE avg(*) > 100
```

---

## How It Works

```
User question → Add prefix + database context → Tokenize → T5 generates SQL tokens → Decode → SQL output
```

1. **Model:** T5-small (60M parameters) — a text-to-text transformer by Google
2. **Dataset:** Spider (7000+ examples) — the standard Text-to-SQL benchmark by Yale
3. **Fine-tuning:** 3 epochs with AdamW optimizer, learning rate 3e-4, batch size 8
4. **Inference:** Beam search (n=5) for better output quality

### Why T5 over BERT?
- BERT is encoder-only → good for classification (sentiment, NER)
- T5 is encoder-decoder → good for generation (translation, summarization, SQL)
- Text-to-SQL is a generation task → T5 is the right fit

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/text-to-sql.git
cd text-to-sql
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Trains for 3 epochs on the Spider dataset. Model saves to `models/t5-sql/`.

## Predict

```bash
# Single question
python predict.py --question "how many employees are in each department"

# With database context
python predict.py --question "how many singers do we have" --db concert_singer

# Run evaluation on validation set
python predict.py --evaluate
```

## Web Demo

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Project Structure

```
text-to-sql/
├── train.py          ← fine-tuning T5 on Spider
├── predict.py        ← inference + evaluation
├── app.py            ← Flask web demo
├── models/           ← saved fine-tuned model
├── requirements.txt  ← dependencies
└── README.md
```

---

## What I Learned

- Fine-tuning adapts a pre-trained model to a specific task using a small dataset — much faster than training from scratch
- The PyTorch training loop: zero_grad → forward pass → loss → backward → step
- Exact match is the standard metric for Text-to-SQL but is harsh — many outputs are functionally correct SQL
- Schema context (database name, table/column info) matters for real-world Text-to-SQL systems
- Beam search improves output quality by considering multiple candidate sequences

---

## Tech Stack

- **PyTorch** — training loop and optimization
- **HuggingFace Transformers** — T5 model and tokenizer
- **HuggingFace Datasets** — Spider dataset
- **Flask** — web demo
- **scikit-learn** — evaluation

## Inspired By

- [Chat2DB](https://github.com/codePhiliaX/Chat2DB) — open source AI-powered SQL client
- [Spider Dataset](https://arxiv.org/abs/1809.08887) — Yale's Text-to-SQL benchmark
- [T5 Paper](https://arxiv.org/abs/1910.10683) — Google's text-to-text transformer
