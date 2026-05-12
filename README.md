# Text → SQL 

I fine-tuned a transformer model on the Spider dataset to turn plain English questions into SQL queries. Built the whole thing from scratch — PyTorch training loop, evaluation pipeline, and a Flask web demo.

Inspired by [Chat2DB](https://github.com/codePhiliaX/Chat2DB).

![Demo](demo.gif)

---

## Results

### v1 — T5-small (60M params)

Started at **4.26%** exact match accuracy. After adding schema context, normalizing evaluation, and beam search — got it up to **9.38%**.

| What I Did | Accuracy | Correct |
|------------|----------|---------|
| Baseline | 4.26% | 44/1034 |
| + Schema-aware input | 4.45% | 46/1034 |
| + Normalized eval | 8.80% | 91/1034 |
| + Beam search (n=5) | **9.38%** | **97/1034** |

Training loss across 3 epochs: **1.14 → 0.56 → 0.41**

### v2 — Flan-T5-base (250M params)

Upgraded base model to `google/flan-t5-base` with schema-aware prompting, gradient checkpointing, and gradient accumulation.

| Metric | Result |
|--------|--------|
| Exact Match | pending |
| Execution Accuracy | *pending* |

Training loss across 3 epochs: **0.52 → 0.25 → 0.16**

> Note: Exact match dropped slightly due to stricter SQL normalization (sqlglot). Execution accuracy (does the SQL return the correct result?) is a more meaningful metric and is expected to be significantly higher.

---

## Example Outputs

```
❯ python predict.py --question "how many singers do we have" --db concert_singer
SQL: SELECT COUNT(*) FROM singer  ✓

❯ python predict.py --question "show the names of students with a grade greater than 90"
SQL: SELECT T1.name FROM student AS T1 JOIN grade AS T2 ON T1.grade = T2.grade WHERE T2.grade > 90

❯ python predict.py --question "find all customers who spent over 100"
SQL: SELECT DISTINCT customer_id FROM customers WHERE avg(*) > 100
```

---

## How It Works

```
English question + schema → tokenize → T5 generates SQL → beam search decode → done
```

- **Model:** Google's Flan-T5-base — instruction-tuned encoder-decoder transformer (250M params)
- **Dataset:** Spider by Yale — 7000+ question/SQL pairs across 200 databases
- **Training:** 3 epochs, AdamW optimizer, lr=3e-4, batch size 2 (effective 8 via gradient accumulation)
- **Inference:** Beam search with 5 beams
- **Schema:** Table/column names passed in the prompt so the model knows the database structure
- **Evaluation:** Exact match + execution accuracy (runs predicted SQL against actual databases)

### Key Improvements Over Baseline
- **Schema-aware prompting** — passes actual table/column names from Spider's tables.json instead of just db_id
- **SQL normalization** — uses sqlglot to parse and normalize SQL before comparison
- **Gradient checkpointing** — trades compute for memory, enabling larger models on limited VRAM
- **Gradient accumulation** — simulates batch size 8 with mini-batches of 2
- **Lazy model loading** — model loads on first request, not at import time
- **HuggingFace Hub fallback** — auto-downloads trained weights if not available locally

### Why T5 and not BERT?
BERT is encoder-only — great for classification stuff like sentiment analysis. T5 is encoder-decoder — built for generation. Text-to-SQL is a generation task so T5 is the right call.

---

## Run It Yourself

```bash
git clone https://github.com/ryanwang-trt/text-to-sql.git
cd text-to-sql
pip install -r requirements.txt
```

### Quick Start (no training needed)
If a pre-trained model is available on HuggingFace, the app will auto-download it:
```bash
python app.py
```
Then open `http://127.0.0.1:5000`

### Download Spider databases (for evaluation)

The Spider SQLite databases are needed for execution accuracy evaluation:

1. Go to [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)
2. Download the dataset zip file
3. Extract the `database/` folder into `data/`:
```
data/database/concert_singer/concert_singer.sqlite
data/database/world_1/world_1.sqlite
...
```

### Train the model
```bash
python train.py
```
Takes ~30 min on GPU (NVIDIA RTX), ~2 hours on CPU. Model saves to `models/t5-sql/`.

### Run predictions
```bash
python predict.py --question "how many singers do we have" --db concert_singer
```

### Evaluate on validation set
```bash
python predict.py --evaluate
```
Reports both exact match accuracy and execution accuracy.

### Launch the web demo
```bash
python app.py
```

---

## Project Structure

```
text-to-sql/
├── train.py          ← training loop + fine-tuning
├── predict.py        ← inference + evaluation (exact match + execution accuracy)
├── app.py            ← Flask web demo (lazy model loading)
├── schema.py         ← schema loading + formatting from Spider's tables.json
├── config.py         ← hyperparameters + settings (centralized)
├── upload_model.py   ← upload trained weights to HuggingFace Hub
├── models/           ← saved model weights
├── data/
│   └── database/     ← Spider SQLite DBs (manually downloaded)
├── requirements.txt  ← dependencies
└── README.md         
```

---

## What I Learned

- **Fine-tuning** = taking a pre-trained model and training it further on your specific task. Way faster than training from scratch.
- **Training loop** = zero_grad → forward → loss → backward → step. Repeat thousands of times.
- **Schema context** is the single highest-impact improvement for Text-to-SQL — the model can't guess column names it's never seen.
- **Exact match is harsh** — execution accuracy is a more meaningful metric because semantically correct SQL with different formatting still fails exact match.
- **Gradient accumulation** lets you simulate large batch sizes on limited hardware by accumulating gradients over multiple mini-batches.
- **SQL normalization** (standardizing quotes, whitespace, parsing with sqlglot) can significantly affect evaluation metrics without changing model behavior.

---

## Stack

PyTorch · HuggingFace Transformers · HuggingFace Datasets · Flask · sqlglot · sqlite3

## Resources

- [Chat2DB](https://github.com/codePhiliaX/Chat2DB) — the project that inspired this
- [Spider Dataset](https://arxiv.org/abs/1809.08887) — Yale's Text-to-SQL benchmark
- [T5 Paper](https://arxiv.org/abs/1910.10683) — Google's text-to-text transformer
- [Flan-T5](https://huggingface.co/google/flan-t5-base) — instruction-tuned T5 by Google
