# SQLator

Natural-language questions in, SQL queries out. I fine-tuned a transformer on the [Spider](https://yale-lily.github.io/spider) dataset, wrapped it in a Flask API, deployed it to Hugging Face Spaces, and built a Chrome extension that calls the API — so you can generate SQL from any tab without leaving the browser.

Inspired by [Chat2DB](https://github.com/codePhiliaX/Chat2DB).

![Demo](demo.gif)

**Try it without installing anything locally:**
- Backend lives on [Hugging Face Spaces](https://huggingface.co/spaces/ryanwang-trt/SQLator), API at <https://ryanwang-trt-sqlator.hf.space> (`curl …/health` returns `{"status":"ok"}`).
- The Chrome extension in [`chrome-extension/`](chrome-extension/) already points at it. Load unpacked at `chrome://extensions`, click the icon, ask a question.

---

## The pipeline

```
┌──────────────────┐    HTTPS POST /predict     ┌────────────────────────┐
│ Chrome extension │ ─────────────────────────► │  Flask app on HF Space │
│  (popup + ctx)   │ ◄───────────────────────── │  (Docker container)    │
└──────────────────┘     { sql } / { error }    └───────────┬────────────┘
                                                            │ lazy load
                                                            ▼
                                          ┌──────────────────────────────┐
                                          │  CodeT5+ 220M fine-tuned     │
                                          │  ryanwang-trt/t5-sql on Hub  │
                                          └──────────────────────────────┘
```

Three layers, each useful on its own:

1. **Model** — CodeT5+ 220M fine-tuned on Spider. Training loop, evaluation pipeline, weights published to the Hub.
2. **Backend** — Flask app exposing both an HTML demo (`/`) and a JSON `/predict` API. Containerized via `Dockerfile` for HF Spaces or any Docker host.
3. **Chrome extension** — Manifest V3 wrapper around the backend: popup with status pill and Ctrl+Enter submit, plus a right-click *Convert to SQL →* that drops a floating panel onto any page.

---

## Results

Benchmarked three models on Spider's validation set (1034 examples). Exact match plateaued at ~9% across all runs — schema generalization is the binding constraint at this parameter scale, not SQL syntax knowledge.

### v1 — T5-small (60M params)

Started at **4.26%** exact match. Iteratively improved through schema context, eval normalization, and beam search.

| What I Did | Exact Match | Correct |
|------------|-------------|---------|
| Baseline | 4.26% | 44/1034 |
| + Schema-aware input | 4.45% | 46/1034 |
| + Normalized eval | 8.80% | 91/1034 |
| + Beam search (n=5) | **9.38%** | **97/1034** |

Training loss across 3 epochs: **1.14 → 0.56 → 0.41**

### v2 — Flan-T5-base (250M params)

Upgraded to an instruction-tuned model with gradient checkpointing and accumulation.

| Metric | Result |
|--------|--------|
| Exact Match | 8.03% (83/1034) |
| Execution Accuracy | pending |

Training loss across 3 epochs: **0.52 → 0.25 → 0.16**

### v3 — CodeT5+ 220M (current)

Code-specialized model, trained for 6 epochs with cosine LR decay and warmup.

| Metric | Result |
|--------|--------|
| Exact Match | **9.38%** (97/1034) |
| Execution Accuracy | **9.77%** (101/1034) |

Training loss across 6 epochs: **0.72 → 0.23 → 0.13 → 0.07 → 0.04 → 0.02**

> Despite CodeT5+'s code-specific pretraining and 2× the epochs, exact match matched T5-small at 9.38%. Loss of 0.02 indicates strong memorization of training patterns — the limiting factor is cross-schema generalization, not SQL syntax. State-of-the-art on Spider (80%+) uses much larger models with execution-guided decoding and explicit schema linking.

---

## Example outputs

```
❯ python predict.py --question "how many singers do we have" --db concert_singer
SQL: SELECT COUNT(*) FROM singer  ✓

❯ python predict.py --question "show the names of students with a grade greater than 90"
SQL: SELECT T1.name FROM student AS T1 JOIN grade AS T2 ON T1.grade = T2.grade WHERE T2.grade > 90

❯ python predict.py --question "find all customers who spent over 100"
SQL: SELECT DISTINCT customer_id FROM customers WHERE avg(*) > 100
```

---

## How the model works

```
English question + db_id + schema → tokenize → CodeT5+ generates SQL → beam search decode
```

- **Model:** [`Salesforce/codet5p-220m`](https://huggingface.co/Salesforce/codet5p-220m) — code-specialized encoder-decoder transformer
- **Dataset:** Spider — 7000+ question/SQL pairs across 200 databases
- **Training:** 6 epochs, AdamW, cosine LR decay with warmup, effective batch size 8 (mini-batch 2 × accumulation 4)
- **Inference:** Beam search with 5 beams
- **Prompt:** `translate English to SQL [database: {db_id} | tables: {schema}]: {question}`
- **Evaluation:** Exact match + execution accuracy (runs predicted SQL against the actual SQLite databases)

### Key improvements over baseline

- **Schema-aware prompting** — passes actual table/column names from Spider's `tables.json`, not just the db_id
- **SQL normalization** — uses sqlglot to parse and normalize SQL before comparing predictions to gold
- **Gradient checkpointing + accumulation** — fit a larger model on limited VRAM, simulate batch 8 on mini-batches of 2
- **Cosine LR schedule with warmup** — smoother convergence than a flat learning rate
- **Lazy model loading + Hub fallback** — the Flask app downloads weights from the Hub on first request if no local checkpoint, which is what makes cold-start deployment to HF Spaces just work

### Why T5 and not BERT?

BERT is encoder-only — great for classification. T5 is encoder-decoder, built for generation. Text-to-SQL is a generation task so T5 is the right call.

---

## Backend

A Flask app with two interfaces:

| Route | What it does |
|-------|--------------|
| `GET  /` | HTML form demo (the gif above) |
| `POST /predict` | JSON API. Body: `{ question, db_id?, schema? }` → `{ sql }` or `{ error }` |
| `GET  /health` | `{ "status": "ok" }` — used by the extension popup to render the online/offline pill |

CORS is enabled (`flask-cors`) so the extension can call it from any origin. On first `/predict` after a cold start the model lazily downloads from the Hub (~30s); subsequent calls are fast.

### Deploying to Hugging Face Spaces

The repo ships a `Dockerfile` and `.dockerignore` ready for HF Spaces. The image uses a CPU-only torch wheel to keep size down, runs gunicorn as a non-root user on port 7860, and sets `HF_HOME` so the Hub cache lands somewhere writable.

```bash
# 1. Create a Space at https://huggingface.co/new-space (SDK: Docker, Blank)
# 2. Clone it next to this repo
git clone https://huggingface.co/spaces/<you>/sqlator sqlator-space
cd sqlator-space

# 3. Copy the backend in (skip chrome-extension/, data/, models/)
cp ../SQLator/app.py ../SQLator/config.py ../SQLator/schema.py \
   ../SQLator/requirements.txt ../SQLator/Dockerfile ../SQLator/.dockerignore .

# 4. Push — auth with an HF access token (write scope)
git add . && git commit -m "Deploy backend" && git push
```

Watch the build on your Space page → *Logs* tab. First build ~5 min. When you see `Listening at: http://0.0.0.0:7860`, hit `/health` to verify, then point the extension at the new URL (see below). My own deployment is at [huggingface.co/spaces/ryanwang-trt/SQLator](https://huggingface.co/spaces/ryanwang-trt/SQLator) for reference.

Same `Dockerfile` works on Render, Railway, Fly.io, or any Docker host — it listens on port 7860 by default.

---

## Chrome extension

[`chrome-extension/`](chrome-extension/) — Manifest V3, dark theme matching the web demo.

**Popup** (click the toolbar icon):
- Question textarea, optional database name, collapsible schema field
- Status pill (Online / Offline) — pings `/health` on open
- `Ctrl+Enter` submits, copy button on the result
- Persists your last question + db + schema to `chrome.storage.local`

**Context menu**: select any text on a webpage → right-click → *Convert to SQL →*. A floating panel slides in from the bottom-right with the generated SQL. The background script reuses the last schema you typed in the popup, so set it once and right-click queries get the same context.

### Install (load unpacked)

1. `chrome://extensions` → toggle **Developer mode**
2. **Load unpacked** → select `chrome-extension/`
3. Click the icon — pill should show **Online** (it's pointing at the hosted Space by default)

### Pointing at a different backend

One file changes when you redeploy: `chrome-extension/config.js`:

```js
const API_BASE = "https://your-backend.example.com";
```

Update `chrome-extension/manifest.json` → `host_permissions` to match. Reload the extension. Done.

Full extension docs in [`chrome-extension/README.md`](chrome-extension/README.md).

---

## Develop locally

The hosted Space is the easiest way to *use* SQLator. This section is for hacking on the model, backend, or extension.

```bash
git clone https://github.com/ryanwang-trt/SQLator.git
cd SQLator
pip install -r requirements.txt
```

### Run the backend

```bash
python app.py
```

Open `http://127.0.0.1:5000` for the web demo. The Flask app auto-downloads my fine-tuned weights from the Hub on first request, so no training is required.

### Reproduce my training

Spider's SQLite databases are needed for execution-accuracy evaluation but not for training itself. Download the dataset from [the Spider site](https://yale-lily.github.io/spider) and extract `database/` into `data/`:

```
data/database/concert_singer/concert_singer.sqlite
data/database/world_1/world_1.sqlite
...
```

Then:

```bash
# Fine-tune CodeT5+ on Spider (~30 min on GPU, ~2 hr on CPU). Saves to models/t5-sql/.
python train.py

# Predict on a single question
python predict.py --question "how many singers do we have" --db concert_singer

# Predict with explicit schema for better results
python predict.py --question "list all employees" --db company \
                  --schema "employee(id, name, dept_id), department(id, name)"

# Evaluate on the validation set (exact match + execution accuracy)
python predict.py --evaluate
```

---

## Project structure

```
SQLator/
├── train.py             ← training loop + fine-tuning
├── predict.py           ← inference + evaluation
├── app.py               ← Flask: HTML demo + /predict + /health
├── schema.py            ← schema loading from Spider's tables.json
├── config.py            ← hyperparameters + model IDs (centralized)
├── upload_model.py      ← push trained weights to HF Hub
├── Dockerfile           ← HF Spaces / any Docker host
├── .dockerignore
├── chrome-extension/    ← Manifest V3 extension (popup + context menu)
├── models/              ← saved checkpoint (gitignored)
└── data/database/       ← Spider SQLite DBs (manually downloaded)
```

---

## What I learned

- **Fine-tuning** = taking a pre-trained model and training it further on your specific task. Way faster than training from scratch.
- **Training loop** = zero_grad → forward → loss → backward → step. Repeat thousands of times.
- **Schema context** is the single highest-impact improvement for text-to-SQL — the model can't guess column names it's never seen. Building the same field into the Chrome extension's popup (and reusing it for context-menu queries) closes the same gap at the UX layer.
- **Exact match is harsh** — execution accuracy is more meaningful because semantically correct SQL with different formatting still fails exact match.
- **Low training loss ≠ better generalization** — CodeT5+ reached 0.02 loss but matched T5-small's exact match, classic overfitting to seen schemas.

---

## Stack

PyTorch · HuggingFace Transformers · HuggingFace Datasets · Flask · flask-cors · gunicorn · sqlglot · sqlite3 · Docker · Chrome Extension (Manifest V3)

## Resources

- [Spider Dataset](https://arxiv.org/abs/1809.08887) — Yale's text-to-SQL benchmark
- [T5 Paper](https://arxiv.org/abs/1910.10683) — Google's text-to-text transformer
- [CodeT5+](https://huggingface.co/Salesforce/codet5p-220m) — Salesforce's code-specialized T5 (the model I fine-tuned)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces) — where the backend lives
- [Chrome Extensions (MV3)](https://developer.chrome.com/docs/extensions/mv3/intro/) — extension docs
- [Chat2DB](https://github.com/codePhiliaX/Chat2DB) — the project that inspired this
