from flask import Flask, request, render_template_string
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

model_path = "models/t5-sql"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()

def predict(question, db_id="unknown"):
    input_text = "translate English to SQL [database: " + db_id + "]: " + question
    tokenized_input = tokenizer(input_text, max_length=128, truncation=True, return_tensors="pt")
    tokenized_outputs = model.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        max_length=128,
        num_beams=5,
    )
    sql = tokenizer.decode(tokenized_outputs[0], skip_special_tokens=True)
    return sql

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Text2SQL — Natural Language to SQL</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'DM Sans', sans-serif;
            min-height: 100vh;
            background: #0a0a0f;
            color: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        /* animated background grid */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image:
                linear-gradient(rgba(56, 189, 248, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56, 189, 248, 0.03) 1px, transparent 1px);
            background-size: 60px 60px;
            z-index: 0;
        }

        /* glow orb */
        body::after {
            content: '';
            position: fixed;
            top: -200px; right: -200px;
            width: 600px; height: 600px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.08), transparent 70%);
            border-radius: 50%;
            z-index: 0;
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 680px;
            padding: 20px;
        }

        .badge {
            display: inline-block;
            padding: 6px 14px;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 100px;
            font-size: 12px;
            font-weight: 500;
            color: #38bdf8;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 20px;
        }

        h1 {
            font-family: 'JetBrains Mono', monospace;
            font-size: 42px;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.1;
            margin-bottom: 8px;
        }

        h1 span {
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: #6b7280;
            font-size: 15px;
            margin-bottom: 40px;
        }

        .card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 32px;
            backdrop-filter: blur(20px);
        }

        label {
            display: block;
            font-size: 13px;
            font-weight: 500;
            color: #9ca3af;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }

        input[type=text] {
            width: 100%;
            padding: 14px 16px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            color: #f0f0f0;
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            outline: none;
            transition: border-color 0.2s;
            margin-bottom: 20px;
        }

        input[type=text]:focus {
            border-color: rgba(56, 189, 248, 0.4);
        }

        input[type=text]::placeholder {
            color: #4b5563;
        }

        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            color: #fff;
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s;
            letter-spacing: 0.3px;
        }

        button:hover { opacity: 0.9; }
        button:active { transform: scale(0.98); }

        .result {
            margin-top: 28px;
            padding-top: 28px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .result-label {
            font-size: 12px;
            font-weight: 500;
            color: #6b7280;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .result-question {
            color: #d1d5db;
            font-size: 15px;
            margin-bottom: 16px;
        }

        .sql-output {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: 10px;
            padding: 16px 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            color: #38bdf8;
            line-height: 1.6;
            overflow-x: auto;
        }

        .footer {
            text-align: center;
            margin-top: 32px;
            font-size: 12px;
            color: #374151;
        }

        .footer a {
            color: #4b5563;
            text-decoration: none;
        }

        /* fade in animation */
        .container { animation: fadeUp 0.6s ease-out; }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="badge">Fine-tuned T5 Model</div>
        <h1>Text → <span>SQL</span></h1>
        <p class="subtitle">Ask a question in plain English. Get a SQL query back.</p>

        <div class="card">
            <form method="POST">
                <label>YOUR QUESTION</label>
                <input type="text" name="question" placeholder="e.g. how many employees are in each department" value="{{ question or '' }}" autofocus>

                <label>DATABASE (OPTIONAL)</label>
                <input type="text" name="db_id" placeholder="e.g. concert_singer" value="{{ db_id or '' }}">

                <button type="submit">Generate SQL →</button>
            </form>

            {% if sql %}
            <div class="result">
                <div class="result-label">Input</div>
                <div class="result-question">{{ question }}</div>

                <div class="result-label">Generated SQL</div>
                <div class="sql-output">{{ sql }}</div>
            </div>
            {% endif %}
        </div>

        <div class="footer">
            Built with T5-small + PyTorch — <a href="https://github.com">View on GitHub</a>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    question = None
    db_id = None
    sql = None

    if request.method == "POST":
        question = request.form["question"]
        db_id = request.form.get("db_id", "unknown")
        if not db_id:
            db_id = "unknown"
        sql = predict(question, db_id)

    return render_template_string(HTML, question=question, db_id=db_id, sql=sql)

if __name__ == "__main__":
    app.run(debug=True)