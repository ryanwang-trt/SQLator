import argparse #reading command lines
import re
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from datasets import load_dataset
from sklearn.metrics import accuracy_score

model_path = "models/t5-sql"

#load tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_path)

#load model
model = T5ForConditionalGeneration.from_pretrained(model_path)

#evalution mode
model.eval()

#predict function
def predict(question, db_id="unknown"):
    input_text = "translate English to SQL [database: " + db_id + "]: " + question

    tokenized_input = tokenizer(input_text, max_length=128, return_tensors="pt") #128 to match the training data

    tokenized_outputs = model.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        max_length=128,
        num_beams=5,
    )

    sql = tokenizer.decode(tokenized_outputs[0], skip_special_tokens=True) #decoded output

    return sql

#evaluate function
def evaluate():
    # load spider dataset and its validation split
    dataset = load_dataset("spider")
    validation_split = dataset["validation"]

    predictions = []
    samples = []

    for i, example in enumerate(validation_split):
        #compare the predicted output and the actual output
        prediction = predict(example["question"], example["db_id"])
        sample = example["query"]

        predictions.append(normalize_sql(prediction))
        samples.append(normalize_sql(sample))

        if i % 100 == 0:
            print(f"Evaluating: {i}/{len(validation_split)}")

    # get accuracy rate
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == samples[i]:
            correct = correct + 1

    accuracy = correct / len(samples) * 100

    print(f"\nExact Match Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{len(samples)}")

    print("\n--- Sample Predictions ---")
    for i in range(5):
        print(f"\nQuestion:  {validation_split[i]['question']}")
        print(f"Predicted: {predictions[i]}")
        print(f"Actual:    {samples[i]}")
        print(f"Match:     {'✓' if predictions[i] == samples[i] else '✗'}")

def normalize_sql(sql):
    sql = sql.strip().lower()
    sql = sql.replace('"', "'")          # standardize quotes
    sql = re.sub(r'\s+', ' ', sql)       # collapse extra spaces
    sql = sql.replace(' ,', ',')         # fix space before comma
    sql = sql.replace(', ', ',')         # remove space after comma
    return sql

#main block 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--db", type=str, default="unknown")
    args = parser.parse_args()

    if args.evaluate:
        evaluate()
    elif args.question:
        sql = predict(args.question, args.db)
        print(f"\nQuestion: {args.question}")
        print(f"SQL:      {sql}")
    else:
        print("Use --question or --evaluate")
    