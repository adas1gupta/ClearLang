import pandas
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import numpy as np

test_df = pandas.read_csv("data/processed/test.csv").head(250)
metric = evaluate.load("sacrebleu")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("models/byt5/final")

MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64

pred_texts = []
ref_texts = []
token_counts = []
latencies = []

for i, row in test_df.iterrows():
    print(f"Processing example {i+1}/{len(test_df)}")
    input_text = row["foreign"]
    reference = row["native"]

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_INPUT_LEN,
    )

    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_length=MAX_TARGET_LEN,
        num_beams=4,
    )
    end = time.time()

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    tokens = tokenizer.tokenize(pred)
    token_counts.append(len(tokens))

    pred_texts.append(pred)
    ref_texts.append(reference)
    latencies.append(end - start)

bleu = metric.compute(predictions=pred_texts, references=[[r] for r in ref_texts])
print(f"BLEU score: {bleu['score']:.2f}")

results_df = pandas.DataFrame({
    "input": test_df["foreign"],
    "reference": ref_texts,
    "prediction": pred_texts,
    "token_count": token_counts,
    "latency_sec": latencies,
})

results_df.to_csv("data/results/byt5_eval_results.csv", index=False)