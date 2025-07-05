import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

TRAIN_PATH = "data/processed/training.csv"
TEST_PATH  = "data/processed/test.csv"
MAX_INPUT_LEN  = 64
MAX_TARGET_LEN = 64

train_df = pd.read_csv(TRAIN_PATH).dropna(subset=["foreign","native"]).reset_index(drop=True)
test_df  = pd.read_csv(TEST_PATH).dropna(subset=["foreign","native"]).reset_index(drop=True)
train_ds = Dataset.from_pandas(train_df).select(range(2000))
test_ds  = Dataset.from_pandas(test_df).select(range(500))

metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": bleu["score"]}

def tokenize_batch(batch, tokenizer):
    inputs = tokenizer(
        batch["foreign"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["native"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )
    inputs["labels"] = labels["input_ids"]
    return inputs

# Training function
def fine_tune(model_name, tokenizer_cls, model_cls, output_dir):
    print(f"\n\n▶️ Fine-tuning {model_name} …")
    tokenizer = tokenizer_cls.from_pretrained(f"google/{model_name}-small")
    tokenized_train = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=train_ds.column_names)
    tokenized_test  = test_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=test_ds.column_names)

    model = model_cls.from_pretrained(f"google/{model_name}-small")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        eval_steps=500,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/final")
    print(f"={model_name} done, checkpoint in {output_dir}/final")

if __name__ == "__main__":
    from transformers import MT5Tokenizer, ByT5Tokenizer, AutoModelForSeq2SeqLM
    fine_tune("mt5",  MT5Tokenizer,  AutoModelForSeq2SeqLM, "models/mt5")
    fine_tune("byt5", ByT5Tokenizer, AutoModelForSeq2SeqLM, "models/byt5")
