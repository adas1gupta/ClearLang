import pandas
from datasets import Dataset
import evaluate
from transformers import MT5Tokenizer
from transformers import ByT5Tokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np

TRAIN_PATH = "data/processed/training.csv"
TEST_PATH = "data/processed/test.csv"
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64

train_df = pandas.read_csv(TRAIN_PATH).dropna(subset=["foreign","native"]).reset_index(drop=True)
test_df  = pandas.read_csv(TEST_PATH).dropna( subset=["foreign","native"]).reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
train_dataset = train_dataset.select(range(2000))
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
test_dataset = test_dataset.select(range(500))

mt5_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
byt5_tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")

metric = evaluate.load("sacrebleu")

model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# Tokenization function
def tokenize_batch(batch, tokenizer):

    model_inputs = tokenizer(
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
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(
    lambda batch: tokenize_batch(batch, mt5_tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_test = test_dataset.map(
    lambda batch: tokenize_batch(batch, mt5_tokenizer),
    batched=True,
    remove_columns=test_dataset.column_names
)

# Compute metrics function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = mt5_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, mt5_tokenizer.pad_token_id)
    decoded_labels = mt5_tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": bleu["score"]}

training_args = Seq2SeqTrainingArguments(
    output_dir="models/mt5",
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
    tokenizer=mt5_tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("models/mt5/final")