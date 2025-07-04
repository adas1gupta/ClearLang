import pandas
from datasets import Dataset
from transformers import AutoTokenizer

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64

train_df = pandas.read_csv(TRAIN_PATH)
test_df = pandas.read_csv(TEST_PATH)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
byt5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

# Tokenization function
def tokenize_batch(batch, tokenizer):
    
    model_inputs = tokenizer(
        batch["src"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["tgt"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(lambda x: tokenize_batch(x, mt5_tokenizer), batched=True)
tokenized_test = test_dataset.map(lambda x: tokenize_batch(x, mt5_tokenizer), batched=True)