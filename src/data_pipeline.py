from datasets import load_dataset

MULTILINGUAL_DATASET = load_dataset("ted_hrlr", "ru_to_en")
MULTILINGUAL_TRAINING_DF = pd.DataFrame(MULTILINGUAL_DATASET["train"])
MULTILINGUAL_VALIDATION_DF = pd.DataFrame(MULTILINGUAL_DATASET["validation"])
MULTILINGUAL_TEST_DF = pd.DataFrame(MULTILINGUAL_DATASET["test"])

def main():
    counter = 0
    for item in MULTILINGUAL_DATASET['train']:
        print(item)
        if counter >= 5:
            break
        counter += 1

main()