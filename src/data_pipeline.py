from datasets import load_dataset
import pandas
import re

MULTILINGUAL_DATASET = load_dataset("ted_hrlr", "ru_to_en")
MULTILINGUAL_TRAINING_DF = pandas.DataFrame(MULTILINGUAL_DATASET["train"])
MULTILINGUAL_VALIDATION_DF = pandas.DataFrame(MULTILINGUAL_DATASET["validation"])
MULTILINGUAL_TEST_DF = pandas.DataFrame(MULTILINGUAL_DATASET["test"])

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]','', text)
    
    return text

def process_entry(dataframe):
    dataframe["foreign"] = dataframe["translation"].apply(lambda x: clean_text(x["ru"]))
    dataframe["native"] = dataframe["translation"].apply(lambda x: clean_text(x["en"]))
    dataframe = dataframe.drop(columns=["translation"])

    return dataframe
    

MULTILINGUAL_TRAINING_DF = process_entry(MULTILINGUAL_TRAINING_DF)
MULTILINGUAL_VALIDATION_DF = process_entry(MULTILINGUAL_VALIDATION_DF)
MULTILINGUAL_TEST_DF = process_entry(MULTILINGUAL_TEST_DF)

MULTILINGUAL_TRAINING_DF.to_csv("data/processed/training.csv", index=False)
MULTILINGUAL_VALIDATION_DF.to_csv("data/processed/validation.csv", index=False)
MULTILINGUAL_TEST_DF.to_csv("data/processed/test.csv", index=False)
