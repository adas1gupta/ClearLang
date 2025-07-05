import argparse
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser(description="ClearLang Translation CLI")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mt5", "byt5"],
        required=True,
        help="Which model to use: mt5 or byt5"
    )
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="The sentence to translate"
    )
    args = parser.parse_args()

    model_map = {
        "mt5": ("google/mt5-small", "models/mt5/final"),
        "byt5": ("google/byt5-small", "models/byt5/final")
    }

    pretrained_name, model_dir = model_map[args.model]

    print(f"Loading {args.model} model...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    MAX_INPUT_LEN = 64
    MAX_TARGET_LEN = 64

    inputs = tokenizer(
        args.input_text,
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

    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    tokens = tokenizer.tokenize(translation)

    print("\n==== ClearLang Translation ====")
    print(f"Input:      {args.input_text}")
    print(f"Translation: {translation}")
    print(f"Token count: {len(tokens)}")
    print(f"Latency:     {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
