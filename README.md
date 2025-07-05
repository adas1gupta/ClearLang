# ClearLang: Comparing Subword and Character-Level Translation with mT5 and ByT5

---

## Project Overview
**ClearLang** is a machine translation experiment comparing **subword-level tokenization** (mT5) and **character-level tokenization** (ByT5) for translating Russian to English.  
The project demonstrates how different tokenization strategies impact translation quality, speed, and output consistency.

---

## Motivation & Inspiration
I have been increasingly interested in how large language models handle character-level information, especially in morphologically rich languages like Russian.  
A recent paper (**ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models**) outlined the potential benefits of character-level tokenization for rare words and out-of-vocabulary tokens.  

This project explores those ideas by fine-tuning and comparing mT5 and ByT5 on a parallel Russian-English corpus.

---

## Dataset
- **Name:** IWSLT TED Talks Russian-English (`ru_to_en`)
- **Source:** Hugging Face Datasets
- **Size:** ~5,000 sentence pairs
- **Preprocessing:**
  - Lowercased text
  - Removed punctuation
  - Filtered sentences between 3 and 50 tokens

---

## Training Details
| Setting               | Value                  |
|-----------------------|------------------------|
| Models                | mT5-small, ByT5-small  |
| Epochs                | 3                      |


---

## Evaluation Results

| Model | BLEU Score | Example Output | Token Count | Latency |
|-------|------------|-----------------|-------------|---------|
| **mT5**  | 4.41       | "it's a wonderful world" | 8           | ~0.50s |
| **ByT5** | 0.34       | "and they have been able to talk about that they have been able" | 63          | ~1.53s |

**Insights:**
- **mT5** performed significantly better in BLEU score with less hallucination.
- **ByT5** generated much longer and often semantically inconsistent outputs on short inputs, highlighting that character-level models typically require more epochs and data to converge.
- Latency was higher with ByT5 due to longer output sequences.

---

## Example CLI Usage

**Translate a sentence with mT5:**

```bash
python src/translate_cli.py --model mt5 --input_text "Привет мир"
```

**Output:**
```
Input: Привет мир
Translation: it's a wonderful world
Token count: 8
Latency: 0.50 seconds
```

### Translate a sentence with ByT5:

```bash
python src/translate_cli.py --model byt5 --input_text "Привет мир"
```

**Output:**
```
Input: Привет мир
Translation: and they have been able to talk about that they have been able
Token count: 63
Latency: 1.53 seconds
```

## 🔍 Insights & Observations

* **Subword tokenization (mT5)** was more stable and efficient on small datasets
* **Character-level tokenization (ByT5)** requires longer training and larger datasets to reach comparable accuracy
* The experiment demonstrates the trade-offs between robustness to out-of-vocabulary tokens and convergence speed

## 🌱 Future Improvements

If I were to extend this project, I would:

* Train for 5–10 epochs to better optimize ByT5
* Benchmark inference on GPU
* Package the project in Docker for reproducibility
* Deploy a REST API behind Kubernetes to scale translations

## 🧠 References

* [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/abs/2010.11934)
* [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/clearlang-translation.git
cd clearlang-translation

# Install dependencies
pip install -r requirements.txt

# Try your own sentence
python src/translate_cli.py --model mt5 --input_text "[Your Russian sentence]"
```