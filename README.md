# Comparing-NLP-Methods-for-News-Article-Summarization

The final project compares three approaches to automatic summarization of news articles: a rule-based baseline, a statistical ranking method, and a pretrained summarization model.

**Team:** Om Patel, Advait Dharampal, Maya Borkar, Yash Patel

## Task & Data

We will generate short summaries of news articles and compare them to human-written ones.

- **Dataset:** [CNN/DailyMail 3.0.0] (https://huggingface.co/datasets/abisee/cnn_dailymail/tree/main/3.0.0)
- **Version:** 1.0.0: Contains ~287k train / 13k validation / 11k test articles

## Methodology

Lead-3 Rule-Based Baseline: a simple extractive baseline that returns the first three sentences of each article as its summary

TextRank: sentences will be represented using TF-IDF features, and cosine similarity will be used to measure similarity between sentences, and TextRank ranks them to find the most important ones

BART: generating new sentences using a pretrained summarization model 

## Evaluation

- **ROUGE-1, ROUGE-2, ROUGE-L**: overlap with the reference summaries
- **Runtime / computational cost** per method
- **Factual consistency**: manual review of a sample of summaries for wrong names, numbers, or claims

## Setup

```bash
git clone https://github.com/omp004/Comparing-NLP-Methods-for-News-Article-Summarization.git
cd Comparing-NLP-Methods-for-News-Article-Summarization
pip install -r requirements.txt
```

## Running

Run all commands from the repo root. They assume `data/processed/` exists
(`python src/data/preprocess.py`).

### Prediction format (all methods)

Every method writes `results/<method>/<split>_predictions.jsonl` using
`save_predictions()` from `src/data/io.py`, so the evaluation scripts work on any method:

```json
{"id": "...", "method": "lead3", "summary": "sent 1\nsent 2\nsent 3", "reference": "...", "time_s": 0.0004}
```

Put one sentence per line in `summary` (needed for ROUGE-Lsum), and set `time_s` to
the seconds spent on that article.

### BART

We use `facebook/bart-large-cnn` (BART-large fine-tuned on CNN/DM) as is, with no extra training.

```bash
# 1. tune generation parameters on validation (beam size, length penalty, min/max length, n-gram blocking)
python -m src.abstractive.tune_generation --limit 100

# 2. summarize the test set (prints ROUGE when done); flags override the default generation settings
python -m src.abstractive.bart --split test --limit 500 --batch-size 8
python -m src.abstractive.bart --split test --model sshleifer/distilbart-cnn-12-6   # faster, smaller
```

Outputs: `results/bart/test_predictions.jsonl` and `results/bart/test_run_meta.json`
(model, device, generation settings, how many articles were truncated, and timing).
Articles longer than 1024 tokens are truncated, since that is BART's input limit.
Speed is about 8–9 s per article on an M1 (MPS) with 4 beams, so the full 2,000-article
test set takes about 5 hours. Use `--limit`, or run it on a CUDA GPU (`--fp16`).

### Evaluation

```bash
# ROUGE-1/2/L/Lsum with 95% bootstrap CIs, summary length, and runtime → results/rouge_summary.csv
python -m src.eval.rouge_eval results/*/test_predictions.jsonl

# automatic checks: novel n-grams (how abstractive) and numbers/names missing from the article
python -m src.eval.error_analysis auto results/*/test_predictions.jsonl

# manual factual-consistency review (see docs/annotation_guidelines.md)
python -m src.eval.error_analysis sample results/*/test_predictions.jsonl --n 50
python -m src.eval.error_analysis report results/error_analysis/annotations_*.csv
```

## Results

TBD

## Tools
Python, Hugging Face datasets and transformers, PyTorch, scikit-learn, NetworkX-learn
