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

## Results

TBD

## Tools
Python, Hugging Face datasets and transformers, PyTorch, scikit-learn, NetworkX-learn
