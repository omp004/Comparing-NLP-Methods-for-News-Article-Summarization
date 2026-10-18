"""
Abstractive summarization with a pretrained BART model.

We use facebook/bart-large-cnn off the shelf (BART-large fine-tuned on the
CNN/DailyMail training set), so no training is done here: the train split is
not needed. The script summarizes a processed split, records per-article runtime,
and writes predictions in the shared format from src/data/io.py.

Example:
    python -m src.abstractive.bart --split test --limit 200
    python -m src.abstractive.bart --split test --num-beams 4 --length-penalty 2.0
"""
import argparse
import os
import time

# torch/transformers must be imported before nltk: the other order segfaults on
# macOS (conflicting native libraries) once the model is loaded.
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize

from src.data.io import RESULTS_DIR, load_split, save_json, save_predictions

DEFAULT_MODEL = "facebook/bart-large-cnn"

# Defaults match the generation config bart-large-cnn ships with, which is what
# was used to get its reported CNN/DM scores.
DEFAULT_GENERATION = {
    "num_beams": 4,
    "max_length": 142,
    "min_length": 56,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

# BART's position embeddings only cover 1024 tokens, so longer articles get truncated.
MAX_INPUT_TOKENS = 1024


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def sync(device):
    """Wait for queued GPU work so wall-clock timings are accurate."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class BartSummarizer:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, fp16=False):
        self.model_name = model_name
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # fp16 is only reliable on CUDA; MPS/CPU stay in fp32.
        if fp16 and self.device == "cuda":
            self.model = self.model.half()
        self.model.to(self.device).eval()
        ensure_punkt()

    def count_truncated(self, articles):
        """How many articles are longer than the 1024-token input limit."""
        lengths = [len(ids) for ids in self.tokenizer(articles, truncation=False)["input_ids"]]
        return sum(n > MAX_INPUT_TOKENS for n in lengths), lengths

    @torch.no_grad()
    def summarize_batch(self, articles, **generation_kwargs):
        gen = {**DEFAULT_GENERATION, **generation_kwargs}
        inputs = self.tokenizer(
            articles,
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        output_ids = self.model.generate(**inputs, **gen)
        summaries = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # Put one sentence per line to match the reference format (needed for ROUGE-Lsum).
        return ["\n".join(sent_tokenize(s.strip())) for s in summaries]

    def summarize(self, articles, batch_size=8, progress=True, **generation_kwargs):
        """
        Summarize a list of articles. Returns (summaries, per_article_seconds).

        Articles are processed in length-sorted batches so each batch needs less
        padding; results are put back in the original order. Per-article time is
        the batch's wall-clock time split evenly across the batch.
        """
        order = sorted(range(len(articles)), key=lambda i: len(articles[i]), reverse=True)
        summaries = [None] * len(articles)
        times = [None] * len(articles)

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            sync(self.device)
            t0 = time.perf_counter()
            batch_out = self.summarize_batch([articles[i] for i in idx], **generation_kwargs)
            sync(self.device)
            elapsed = time.perf_counter() - t0
            for i, s in zip(idx, batch_out):
                summaries[i] = s
                times[i] = elapsed / len(idx)
            if progress:
                done = min(start + batch_size, len(order))
                print(f"  {done}/{len(order)} articles ({elapsed:.1f}s for last batch)", flush=True)

        return summaries, times


def parse_args():
    p = argparse.ArgumentParser(description="Summarize CNN/DM articles with pretrained BART.")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--limit", type=int, default=None, help="only use the first N articles")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="e.g. sshleifer/distilbart-cnn-12-6 for a faster, smaller model")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected by default)")
    p.add_argument("--fp16", action="store_true", help="half precision (CUDA only)")
    p.add_argument("--out", default=None, help="predictions path (default results/bart/{split}_predictions.jsonl)")
    p.add_argument("--no-eval", action="store_true", help="skip computing ROUGE at the end")

    g = p.add_argument_group("generation")
    g.add_argument("--num-beams", type=int, default=DEFAULT_GENERATION["num_beams"])
    g.add_argument("--max-length", type=int, default=DEFAULT_GENERATION["max_length"])
    g.add_argument("--min-length", type=int, default=DEFAULT_GENERATION["min_length"])
    g.add_argument("--length-penalty", type=float, default=DEFAULT_GENERATION["length_penalty"])
    g.add_argument("--no-repeat-ngram-size", type=int, default=DEFAULT_GENERATION["no_repeat_ngram_size"])
    return p.parse_args()


def main():
    args = parse_args()
    generation = {
        "num_beams": args.num_beams,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "length_penalty": args.length_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": True,
    }
    out_path = args.out or os.path.join(RESULTS_DIR, "bart", f"{args.split}_predictions.jsonl")

    examples = load_split(args.split, args.limit)
    articles = [ex["article"] for ex in examples]
    print(f"Loaded {len(examples)} articles from the {args.split} split")

    t0 = time.perf_counter()
    summarizer = BartSummarizer(args.model, device=args.device, fp16=args.fp16)
    load_time = time.perf_counter() - t0
    print(f"Loaded {args.model} on {summarizer.device} in {load_time:.1f}s")

    n_truncated, token_lengths = summarizer.count_truncated(articles)
    print(f"{n_truncated}/{len(articles)} articles exceed {MAX_INPUT_TOKENS} tokens and will be truncated")

    t0 = time.perf_counter()
    summaries, times = summarizer.summarize(articles, batch_size=args.batch_size, **generation)
    total_time = time.perf_counter() - t0

    records = [
        {
            "id": ex["id"],
            "method": "bart",
            "summary": summary,
            "reference": ex["highlights"],
            "time_s": round(t, 4),
            "input_tokens": n_tokens,
            "truncated": n_tokens > MAX_INPUT_TOKENS,
        }
        for ex, summary, t, n_tokens in zip(examples, summaries, times, token_lengths)
    ]
    save_predictions(records, out_path)

    meta = {
        "method": "bart",
        "model": args.model,
        "device": summarizer.device,
        "fp16": args.fp16 and summarizer.device == "cuda",
        "split": args.split,
        "n_articles": len(examples),
        "batch_size": args.batch_size,
        "generation": generation,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "n_truncated": n_truncated,
        "model_load_s": round(load_time, 2),
        "total_generation_s": round(total_time, 2),
        "mean_s_per_article": round(total_time / len(examples), 4),
        "articles_per_s": round(len(examples) / total_time, 3),
        "n_parameters": sum(p.numel() for p in summarizer.model.parameters()),
    }
    meta_path = out_path.replace("_predictions.jsonl", "_run_meta.json")
    save_json(meta, meta_path)
    print(f"Wrote {out_path} and {meta_path}")
    print(f"Generation took {total_time:.1f}s ({meta['mean_s_per_article']:.3f}s per article)")

    if not args.no_eval:
        from src.eval.rouge_eval import evaluate_file, print_table
        print_table([evaluate_file(out_path)])


if __name__ == "__main__":
    main()
