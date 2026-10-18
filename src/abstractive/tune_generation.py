"""
Grid search over BART generation parameters on a validation subset.

Tuning is done on validation (never test) so the test-set numbers stay unbiased.
Each config is scored with ROUGE and timed, so the trade-off between quality and
decoding cost (e.g. beam size) can be reported.

Example:
    python -m src.abstractive.tune_generation --limit 100
"""
import argparse
import csv
import itertools
import os
import statistics
import time

from src.abstractive.bart import DEFAULT_GENERATION, DEFAULT_MODEL, BartSummarizer
from src.data.io import RESULTS_DIR, load_split
from src.eval.rouge_eval import ROUGE_TYPES, score_pairs

# Each key is varied on its own, holding the others at the default value, to keep
# the number of runs small. Use --full-grid to try every combination instead.
SEARCH_SPACE = {
    "num_beams": [1, 2, 4, 6],
    "length_penalty": [1.0, 2.0, 3.0],
    "min_length": [30, 56],
    "max_length": [100, 142],
    "no_repeat_ngram_size": [0, 3],
}


def one_at_a_time_configs():
    configs = [dict(DEFAULT_GENERATION)]
    for key, values in SEARCH_SPACE.items():
        for v in values:
            if v != DEFAULT_GENERATION[key]:
                configs.append({**DEFAULT_GENERATION, key: v})
    return configs


def full_grid_configs():
    keys = list(SEARCH_SPACE)
    return [
        {**DEFAULT_GENERATION, **dict(zip(keys, combo))}
        for combo in itertools.product(*(SEARCH_SPACE[k] for k in keys))
    ]


def main():
    p = argparse.ArgumentParser(description="Tune BART generation parameters on validation.")
    p.add_argument("--limit", type=int, default=100, help="number of validation articles")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--full-grid", action="store_true")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "bart", "generation_tuning.csv"))
    args = p.parse_args()

    examples = load_split("validation", args.limit)
    articles = [ex["article"] for ex in examples]
    references = [ex["highlights"] for ex in examples]
    summarizer = BartSummarizer(args.model, device=args.device)

    configs = full_grid_configs() if args.full_grid else one_at_a_time_configs()
    print(f"Trying {len(configs)} configs on {len(examples)} validation articles")

    rows = []
    for i, gen in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {gen}")
        t0 = time.perf_counter()
        summaries, _ = summarizer.summarize(articles, batch_size=args.batch_size, progress=False, **gen)
        elapsed = time.perf_counter() - t0
        scores = score_pairs(summaries, references)
        row = {**gen}
        for t in ROUGE_TYPES:
            row[t] = round(100 * statistics.mean(s[t] for s in scores), 2)
        row["avg_summary_words"] = round(statistics.mean(len(s.split()) for s in summaries), 1)
        row["s_per_article"] = round(elapsed / len(articles), 4)
        rows.append(row)
        print(f"    R1 {row['rouge1']}  R2 {row['rouge2']}  RLsum {row['rougeLsum']}  "
              f"{row['avg_summary_words']} words  {row['s_per_article']}s/article")

    rows.sort(key=lambda r: r["rouge2"], reverse=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    best = rows[0]
    print(f"\nWrote {args.out}")
    print("Best config by ROUGE-2:", {k: best[k] for k in DEFAULT_GENERATION})
    flags = " ".join(f"--{k.replace('_', '-')} {best[k]}" for k in DEFAULT_GENERATION if k != "early_stopping")
    print(f"Run it on test with:\n  python -m src.abstractive.bart --split test {flags}")


if __name__ == "__main__":
    main()
