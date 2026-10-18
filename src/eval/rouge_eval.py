"""
ROUGE evaluation for any method's prediction file (format described in src/data/io.py).

Reports ROUGE-1/2/L F1 plus ROUGE-Lsum (the summary-level LCS that splits on
newlines, which is what most CNN/DM papers report as "ROUGE-L"), with 95%
bootstrap confidence intervals, average summary length and runtime.

Example:
    python -m src.eval.rouge_eval results/bart/test_predictions.jsonl results/lead3/test_predictions.jsonl
"""
import argparse
import csv
import os
import random
import statistics

from rouge_score import rouge_scorer

from src.data.io import RESULTS_DIR, load_predictions, save_predictions

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def score_pairs(predictions, references):
    """Per-example ROUGE F1 scores as a list of {rouge_type: f1} dicts."""
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    return [
        {t: s.fmeasure for t, s in scorer.score(ref, pred).items()}
        for pred, ref in zip(predictions, references)
    ]


def bootstrap_ci(values, n_resamples=1000, seed=0):
    """95% percentile bootstrap interval for the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


def evaluate_file(path, save_per_example=True):
    records = load_predictions(path)
    scores = score_pairs([r["summary"] for r in records], [r["reference"] for r in records])

    if save_per_example:
        per_example = [{"id": r["id"], **s} for r, s in zip(records, scores)]
        save_predictions(per_example, path.replace("_predictions.jsonl", "_rouge_per_example.jsonl"))

    result = {"method": records[0].get("method", os.path.basename(path)), "n": len(records)}
    for t in ROUGE_TYPES:
        values = [s[t] for s in scores]
        lo, hi = bootstrap_ci(values)
        result[t] = 100 * statistics.mean(values)
        result[f"{t}_ci"] = (100 * lo, 100 * hi)
    result["avg_summary_words"] = statistics.mean(len(r["summary"].split()) for r in records)
    result["avg_reference_words"] = statistics.mean(len(r["reference"].split()) for r in records)
    times = [r["time_s"] for r in records if r.get("time_s") is not None]
    result["mean_s_per_article"] = statistics.mean(times) if times else None
    return result


def print_table(results):
    header = f"{'method':<12}{'n':>6}" + "".join(f"{t:>20}" for t in ROUGE_TYPES) + f"{'words':>8}{'s/art':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['method']:<12}{r['n']:>6}"
        for t in ROUGE_TYPES:
            lo, hi = r[f"{t}_ci"]
            row += f"{r[t]:>8.2f} [{lo:5.2f},{hi:5.2f}]"
        s = r["mean_s_per_article"]
        row += f"{r['avg_summary_words']:>8.1f}" + (f"{s:>9.4f}" if s is not None else f"{'-':>9}")
        print(row)


def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["method", "n"]
    for t in ROUGE_TYPES:
        fields += [t, f"{t}_ci_low", f"{t}_ci_high"]
    fields += ["avg_summary_words", "avg_reference_words", "mean_s_per_article"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if not k.endswith("_ci")}
            for t in ROUGE_TYPES:
                row[f"{t}_ci_low"], row[f"{t}_ci_high"] = r[f"{t}_ci"]
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Compute ROUGE for one or more prediction files.")
    p.add_argument("prediction_files", nargs="+")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "rouge_summary.csv"))
    args = p.parse_args()

    results = [evaluate_file(path) for path in args.prediction_files]
    print_table(results)
    save_csv(results, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
