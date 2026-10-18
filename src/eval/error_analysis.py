"""
Factual-consistency / error analysis across summarization methods.

Three subcommands:

  auto    Automatic checks for every summary: how abstractive it is (novel n-grams
          not found in the article) and simple hallucination flags (numbers and
          capitalized names in the summary that never appear in the article).
          Extractive methods should score ~0 on all of these, so the numbers mostly
          tell us where to look in BART output.

  sample  Draw the same random set of articles for every method and write a blinded
          annotation sheet (methods shown as System A/B/C in random order) plus a
          key file for un-blinding. See docs/annotation_guidelines.md.

  report  Read one or more filled-in annotation sheets, un-blind them with the key,
          and report error rates per method (and Cohen's kappa if two annotators
          labeled the same sheet).

Example:
    python -m src.eval.error_analysis auto   results/*/test_predictions.jsonl
    python -m src.eval.error_analysis sample results/*/test_predictions.jsonl --n 50
    python -m src.eval.error_analysis report results/error_analysis/annotations_yash.csv
"""
import argparse
import csv
import os
import random
import re
import statistics
from collections import defaultdict

from src.data.io import RESULTS_DIR, load_articles_by_id, load_predictions, save_predictions

OUT_DIR = os.path.join(RESULTS_DIR, "error_analysis")

# Columns annotators fill in with 0/1 (see docs/annotation_guidelines.md).
ERROR_LABELS = [
    "entity_error",       # wrong person / place / organization, or names swapped
    "number_error",       # wrong number, date, quantity, age, money amount
    "relation_error",     # right entities, wrong relation: who did what to whom, wrong attribution of a quote
    "fabricated",         # content that is not in the article at all (and not general knowledge)
    "contradiction",      # directly contradicts the article (incl. dropped negation)
]
OTHER_LABELS = ["incoherent"]  # not a factual error, but unreadable / broken summaries
ALL_LABELS = ERROR_LABELS + OTHER_LABELS

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "first": "1", "second": "2", "third": "3",
}
STOP_CAPS = {
    "The", "A", "An", "He", "She", "It", "They", "We", "I", "His", "Her", "Their", "This",
    "That", "These", "Those", "In", "On", "At", "But", "And", "Mr", "Mrs", "Ms", "Dr",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}


# ---------------------------------------------------------------------------
# automatic checks
# ---------------------------------------------------------------------------

def tokens(text):
    return re.findall(r"\w+", text.lower())


def ngrams(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def novel_ngram_ratio(summary, article, n):
    summ = ngrams(tokens(summary), n)
    if not summ:
        return 0.0
    return len(summ - ngrams(tokens(article), n)) / len(summ)


def extract_numbers(text):
    """Numbers in the text, normalized so '1,000' == '1000' and 'three' == '3'."""
    nums = {m.replace(",", "") for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    nums |= {NUMBER_WORDS[w] for w in tokens(text) if w in NUMBER_WORDS}
    return nums


def capitalized_words(text):
    """Capitalized words that are not the first word of a sentence (a rough proxy for names)."""
    words = set()
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        for w in re.findall(r"[A-Za-z][\w'-]*", sent)[1:]:
            if w[0].isupper() and w not in STOP_CAPS:
                words.add(w)
    return words


def auto_checks(summary, article):
    article_tokens = set(tokens(article))
    return {
        "novel_1gram": novel_ngram_ratio(summary, article, 1),
        "novel_2gram": novel_ngram_ratio(summary, article, 2),
        "novel_3gram": novel_ngram_ratio(summary, article, 3),
        "unsupported_numbers": sorted(extract_numbers(summary) - extract_numbers(article)),
        "unsupported_names": sorted(
            w for w in capitalized_words(summary)
            if not all(t in article_tokens for t in tokens(w))
        ),
    }


def flag_string(checks):
    parts = []
    if checks["unsupported_numbers"]:
        parts.append("numbers not in article: " + ", ".join(checks["unsupported_numbers"]))
    if checks["unsupported_names"]:
        parts.append("names not in article: " + ", ".join(checks["unsupported_names"]))
    return "; ".join(parts)


def cmd_auto(args):
    articles = load_articles_by_id(args.split)
    rows = []
    for path in args.prediction_files:
        records = load_predictions(path)
        method = records[0]["method"]
        per_example = []
        for r in records:
            checks = auto_checks(r["summary"], articles[r["id"]]["article"])
            per_example.append({"id": r["id"], **checks})
        save_predictions(per_example, os.path.join(OUT_DIR, f"{method}_auto_checks.jsonl"))

        n = len(per_example)
        rows.append({
            "method": method,
            "n": n,
            "novel_1gram_%": 100 * statistics.mean(e["novel_1gram"] for e in per_example),
            "novel_2gram_%": 100 * statistics.mean(e["novel_2gram"] for e in per_example),
            "novel_3gram_%": 100 * statistics.mean(e["novel_3gram"] for e in per_example),
            "has_unsupported_number_%": 100 * sum(bool(e["unsupported_numbers"]) for e in per_example) / n,
            "has_unsupported_name_%": 100 * sum(bool(e["unsupported_names"]) for e in per_example) / n,
        })

    write_csv(rows, os.path.join(OUT_DIR, "auto_checks_summary.csv"))
    print_rows(rows)
    print(f"\nWrote {OUT_DIR}/auto_checks_summary.csv and per-example *_auto_checks.jsonl files")


# ---------------------------------------------------------------------------
# blinded annotation sheet
# ---------------------------------------------------------------------------

def cmd_sample(args):
    rng = random.Random(args.seed)
    articles = load_articles_by_id(args.split)
    by_method = {}
    for path in args.prediction_files:
        records = load_predictions(path)
        by_method[records[0]["method"]] = {r["id"]: r for r in records}

    common_ids = sorted(set.intersection(*(set(recs) for recs in by_method.values())))
    if len(common_ids) < args.n:
        raise SystemExit(f"Only {len(common_ids)} articles are shared by all methods, asked for {args.n}")
    sample_ids = rng.sample(common_ids, args.n)
    methods = sorted(by_method)
    system_names = [f"System {chr(ord('A') + i)}" for i in range(len(methods))]

    sheet, key = [], []
    for item, article_id in enumerate(sample_ids, 1):
        shuffled = methods[:]
        rng.shuffle(shuffled)  # different order per article so systems can't be recognized by position
        article = articles[article_id]["article"]
        for system, method in zip(system_names, shuffled):
            summary = by_method[method][article_id]["summary"]
            sheet.append({
                "item": item,
                "article_id": article_id,
                "system": system,
                "article": article,
                "reference": articles[article_id]["highlights"],
                "summary": summary,
                "auto_flags": flag_string(auto_checks(summary, article)),
                **{label: "" for label in ALL_LABELS},
                "consistent": "",
                "notes": "",
            })
            key.append({"article_id": article_id, "system": system, "method": method})

    os.makedirs(OUT_DIR, exist_ok=True)
    sheet_path = os.path.join(OUT_DIR, "annotation_sheet.csv")
    key_path = os.path.join(OUT_DIR, "annotation_key.csv")
    write_csv(sheet, sheet_path)
    write_csv(key, key_path)
    print(f"Wrote {len(sheet)} rows ({args.n} articles x {len(methods)} methods: {', '.join(methods)})")
    print(f"  sheet: {sheet_path}  (copy one per annotator, e.g. annotations_<name>.csv)")
    print(f"  key:   {key_path}  (don't open until annotation is done)")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def read_annotations(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for label in ALL_LABELS + ["consistent"]:
            value = r.get(label, "").strip()
            r[label] = int(value) if value in ("0", "1") else None
        # If "consistent" was left blank, derive it from the error columns.
        if r["consistent"] is None and all(r[l] is not None for l in ERROR_LABELS):
            r["consistent"] = int(not any(r[l] for l in ERROR_LABELS))
    return rows


def cohen_kappa(a, b):
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    p_a, p_b = sum(a) / n, sum(b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def cmd_report(args):
    with open(args.key, newline="") as f:
        key = {(r["article_id"], r["system"]): r["method"] for r in csv.DictReader(f)}
    sheets = [read_annotations(path) for path in args.annotation_files]

    # Per-method error rates, pooling all annotators.
    by_method = defaultdict(list)
    for sheet in sheets:
        for r in sheet:
            by_method[key[(r["article_id"], r["system"])]].append(r)

    rows = []
    for method, items in sorted(by_method.items()):
        row = {"method": method, "n_judgments": len(items)}
        for label in ["consistent"] + ALL_LABELS:
            labeled = [r[label] for r in items if r[label] is not None]
            row[f"{label}_%"] = 100 * sum(labeled) / len(labeled) if labeled else None
        row["n_unlabeled"] = sum(r["consistent"] is None for r in items)
        rows.append(row)
    write_csv(rows, os.path.join(OUT_DIR, "error_report.csv"))
    print_rows(rows)

    # Inter-annotator agreement when two people labeled the same sheet.
    if len(sheets) == 2:
        index_b = {(r["article_id"], r["system"]): r for r in sheets[1]}
        print("\nCohen's kappa between the two annotators:")
        for label in ["consistent"] + ERROR_LABELS:
            pairs = [
                (r[label], index_b[k][label])
                for r in sheets[0]
                if (k := (r["article_id"], r["system"])) in index_b
                and r[label] is not None and index_b[k][label] is not None
            ]
            if pairs:
                a, b = zip(*pairs)
                print(f"  {label:<16} kappa={cohen_kappa(a, b):.3f}  (n={len(pairs)})")
    print(f"\nWrote {OUT_DIR}/error_report.csv")


# ---------------------------------------------------------------------------

def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_rows(rows):
    for r in rows:
        print("  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def main():
    p = argparse.ArgumentParser(description="Factual-consistency error analysis.")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("auto", help="automatic abstractiveness / hallucination checks")
    a.add_argument("prediction_files", nargs="+")
    a.add_argument("--split", default="test")
    a.set_defaults(func=cmd_auto)

    s = sub.add_parser("sample", help="write a blinded annotation sheet")
    s.add_argument("prediction_files", nargs="+")
    s.add_argument("--split", default="test")
    s.add_argument("--n", type=int, default=50, help="number of articles to sample")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_sample)

    r = sub.add_parser("report", help="summarize filled-in annotation sheets")
    r.add_argument("annotation_files", nargs="+")
    r.add_argument("--key", default=os.path.join(OUT_DIR, "annotation_key.csv"))
    r.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
