"""
Shared helpers for loading the processed CNN/DailyMail splits and for reading /
writing prediction files.

Every summarization method (Lead-3, TextRank, BART) should write its output with
save_predictions() so that the ROUGE and error-analysis scripts can compare them.
One JSON object per line:

    {"id": ..., "method": ..., "summary": ..., "reference": ..., "time_s": ...}

- summary / reference: sentences separated by "\n" (this is what ROUGE-Lsum uses)
- time_s: seconds spent producing that summary
"""
import json
import os

DATA_DIR = "data/processed"
RESULTS_DIR = "results"


def load_split(split, limit=None):
    """Load data/processed/{split}.jsonl as a list of dicts (article, highlights, id, sentences)."""
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    examples = []
    with open(path) as f:
        for line in f:
            if limit is not None and len(examples) >= limit:
                break
            examples.append(json.loads(line))
    return examples


def load_articles_by_id(split):
    """Map article id -> processed example, used to join predictions back to their source article."""
    return {ex["id"]: ex for ex in load_split(split)}


def save_predictions(records, path):
    """Write prediction records (see module docstring) to a jsonl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_predictions(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
