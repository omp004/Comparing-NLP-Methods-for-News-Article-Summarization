# Factual-consistency annotation guidelines

We are checking whether each summary says only things that the **article** supports.
The reference summary is shown for context only. A summary can be faithful and still
differ from the reference.

## Setup

1. Run `python -m src.eval.error_analysis sample results/*/test_predictions.jsonl --n 50`.
2. Each annotator copies `results/error_analysis/annotation_sheet.csv` to
   `results/error_analysis/annotations_<name>.csv` and fills it in on their own.
3. Don't open `annotation_key.csv` until everyone is done. Summaries are shown as
   System A/B/C, in a different order for each article, so we can't tell which
   method wrote them.
4. Run `python -m src.eval.error_analysis report results/error_analysis/annotations_*.csv`.

## How to label

Read the article first, then judge each summary sentence by sentence. Put `1` in a
column if the summary has at least one error of that type, `0` otherwise. A summary
can have more than one type. `auto_flags` lists numbers and names that never appear
in the article. These are hints only: confirm each one yourself (for example,
"Obama" for "Barack Obama" is fine).

| column           | mark 1 when the summary…                                                      | example                                                  |
|------------------|-------------------------------------------------------------------------------|----------------------------------------------------------|
| `entity_error`   | names the wrong person, place or organization, or swaps two of them           | article by Sanjay Gupta → "CNN's John Sutter says…"     |
| `number_error`   | gets a number, date, age, amount or count wrong                               | "53% favor" → "35% favor"                                |
| `relation_error` | has the right entities but the wrong relation: who did what, who said a quote | police arrested X → "X arrested police"                  |
| `fabricated`     | adds content the article doesn't contain and that isn't common knowledge      | invents a motive, an outcome or a location               |
| `contradiction`  | states the opposite of the article, including a dropped or added negation     | "was not charged" → "was charged"                        |
| `incoherent`     | is ungrammatical, cut off or doesn't make sense (not a factual error)         | sentence ends mid-clause                                 |

`consistent`: `1` if all five factual-error columns are `0`, otherwise `0`. If you
leave it blank, the report fills it in from the other columns.

`notes`: quote the bad span, e.g. `"John Sutter" should be Sanjay Gupta`. We use
these quotes as examples in the report.

### Edge cases

- **Extractive methods (Lead-3, TextRank)** copy whole sentences, so they rarely make
  factual errors. They can still mislead when a pronoun ("he") loses its referent once
  the sentence is taken out of context. Mark that as `relation_error` and say so in
  `notes`.
- **Paraphrases** that keep the meaning are fine ("more than 3,000" for "3,124").
- **Opinions** written in the article's own voice: if the summary turns a quoted opinion
  into a stated fact, mark `relation_error` (wrong attribution).
- If you aren't sure, go with the label you lean toward and explain in `notes`.
