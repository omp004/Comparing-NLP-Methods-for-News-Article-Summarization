import json
from rouge_score import rouge_scorer

def loadSummaries(filePath):
    summaries = []

    with open(filePath, "r") as file:
        for line in file:
            summaries.append(json.loads(line))

    return summaries

def evaluateSummaries(summaries, summaryKey):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    rouge1 = 0
    rouge2 = 0
    rougeL = 0

    for summary in summaries:
        scores = scorer.score(summary["highlight"], summary[summaryKey])

        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure

    total = len(summaries)

    return rouge1 / total, rouge2 / total, rougeL / total

if __name__ == "__main__":
    lead3Summaries = loadSummaries("data/lead3/test.jsonl")
    textRankSummaries = loadSummaries("data/text_rank/test.jsonl")

    lead3Scores = evaluateSummaries(lead3Summaries, "lead3")
    textRankScores = evaluateSummaries(textRankSummaries, "text_rank")

    print("Lead-3 ROUGE Scores:", lead3Scores)
    print("TextRank ROUGE Scores:", textRankScores)