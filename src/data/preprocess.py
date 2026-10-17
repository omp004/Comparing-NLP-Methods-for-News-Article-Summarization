from datasets import load_dataset
import re
import nltk
import os
from nltk.tokenize import sent_tokenize

# Download the punkt tokenizer to check for validation
# on where a sentence ends and begins. Useful when verifying with 
# training and testing data.
nltk.download("punkt_tab")
# list of words that can be ended on when a line is broken in the middle of a sentence
BROKEN_LABELS = ["and", "or", "but", "of", "the", "a", "to", "in", "on", "for", "with"]

def process_article(text):
    """
    Clean an article by:
    - removing newlines and extra spaces and converting to single space
    - remove " ." between the paragraphs in Daily Mail articles
    """
    # Remove newlines and extra spaces and fix periods using built in regex
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r" \.(?= )", "", text)
    return text.strip()

def process_summaries(text):
    """
    - Clean the summaries for an article by removing " ." at the end of 
    the reference summary's line, and rejoin sentences that were broken
    by two lines in raw data:
    - Re-attaches the summary together if starts with lowercase letter or 
    if the previous line ended with a word within the BROKEN_LABELS list.
    - Add a single period back into the end of each sentence within the summary
    and return the summary as a string.
    """
    lines = []
    for line in text.split("\n"):
        line = line.replace(" .", "").strip()
        if line == "":
            continue

        #Check if line ends with a broken label or is lowercase
        # if so, then make second half continue from the first half
        if lines and (line[0].islower() or lines[-1].split()[-1] in BROKEN_LABELS):
            lines[-1] += " " + line
        else:
            lines.append(line)
    return "\n".join(line + "." for line in lines)


def processed_ex(example):
    """
    Process an example from the dataset, clean the article 
    and the summaries and add sentences field that contains
    the article split into sentences using nltk's sent_tokenize function.
    """
    example["article"] = process_article(example["article"])
    example["highlights"] = process_summaries(example["highlights"])
    example["sentences"] = sent_tokenize(example["article"])
    return example

# Load dataset, shuffle and select a subset for training, validation, and testing
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
train = dataset["train"].shuffle(seed=42).select(range(20000))
validation = dataset["validation"].shuffle(seed=42).select(range(1000))
test = dataset["test"].shuffle(seed=42).select(range(2000))

# set a regular article to compare before and after processing
reg_article = train[0]["article"]

train = train.map(processed_ex)
validation = validation.map(processed_ex)
test = test.map(processed_ex)

# Check the pre processing and sentence splitting worked on the article

# print("\nBEFORE:", reg_article[:300])
# print("\nAFTER: ", train[0]["article"][:300])
# print("\nHIGHLIGHTS:", train[0]["highlights"])

# print("\nNumber of sentences:", len(train[0]["sentences"]))
# print("\nFIRST 5 SENTENCES:")
# for s in train[0]["sentences"][:5]:
#     print("-", s)

# Save each processed subset to a file (one article per line)
# So others can load the data without needing to process it again.
os.makedirs("data/processed", exist_ok=True)
train.to_json("data/processed/train.jsonl")
validation.to_json("data/processed/validation.jsonl")
test.to_json("data/processed/test.jsonl")

print("\nData processing complete. Check data/data/processed for the processed files.")