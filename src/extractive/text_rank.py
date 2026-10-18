import numpy as np
import networkx as nx
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.helpers.file_io import load_jsonl, save_jsonl

# load data
train_data = load_jsonl("data/processed/train.jsonl")
test_data = load_jsonl("data/processed/test.jsonl")
val_data = load_jsonl("data/processed/validation.jsonl")

vectorizer = TfidfVectorizer()

def text_rank(ex):
    sentences = ex["sentences"]

    # convert sentences into tf_idf vectors
    vectorizer = TfidfVectorizer()
    tf_idf_vecs = vectorizer.fit_transform(sentences)

    # compute cosine similarity for each vector
    similarity_matrix = cosine_similarity(tf_idf_vecs)

    # ensures that there is no self-similarity
    np.fill_diagonal(similarity_matrix, 0)

    # create graph from similarity matric and then apply page rank to it
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # get the top 3 highest scores 
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]

    # get the originals entences from top 3 and create a summary 
    summary = " ".join(sentences[i] for i in sorted(top3))
    return summary

def format_data(ex):
    '''
    Applies TextRank to an example and returns an object containing the TextRank summary, and 
    the highlight summary from the original article for ROUGE evaluations. 
    '''
    text_rank_ex = {
        "text_rank": text_rank(ex),
        "highlight": ex["highlights"]
    }
    return text_rank_ex

# apply Lead3 and format for each dataset
train = [format_data(ex) for ex in train_data]
test = [format_data(ex) for ex in test_data]
val = [format_data(ex) for ex in val_data]   

# save outputs in new Lead3 directory
os.makedirs("data/text_rank", exist_ok=True)
save_jsonl("data/text_rank/train.jsonl", train)
save_jsonl("data/text_rank/test.jsonl", test)
save_jsonl("data/text_rank/validation.jsonl", val)

print("\TextRank complete. Check data/text_rank for the processed files.")