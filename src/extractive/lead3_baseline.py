from src.helpers.file_io import load_jsonl, save_jsonl
import os

def lead3(ex):
    ''' 
    Method 1: Lead 3 Baseline. Takes the first 3 sentences in an article, 
    and uses them as the summary for the article.
    '''
    return " ".join(ex["sentences"][:3])

def format_data(ex):
    '''
    Applies Lead3 to an example and returns an object containing the Lead3 summary, and 
    the highlight summary from the original article for ROUGE evaluations. 
    '''
    lead3_ex = {
        "lead3": lead3(ex),
        "highlight": ex["highlights"]
    }
    return lead3_ex

# load data
train_data = load_jsonl("data/processed/train.jsonl")
test_data = load_jsonl("data/processed/test.jsonl")
val_data = load_jsonl("data/processed/validation.jsonl")

# apply Lead3 and format for each dataset
test = [format_data(ex) for ex in test_data]
val = [format_data(ex) for ex in val_data] 

# given that Lead3 is a baseline that we will be comparing with performances with the other models, 
# we do not need to apply it to the train data, but for completeness I am still adding it.  
train = [format_data(ex) for ex in train_data]

# save outputs in new Lead3 directory
os.makedirs("data/lead3", exist_ok=True)
save_jsonl("data/lead3/train.jsonl", train)
save_jsonl("data/lead3/test.jsonl", test)
save_jsonl("data/lead3/validation.jsonl", val)

print("\nLead3 complete. Check data/lead3 for the processed files.")