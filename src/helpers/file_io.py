import json 
def save_jsonl(filepath, data):
    '''
    Helper function to save data into a jsonl file.
    '''
    with open(filepath, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

def load_jsonl(filepath):
    '''
    Helper function to load each preprocessed jsonl file. 
    '''
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]