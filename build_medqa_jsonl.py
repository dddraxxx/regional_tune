import json
import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'dev': 'data/dev-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/openlifescienceai/medqa/" + splits["test"])

# Extract the 'data' column and write each dict to jsonl
with open("data/test/medqa_test.jsonl", "w", encoding="utf-8") as f:
    for data_dict in df["data"]:
        f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
