import pandas as pd

df = pd.read_parquet("train-00000-of-00003.parquet")

human_code = df["code"].head(20016)

human_labeled = pd.DataFrame({
    "text": human_code,
    "label": 0
})

human_labeled.to_json("human_code_20357.jsonl", orient="records", lines=True)