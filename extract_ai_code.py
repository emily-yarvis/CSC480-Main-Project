import json

# Loads AI code JSON file
with open("code_alpaca_20k.json", "r") as f:
    data = json.load(f)

# Write JSONL with only the output and label: 1 for AI-generated
with open("ai_code_20022.jsonl", "w") as f:
    for entry in data:
        code = entry.get("output", "").strip()
        if code:  # skip empty outputs
            json.dump({"text": code, "label": 1}, f)
            f.write("\n")
