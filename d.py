import json

with open("training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))