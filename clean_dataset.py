import json

CUT_MARKERS = [
    "\nUser:",
    "\nHuman:",
    "\nAI:",
    "\nAssistant:",
    "\nAnswerer:",
    "\nReviewer:",
    "\nComment:",
    "\nQuestion:"
]

print("Loading dataset...")

with open("training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
contaminated = 0

for item in data:

    output = item["output"]

    original_output = output

    for marker in CUT_MARKERS:
        if marker in output:
            output = output.split(marker)[0]

    output = output.strip()

    if output != original_output:
        contaminated += 1

    item["output"] = output
    cleaned.append(item)

with open("training_data_clean.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print(f"Total examples: {len(data)}")
print(f"Contaminated examples fixed: {contaminated}")
print("Saved: training_data_clean.json")