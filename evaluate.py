import json

with open("results_finetuned.json", "r", encoding="utf-8") as f:
    results = json.load(f)

score = 0

for item in results:

    expected = item["expected"].lower().strip()
    answer = item["answer"].lower().strip()

    expected_words = set(
        w for w in expected.split()
        if len(w) > 4
    )

    matched = sum(
        1 for w in expected_words
        if w in answer
    )

    similarity = matched / max(len(expected_words), 1)

    if similarity >= 0.40:
        score += 1
        print(f"Q{item['id']}: PASS ({similarity:.2f})")
    else:
        print(f"Q{item['id']}: FAIL ({similarity:.2f})")

print("\n==========")
print(f"Score: {score}/{len(results)}")
print(f"Accuracy: {(score/len(results))*100:.2f}%")