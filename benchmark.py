import json
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

with open("benchmark.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

for q in questions:

    prompt = f"""
Code:
{q['code']}

Find the bug in the code.
"""

    response = client.chat.completions.create(
        model="qwen/qwen3-4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=300
    )

    message = response.choices[0].message

    answer = message.content

    # Qwen 3 often stores its answer in reasoning_content
    if not answer:
        answer = getattr(message, "reasoning_content", "")

    if not answer:
        answer = ""

    answer = answer.strip()

    results.append({
        "id": q["id"],
        "language": q["language"],
        "expected": q["expected"],
        "answer": answer
    })

    print(f"Finished Question {q['id']}")

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nBenchmark Complete!")
print(f"Processed {len(questions)} questions.")