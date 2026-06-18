import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "./qwen2.5-1.5b"
ADAPTER = "./qwen-code-review-lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER
)

model.eval()

print("Loading benchmark questions...")

with open("benchmark.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

for q in questions:

    prompt = f"""### Instruction:
Review the following code and identify the specific bug. Explain why it occurs and how to fix it.

### Input:
{q["code"]}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    # ONLY keep newly generated tokens
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    results.append({
        "id": q["id"],
        "language": q["language"],
        "expected": q["expected"],
        "answer": answer
    })

    print(f"Finished {q['id']} -> {answer[:60]}")

with open("results_finetuned.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nBenchmark Complete!")
print(f"Processed {len(questions)} questions.")
print("Saved to results_finetuned.json")