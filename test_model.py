from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

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

prompt = """
### Instruction:
Review the following code and identify the specific bug. Explain why it occurs and how to fix it.

### Input:
def is_even(n):
    return n % 2 == 1

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...")

output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id
)

print("\nRESULT:")
print(tokenizer.decode(output[0], skip_special_tokens=True))