import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

MODEL_PATH = "./qwen2.5-1.5b"

print("Loading dataset...")

with open("training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

formatted = []

for item in data:
    text = f"""### Instruction:
{item["instruction"]}

### Input:
{item["input"]}

### Response:
{item["output"]}"""

    formatted.append({"text": text})

dataset = Dataset.from_list(formatted)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
)

print("Applying LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
    )

print("Tokenizing...")
tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=["text"]
)

training_args = TrainingArguments(
    output_dir="./qwen-code-review-lora-v2",

    num_train_epochs=5,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=1e-4,

    fp16=True,

    logging_steps=10,
    save_strategy="epoch",

    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

print("Starting training...")
trainer.train()

print("Saving model...")

model.save_pretrained("./qwen-code-review-lora-v2")
tokenizer.save_pretrained("./qwen-code-review-lora-v2")

print("Done!")