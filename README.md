# AI Code Reviewer - LoRA Fine-Tuning Project

## Overview

AI Code Reviewer is a custom fine-tuned Large Language Model (LLM) designed to identify bugs in source code and provide explanations for why they occur.

The project demonstrates the complete workflow of:

* Dataset creation
* Benchmark generation
* Local LLM inference
* LoRA fine-tuning
* Model evaluation
* Performance analysis

The model was fine-tuned locally on consumer hardware using Qwen2.5-1.5B and PEFT LoRA.

---

## Features

* Detects bugs in source code snippets
* Explains why the bug occurs
* Suggests possible fixes
* Automated benchmarking pipeline
* Local inference without external APIs
* GPU-accelerated fine-tuning

---

## Tech Stack

### Models

* Qwen2.5-1.5B-Instruct
* LoRA (Low-Rank Adaptation)

### Libraries

* PyTorch
* Hugging Face Transformers
* PEFT
* Datasets

---

## Dataset

### Training Dataset

A custom dataset containing approximately 452 code-review examples.

Each example follows the format:

```json
{
  "instruction": "Review the following code and identify the specific bug. Explain why it occurs and how to fix it.",
  "input": "source code here",
  "output": "Bug explanation here"
}
```

The dataset primarily focuses on:

* Data Structures & Algorithms
* Python programming
* Logic errors
* Off-by-one errors
* Tree and graph bugs
* Recursion mistakes
* Dynamic programming issues

---

### Benchmark Dataset

A separate benchmark dataset containing 100 code-review examples was created to evaluate model performance.

The benchmark includes:

* Python
* SQL
* Security issues
* Language edge cases
* Common software bugs

---

## Fine-Tuning Configuration

### LoRA Settings

```python
r = 8
lora_alpha = 16
lora_dropout = 0.05
```

### Target Modules

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

### Training Settings
<img width="1337" height="305" alt="image" src="https://github.com/user-attachments/assets/4136018a-d5ea-431a-913d-a6641c66a4c3" />


```python
Epochs: 2
Learning Rate: 2e-4
Batch Size: 1
Gradient Accumulation: 4
Mixed Precision: FP16
```

Training time:

```text
~18 minutes
```

---

## Project Structure

```text
AI-code-reviewer/
│
├── training_data.json
├── benchmark.json
│
├── benchmark.py
├── benchmark_finetuned.py
├── evaluate.py
├── finetune.py
│
├── results.json
├── results_finetuned.json
│
├── qwen2.5-1.5b/
│
└── qwen-code-review-lora/
```

---

## Results

### Base Model

The base Qwen model demonstrated strong general-purpose bug detection capabilities due to its broad pretraining knowledge.

### Fine-Tuned Model

The fine-tuned model showed improved specialization toward the training domain but reduced performance on benchmark tasks outside the dataset distribution.

### Key Observation

Fine-tuning improved task specialization but highlighted an important machine learning lesson:

> A model trained on a narrow domain may perform worse on a broad benchmark containing tasks it was never trained to solve.

---

## Lessons Learned

This project provided practical experience with:

* LLM fine-tuning
* LoRA adapters
* Dataset engineering
* Prompt formatting
* Benchmark design
* Model evaluation
* GPU training workflows

Key takeaways:

1. Dataset quality matters more than model size.
2. Benchmark datasets must match training objectives.
3. Fine-tuning can increase specialization while reducing generalization.
4. Evaluation pipelines are often harder than training itself.
5. Consumer GPUs can successfully fine-tune modern language models using LoRA.

---

## Future Improvements

* Expand dataset to 2,000+ examples
* Add Java, JavaScript, and SQL bug examples
* Create domain-specific benchmark sets
* Implement response-only loss masking
* Improve evaluation metrics using semantic similarity

---

## Author

Meghamsh Anirudh Pulivendala

B.Tech CSE Student

Built as a hands-on project to explore practical LLM fine-tuning, evaluation, and model benchmarking.
