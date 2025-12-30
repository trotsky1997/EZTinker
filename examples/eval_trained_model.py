#!/usr/bin/env python3
"""评估训练好的模型"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simple_math_dataset import SimpleMathDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

print("评估训练好的rejection model...")

import torch

model = AutoModelForCausalLM.from_pretrained(
    "./rejection_output/final_model",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("./rejection_output/final_model", trust_remote_code=True)
print("✓ Loaded fine-tuned model")

eval_data = SimpleMathDataset(num_samples=20, seed=1000)
correct = 0

for i in range(len(eval_data)):
    _, p, exp = eval_data.get_example_question(i)

    inputs = tokenizer(p, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )

    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    nums = re.findall(r"-?\d+", resp)
    res = exp in nums if nums else False

    print(
        f"[{i + 1:2d}/20] Expected: {exp:>3s}, Got: {nums[-1] if nums else 'NONE':>4s}, {'✓' if res else '✗'}"
    )

    if res:
        correct += 1

print(f"\n评估结果: {correct}/20 = {correct / 20 * 100:.1f}%")
