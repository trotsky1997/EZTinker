#!/usr/bin/env python3
"""快速测试版本，只处理几个样本"""

import sys

sys.path.insert(0, str(__file__).replace("/examples/test_quick.py", "/src"))

print("Step 1: 加载模型")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✓ 模型加载完成")

print("\nStep 2: 加载数据集")
from eztinker.dataset.gsm8k import GSM8KDataset

dataset = GSM8KDataset(split="train", max_samples=5)
print(f"✓ 数据集加载完成: {len(dataset)} samples")

print("\nStep 3: 生成测试")
q, p, exp = dataset.get_example_question(0)
print(f"Question: {q[:60]}...")
print(f"Expected: {exp}")

inputs = tokenizer(p, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=True)
resp = tokenizer.decode(out[0], skip_special_tokens=True)

print(f"Response: {resp[:100]}...")

import re


def extract_answer(text):
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else "UNKNOWN"


answer = extract_answer(resp)
print(f"Extracted: {answer}, Expected: {exp}, Match: {answer == exp}")
print("\n测试完成！")
