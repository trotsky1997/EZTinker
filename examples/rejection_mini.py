#!/usr/bin/env python3
"""Rejection SFT Mini - 从5个样本开始测试流程"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 80)
print("Rejection SFT Mini - 5 samples test")
print("=" * 80)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eztinker.dataset.gsm8k import GSM8KDataset

# 1. 加载
print("\n1. 加载模型")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Model loaded")

print("\n2. 加载数据集")
train_data = GSM8KDataset(split="train", max_samples=5)
print(f"✓ Train: {len(train_data)}")

# 2. 生成和验证
collected = []
gen_count = 0
acc_count = 0

print("\n3. 生成和验证")


def extract_answer(text):
    patterns = [
        r"(?:the\s+)?answer\s+is\s+(-?\d+)",
        r"answer\s*:\s*(-?\d+)",
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return m.group(1)
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else ""


for idx in range(len(train_data)):
    q, p, exp = train_data.get_example_question(idx)
    print(f"\n[{idx + 1}/5] Q: {q[:50]}...")
    print(f"     Expected: {exp}")

    found = False
    for try_idx in range(3):
        temp = 0.3 + try_idx * 0.4
        inputs = tokenizer(p, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        gen_count += 1

        ans = extract_answer(resp)
        is_correct = ans == exp

        print(f"  Try {try_idx + 1} (temp={temp:.1f}): {ans} {'✓' if is_correct else '✗'}")

        if is_correct:
            collected.append((p, resp, q, exp))
            acc_count += 1
            found = True
            break

    if not found:
        print("  ✗ All tries failed for this question")

# 3. 总结
print("\n" + "=" * 80)
print(f"Accepted: {acc_count}/{gen_count} = {acc_count / gen_count * 100:.1f}%")
print(f"Collected: {len(collected)} positive examples")
print("=" * 80)

if collected:
    print("\nSample accepted:")
    p, r, q, e = collected[0]
    print(f"Q: {q[:60]}...")
    print(f"A: {r[:100]}...")

# 保存结果
results = {
    "generated": gen_count,
    "accepted": acc_count,
    "accuracy": acc_count / gen_count * 100,
    "collected": len(collected),
}
with open("rejection_mini_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to: rejection_mini_results.json")
