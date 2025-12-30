#!/usr/bin/env python3
"""激进版rejection - 生成更多次，降低验证标准"""
import json
import re
import sys
sys.path.insert(0, str(__file__).replace('/examples/rejection_aggressive.py', '/src'))

print("="*80)
print("Rejection SFT Aggressive - 生成20次/样本")
print("="*80)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from eztinker.dataset.gsm8k import GSM8KDataset

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Model loaded")

# 先测试5个中等难度的题
train_data = GSM8KDataset(split="train", max_samples=10)
print(f"✓ Dataset: {len(train_data)} samples\n")

collected = []
stats = {"correct": 0, "wrong": 0, "partial": 0}

for idx in range(len(train_data)):
    q, p, exp = train_data.get_example_question(idx)
    print(f"[{idx+1}/10] Q: {q[:50]}...")
    print(f"         Expected: {exp}")

    # 生成20次
    tried = 0
    for try_idx in range(20):
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300,
                                 temperature=0.7, do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0], skip_special_tokens=True)

        # 验证
        nums = re.findall(r'-?\d+', resp)
        is_correct = exp in nums if nums else False

        if is_correct:
            print(f"  ✓ Try {try_idx+1}: CORRECT")
            collected.append((p, resp))
            stats["correct"] += 1
            break
        else:
            # 部分正确？（包含部分答案数字也算）
            partial = False
            if nums:
                # 检查是否有数字接近
                for n in nums:
                    if n.isdigit() and abs(int(n) - int(exp)) < 50:
                        partial = True
                        break

            if partial and tried < 1:  # 最多接受1个部分对的
                print(f"  ~ Try {try_idx+1}: PARTIAL ({nums[:3]})")
                collected.append((p, resp))
                stats["partial"] += 1
                tried += 1
                break

    else:
        print(f"  ✗ All 20 tries failed")
        stats["wrong"] += 1

print("\n" + "="*80)
print(f"Results:")
print(f"  Correct: {stats['correct']}/10")
print(f"  Partial: {stats['partial']}/10")
print(f"  Wrong: {stats['wrong']}/10")
print(f"  Collected: {len(collected)}/10")
print("="*80)

if collected:
    results = {
        "stats": stats,
        "collected": len(collected),
        "accuracy": (stats['correct'] + stats['partial'] * 0.5) / 10 * 100,
    }
    with open("rejection_aggressive.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to: rejection_aggressive.json")