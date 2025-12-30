#!/usr/bin/env python3
"""
Rejection SFT Training - 简化版本

逻辑：
1. 256个训练样本
2. 每个样本生成答案，验证正确性
3. 攒够8个正确样本做一次梯度下降
4. 每10次梯度下降在25个测试样本上评估
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from eztinker.dataset.gsm8k import GSM8KDataset
from eztinker import EZTinkerClient
import requests


print("="*80)
print("Rejection SFT - 256 training samples, 8 pos/batch, eval every 10 steps")
print("="*80)

# 加载模型
print("\nLoading Qwen/Qwen2-0.5B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Model loaded")

#柱加载数据集
print("\nLoading GSM8K datasets...")
train_data = GSM8KDataset(split="train", max_samples=256)
eval_data = GSM8KDataset(split="test", max_samples=25)
print(f"✓ Train: {len(train_data)}, Eval: {len(eval_data)}")

# 初始化EZTinker客户端
print("\nInitializing EZTinker client...")
client = EZTinkerClient(base_url="http://localhost:8000")
try:
    health = client.health()
    print(f"✓ Server online: {health}")
except Exception as e:
    print(f"✗ Server error: {e}")
    print("Run: uvicorn src.eztinker.api.server:app --host 0.0.0.0 --port 8000")
    sys.exit(1)

# 创建训练run
run_id = f"rejection_sft_{random.randint(1000, 9999)}"
request_data = {
    "base_model": "Qwen/Qwen2-0.5B-Instruct",
    "lora_config": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "run_id": run_id,
}

resp = requests.post(f"{client.base_url}/v1/runs", json=request_data)
if resp.status_code != 200:
    print(f"✗ Failed to create run: {resp.text}")
    sys.exit(1)

print(f"✓ Training run created: {run_id}")


def extract_answer(text):
    """提取答案数字"""
    patterns = [
        r'(?:the\s+)?answer\s+is\s+(-?\d+\.?\d*)',
        r'answer\s*:\s*(-?\d+\.?\d*)',
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return m.group(1)
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ""


def math_verify(response, expected):
    """验证答案"""
    return extract_answer(response) == expected


print("\n" + "="*80)
print("Training Loop")
print("="*80)

collected = []
grad_steps = 0
total_gen = 0
total_acc = 0
step_metrics = []

for idx in range(len(train_data)):
    q, p, exp = train_data.get_example_question(idx)

    # 生成答案
    for try_idx in range(3):
        temp = 0.3 + try_idx * 0.3
        inputs = tokenizer(p, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        total_gen += 1

        if math_verify(resp, exp):
            print(f"✓ {idx+1}/256: ACCEPTED (try={try_idx+1})")
            total_acc += 1

            # 准备训练数据
            train_text = p + resp
            input_ids = tokenizer(train_text, return_tensors="pt")["input_ids"][0][:512].tolist()
            collected.append(input_ids)
            break
    else:
        print(f"✗ {idx+1}/256: REJECTED (all tries failed)")

    # 攒够8个做梯度下降
    if len(collected) >= 8:
        print(f"\n{'='*80}")
        print(f"Gradient Step {grad_steps + 1} (accepted: {total_acc}/{total_gen})")
        print(f"{'='*80}")

        batch = collected[:8]
        collected = collected[8:]

        for item in batch:
            result = client.forward_backward(run_id, item)
            if result["status"] != "completed":
                print(f"⚠ Forward/backward failed: {result}")

        client.optim_step(run_id, learning_rate=5e-5, weight_decay=0.01)
        grad_steps += 1
        print(f"✓ Step {grad_steps} done\n")

        # 每10步eval
        if grad_steps % 10 == 0:
            print(f"{'='*80}")
            print(f"Evaluation at step {grad_steps}")
            print(f"{'='*80}")

            eval_correct = 0
            for i in range(len(eval_data)):
                _, p2, exp2 = eval_data.get_example_question(i)

                inputs = tokenizer(p2, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True,
                    )
                resp2 = tokenizer.decode(out[0], skip_special_tokens=True)

                if math_verify(resp2, exp2):
                    eval_correct += 1

            eval_acc = eval_correct / len(eval_data) * 100
            print(f"Eval: {eval_correct}/{len(eval_data)} = {eval_acc:.1f}%")
            print(f"{'='*80}\n")

            step_metrics.append({
                "step": grad_steps,
                "train_acc": total_acc/total_gen*100,
                "eval_acc": eval_acc,
            })

print("\n" + "="*80)
print("Training Complete!")
print("="*80)
print(f"Gradient steps: {grad_steps}")
print(f"Train accuracy: {total_acc}/{total_gen} = {total_acc/total_gen*100:.1f}%")

results = {
    "run_id": run_id,
    "grad_steps": grad_steps,
    "metrics": step_metrics,
}
with open(f"rejection_results_{run_id}.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: rejection_results_{run_id}.json")
print("="*80)