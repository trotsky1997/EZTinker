#!/usr/bin/env python3
"""
Rejection SFT with Simple Math (100以内加减法)

- 256训练样本
- 25测试样本
- 每个样本生成最多10次
- 攒8个positive做梯度下降
- 每10步eval
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 80)
print("Rejection SFT - Simple Math (100以内加减法)")
print("=" * 80)

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eztinker import EZTinkerClient

# 导入自定义数据集
sys.path.insert(0, str(Path(__file__).parent))
from simple_math_dataset import SimpleMathDataset

# 加载模型
print("\n[1/6] 加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Model loaded")

# 加载数据集
print("\n[2/6] 加载数据集...")
train_data = SimpleMathDataset(num_samples=256, seed=42)
eval_data = SimpleMathDataset(num_samples=25, seed=1000)
print(f"✓ Train: {len(train_data)}, Eval: {len(eval_data)}")

# 初始化EZTinker
print("\n[3/6] 初始化EZTinker...")
client = EZTinkerClient(base_url="http://localhost:8000")
try:
    health = client.health()
    print(f"✓ Server: {health}")
except Exception as e:
    print(f"✗ Server error: {e}")
    print("Run: uvicorn src.eztinker.api.server:app --host 0.0.0.0 --port 8000")
    sys.exit(1)

# 创建训练run
print("\n[4/6] 创建训练run...")
run_id = f"simple_math_{random.randint(1000, 9999)}"
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
    print(f"✗ Failed: {resp.text}")
    sys.exit(1)
print(f"✓ Run: {run_id}")


# 验证函数
def verify_answer(resp_text, expected):
    """验证答案"""
    nums = re.findall(r"-?\d+", resp_text)
    return expected in nums if nums else False


# 收集positive样本
print("\n[5/6] 收集positive样本...")
collected = []
stats = {"correct": 0, "failed": 0, "total_gen": 0}

for idx in range(len(train_data)):
    q, p, exp = train_data.get_example_question(idx)

    if (idx + 1) % 20 == 0:
        print(f"[{idx + 1}/256] Correct: {stats['correct']}, Failed: {stats['failed']}")

    # 生成最多10次
    found = False
    for try_idx in range(10):
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        stats["total_gen"] += 1

        if verify_answer(resp, exp):
            # 准备训练数据
            train_text = p + resp
            input_ids = tokenizer(train_text, return_tensors="pt")["input_ids"][0][:512].tolist()
            collected.append(input_ids)
            stats["correct"] += 1
            found = True
            break

    if not found:
        stats["failed"] += 1

print(f"✓ Collected: {len(collected)}/256 = {len(collected) / 256 * 100:.1f}%")

# 训练循环
print("\n[6/6] 训练...")
grad_steps = 0
batch_size = 8
eval_freq = 10
metrics = []
max_steps = 32  # 256/8 = 32步

while len(collected) >= batch_size and grad_steps < max_steps:
    # 取一批样本
    batch = collected[:batch_size]
    collected = collected[batch_size:]

    # 梯度下降
    avg_loss = 0
    count = 0
    for item in batch:
        result = client.forward_backward(run_id, item)
        if result["status"] == "completed":
            avg_loss += result["result"].get("loss", 0.0)
            count += 1

    if count > 0:
        avg_loss /= count
        client.optim_step(run_id, learning_rate=5e-5, weight_decay=0.01)
        grad_steps += 1
    else:
        print("⚠ No successful forward passes")
        break

    if grad_steps % 5 == 0:
        print(f"  Step {grad_steps}: loss={avg_loss:.4f}, collected={len(collected)}")

    # Eval
    if grad_steps % eval_freq == 0:
        print(f"  Eval at step {grad_steps}...")
        eval_correct = 0

        for i in range(len(eval_data)):
            _, p2, exp2 = eval_data.get_example_question(i)

            inputs = tokenizer(p2, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            resp2 = tokenizer.decode(out[0], skip_special_tokens=True)

            if verify_answer(resp2, exp2):
                eval_correct += 1

        eval_acc = eval_correct / len(eval_data) * 100
        print(f"    Eval: {eval_correct}/{len(eval_data)} = {eval_acc:.1f}%")

        metrics.append(
            {
                "step": grad_steps,
                "train_acc": stats["correct"] / 256 * 100,
                "eval_acc": eval_acc,
                "loss": avg_loss,
            }
        )

# 保存结果
print("\n" + "=" * 80)
print("训练完成！")
print("=" * 80)
print(f"Gradient steps: {grad_steps}")
print(f"Train accuracy: {stats['correct']}/{256} = {stats['correct'] / 256 * 100:.1f}%")
print(f"Total generations: {stats['total_gen']}")
print(
    f"Success rate: {stats['correct']}/{stats['total_gen']} = {stats['correct'] / stats['total_gen'] * 100:.1f}%"
)

results = {
    "run_id": run_id,
    "grad_steps": grad_steps,
    "training_stats": stats,
    "metrics": metrics,
}
with open(f"rejection_simple_math_{run_id}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: rejection_simple_math_{run_id}.json")
print("=" * 80)
