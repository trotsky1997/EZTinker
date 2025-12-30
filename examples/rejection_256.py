#!/usr/bin/env python3
"""
Rejection SFT Training - 256 samples

策略:
- 生成20次/样本
- 接受正确答案或部分答案
- 攒8个做梯度下降
- 每10步eval 25测试样本
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 80)
print("Rejection SFT Training - 256 train, 25 eval, 8 pos/batch")
print("=" * 80)

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eztinker import EZTinkerClient
from eztinker.dataset.gsm8k import GSM8KDataset

# 加载模型
print("\n[1/6] Loading model...")
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
print("\n[2/6] Loading datasets...")
train_data = GSM8KDataset(split="train", max_samples=256)
eval_data = GSM8KDataset(split="test", max_samples=25)
print(f"✓ Train: {len(train_data)}, Eval: {len(eval_data)}")

# 初始化EZTinker
print("\n[3/6] Initializing EZTinker...")
client = EZTinkerClient(base_url="http://localhost:8000")
try:
    health = client.health()
    print(f"✓ Server: {health}")
except Exception as e:
    print(f"✗ Server error: {e}")
    sys.exit(1)

# 创建训练run
print("\n[4/6] Creating training run...")
run_id = f"rejection_256_{random.randint(1000, 9999)}"
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
def verify_response(resp_text, expected_answer):
    """验证答案（灵活标准）"""
    nums = re.findall(r"-?\d+", resp_text)
    if not nums:
        return False, False

    # 完全匹配
    if expected_answer in nums:
        return True, True

    # 部分匹配（数值接近）
    try:
        exp_num = int(expected_answer)
        for n in nums:
            if n.isdigit() and abs(int(n) - exp_num) < 50:
                return True, False
    except:
        pass

    return False, False


# 收集positive样本
print("\n[5/6] Collecting positive samples...")
collected = []
stats = {"correct": 0, "partial": 0, "failed": 0, "total_gen": 0}

for idx in range(len(train_data)):
    q, p, exp = train_data.get_example_question(idx)

    if (idx + 1) % 20 == 0:
        print(
            f"[{idx + 1}/256] Correct: {stats['correct']}, Partial: {stats['partial']}, Failed: {stats['failed']}"
        )

    # 生成最多20次
    for try_idx in range(20):
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        stats["total_gen"] += 1

        is_ok, is_correct = verify_response(resp, exp)

        if is_ok:
            if is_correct:
                stats["correct"] += 1
            else:
                stats["partial"] += 1

            # 准备训练数据
            train_text = p + resp
            input_ids = tokenizer(train_text, return_tensors="pt")["input_ids"][0][:512].tolist()
            collected.append(input_ids)
            break
    else:
        stats["failed"] += 1

print(f"✓ Collected: {len(collected)}/256 = {len(collected) / 256 * 100:.1f}%")

# 训练循环
print("\n[6/6] Training...")
grad_steps = 0
batch_size = 8
eval_freq = 10
metrics = []

while len(collected) >= batch_size and grad_steps < 50:  # 最多50步
    # 取一批样本
    batch = collected[:batch_size]
    collected = collected[batch_size:]

    # 梯度下降
    avg_loss = 0
    for item in batch:
        result = client.forward_backward(run_id, item)
        if result["status"] == "completed":
            avg_loss += result["result"].get("loss", 0.0)

    client.optim_step(run_id, learning_rate=5e-5, weight_decay=0.01)
    grad_steps += 1
    avg_loss /= batch_size

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
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            resp2 = tokenizer.decode(out[0], skip_special_tokens=True)

            is_ok, _ = verify_response(resp2, exp2)
            if is_ok:
                eval_correct += 1

        eval_acc = eval_correct / len(eval_data) * 100
        print(f"    Eval: {eval_correct}/{len(eval_data)} = {eval_acc:.1f}%")

        metrics.append(
            {
                "step": grad_steps,
                "train_acc": (stats["correct"] + stats["partial"]) / 256 * 100,
                "eval_acc": eval_acc,
                "loss": avg_loss,
            }
        )

# 保存结果
print("\n" + "=" * 80)
print("Training Complete!")
print("=" * 80)
print(f"Gradient steps: {grad_steps}")
print(
    f"Train accuracy: {stats['correct'] + stats['partial']}/{256} = {(stats['correct'] + stats['partial']) / 256 * 100:.1f}%"
)
print(f"Collections: {len(collected)} remaining")

results = {
    "run_id": run_id,
    "grad_steps": grad_steps,
    "training_stats": stats,
    "metrics": metrics,
}
with open(f"rejection_256_{run_id}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: rejection_256_{run_id}.json")
print("=" * 80)
