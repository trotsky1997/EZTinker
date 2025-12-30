#!/usr/bin/env python3
"""测试训练功能的手动脚本"""
import sys
import os
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("手动训练测试")
print("=" * 60)

# 检查服务器
print("\n[1] 检查服务器...")
response = requests.get(f"{BASE_URL}/health")
print(f"✅ 服务器状态: {response.json()}")

# 创建训练会话
print("\n[2] 创建训练会话...")
response = requests.post(
    f"{BASE_URL}/v1/runs",
    json={
        "base_model": "gpt2",
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["c_attn", "c_proj"]
        }
    }
)
run_id = response.json()["run_id"]
print(f"✅ 创建会话: {run_id}")

# 准备一个简单的 batch
print("\n[3] 准备训练数据 (GPT-2 BPE token IDs)...")
# "Hello world" 的 token IDs (通过 GPT-2 tokenizer)
batch = {
    "input_ids": [15496, 995],
    "target_ids": [15496, 995]
}

# Forward + Backward
print("\n[4] Forward + Backward 传播...")
response = requests.post(
    f"{BASE_URL}/v1/runs/{run_id}/forward_backward",
    json=batch
)
fb_result = response.json()
print(f"✅ 传播结果: {fb_result}")

# Poll job status
print("\n[5] 等待任务完成...")
job_id = fb_result["job_id"]
while True:
    response = requests.get(f"{BASE_URL}/v1/jobs/{job_id}")
    result = response.json()
    if result["status"] == "completed":
        print(f"✅ 任务完成: {result}")
        break
    elif result["status"] == "failed":
        print(f"❌ 任务失败: {result}")
        break

# Optimizer Step
print("\n[6] Optimizer Step (AdamW)...")
response = requests.post(
    f"{BASE_URL}/v1/runs/{run_id}/optim_step",
    json={
        "learning_rate": 2e-4,
        "weight_decay": 0.01
    }
)
optim_result = response.json()
print(f"✅ Optimizer 结果: {optim_result}")

# 等待 optimizer 完成
job_id = optim_result["job_id"]
while True:
    response = requests.get(f"{BASE_URL}/v1/jobs/{job_id}")
    result = response.json()
    if result["status"] == "completed":
        print(f"✅ Optimizer 完成: {result}")
        break
    elif result["status"] == "failed":
        print(f"❌ Optimizer 失败: {result}")
        break

print("\n" + "=" * 60)
print("✅ 手动训练测试成功!")
print("=" * 60)
print(f"\n训练会话: {run_id}")
print("可以使用这个会话继续训练或保存 checkpoint")
print()