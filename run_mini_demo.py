#!/usr/bin/env python3
"""
极小的 Rejection SFT demo (用于快速验证)
只处理 2 个样本，1 轮训练
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 70)
print("Rejection SFT 微型 Demo")
print("=" * 70)

from eztinker.rl.rejection_sampler import (
    create_training_run,
    generate_candidates,
    select_best_candidate_and_train,
    wait_for_job,
)
from eztinker.dataset.gsm8k import GSM8KDataset

# 配置
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"

print("\n[步骤 1/5] 检查服务器...")
import requests
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    if response.status_code == 200:
        print("✅ 服务器正在运行")
    else:
        print("❌ 服务器响应异常")
        sys.exit(1)
except Exception as e:
    print(f"❌ 服务器未运行: {e}")
    print("\n请先启动服务器: uv run eztinker server")
    sys.exit(1)

print(f"\n[步骤 2/5] 创建训练会话 (Rank-1 LoRA, {MODEL_ID})...")
try:
    run_id = create_training_run(base_model=MODEL_ID, lora_rank=1)
    print(f"✅ 会话创建成功: {run_id}")
except Exception as e:
    print(f"❌ 创建失败: {e}")
    sys.exit(1)

print("\n[步骤 3/5] 加载 GSM8K 数据集 (2 个样本)...")
try:
    dataset = GSM8KDataset(split="train", max_samples=2)
    print(f"✅ 成功加载 {len(dataset)} 个样本")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    sys.exit(1)

print("\n[步骤 4/5] 生成和评估候选答案...")
for i in range(len(dataset)):
    print(f"\n  处理样本 {i+1}/{len(dataset)}...")
    question, prompt, ground_truth = dataset.get_example_question(i)

    try:
        print(f"    问题: {question[:50]}...")
        print(f"    正确答案: {ground_truth}")

        # 生成 2 个候选答案
        print("    生成 2 个候选答案...")
        candidates = generate_candidates(
            prompt=prompt,
            question=question,
            run_id=run_id,
            num_candidates=2,
            temperature=0.8
        )
        print(f"    已生成 {len(candidates)} 个候选")

        # 评估并选择最佳
        print("    评估候选答案...")
        result = select_best_candidate_and_train(
            run_id=run_id,
            prompt=prompt,
            candidates=candidates,
            ground_truth=ground_truth,
            question=question,
            dataset=dataset,
            learning_rate=2e-4,
        )

        print(f"    最佳分数: {result['selected_score']:.2f}")
        print(f"    是否正确: {result['selected_is_correct']}")
        print(f"    是否训练: {result.get('trained', False)}")

    except Exception as e:
        print(f"    ⚠️  处理失败: {e}")
        continue

print("\n[步骤 5/5] 生成示例文本...")
try:
    from eztinker.rl.rejection_sampler import generate_candidate_single
    response = generate_candidate_single(
        prompt="What is 2 + 2?",
        run_id=run_id,
        temperature=0.7,
        max_new_tokens=50
    )
    print(f"生成结果: {response}")
except Exception as e:
    print(f"⚠️  生成失败: {e}")

print("\n" + "=" * 70)
print("✅ 微型 Demo 完成!")
print("=" * 70)
print("\n📊 如果所有步骤都成功，你的 Rejection SFT 就配置好了！")
print("\n🚀 运行完整 demo:")
print("   uv run python rejection_sft_demo.py --max-samples 50 --num-candidates 4 --epochs 3")
print()