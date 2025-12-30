#!/usr/bin/env python3
"""
快速测试 Rejection SFT demo 功能的脚本
（不需要真的运行训练，只是测试导入和数据加载）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 60)
print("Rejection SFT Demo 功能测试")
print("=" * 60)

# 测试 1: 导入模块
print("\n[1/6] 测试导入模块...")
try:
    from eztinker.rl.rejection_sampler import (
        create_training_run,
        generate_candidates,
        select_best_candidate_and_train,
        wait_for_job,
    )
    from eztinker.dataset.gsm8k import GSM8KDataset
    from transformers import AutoTokenizer

    print("✅ 所有模块导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试 2: 检查服务器
print("\n[2/6] 测试服务器连接...")
try:
    import requests
    response = requests.get("http://localhost:8000/health", timeout=2)
    if response.status_code == 200:
        print("✅ 服务器正在运行")
        server_running = True
    else:
        print(f"⚠️  服务器响应异常: {response.status_code}")
        server_running = False
except Exception as e:
    print(f"❌ 服务器未运行: {e}")
    print("   需要先运行: uv run eztinker server")
    server_running = False

# 测试 3: 加载 GSM8K 数据集
print("\n[3/6] 测试 GSM8K 数据加载...")
try:
    dataset = GSM8KDataset(split="train", max_samples=5)
    print(f"✅ 成功加载 {len(dataset)} 个样本")

    # 测试一个样本
    question, prompt, ground_truth = dataset.get_example_question(0)
    print(f"   - 问题: {question[:50]}...")
    print(f"   - Prompt: {prompt[:50]}...")
    print(f"   - 答案: {ground_truth}")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)

# 测试 4: 测试 Math-Verify 评估
print("\n[4/6] 测试 Math-Verify 评估...")
try:
    eval_result = dataset.evaluate_answer(
        model_response="The answer is 42.",
        ground_truth_str="42",
        question=question
    )
    print(f"✅ 评估成功: {eval_result}")
except Exception as e:
    print(f"⚠️  评估失败: {e}")

# 测试 5: 测试 Tokenizer
print("\n[5/6] 测试 Tokenizer 加载...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    print("✅ Tokenizer 加载成功")

    # 测试 tokenization
    tokens = tokenizer("Hello world", return_tensors="pt")
    print(f"   - Token 数量: {tokens['input_ids'].shape[1]}")
except Exception as e:
    print(f"❌ Tokenizer 失败: {e}")
    sys.exit(1)

# 测试 6: 如果服务器运行，测试创建 run
if server_running:
    print("\n[6/6] 测试创建训练会话...")
    try:
        print("   尝试创建训练会话 (rank-1 LoRA)...")
        # 注意：这个会真的创建一个会话，需要服务器正在运行
        run_id = create_training_run(
            base_model="Qwen/Qwen2-0.5B-Instruct",
            lora_rank=1
        )
        print(f"✅ 训练会话创建成功: {run_id}")
    except Exception as e:
        print(f"⚠️  创建会话失败: {e}")
        print("   但这可能只是网络问题")
else:
    print("\n[6/6] 跳过服务器测试（服务器未运行）")

print("\n" + "=" * 60)
print("✅ 功能测试完成!")
print("=" * 60)
print("\n🚀 可以运行完整 demo:")
print("   Terminal 1: uv run eztinker server")
print("   Terminal 2: uv run python rejection_sft_demo.py --max-samples 5 --epochs 1")
print()