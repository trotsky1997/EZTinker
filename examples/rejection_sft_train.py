#!/usr/bin/env python3
"""
Rejection SFT Training - Full Pipeline

训练逻辑:
- 训练集: 256 examples
- 测试集: 25 examples
- Batch策略: 攒8个positive example做一次梯度下降
- Eval频率: 每10次梯度下降eval一次
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests

from eztinker import EZTinkerClient
from eztinker.dataset.gsm8k import GSM8KDataset

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠ transformers not installed: pip install transformers torch")


def extract_answer(text: str) -> str:
    """提取答案中的数字"""
    # 找 "answer is X" 模式
    patterns = [
        r"(?:the\s+)?answer\s+is\s+(-?\d+\.?\d*)",
        r"answer\s*:\s*(-?\d+\.?\d*)",
        r"(?:final\s+)?answer\s*=\s*(-?\d+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1)

    # fallback: 找纯数字
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]  # 最后一个数字

    return ""


def math_verify(answer_text: str, expected_answer: str) -> bool:
    """验证答案是否正确"""
    extracted = extract_answer(answer_text)
    return extracted == expected_answer


def generate_with_rejection(
    model, tokenizer, prompt: str, expected_answer: str, max_tries: int = 3
) -> tuple[str, bool, int]:
    """生成答案，直到找到正确的或达到最大尝试次数

    Returns:
        (answer, is_correct, tries_used)
    """
    for try_idx in range(max_tries):
        # 逐步增加temperature
        temp = 0.3 + try_idx * 0.3  # 0.3, 0.6, 0.9

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 只保留assistant的回答部分
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        # 验证
        if math_verify(response, expected_answer):
            return response, True, try_idx + 1

    # 所有尝试都失败
    return response, False, max_tries


def run_training_loop(
    client: EZTinkerClient,
    run_id: str,
    training_samples: list[dict],
    eval_samples: list[dict],
    model,
    tokenizer,
    batch_size: int = 8,
    eval_freq: int = 10,
    num_grad_steps: int = 100,
):
    """主训练循环"""

    print("\n" + "=" * 80)
    print("Rejection SFT Training Loop")
    print("=" * 80)
    print(f"Batch size (positive examples): {batch_size}")
    print(f"Eval frequency: Every {eval_freq} gradient steps")
    print(f"Max gradient steps: {num_grad_steps}")
    print("=" * 80 + "\n")

    collected_positives = []
    grad_step_count = 0
    total_generated = 0
    total_accepted = 0

    step_metrics = []

    # 采样训练样本顺序
    sample_indices = list(range(len(training_samples)))

    step_idx = 0
    while grad_step_count < num_grad_steps:
        # 按顺序取样本
        idx = sample_indices[step_idx % len(sample_indices)]
        step_idx += 1

        question, prompt, expected_answer = training_samples[idx]

        # 生成并验证
        response, is_correct, tries = generate_with_rejection(
            model, tokenizer, prompt, expected_answer
        )

        total_generated += tries

        if is_correct:
            # 收集positive example
            collected_positives.append(
                {
                    "question": question,
                    "prompt": prompt,
                    "response": response,
                    "answer": expected_answer,
                    "input_ids": tokenizer(prompt + response, return_tensors="pt")["input_ids"][
                        0
                    ].tolist(),
                }
            )
            total_accepted += 1
            print(f"✓ Accepted: {question[:50]}... (tries={tries})")

            # 攒够8个，做一次梯度下降
            if len(collected_positives) >= batch_size:
                print(f"\n{'=' * 80}")
                print(f"Gradient Step {grad_step_count + 1}: {len(collected_positives)} positives")
                print(f"{'=' * 80}")

                # 取出8个
                batch = collected_positives[:batch_size]
                collected_positives = collected_positives[batch_size:]

                # 做梯度下降
                try:
                    # 对每个positive example做forward/backward
                    avg_loss = 0
                    for item in batch:
                        input_ids = item["input_ids"][:512]  # Truncate to 512
                        result = client.forward_backward(run_id, input_ids)
                        if result["status"] == "completed" and result.get("result"):
                            loss = result["result"].get("loss", 0.0)
                            avg_loss += loss

                    avg_loss /= len(batch)

                    # 做optimizer step
                    client.optim_step(run_id, learning_rate=5e-5, weight_decay=0.01)

                    grad_step_count += 1

                    print(f"✓ Gradient step {grad_step_count} completed")
                    print(f"  Avg loss: {avg_loss:.4f}")
                    print(
                        f"  Collected: {total_accepted}/{total_generated} = {total_accepted / total_generated * 100:.1f}%"
                    )
                    print(f"  Pending positives: {len(collected_positives)}")

                    # Eval
                    if grad_step_count % eval_freq == 0:
                        print(f"\n{'=' * 80}")
                        print(f"Step {grad_step_count}: Evaluation")
                        print(f"{'=' * 80}")

                        eval_correct, eval_total, eval_generated = run_evaluation(
                            model, tokenizer, eval_samples[:25]
                        )

                        step_metrics.append(
                            {
                                "grad_step": grad_step_count,
                                "train_accepted": total_accepted,
                                "train_generated": total_generated,
                                "eval_correct": eval_correct,
                                "eval_total": eval_total,
                                "eval_generated": eval_generated,
                            }
                        )

                        print(
                            f"Eval result: {eval_correct}/{eval_total} = {eval_correct / eval_total * 100:.1f}% (generated {eval_generated}x)"
                        )
                        print(f"{'=' * 80}\n")

                except Exception as e:
                    print(f"⚠ Gradient step failed: {e}")

        else:
            print(f"✗ Rejected: {question[:50]}... (tries={tries})")

    # 最终eval
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    eval_correct, eval_total, eval_generated = run_evaluation(model, tokenizer, eval_samples[:25])

    print(f"Final Eval: {eval_correct}/{eval_total} = {eval_correct / eval_total * 100:.1f}%")
    print(f"Total generated: {eval_generated}x")

    return {
        "grad_steps": grad_step_count,
        "train_accepted": total_accepted,
        "train_generated": total_generated,
        "final_eval": {"correct": eval_correct, "total": eval_total},
        "step_metrics": step_metrics,
    }


def run_evaluation(model, tokenizer, eval_samples):
    """在测试集上评估"""
    correct = 0
    total = 0
    generated = 0

    for question, prompt, expected_answer in eval_samples:
        response, is_correct, tries = generate_with_rejection(
            model,
            tokenizer,
            prompt,
            expected_answer,
            max_tries=1,  # Eval只生成1次
        )
        total += 1
        generated += tries

        if is_correct:
            correct += 1

    return correct, total, generated


def main():
    """主入口"""
    print("=" * 80)
    print("Rejection SFT Training - GSM8K")
    print("=" * 80)
    print("Training: 256 examples")
    print("Eval: 25 examples")
    print("Batch: 8 positive examples per gradient step")
    print("Eval frequency: Every 10 gradient steps")
    print("=" * 80)

    # 检查依赖
    if not HAS_TRANSFORMERS:
        print("⚠ Install: pip install transformers torch")
        return

    # 加载数据集
    print("\nLoading GSM8K datasets...")
    train_dataset = GSM8KDataset(split="train", max_samples=256)
    test_dataset = GSM8KDataset(split="test", max_samples=25)

    train_samples = [train_dataset.get_example_question(i) for i in range(len(train_dataset))]
    eval_samples = [test_dataset.get_example_question(i) for i in range(len(test_dataset))]

    print(f"✓ Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    # 加载模型
    print("\nLoading Qwen/Qwen2-0.5B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded")

    # 初始化EZTinker客户端
    print("\n" + "=" * 80)
    print("Initialize EZTinker Client")
    print("=" * 80)
    client = EZTinkerClient(base_url="http://localhost:8000")

    try:
        server_health = client.health()
        print(f"✓ Server online: {server_health}")
    except Exception as e:
        print(f"✗ Server not reachable: {e}")
        print("Run: uvicorn src.eztinker.api.server:app --host 0.0.0.0 --port 8000")
        return

    # 创建训练run
    print("\n" + "=" * 80)
    print("Create Training Run")
    print("=" * 80)
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

    response = requests.post(f"{client.base_url}/v1/runs", json=request_data)
    if response.status_code != 200:
        print(f"✗ Failed to create run: {response.text}")
        return

    print(f"✓ Training run: {run_id}")

    # 运行训练
    results = run_training_loop(
        client,
        run_id,
        train_samples,
        eval_samples,
        model,
        tokenizer,
        batch_size=8,
        eval_freq=10,
        num_grad_steps=20,  # 先跑20步看看
    )

    # 保存结果
    output_file = f"rejection_sft_results_{run_id}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"Run ID: {run_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()
