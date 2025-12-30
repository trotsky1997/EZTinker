#!/usr/bin/env python3
"""
Rejection SFT - 完整整合版

核心流程：
1. 生成多个候选答案（rejection sampling）
2. 验证并筛选正确答案
3. 用筛选后的高质量数据训练模型
4. 评估训练好的模型

特点：
- 简单数学题（100以内加减法）
- 验证驱动的数据筛选
- 无reward model需求
"""

import sys
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch


# ============================================================================
# 1. 简单数学数据集
# ============================================================================

class SimpleMathDataset:
    """100以内加减法数据集"""

    def __init__(self, num_samples: int, seed: int = 42):
        random.seed(seed)
        self.samples = []

        for i in range(num_samples):
            if random.random() < 0.5:
                a = random.randint(1, 99)
                b = random.randint(1, 100 - a)
                q = f"What is {a} + {b}?"
                ans = str(a + b)
            else:
                a = random.randint(1, 99)
                b = random.randint(1, a)
                q = f"What is {a} - {b}?"
                ans = str(a - b)

            prompt = f"Please solve: {q}\nAnswer: "

            self.samples.append({
                "question": q,
                "prompt": prompt,
                "answer": ans,
            })

    def __len__(self):
        return len(self.samples)

    def get_example_question(self, idx):
        item = self.samples[idx]
        return item["question"], item["prompt"], item["answer"]


# ============================================================================
# 2. Rejection Sampling核心逻辑
# ============================================================================

def verify_answer(resp: str, expected: str) -> bool:
    """验证答案是否正确（简单数字匹配）"""
    nums = re.findall(r'-?\d+', resp)
    return expected in nums if nums else False


def rejection_collect_data(
    model,
    tokenizer,
    dataset: SimpleMathDataset,
    max_generations_per_sample: int = 10
) -> Tuple[List[Dict], Dict]:
    """
    执行rejection sampling收集高质量训练数据

    Returns:
        (training_samples, stats)
    """
    print("="*80)
    print("Rejection Sampling - 收集高质量训练数据")
    print("="*80)
    print(f"每个样本最多生成{max_generations_per_sample}次\n")

    collected = []
    stats = {
        "total_samples": len(dataset),
        "collected": 0,
        "failed": 0,
        "total_generations": 0,
        "avg_generations_per_sample": 0,
    }

    print(f"{'Sample':<8} {'Expected':<10} {'Got':<10} {'Tries':<6} {'Status'}")
    print("-" * 60)

    for idx in range(len(dataset)):
        q, p, exp = dataset.get_example_question(idx)

        # 生成并验证
        found = False
        for try_i in range(max_generations_per_sample):
            inputs = tokenizer(p, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            stats["total_generations"] += 1

            # 提取答案数字
            nums = re.findall(r'-?\d+', resp)
            got = nums[-1] if nums else "NONE"

            if verify_answer(resp, exp):
                # 保存正确答案用于训练
                train_text = f"{p}{resp}"
                collected.append({
                    "text": train_text,
                    "question": q,
                    "answer": exp,
                })

                print(f"{idx+1:<8} {exp:<10} {got:<10} {try_i+1:<6} ✓")
                stats["collected"] += 1
                found = True
                break

        if not found:
            print(f"{idx+1:<8} {exp:<10} {got:<10} {'FAIL':<6} ✗")
            stats["failed"] += 1

    stats["avg_generations_per_sample"] = stats["total_generations"] / len(dataset)

    print("\n" + "="*80)
    print(f"收集结果:")
    print(f"  成功率: {stats['collected']}/{stats['total_samples']} = {stats['collected']/stats['total_samples']*100:.1f}%")
    print(f"  总生成次数: {stats['total_generations']}")
    print(f"  平均生成次数/样本: {stats['avg_generations_per_sample']:.1f}")
    print("="*80 + "\n")

    return collected, stats


# ============================================================================
# 3. SFT训练
# ============================================================================

def train_on_collected_data(
    model,
    tokenizer,
    training_samples: List[Dict],
    output_dir: str = "./rejection_output"
):
    """在筛选后的数据上训练模型"""

    print("="*80)
    print("SFT Training on Filtered Data")
    print("="*80)
    print(f"训练样本数: {len(training_samples)}")

    if len(training_samples) < 10:
        print("⚠ 样本太少，跳过训练")
        return None

    # 转换为huggingface dataset格式
    dataset = Dataset.from_list(training_samples)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    print("Tokenizing...")
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # 配置训练
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    print("Setup training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,  # Conservative LR
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
        max_grad_norm=1.0,  # Gradient clipping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n" + "="*80)
    print("Training...")
    print("="*80)
    train_result = trainer.train()

    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    print(f"最终损失: {train_result.training_loss:.4f}")
    print(f"训练步数: {train_result.global_step}")

    # 保存模型
    model.save_pretrained(f"{output_dir}/model")
    tokenizer.save_pretrained(f"{output_dir}/model")
    print(f"\n✓ Model saved to: {output_dir}/model")

    return train_result


# ============================================================================
# 4. 评估
# ============================================================================

def evaluate_model(
    model,
    tokenizer,
    eval_dataset: SimpleMathDataset,
    max_samples: int = 20
):
    """评估训练好的模型"""

    print("\n" + "="*80)
    print("评估模型性能")
    print("="*80)

    correct = 0
    total = min(len(eval_dataset), max_samples)

    for i in range(total):
        _, p, exp = eval_dataset.get_example_question(i)

        inputs = tokenizer(p, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        nums = re.findall(r'-?\d+', resp)
        got = nums[-1] if nums else "NONE"
        res = verify_answer(resp, exp)

        status = "✓" if res else "✗"
        print(f"[{i+1:2d}/{total}] Expected: {exp:>3s}, Got: {got:>4s} {status}")

        if res:
            correct += 1

    print("\n" + "="*80)
    print(f"评估结果: {correct}/{total} = {correct/total*100:.1f}%")
    print("="*80)

    return correct / total


# ============================================================================
# 5. 主流程
# ============================================================================

def main():
    """完整Rejection SFT流程"""

    print("\n" + "="*80)
    print("Rejection SFT - 完整整合版")
    print("="*80)
    print("模型: Qwen/Qwen2-0.5B-Instruct")
    print("数据集: 100以内加减法")
    print("="*80 + "\n")

    # 配置
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
    NUM_TRAIN_SAMPLES = 100
    NUM_EVAL_SAMPLES = 20
    OUTPUT_DIR = "./rejection_sft_output"

    # Step 1: 加载模型
    print("[1/4] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Loaded: {MODEL_NAME}\n")

    # Step 2: 准备数据集
    print("[2/4] 准备数据集...")
    train_dataset = SimpleMathDataset(num_samples=NUM_TRAIN_SAMPLES, seed=42)
    eval_dataset = SimpleMathDataset(num_samples=NUM_EVAL_SAMPLES, seed=1000)
    print(f"✓ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}\n")

    # Step 3: Rejection sampling收集高质量数据
    print("[3/4] Rejection sampling收集数据...")
    training_samples, collection_stats = rejection_collect_data(
        model, tokenizer, train_dataset, max_generations_per_sample=10
    )

    # Step 4: 训练
    if len(training_samples) >= 10:
        print("[4/4] 训练...")
        train_result = train_on_collected_data(
            model, tokenizer, training_samples, output_dir=OUTPUT_DIR
        )

        # 评估
        eval_acc = evaluate_model(model, tokenizer, eval_dataset, max_samples=NUM_EVAL_SAMPLES)

        # 保存结果
        results = {
            "collection_stats": collection_stats,
            "train_result": {
                "loss": train_result.training_loss,
                "steps": train_result.global_step,
            } if train_result else None,
            "eval_accuracy": eval_acc,
        }

        with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ All results saved to: {OUTPUT_DIR}/")
    else:
        print("⚠ 收集样本不足，跳过训练")

    print("\n" + "="*80)
    print("Rejection SFT Demo Complete!")
    print("="*80)


if __name__ == "__main__":
    main()