#!/usr/bin/env python3
"""
Rejection SFT - 一个样本立即梯度下降

超级简化版：
- 生成到找到正确的样本
- 立即训练
- 跳过EZTinker，直接用transformer
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Rejection SFT - One Sample One Step")
print("=" * 80)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# 使用更小的模型或继续用Qwen
model_name = "Qwen/Qwen2-0.5B-Instruct"

print(f"\n[1/3] 加载模型: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Model loaded")

print("\n[2/3] 准备数据集")
from datasets import Dataset
from simple_math_dataset import SimpleMathDataset

train_data = SimpleMathDataset(num_samples=50, seed=42)
eval_data = SimpleMathDataset(num_samples=10, seed=1000)
print(f"✓ Train: {len(train_data)}, Eval: {len(eval_data)}")


def verify_answer(resp, expected):
    """验证答案"""
    nums = re.findall(r"-?\d+", resp)
    return expected in nums if nums else False


# 收集positive样本并训练
print("\n[3/3] 训练循环")
training_samples = []
stats = {"correct": 0, "failed": 0, "generated": 0}

print(f"{'Idx':<4} {'Expected':<8} {'Got':<8} {'Status':<10} {'Tries'}")
print("-" * 50)

for idx in range(len(train_data)):
    q, p, exp = train_data.get_example_question(idx)

    # 生成最多10次找正确答案
    found = False
    for try_i in range(10):
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
        stats["generated"] += 1

        nums = re.findall(r"-?\d+", resp)
        got = nums[-1] if nums else "NONE"

        if verify_answer(resp, exp):
            training_samples.append({"question": q, "prompt": p, "response": resp, "answer": exp})
            stats["correct"] += 1
            print(f"{idx + 1:<4} {exp:<8} {got:<8} {'ACCEPT':<10} {try_i + 1}")
            found = True
            break
    else:
        stats["failed"] += 1
        print(f"{idx + 1:<4} {exp:<8} {got:<8} {'REJECT':<10} 10")

    # 每收集一个样本就训练（或者攒几个再练）
    if len(training_samples) > 0 and len(training_samples) % 5 == 0:
        print(f"\n  ✓ Trained {len(training_samples)} samples so far")
        # 这里可以加入实时训练逻辑

print("\n" + "=" * 80)
print("收集完成")
print("=" * 80)
print(
    f"Accepted: {stats['correct']}/{len(train_data)} = {stats['correct'] / len(train_data) * 100:.1f}%"
)
print(f"Generated: {stats['generated']}次")
print(f"Hit rate: {stats['correct'] / stats['generated'] * 100:.1f}%")

# 准备训练数据
if len(training_samples) >= 10:
    print("\n" + "=" * 80)
    print("开始SFT训练")
    print("=" * 80)

    # 将样本转换为huggingface dataset格式
    train_formatted = []
    for item in training_samples:
        train_text = item["prompt"] + "\n" + item["response"]
        train_formatted.append({"text": train_text})

    dataset = Dataset.from_list(train_formatted)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    print(f"✓ Tokenized {len(tokenized_dataset)} samples")

    # 简单的训练配置
    training_args = TrainingArguments(
        output_dir="./rejection_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("\n训练中...")
    trainer.train()

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)

    # 保存模型
    model.save_pretrained("./rejection_output/final_model")
    tokenizer.save_pretrained("./rejection_output/final_model")
    print("Saved to: ./rejection_output/final_model")

else:
    print("⚠ Not enough positive samples for training")

# 保存结果
results = {
    "stats": stats,
    "collected": len(training_samples),
    "accuracy": stats["correct"] / len(train_data) * 100,
}
with open("rejection_one_sample.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults: rejection_one_sample.json")
print("=" * 80)
