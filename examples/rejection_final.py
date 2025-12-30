#!/usr/bin/env python3
"""
Rejection SFT - 完整版（带正确训练）

前一部分完美：50/50 samples collected!
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("Rejection SFT - 简单数学 (已验证收集100%成功)")
print("="*80)

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

model_name = "Qwen/Qwen2-0.5B-Instruct"

print(f"\n[1/3] 模型: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Model loaded")

print("\n[2/3] 生成训练数据...")
from simple_math_dataset import SimpleMathDataset
from datasets import Dataset

train_data = SimpleMathDataset(num_samples=50, seed=42)

def verify_answer(resp, expected):
    nums = re.findall(r'-?\d+', resp)
    return expected in nums if nums else False

# 收集positive样本
print(f"{'Progress':<10} {'Expected':<8} {'Got':<8} {'Tries':<6}")
print("-" * 40)

training_samples = []
for idx in range(len(train_data)):
    q, p, exp = train_data.get_example_question(idx)

    # 生成最多10次
    for try_i in range(10):
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        nums = re.findall(r'-?\d+', resp)
        got = nums[-1] if nums else "NONE"

        if verify_answer(resp, exp):
            training_samples.append({"text": p + "\n" + resp})
            print(f"[{idx+1}/50]  {exp:<8} {got:<8} {try_i+1:<6}")
            break
    else:
        print(f"[{idx+1}/50]  {exp:<8} {got:<8} FAILED")

print(f"✓ Collected: {len(training_samples)}/50 samples")

# 准备训练数据
print("\n[3/3] SFT训练...")
dataset = Dataset.from_list(training_samples)

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

print("Tokenizing...")
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# 正确的训练配置（带language modeling loss）
print("Setup training...")

# 使用DataCollatorForLanguageModeling来处理labels
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM
)

training_args = TrainingArguments(
    output_dir="./rejection_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=1,
    report_to=[],
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\n" + "="*80)
print("训练中...")
print("="*80)
train_result = trainer.train()

print("\n" + "="*80)
print("训练完成！")
print("="*80)
print(f"训练损失: {train_result.training_loss:.4f}")
print(f"训练步数: {train_result.global_step}")

# 保存模型
model.save_pretrained("./rejection_output/final_model")
tokenizer.save_pretrained("./rejection_output/final_model")
print("\n✓ Model saved to: ./rejection_output/final_model")

# 简单evaluation
print("\n" + "="*80)
print("评估（10个样本，不sample直接greedy）...")
print("="*80)

eval_data = SimpleMathDataset(num_samples=10, seed=1000)
correct = 0

for i in range(len(eval_data)):
    _, p, exp = eval_data.get_example_question(i)

    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=150,
            do_sample=False,  # 禁用temperature sampling避免nan
            pad_token_id=tokenizer.eos_token_id
        )
    resp = tokenizer.decode(out[0], skip_special_tokens=True)

    got = re.findall(r'-?\d+', resp)
    res = exp in got if got else False
    print(f"[{i+1}/10] Expected: {exp}, Got: {got[-1] if got else 'NONE'}, {'✓' if res else '✗'}")
    if res:
        correct += 1

print(f"\n评估结果: {correct}/10 = {correct/10*100:.1f}%")
print("="*80)

results = {
    "collected": len(training_samples),
    "train_loss": train_result.training_loss,
    "eval_acc": correct/10*100,
}
with open("rejection_final.json", 'w') as f:
    json.dump(results, f, indent=2)

print("Results: rejection_final.json")