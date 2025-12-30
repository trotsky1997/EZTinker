#!/usr/bin/env python3
"""
Rejection Sampling for GSM8K Math Training

说白了就是：
1. 每个问题生成多个答案
2. 用math-verify验证哪个对
3. 只训练对的答案（过滤错的）

比标准SFT多一步sample和验证而已
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eztinker.dataset.gsm8k import GSM8KDataset


def math_verify(question: str, answer_text: str, expected_answer: str) -> bool:
    """简单的数学验证，检查是否包含了正确答案。

    Production就用真正的math verification工具
    """
    # 提取数字（简单的正则版本）
    import re

    numbers = re.findall(r"-?\d+\.?\d*", answer_text)

    # 看看答案里有没有正确答案的数字
    if expected_answer in numbers:
        return True

    # 如果用户明确的答案格式
    expected_words = [
        f"the answer is {expected_answer}",
        f"answer is {expected_answer}",
        f"answer: {expected_answer}",
        f"is {expected_answer}",
    ]

    answer_lower = answer_text.lower()
    return any(word in answer_lower for word in expected_words)


def rejection_sft(mini_batch_size: int = 4, max_tries: int = 3):
    """Rejection SFT核心逻辑

    Args:
        mini_batch_size: 生成几个答案选最好的
        max_tries: 最多试几次找正确的答案
    """
    print("=" * 80)
    print("Rejection Sampling SFT - GSM8K")
    print("=" * 80)
    print(f"mini_batch_size: {mini_batch_size}, max_tries: {max_tries}")
    print()

    # 1. Load GMSK
    gsm8k = GSM8KDataset(split="train", max_samples=5)
    print(f"Loaded {len(gsm8k)} examples\n")

    # 2. Load model (Qwen2-0.5B)
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded\n")

    # 3. Temperature sampling生成多个答案
    accepted_count = 0
    total_generated = 0

    for i in range(len(gsm8k)):
        question, prompt, expected_answer = gsm8k.get_example_question(i)

        print(f"Example {i + 1}: {question[:60]}...")
        print(f"Expected: {expected_answer}")

        # 生成多个候选，看哪个对
        candidates = []
        for try_idx in range(max_tries):
            # 用不同temperature生成
            temp = 0.3 + try_idx * 0.2  # 0.3, 0.5, 0.7

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=temp,
                    do_sample=True,
                    num_return_sequences=1,
                )

            response = tokenizer.decode(outputs[0])
            total_generated += 1

            # 验证答案
            is_correct = math_verify(question, response, expected_answer)

            if is_correct:
                print(f"  ✓ Try {try_idx + 1}: CORRECT (temp={temp:.1f})")
                accepted_count += 1

                # 保存对的答案用于训练
                # （这里就显示逻辑，实际训练用EZTinker的forward_backward等API）
                candidates.append(response)
                break
            else:
                print(f"  ✗ Try {try_idx + 1}: WRONG (temp={temp:.1f})")

        if not candidates:
            print(f"  ⚠ Failed to generate correct answer after {max_tries} tries")

        print()

    # 4. Summary
    print("=" * 80)
    print(f"Accepted: {accepted_count}/{len(gsm8k)} = {accepted_count / len(gsm8k) * 100:.1f}%")
    print(f"Total generations: {total_generated}")
    print("=" * 80)

    print("\n训练数据这样准备:")
    print("- 对的问题: 用SFT training")
    print("- 错的问题: 丢弃 or 负向sample")
    print("- 然后在EZTinker里运行gsm8k_sft.py那套就完事了")


if __name__ == "__main__":
    # 就这么简单：mini_batch_size个答案，选第一个对的
    rejection_sft(mini_batch_size=4, max_tries=3)
