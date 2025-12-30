#!/usr/bin/env python3
"""
简单的100以内加减法数据集生成器
"""

import json
import random


class SimpleMathDataset:
    """生成100以内的加减法"""

    def __init__(self, num_samples: int = 256, seed: int = 42):
        random.seed(seed)
        self.samples = self._generate_samples(num_samples)

    def _generate_samples(self, num_samples):
        """生成样本"""
        samples = []

        for i in range(num_samples):
            # 50%加法，50%减法
            if random.random() < 0.5:
                a = random.randint(1, 99)
                b = random.randint(1, 100 - a)
                question = f"What is {a} + {b}?"
                answer = str(a + b)
            else:
                a = random.randint(1, 99)
                b = random.randint(1, a)
                question = f"What is {a} - {b}?"
                answer = str(a - b)

            prompt = f"Please solve the following math problem step by step and provide your final answer.\n\nQuestion: {question}"

            samples.append(
                {
                    "question": question,
                    "prompt": prompt,
                    "answer": answer,
                    "full_text": f"{prompt}\nAnswer: Let me solve this. {question.replace('What is ', '')} = {answer}. Therefore, the answer is {answer}.",
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_example_question(self, idx):
        """兼容GSM8K接口"""
        item = self.samples[idx]
        return item["question"], item["prompt"], item["answer"]

    def get_full_text(self, idx):
        """获取完整格式化的文本"""
        return self.samples[idx]["full_text"]

    def save(self, filename: str):
        """保存数据集"""
        with open(filename, "w") as f:
            json.dump(self.samples, f, indent=2)


def test_dataset():
    """测试数据集"""
    import sys

    sys.path.insert(0, ".")

    print("Testing SimpleMathDataset...")
    dataset = SimpleMathDataset(num_samples=20)

    print(f"Generated {len(dataset)} samples\n")

    for i in range(5):
        q, p, a = dataset.get_example_question(i)
        print(f"[{i + 1}] Q: {q}")
        print(f"    A: {a}")
        print()

    dataset.save("simple_math_train_20.json")
    print("Saved to simple_math_train_20.json")


if __name__ == "__main__":
    test_dataset()
