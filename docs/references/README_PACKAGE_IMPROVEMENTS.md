# EZTinker - Python 包改进

## 🎯 改进概览

### 问题
之前 EZTinker 的用户体验存在问题：
- 需要手动调用 HTTP API（不够优雅）
- CLI 功能有限
- 没有统一的 Python API

### 解决方案
现在 EZTinker 提供了三种使用方式：

## 1️⃣ 优雅的 Python Client API（推荐）

### 简单的开始

```python
from eztinker import EZTinkerClient

# 使用上下文管理器（自动清理资源）
with EZTinkerClient() as client:
    # 创建训练运行
    run_id = client.create_run("Qwen/Qwen2-0.5B-Instruct")

    # 生成文本
    text = client.sample("Hello!", max_new_tokens=100)

    print(text)
```

### 完整的训练循环示例

```python
from eztinker import EZTinkerClient, GSM8KDataset

# 1. 初始化
client = EZTinkerClient()

# 2. 加载数据集
dataset = GSM8KDataset(split="train", max_samples=100)

# 3. 创建训练运行
run_id = client.create_run(
    base_model="Qwen/Qwen2-0.5B-Instruct",
    lora_rank=1,
    lora_alpha=2
)

# 4. 训练循环
for i in range(len(dataset)):
    question, prompt, ground_truth = dataset.get_example_question(i)

    # 分词
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].tolist()[0]

    # 训练步骤
    client.forward_backward(run_id, input_ids)
    client.optim_step(run_id, learning_rate=2e-4)

# 5. 保存检查点
client.save_checkpoint(run_id, "checkpoint_final")

# 6. 关闭客户端
client.close()
```

## 2️⃣ 强大的 CLI 工具

### CLI 命令概览

```bash
# 启动服务器
eztinker server

# 版本信息
eztinker version

# 服务器状态
eztinker health
eztinker status

# 训练运行管理
eztinker create --model MODEL         # 创建运行
eztinker list-runs                    # 列出所有运行
eztinker delete RUN_ID                # 删除运行

# 推理
eztinker sample "Your prompt"         # 生成文本

# 检查点管理
eztinker save RUN_ID NAME             # 保存检查点
eztinker checkpoints                  # 列出检查点
eztinker checkpoints --run-id RUN_ID  # 列出指定运行的检查点

# 演示
eztinker demo                         # 运行拒绝采样演示
```

### CLI 使用示例

```bash
# 终端 1: 启动服务器
eztinker server \
    --host 127.0.0.1 \
    --port 8080 \
    --workers 4 \
    --checkpoints-dir data/checkpoints

# 终端 2: 创建训练运行
eztinker create \
    --model Qwen/Qwen2-0.5B-Instruct \
    --run-id custom123

# 使用模型
eztinker sample "你好世界！" \
    --max-tokens 200 \
    --temperature 0.8

# 查看状态
eztinker status

# 运行演示
eztinker demo
```

## 3️⃣ 原始 HTTP API（仍然可用，但不推荐）

```python
import requests

# 创建运行
response = requests.post(
    "http://localhost:8000/v1/runs",
    json={"base_model": "Qwen/Qwen2-0.5B-Instruct"}
)
run_id = response.json()["run_id"]

# 推理
response = requests.post(
    "http://localhost:8000/v1/sample",
    json={
        "prompt": "Hello!",
        "max_new_tokens": 100
    }
)
job_id = response.json()["job_id"]

# 轮询
result = requests.get(f"http://localhost:8000/v1/jobs/{job_id}").json()
# ...
```

## 📦 安装和使用

### 本地安装

```bash
# 1. 确保在项目目录
cd /path/to/eztinker

# 2. 安装开发版本
uv sync  # 或 pip install -e .

# 3. 测试安装
eztinker --help
uv run python -c "from eztinker import EZTinkerClient; print('✓ 安装成功')"
```

### Python 脚本使用

```python
# my_training.py
from eztinker import (
    EZTinkerClient,
    GSM8KDataset,
    ShareGPTDataset
)
from transformers import AutoTokenizer

# 1. 准备
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# 2. 选择数据集
## GSM8K
dataset = GSM8KDataset(split="train", max_samples=100)

## ShareGPT
sharegpt = ShareGPTDataset(
    file_path="data.json",
    tokenizer=tokenizer,
    max_samples=100
)

# 3. 训练
with EZTinkerClient() as client:
    run_id = client.create_run("Qwen/Qwen2-0.5B-Instruct", lora_rank=8)
    # ... 你的训练代码
```

## 🔧 数据集集成

### GSM8K 数据集

```python
from eztinker import GSM8KDataset

dataset = GSM8KDataset(
    split="train",           # 或 "test"
    max_samples=100,
    use_math_verify=True     # 使用 Math-Verify 评估
)

question, prompt, answer = dataset.get_example_question(0)
print(question)  # 数学问题
print(prompt)    # 格式化的提示
print(answer)    # 正确答案

# 评估生成
result = dataset.evaluate_answer(generated_text, answer, question)
score = result['score']  # 准确度分数
```

### ShareGPT 数据集

```python
from eztinker import ShareGPTDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# 支持两种方言格式：
# - Dialect A: from/value (原始 ShareGPT)
# - Dialect B: role/content (OpenAI 风格)

dataset = ShareGPTDataset(
    file_path="data.json",  # 或 data.jsonl
    tokenizer=tokenizer,
    max_samples=100,
    strict=True  # 严格的验证
)

# 统计数据
print(dataset.stats)
"""
{
    'total_loaded': 100,
    'valid_conversations': 99,
    'invalid_conversations': 1,
    'total_turns': 500,
    'dialect_counts': {
        'from_value': 30,
        'role_content': 70
    }
}
"""

# 获取对话
conv_id, formatted_text, num_turns = dataset.get_conversation_turns(0)
print(formatted_text)  # Qwen2 模板格式化
```

## 🚀 拒绝采样训练

### 提供的演示脚本

支持 **GSM8K** 和 **ShareGPT**:

```bash
# 使用 GSM8K 数据集
uv run python rejection_sft_demo.py \
    --max-samples 50 \
    --num-candidates 4 \
    --epochs 3

# 使用 ShareGPT 数据集
uv run python rejection_sft_demo_sharegpt.py \
    --dataset-type sharegpt \
    --data-path examples/sharegpt_dialect_b.json \
    --max-samples 50 \
    --epochs 3
```

### 直接代码使用

```python
from eztinker import (
    create_training_run,
    generate_candidates,
    select_best_candidate_and_train,
    GSM8KDataset
)

# 1. 加载数据集
dataset = GSM8KDataset(split="train", max_samples=100)

# 2. 创建运行
run_id = create_training_run(
    "Qwen/Qwen2-0.5B-Instruct",
    lora_rank=1
)

# 3. 处理示例
for i in range(len(dataset)):
    question, prompt, ground_truth = dataset.get_example_question(i)

    # 生成候选
    candidates = generate_candidates(
        prompt=prompt,
        question=question,
        run_id=run_id,
        num_candidates=4,
        temperature=0.8
    )

    # 选择最佳并训练
    result = select_best_candidate_and_train(
        run_id=run_id,
        prompt=prompt,
        candidates=candidates,
        ground_truth=ground_truth,
        question=question,
        dataset=dataset,
        learning_rate=2e-4
    )

    print(f"Candidate score: {result['selected_score']:.3f}")
```

## 📝 完整的工作流程示例

### 训练一个新模型

```bash
# 终端 1: 启动服务器
eztinker server --checkpoints-dir my_checkpoints &

# 终端 2: 运行训练
python my_training.py

# 终端 3 (可选): 监控进度
eztinker status
eztinker checkpoints
```

### 使用 Jupyter Notebook

```python
from eztinker import EZTinkerClient, GSM8KDataset

# 初始化
client = EZTinkerClient()
dataset = GSM8KDataset(split="train", max_samples=100)
run_id = client.create_run("Qwen/Qwen2-0.5B-Instruct", lora_rank=1)

# 小批次训练
for i in range(10):
    question, prompt, ground_truth = dataset.get_example_question(i)

    # 生成
    generated = client.sample(prompt, max_new_tokens=200)

    # 评估
    result = dataset.evaluate_answer(generated, ground_truth, question)

    print(f"Sample {i}: Score = {result['score']:.3f}")

    # 训练（如果满足条件）
    # ...

# 保存
client.save_checkpoint(run_id, "notebook_run")
```

## 📚 更多示例

查看 `examples/example_client_api.py` 获取完整的交互式示例:

```bash
uv run python examples/example_client_api.py
```

这个脚本演示了:
1. Server health checking
2. Dataset loading (GSM8K & ShareGPT)
3. Training run creation
4. Sample generation
5. Rejection sampling workflow
6. HTTP API vs Client API comparison

## 🎉 总结

现在 EZTinker 提供了完整个 PyPI 包的方法：

✅ **优雅的客户端 API** - 语法糖和上下文管理器
✅ **强大的 CLI** - 完整的服务管理、监控命令
✅ **多种数据集支持** - GSM8K、ShareGPT (JSON/JSONL)
✅ **完整的工作流程** - 演示脚本、测试案例
✅ **结构化导出** - 轻松的模块导入
✅ **优秀文档** - 代码示例、使用指南、类型提示

不需要再手动处理 HTTP API - 只需用 `from eztinker import EZTinkerClient` 就能开始！