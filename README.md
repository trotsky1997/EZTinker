# EZTinker

A **minimal Tinker** clone for distributed model training - **用户在本地写训练循环/算法，服务端负责把操作可靠地跑在 GPU 集群上**。

## MVP 核心功能

✨ **四核心 API 原语**（完全对齐 Tinker 设计）:
- `forward_backward()`: 前向 + 反向、梯度累积
- `optim_step()`: 参数更新
- `sample()`: 推理采样
- `save_state()` / `load_state()`: 保存/加载 checkpoints

✨ **LoRA 适配训练**: （节省成本、快速迭代）
- Base model 只读加载
- LoRA adapter 可训练
- 支持 checkpoint 断点续训

✨ **Job/Future 模式**:
- 异步提交训练任务
- 轮询获取结果
- 可靠的异步执行

## 架构设计

```
EZTinker Service
├── 🚀 API Layer (FastAPI)
├── 🧠 Training Engine (PyTorch + LoRA)
├── 🔮 Sampling Engine (Inference)
├── 💾 Checkpoint Manager
└── 🖥️  CLI (Typer)

核心数据流:
Client <--HTTP--> API <--State--> TrainingRun <--LoRA--> Model
                               └--Sampler--> Inference
```

## 安装

```bash
# 使用 uv 创建项目
uv init --lib eztinker

# 安装依赖
uv add fastapi uvicorn typer pydantic torch transformers peft accelerate redis

# 安装 Ruff
uv add --dev ruff
```

## 快速开始

### 1. 启动 EZTinker 服务

```bash
uv run eztinker server
# 或在开发模式（自动重载）
uv run --reload eztinker server --reload
```

服务启动在 `http://localhost:8000`，你可以：
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 2. 创建训练会话

```python
import requests

# 创建基于 GPT-2 的训练会话
response = requests.post(
    "http://localhost:8000/v1/runs",
    json={
        "base_model": "gpt2",
    }
)
run_id = response.json()["run_id"]
print(f"Training run created: {run_id}")
```

### 3. 执行训练循环

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 准备数据
text = "This is a training example"
batch = tokenizer(text, return_tensors="pt")

# Tinker 风格：用户写循环逻辑
for _ in range(10):
    # 1. forward + backward + accumulation
    fb_response = requests.post(
        f"http://localhost:8000/v1/runs/{run_id}/forward_backward",
        json={
            "input_ids": batch["input_ids"].tolist()[0],
            "target_ids": batch["input_ids"].tolist()[0],
        }
    )
    job_id = fb_response.json()["job_id"]

    # 2. 等待梯度累积完成
    # ... 这里可以异步优化 (polling / callback)

    # 3. optimizer step
    optim_response = requests.post(
        f"http://localhost:8000/v1/runs/{run_id}/optim_step",
        json={
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
        }
    )

    print(f"Step: {_}, Loss: ...")
```

### 4. 生成评测

```python
# 推理采样
response = requests.post(
    "http://localhost:8000/v1/sample",
    json={
        "prompt": "Once upon a time,",
        "max_new_tokens": 50,
        "temperature": 0.7,
    }
)
print(f"Generated: {response.json()}")
```

### 5. 保存和加载 checkpoint

```python
# 保存
save_response = requests.post(
    f"http://localhost:8000/v1/runs/{run_id}/save",
    json={"name": "checkpoint_v1"}
)
print(save_response.json())

# 文件保存在: checkpoints/{run_id}/checkpoint_v1.adapter.pt
#               checkpoints/{run_id}/checkpoint_v1.optimizer.pt
```

## CLI 使用

```bash
# 启动服务器
uv run eztinker server

# 创建训练会话
uv run eztinker create --model gpt2

# 列出所有会话
uv run eztinker list

# 生成 sample
uv run eztinker sample "Once upon a time," --max-tokens 50

# 保存 checkpoint
uv run eztinker save my_run_id checkpoint_v1

# 删除会话
uv run eztinker delete my_run_id
```

## Directory Structure

```
eztinker/
├── src/eztinker/
│   ├── api/          # FastAPI server
│   ├── engine/       # Training & sampling engines
│   ├── models/       # Pydantic models
│   ├── core/         # State + checkpoint mgmt
│   └── cli/          # CLI (typer)
├── checkpoints/      # Checkpoint files
├── .env              # Environment config
├── pyproject.toml    # uv project config
└── README.md
```

## Environment Variables

```bash
# Set checkpoint directory (default: checkpoints)
export CHECKPOINTS_DIR=/path/to/checkpoints
export EZTINKER_BASE_URL=http://localhost:8000

# GPU support (CUDA)
export CUDA_VISIBLE_DEVICES=0
```

## 开发工具链 (uv + ruff + ty) 🚀

我们使用**现代 Python 最快速**的开发工具链以获得最佳开发体验：

- **uv**: 极速包管理和项目构建 (Rust 实现，100x faster than pip)
- **ruff**: 极速的 Python linter 和格式化器 (1000x faster than black + isort + autoflake + ...)
- **ty** (astral-sh/ty): 极速类型检查器 (100x faster than mypy)

### 配置文件

项目已配置完善的工具配置文件：
- `.ruff.toml` - Ruff 格式化和 lint 配置
- `.ty.toml` - Ty 类型检查配置
- `pyproject.toml` - 项目依赖和打包配置

### 开发工作流

```bash
# 1. 格式化代码 (auto-format)
uv run ruff format src/

# 2. Lint 检查 (static analysis)
uv run ruff check src/

# 3. 类型检查 (runtime correctness)
uv run ty check

# 4. 自动修复所有 lint 问题
uv run ruff check src/ --fix

# 5. 手动测试
uv run eztinker server --reload

# 6. 完整的质量检查 (format + lint + type)
uv run ruff format src/ && uv run ruff check src/ && uv run ty check
```

### 环境变量配置示例

```bash
# .bashrc / .zshrc
# 配置便捷的 shell alias
alias ezt-lint='uv run ruff check src/'
alias ezt-fmt='uv run ruff format src/'
alias ezt-type='uv run ty check'
alias ezt-qc='ezt-fmt && ezt-lint && ezt-type'
alias ezt-dev='uv run eztinker server --reload'
alias ezt-add='uv add'
alias ezt-rm='uv remove'
```

### 典型的上手流程

```bash
# 1. 添加新依赖
uv add <package-name>

# 2. 格式化所有代码
ezt-fmt

# 3. 运行 linter
ezt-lint

# 4. 类型检查
ezt-type

# 5. 启动开发服务器
ezt-dev
```



## TODO (Future Enhancements)

- [ ] **LoRA Loading**: Load adapter from checkpoint `.adapter.pt` to sampler
- [ ] **Batch Training**: Batch processing for `forward_backward`
- [ ] **Multi-GPU**: Distributed training support
- [ ] **OpenAI Compatible API**: Inference interface support
- [ ] **Web UI Console**: Visualize training state
- [ ] **Scheduler**: Clock cycle scheduling like Tinker
- [ ] **Advanced Losses**: PPO/CISPO/DRO RL losses

## EZTinker vs Tinker

| Feature | EZTinker (MVP) | Tinker (Full) |
|---------|----------------|---------------|
| ✅ LoRA Fine-tuning | ✅ | ✅ |
| ✅ Checkpoint Management | ✅ | ✅ |
| ✅ Async/Future Pattern | ✅ | ✅ |
| ❌ Multi-GPU Worker Pool | ❌ | ✅ |
| ❌ Clock Cycle Scheduler | ❌ | ✅ |
| ❌ OpenAI Compatible | ❌ | ✅ |
| ❌ Custom Losses | ❌ | ✅ |

## License

MIT License - free to use, modify, distribute.# EZTinker
# EZTinker
