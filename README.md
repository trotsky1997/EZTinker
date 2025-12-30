# EZTinker

A **minimal Tinker** clone for distributed model training - **用户在本地写训练循环/算法，服务端负责把操作可靠地跑在 GPU 集群上**。

## 🎯 核心功能

✨ **四核心 API 原语**（完全对齐 Tinker 设计）:
- `forward_backward()`: 前向 + 反向、梯度累积
- `optim_step()`: 参数更新
- `sample()`: 推理采样
- `save_checkpoint()` / `load_checkpoint()`: 保存/加载 LoRA adapter + optimizer

✨ **LoRA Fine-tuning 训练**:
- ✅ Base model 只读加载（省显存）
- ✅ LoRA adapter 高效训练（rank=1-8, alpha=2-16）
- ✅ 支持 Qwen2/GPT2/Phi-2 等主流模型
- ✅ 支持多种 LoRA 配置（rank, alpha, dropout, target_modules）
- ✅ 完整 checkpoint 支持（断点续训 + 多个检查点）

✨ **标准化 Loss Function 接口** (Protocol-based):
- ✅ 类型安全：固定参数签名 `(logits, labels, weights=None) -> Tensor`
- ✅ 5种内置损失：cross_entropy, weighted_cross_entropy, focal_loss, smooth_l1, contrastive_loss
- ✅ 程序化注册：`register_loss_function(name, func)`
- ✅ 无需字符串注入：更安全可维护
- ✅ 完整类型检查：IDE自动补全和验证

✨ **Job/Future 异步模式**:
- 低延迟异步提交训练任务
- 轮询获取结果
- 可靠的执行（失败自动回退）

✨ **数据集支持**:
- ✅ GSM8K: 数学问题数据集
- ✅ ShareGPT: 对话格式，支持多种方言
- 🔄 扩展性强：统一的 Dataset 接口

## 🏗️ 架构设计

```
EZTinker 服务
├── 🚀 API Layer (FastAPI) - 提供 RESTful 接口
├── 🧠 Training Engine (PyTorch + LoRA + Loss Functions)
│   ├── TrainingRun: 训练状态管理
│   ├── Loss Functions: 标准化损失函数协议
│   └── Model Manager: LoRA adapter + Base model
├── 🔮 Sampling Engine (Inference) - 独立采样服务
├── 💾 Checkpoint Manager - Adapter + Optimizer 保存/恢复
└── 🖥️  CLI (Typer) - 命令行工具

核心数据流:
Client ←HTTP→ API ←State→ TrainingRun ←LoRA→ Model → Training
                           ↑                              ↓
                         Loss Function Protocol ←--[logits, labels]
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

## 🚀 快速开始

### 1. 启动 EZTinker 服务

```bash
# 启动服务器
uv run eztinker server

# 开发模式（自动重载）
uv run --reload eztinker server --reload
```

服务启动在 `http://localhost:8000`，你可以：
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 2. 使用 Python Client API (推荐)

```python
from eztinker import EZTinkerClient, LossFunctionConfig

# 创建客户端
with EZTinkerClient(base_url="http://localhost:8000") as client:
    # 健康检查
    print(client.health())

    # 创建训练 run (默认使用 cross_entropy loss)
    run_id = client.create_run(
        base_model="Qwen/Qwen2-0.5B-Instruct",
        lora_config={"r": 1, "lora_alpha": 2, "lora_dropout": 0.05}
    )
    print(f"Training run created: {run_id}")

    # 或使用自定义 loss function
    run_id = client.create_run(
        base_model="gpt2",
        lora_config={"r": 8},
        loss_config=LossFunctionConfig(
            loss_type="focal_loss",
            focal_alpha=0.3,
            focal_gamma=2.5
        )
    )

    # 生成文本
    text = client.sample("Hello world", max_new_tokens=50, temperature=0.8)
    print(text)

    # 获取所有 runs
    runs = client.get_runs()
    print(runs)
```

### 3. 训练循环

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "This is a training example"
input_ids = tokenizer(text)["input_ids"]

# Tinker 风格：用户在客户端写循环逻辑
for step in range(10):
    # 1. forward + backward + accumulation
    fb_response = client.forward_backward(run_id, input_ids=input_ids, target_ids=input_ids)
    job_id = fb_response["job_id"]

    # 2. optimizer step
    optim_response = client.optim_step(run_id, learning_rate=2e-4, weight_decay=0.01)
    print(f"Step: {step}, Status: {optim_response['status']}")
```

### 4. 保存 Checkpoint

```python
# 保存当前 adapter 和 optimizer
save_response = client.save_checkpoint(run_id, name="checkpoint_v1")
print(save_response)
# {"status": "completed", "adapter_path": "...", "optimizer_path": "..."}

# 检查点保存在: checkpoints/{run_id}/checkpoint_v1.adapter.pt
#               checkpoints/{run_id}/checkpoint_v1.optimizer.pt
```

### 5. 使用自定义 Loss Function

```python
import torch
from eztinker.engine import register_loss_function, get_loss_function

# 定义自定义损失函数（遵循 LossFunction Protocol）
def my_custom_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Custom loss with temperature scaling."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Apply temperature
    scaled_logits = shift_logits / temperature

    # Compute cross-entropy
    loss = torch.nn.functional.cross_entropy(
        scaled_logits.view(-1, scaled_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=kwargs.get("ignore_index", -100),
    )
    return loss

# 注册自定义损失函数
register_loss_function("temperature_scaled", my_custom_loss)

# 使用自定义损失函数创建 run
run_id = client.create_run(
    base_model="gpt2",
    loss_config=LossFunctionConfig(
        loss_type="temperature_scaled",
        # 额外的参数会传递给 kwargs
        # temperature 需要在 kwargs 初始化时指定
    )
)
```

## 📚 API 参考文档

运行 `uv run nox -s docs` 自动生成完整的 API 文档到 `/documents` 目录。
文档包含所有模块、类、方法的详细说明和示例。

```bash
# 生成文档
uv run nox -s docs

# 查看文档（浏览器打开）
open documents/eztinker.html

# 或运行本地服务器
cd documents && python -m http.server 8000
```

## 📦 目录结构

```
eztinker/
├── src/eztinker/
│   ├── api/           # FastAPI server (RESTful endpoints)
│   ├── engine/        # Training & sampling engines
│   │   ├── loss.py        # Standardized loss function interface
│   │   ├── run_manager.py # TrainingRun management
│   │   └── sampler.py     # Inference sampling
│   ├── models/        # Pydantic schemas
│   │   └── api.py         # API models (LoRAConfig, LossFunctionConfig, etc.)
│   ├── core/          # State management
│   ├── dataset/       # Dataset loaders (GSM8K, ShareGPT)
│   ├── rl/            # Rejection sampling utilities
│   └── client.py      # EZTinkerClient API
├── checkpoints/       # Checkpoint files (gitignored)
├── documents/         # Auto-generated API docs (gitignored)
├── tests/             # Comprehensive test suite (32 tests)
├── .ruff.toml         # Ruff configuration
├── .ty.toml           # Ty type checker configuration
├── pyproject.toml     # Project configuration
└── README.md
```

## 🔧 开发工具链

使用现代 Python 最快速的开发工具链：

- **uv**: 极速包管理 (Rust 实现, 100x faster)
- **ruff**: 极速 linter 和 formatter (1000x faster than black+isort+flake8)
- **ty** (astral-sh/ty): 极速类型检查 (100x faster than mypy)

### 开发工作流

```bash
# 1. 一键格式化 + lint + 修复
uv run nox -s fix

# 2. 完整的质量检查流程（CI 自动化）
uv run nox

# 包括：
#   - format: 格式化代码
#   - lint: 静态分析
#   - type-check: 类型检查
#   - security: 安全扫描 (Semgrep)
#   - test: 运行测试
#   - docs: 生成 API 文档

# 3. 快速开发常用命令
uv run nox -s fmt          # 只格式化
uv run nox -s lint         # 只检查 lint
uv run nox -s type-check   # 只类型检查
uv run nox -s test-fast    # 运行快速测试（跳过慢测试）
```

### 环境变量配置示例

```bash
# Shell aliases (添加到 ~/.bashrc 或 ~/.zshrc)
alias ezt-lint='uv run ruff check src/'
alias ezt-fmt='uv run ruff format src/'
alias ezt-type='uv run ty check'
alias ezt-qc='ezt-fmt && ezt-lint && ezt-type'
alias ezt-dev='uv run eztinker server --reload'
```

## 🧪 测试

```bash
# 运行所有测试
uv run pytest tests/

# 运行快速测试（跳过 @pytest.mark.slow）
uv run pytest tests/ -m "not slow"

# 运行特定测试
uv run pytest tests/unit/test_api_server.py::TestCustomLossFunctions
```

测试覆盖:
- ✅ 32个单元测试和集成测试
- ✅ LoRA rank=1/Qwen2/LossFunction 兼容性测试
- ✅ 自定义损失函数测试（6个测试）
- ✅ API 字段验证测试

## 📊 EZTinker vs Tinker

| 特性 | EZTinker (当前) | Tinker (完整版) |
|------|-----------------|----------------|
| ✅ LoRA Fine-tuning | ✅ | ✅ |
| ✅ Checkpoint Management | ✅ | ✅ |
| ✅ Async/Future Pattern | ✅ | ✅ |
| ✅ Custom Loss Functions | ✅ (5内置 + 注册系统) | ✅ |
| ❌ Multi-GPU Worker Pool | ❌ | ✅ |
| ❌ Clock Cycle Scheduler | ❌ | ✅ |
| ❌ OpenAI Compatible | ❌ | ✅ |

## 🚧 TODO (未来增强)

- [ ] **Batch 训练**: Optimize forward_backward batch processing
- [ ] **Multi-GPU**: Distributed training support
- [ ] **OpenAI 兼容 API**: Inference API 兼容
- [ ] **Web UI**: 训练状态可视化
- [ ] **更多损失函数**: PPO/CISPO/DRO 强化学习损失
- [ ] **Scheduler**: Clock 周期调度（类 Tinker）

## 📄 License

MIT License - free to use, modify, distribute.