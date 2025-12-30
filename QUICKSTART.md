# EZTinker Quick Start Guide

## 安装和运行（3 步）

### 1. 安装依赖
```bash
uv add fastapi uvicorn typer pydantic torch transformers peft
```

### 2. 启动服务器
```bash
uv run eztinker server
```

### 3. 测试（新终端）
```bash
# 创建训练会话
uv run eztinker create

# 生成示例
uv run eztinker sample "Hello world"

# 列出所有会话
uv run eztinker list-runs
```

## REST API 使用（对齐 Tinker 四原语）

### 1. 创建训练会话
```bash
curl -X POST http://localhost:8000/v1/runs \\
  -H "Content-Type: application/json" \\
  -d '{"base_model": "gpt2"}'
```

Response:
```json
{
  "run_id": "run_abc123",
  "status": "created",
  "message": "Training session initialized"
}
```

### 2. Forward + Backward（梯度累积）
```bash
curl -X POST http://localhost:8000/v1/runs/run_abc123/forward_backward \\
  -H "Content-Type: application/json" \\
  -d '{
    "input_ids": [101, 102, 103, ...],
    "target_ids": [102, 103, 104, ...]
  }'
```

Response:
```json
{"job_id": "job_xyz456", "status": "completed"}
```

### 3. Optimizer Step（参数更新）
```bash
curl -X POST http://localhost:8000/v1/runs/run_abc123/optim_step \\
  -H "Content-Type: application/json" \\
  -d '{
    "learning_rate": 0.0002,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8
  }'
```

Response:
```json
{"job_id": "job_xyz789", "status": "completed"}
```

### 4. Sample（推理/评测）
```bash
curl -X POST http://localhost:8000/v1/sample \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Once upon a time,",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'
```

### 5. Save Checkpoint
```bash
curl -X POST http://localhost:8000/v1/runs/run_abc123/save \\
  -H "Content-Type: application/json" \\
  -d '{"name": "checkpoint_epoch1"}'
```

Files:
```
checkpoints/run_abc123/checkpoint_epoch1.adapter.pt
checkpoints/run_abc123/checkpoint_epoch1.optimizer.pt
```

### 6. Query Job Result（轮询结果）
```bash
curl http://localhost:8000/v1/jobs/job_xyz456
```

Response:
```json
{
  "job_id": "job_xyz456",
  "status": "completed",
  "result": {"loss": 1.23, "batches": 4},
  "error": null
}
```

## Python 完整示例（类似 Tinker 工作流）

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. 创建训练会话
run = requests.post(f"{BASE_URL}/v1/runs", json={"base_model": "gpt2"})
run_id = run.json()["run_id"]
print(f"Run ID: {run_id}")

# 2. 准备数据集
data = {
    "input_ids": [...],  # 训练输入
    "target_ids": [...],  # 标签
}

# 3. Tinker 循环：forward -> optimizer -> save
for epoch in range(100):
    # 前向 + 反向传播
    fb = requests.post(
        f"{BASE_URL}/v1/runs/{run_id}/forward_backward",
        json=data
    )
    job_id = fb.json()["job_id"]

    # 等待完成（可用 async/callback 替代）
    while True:
        result = requests.get(f"{BASE_URL}/v1/jobs/{job_id}").json()
        if result["status"] == "completed":
            loss = result["result"]["loss"]
            break
        time.sleep(0.1)

    # 参数更新
    requests.post(
        f"{BASE_URL}/v1/runs/{run_id}/optim_step",
        json={"learning_rate": 2e-4}
    )

    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# 4. 保存模型（可选：sampler-optimized）
requests.post(f"{BASE_URL}/v1/runs/{run_id}/save", json={"name": "model_v1"})
```

## 环境变量

```bash
# Checkpoint 存储位置
export CHECKPOINTS_DIR=/path/to/checkpoints

# API 服务器基础 URL
export EZTINKER_BASE_URL=http://your-server:8000

# GPU 设备
export CUDA_VISIBLE_DEVICES=0
```

## 故障排查

### 问题 1：端口 8000 被占用
```bash
# 改用其他端口
uv run eztinker server --host 127.0.0.1 --port 8080
```

### 问题 2：GPU 内存不足
- 减小 batch size
- 使用更小的模型（如 gpt2-medium -> gpt2）
- 使用多 GPU 或分布式

### 问题 3：请求超时
- 增大 batch size 减少请求次数
- 使用异步批处理
- 启用 pipeline 优化

## 下一步

- 阅读 [README.md](README.md) 了解完整功能
- 检查 [example.py](example.py) 运行示例训练
- 探索 CLI: `uv run eztinker --help`
- 访问 Swagger API docs: `http://localhost:8000/docs`