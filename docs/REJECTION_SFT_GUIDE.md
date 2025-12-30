# Rejection SFT Demo - 快速上手指南

这是一个完整的 Qwen2-0.5B + GSM8K + Rank-1 LoRA 拒绝采样监督微调演示。

## 准备工作

确保已完成依赖安装：

```bash
cd /path/to/eztinker
uv add datasets math-verify
```

## 快速开始（3个步骤）

### Step 1: 启动 EZTinker 服务器

```bash
# Terminal 1
uv run eztinker server
```

等待服务器启动（约5-10秒），看到：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 2: 运行小规模测试（推荐先测试）

```bash
# Terminal 2
uv run python rejection_sft_demo.py --max-samples 5 --num-candidates 2 --epochs 1
```

这个命令会：
- 加载 5 个 GSM8K 训练样本
- 每个样本生成 2 个候选响应
- 使用 Math-Verify 评估并选择最佳
- 训练 1 轮
- 预计用时：2-3 分钟

### Step 3: 完整训练（100 样本，3 轮）

```bash
# Terminal 2（等待小规模测试成功后）
uv run python rejection_sft_demo.py \
  --max-samples 100 \
  --num-candidates 4 \
  --epochs 3 \
  --checkpoint \
  --temperature 0.8
```

## 命令行参数说明

| 参数 | 默认值 | 说明 | 建议 |
|-----|-------|------|------|
| `--max-samples` | 100 | 训练样本数量 | 测试: 5-10, 完整: 100-500 |
| `--num-candidates` | 4 | 每个样本的候选数 | 测试: 2, 完整: 4-8 |
| `--epochs` | 3 | 训练轮数 | 测试: 1, 完整: 3 |
| `--learning-rate` | 2e-4 | 学习率 | 0.5B模型推荐 |
| `--temperature` | 0.8 | 生成温度 | 0.7-0.9 |
| `--checkpoint` | 否 | 每轮保存 | 完整训练时启用 |
| `--eval-size` | 100 | 测试样本数 | 评估准确率 |

## 预期输出

运行时会看到类似输出：

```
=== Phase 1: Creating Training Run ===

✓ Created training run: abc123

=== Phase 2: Populating Rejection Buffer ===

[1/100] Processing example...
  Generating candidates...
  Evaluating candidates...
  Best candidate: score=3.45, is_correct=True, trained=True

=== Phase 3: Rejection SFT Training Loop ===

--- Epoch 1/3 ---
  Trained on 85/100 examples
  Saving checkpoint...

=== Phase 4: Final Evaluation ===

Total: 100 examples
Correct: 68 examples
Accuracy: 68.00%
```

## 输出文件

运行完成后，在 `data/` 目录下会生成：

```
data/
├── rejection_buffer.jsonl          # 拒绝采样缓冲区
│   └── {"question": "...", "best_response": "...", "is_correct": true}
├── rejection_sft_results.json      # 完整结果
│   └── {"run_id": "...", "evaluation_metrics": {"accuracy": 0.68}}
└── checkpoints/
    └── abc123/
        ├── rejection_sft_epoch_1.adapter.pt
        ├── rejection_sft_epoch_1.optimizer.pt
        ├── rejection_sft_epoch_2.adapter.pt
        └── rejection_sft_epoch_3.adapter.pt
```

## 故障排查

### 问题 1: 服务器未响应

```
Error: Server not found
✗ Failed to create training run
```

**解决**:
1. 确保 Terminal 1 的服务器正在运行
2. 访问 `http://localhost:8000/health` 确认返回 `{"status":"ok"}`
3. 重启服务器：`uv run eztinker server --reload`

### 问题 2: CUDA OOM

```
RuntimeError: CUDA out of memory
```

**解决**:
1. 减少样本数量：`--max-samples 25`
2. 减少候选数：`--num-candidates 2`
3. 关闭其他 GPU 程序

### 问题 3: GSM8K 下载慢

首次运行会下载 100MB+ 数据集，可以：
- 使用镜像或代理
- 提前下载：在 `.cache/huggingface/datasets/` 检查
- 使用小数据集测试：`--max-samples 5`

### 问题 4: Math-Verify 错误

```
Warning: Math-Verify evaluation failed: ...
```

**解决**:
- 检查 `math-verify` 是否成功安装：`uv add math-verify --frozen`
- 升级依赖：`uv add math-verify --upgrade`
- 代码会自动 fallback 到简单的数值比较，不会崩溃

## 进阶使用

### 调整训练参数

```bash
# 更高的学习率（尝试突破局部最优）
python rejection_sft_demo.py --max-samples 200 --learning-rate 5e-4

# 更多候选（提高选择质量）
python rejection_sft_demo.py --max-samples 50 --num-candidates 8 --epochs 5

# 更高温度（更多样化）
python rejection_sft_demo.py --temperature 1.0 --num-candidates 8
```

### 评估保存的 checkpoint

```bash
# 查看有哪些 checkpoint
ls data/checkpoints/*/

# 使用某个 checkpoint 评估
# TODO: 需要创建 evaluation-only 脚本
```

### 批量运行多个配置

创建 `run_configs.sh`:

```bash
#!/bin/bash

# Config 1: 小数据、多候选
python rejection_sft_demo.py --max-samples 50 --num-candidates 8 --output-dir data/exp_1

# Config 2: 大数据、少候选
python rejection_sft_demo.py --max-samples 200 --num-candidates 4 --output-dir data/exp_2

# Config 3: 中等数据、中等候选、更多轮次
python rejection_sft_demo.py --max-samples 100 --num-candidates 6 --epochs 5 --output-dir data/exp_3
```

## 性能优化

### 使用多进程加速候选生成

在 `src/eztinker/rl/rejection_sampler.py` 里面调整：

```python
MAX_WORKERS = 8  # 增加到 GPU 数量或 CPU 核数
```

### 批量评估

服务器端 `evaluate_responses` endpoint 已支持批量，但客户端可以改进：

```python
# 当前：逐个评估
for candidate in candidates:
    results.append(api.evaluate(candidate))

# 优化：一次评估所有
all_results = api.batch_evaluate(candidates)
```

## 监控训练

可以用 `watch` 查看生成的 buffer 文件：

```bash
# Terminal 3
watch -n 5 "wc -l data/rejection_buffer.jsonl && tail -3 data/rejection_buffer.jsonl | jq ."
```

或实时跟踪准确率：

```bash
tail -f rejection_sft.stdout | grep "Epoch.*accuracy"
```

## 下一步

1. **分析结果**：检查 `data/rejection_sft_results.json` 中的训练曲线
2. **调试问题**：查看哪些样本被错误选择（查看 `rejection_buffer.jsonl`）
3. **调整超参**：学习率、候选数量、轮数等
4. **扩展规模**：尝试 500、1000 样本
5. **对比实验**：
   - Baseline: 传统 SFT（不使用 rejection sampling）
   - 不同候选数的影响
   - 不同模型大小（Qwen2-1.5B, Qwen2-7B）

## 提示

- 📊 **先小规模测试**：确保 pipeline 能跑起来
- 📈 **监控准确率**：每轮后验证增益
- 💾 **频繁保存**：使用 `--checkpoint` 参数
- 🐛 **调试时单步**：添加 `breakpoint()` 查看中间结果
- 📝 **记录实验**：保存命令行参数和结果到笔记