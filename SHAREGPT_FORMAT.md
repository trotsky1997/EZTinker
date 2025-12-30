# ShareGPT Format Data Specifications

## 📋 标准 ShareGPT 数据格式

ShareGPT 是一种常见的对话数据集格式，用于训练指令跟随和对话模型（如 Vicuna, LLaMA-2-Chat 等）。

### 1️⃣ 单条对话示例

```json
{
  "id": "unique-conversation-id",
  "conversations": [
    {
      "from": "human",
      "value": "What is the capital of France?"
    },
    {
      "from": "gpt",
      "value": "The capital of France is Paris."
    },
    {
      "from": "human",
      "value": "Tell me more about Paris."
    },
    {
      "from": "gpt",
      "value": "Paris is the capital and most populous city of France..."
    }
  ]
}
```

### 2️⃣ 标准字段定义

#### 方言 A: `from`/`value` (原始 ShareGPT)

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `id` | string | ✅ | 对话唯一标识符 |
| `conversations` | array | ✅ | 对话轮次列表 |
| └─ `from` | string | ✅ | 说话者："human" 或 "gpt" |
| └─ `value` | string | ✅ | 对话内容 |
| `system` | string | ❌ | 系统提示（可选） |
| `dataset` | string | ❌ | 数据集来源（可选） |

#### 方言 B: `role`/`content` (OpenAI 风格)

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `id` | string | ✅ | 对话唯一标识符 |
| `messages` or `conversations` | array | ✅ | 对话轮次列表 |
| └─ `role` | string | ✅ | 角色："user" 或 "assistant" |
| └─ `content` | string | ✅ | 消息内容 |
| `system` | string | ❌ | 系统提示（可选） |

**方言对照表**:

| from/value | role/content |
|-----------|-------------|
| `"human"` | `"user"` |
| `"gpt"` | `"assistant"` |

**统一映射规则**:
- `from: "human"` → `role: "user"`
- `from: "gpt"` → `role: "assistant"`
- `value` → `content`

### 3️⃣ 对话轮次规则

**严格交替**:
- 必须以 `human`/`user` 开始
- 必须以 `gpt`/`assistant` 结束
- 角色必须交替出现

```json
// from/value 方言
✅ 有效: [human, gpt, human, gpt]
✅ 有效: [human, gpt]

// role/content 方言
✅ 有效: [user, assistant, user, assistant]
✅ 有效: [user, assistant]

// 无效示例
❌ 无效: [gpt, user]          // 必须以用户开始
❌ 无效: [user, user]         // 不能连续用户
❌ 无效: [assistant, assistant] // 不能连续助手
```

### 4️⃣ 完整数据集文件示例

#### 方言 A: from/value

```json
[
  {
    "id": "conv_001",
    "conversations": [
      {"from": "human", "value": "Write a poem about coding."},
      {"from": "gpt", "value": "In the silence of the night..."}
    ],
    "dataset": "dolly-15k"
  }
]
```

#### 方言 B: role/content

```json
[
  {
    "id": "conv_002",
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "2+2=4"}
    ]
  },
  {
    "id": "multi_turn",
    "conversations": [  // 可以是 messages 或 conversations
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]
```

### 5️⃣ 多轮对话示例

```json
{
  "id": "multi-turn-demo",
  "conversations": [
    {
      "from": "human",
      "value": "Recommend a good book to read."
    },
    {
      "from": "gpt",
      "value": "I recommend 'The Pragmatic Programmer' by Andy Hunt and Dave Thomas."
    },
    {
      "from": "human",
      "value": "What is it about?"
    },
    {
      "from": "gpt",
      "value": "It's a classic software engineering book that covers practical advice..."
    },
    {
      "from": "human",
      "value": "Is it suitable for beginners?"
    },
    {
      "from": "gpt",
      "value": "Yes, it's great for developers at all levels..."
    }
  ]
}
```

### 6️⃣ 真实数据集示例

**Open-Orca** (100k+ samples):
```json
{
  "id": "OpenOrca/0",
  "system": "You are an AI assistant...",
  "conversations": [
    {"from": "human", "value": "How do I..."},
    {"from": "gpt", "value": "Here's how..."}
  ]
}
```

**Vicuna** (70k+ samples):
```json
{
  "id": "vicuna_1",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

**Alpaca-ShareGPT**:
```json
{
  "id": "alpaca_sharegpt_0",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

### 7️⃣ 模板化和分词

#### 训练模板（以 Qwen2 为例）

```python
def format_conversation(conversations, tokenizer):
    """将 ShareGPT 对话格式化为训练文本。"""
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"

    # Detect if system message exists
    # ...

    parts = [system_prompt]

    for idx, turn in enumerate(conversations):
        role = turn["from"]
        content = turn["value"]

        if role == "human":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "gpt":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

    return "".join(parts)
```

**输出示例**:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is Paris?<|im_end|>
<|im_start|>assistant
Paris is the capital of France.<|im_end|>
<|im_start|>user
Tell me more.<|im_end|>
<|im_start|>assistant
Paris is known for the Eiffel Tower...<|im_end|>
```

### 8️⃣ EZTinker 支持计划

#### 8.1 数据集加载器
```python
from eztinker.dataset.sharegpt import ShareGPTDataset

# 从 JSONL 文件加载
dataset = ShareGPTDataset(
    file_path="sharegpt_data.jsonl",
    tokenizer=tokenizer,
    max_length=2048,
    system_template="You are a helpful assistant."
)
```

#### 8.2 数据验证
- ✅ 验证 JSON 格式
- ✅ 检查必要字段
- ✅ 验证对话轮次交替
- ✅ 统计数据集信息

#### 8.3 训练集成
```python
# 在 rejection_sft_demo.py 中的用法
if args.dataset == "sharegpt":
    dataset = ShareGPTDataset(
        file_path=args.sharegpt_file,
        tokenizer=tokenizer,
        max_length=2048
    )
elif args.dataset == "gsm8k":
    dataset = GSM8KDataset(...)
```

#### 8.4 API 支持
```python
# FastAPI endpoint
@app.post("/v1/forward_backward")
async def forward_backward(
    batch: ShareGPTBatchInput  # 支持 ShareGPT 格式的输入
):
    ...
```

### 9️⃣ 支持的格式变体

#### 9.1 JSONL 格式（一行一个对话）

**方言 A (from/value)**:
```jsonl
{"id": "1", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
{"id": "2", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

**方言 B (role/content)**:
```jsonl
{"id": "1", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"id": "2", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

#### 9.2 单文件 JSON 格式（全部对话在一个数组中）
```json
[
  {"id": "1", ...},
  {"id": "2", ...}
]
```

#### 9.3 混合字段名支持

✅ **自动识别**: 支持 `messages` 和 `conversations` 两种字段名
- 优先使用 `messages`，如果不存在则使用 `conversations`

✅ **自动映射**: 混用 from/value 和 role/content
- `{"from": "human", "value": "..."}` → 统一为 `(user, "...")`
- `{"role": "user", "content": "..."}` → 统一为 `(user, "...")`

📝 **示例混合文件**:
```json
[
  {
    "id": "mixed_1",
    "conversations": [  // 使用的字段名
      {"from": "human", "value": "..."},  // 方言 A
      {"role": "assistant", "content": "..."}  // 方言 B (混用)
    ]
  }
]
```

### 🔟 验证规则

**严格验证**（默认）：

1. ✅ 必须有 `id` 字段
2. ✅ 必须有 `messages` 或 `conversations` 字段（二选一）
3. ✅ 必须以 `human`/`user` 开始，以 `gpt`/`assistant` 结束
4. ✅ 必须交替出现
5. ✅ 角色检查：
   - `from`: 只能为 `human` 或 `gpt`
   - `role`: 只能为 `user` 或 `assistant`
6. ✅ 内容检查：
   - `value` 或 `content` 必须是字符串且非空

**宽松验证**（可选）：
- ✅ 允许自动修正：删除无效轮次、补齐缺失字段
- ✅ 自动检测并映射方言
- ✅ 支持混合格式

**方言检测优先级**：
1. 检测 `messages` 字段是否存在
2. 如果存在，使用 `role`/`content` 方言
3. 否则使用 `conversations` + `from`/`value`

**角色映射表**:
| from/value | role/content | 统一角色 | 统一字段 | 说明 |
|-----------|-------------|---------|---------|------|
| `"human"` | `"user"` | `"user"` | `(role, content)` | 用户消息 |
| `"gpt"` | `"assistant"` | `"assistant"` | `(role, content)` | AI 回复 |

**字段映射**:
```python
# 内部统一表示
{
    "id": "...",
    "turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]
}
```

---

## 📝 EZTinker 实现计划

### 阶段 1：数据集加载器
- `ShareGPTDataset` 类
- 格式验证
- 模板化

### 阶段 2：训练集成
- `rejection_sft_demo.py` 支持 ShareGPT
- 命令行参数
- 评估指标

### 阶段 3：API 支持
- 新的 FastAPI endpoint
- Pydantic models
- Batch 输入支持

### 阶段 4：示例和文档
- ShareGPT 训练脚本
- 数据准备指南
- 训练效果的评估

---

**请确认以上约定是否符合你的预期？**

特别是：
1. ✅ ShareGPT 格式定义是否正确？
2. ✅ 字段和验证规则是否合理？
3. ✅ 模板化方式是否适合 Qwen2 模型？
4. ✅ 实现计划的优先级是否合理？