# 🚀 EZTinker 简化 CI/CD 指南

> 🎯 只保留核心工具：Ruff + Pyright + Pytest

## 🏃 快速开始（2分钟）

```bash
# 1️⃣ 一键设置
bash setup-dev.sh

# 2️⃣ 查看命令
make help

# 3️⃣ 快速检查
make check
```

## 🛠️ 核心工具

| 工具 | 作用 | 命令 |
|-----|------|------|
| 🧹 **Ruff** | 格式化 + Lint | `make fmt` / `make lint` |
| 🧠 **Pyright** | 类型检查 | `make tc` |
| 🧪 **Pytest** | 测试 | `make test` |

## 📋 可用命令

```
make help     # 显示所有命令
make check    # 快速检查（格式化 + lint + 类型检查）
make dev      # 开发模式（检查 + 快速测试）
make ci       # CI 完整流程
make fmt      # 格式化
make lint     # Lint
make tc       # 类型检查
make test     # 运行所有测试
make test-fast # 只跑不慢的测试
make clean    # 清理缓存
```

## 🪝 Pre-commit

每次 `git commit` 自动触发：

```bash
git commit -m "feat: your message"
# ↓ 自动运行 ↓
# 1. ruff format (格式化)
# 2. ruff check (lint + 自动修复)
# 3. pyright (类型检查)
```

如果检查失败，提交会被阻止。

## 🎯 常见场景

### 开发流程

```bash
# 1. 修改代码
vim src/eztinker/engine/run_manager.py

# 2. 快速检查（30秒）
make check

# 3. 提交
git add .
git commit -m "feat: improve model loading"
# ↑ 自动运行 ruff + pyright

# 4. 推送到远程
git push
```

### 提交前完整检查

```bash
# 运行完整的 CI 流程
make ci
# ↑ 包括格式化 + lint + 类型检查 + 测试
```

### 测试

```bash
# 运行所有测试
make test

# 快速测试（跳过慢测试）
make test-fast

# 查看测试详情
pytest tests/ -v

# 查看某个文件
pytest tests/unit/test_api_server.py -v
```

## ⚠️ 修复错误

如果 pre-commit 失败：

```bash
# 1. 查看具体错误
make lint    # Ruff 问题
make tc      # Pyright 类型问题

# 2. 自动修复（Ruff）
make check   # 会自动修复格式和 lint

# 3. 手动修复类型错误
# 添加类型注解到代码中

# 4. 再次提交
git commit -m "feat: ..."
```

## 🔧 VS Code 集成

在 `.vscode/settings.json`：

```json
{
  "python.linting.enabled": true,
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": true,
      "source.organizeImports": true
    }
  }
}
```

然后用 VS Code 内置的 Ruff 扩展，无需手动运行命令。

## 📚 完整的 Nox 命令（可选）

需要更多功能时：

```bash
# 格式化
nox -s fmt

# Lint
nox -s lint

# 类型检查
nox -s type-check

# 测试
nox -s test

# 快速测试
nox -s test-fast

# 清理
nox -s clean
```

## 🎯 为什么简化？

之前版本包含太多工具：
- ❌ 安全扫描（Bandit）
- ❌ 依赖扫描（pip-audit）
- ❌ 文档检查（Pydocstyle）
- ❌ 提交格式检查
- ❌ 性能基准测试
- ❌ 覆盖率报告

**简化后保留：**
- ✅ Ruff：格式化 + Lint（一个工具完成所有代码质量）
- ✅ Pyright：严格的类型检查（防止 AI 生成类型错误代码）
- ✅ Pytest：测试框架

**够用就好！** 🚀

## 📖 更多信息

- `make help` - 查看所有命令
- `nox --list` - 查看完整的 nox 命令
- `.pre-commit-config.yaml` - pre-commit 配置
- `noxfile.py` - nox 自动化配置