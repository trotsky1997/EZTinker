# EZTinker 开发指南

本地 CI/CD 和小号自动质量保障系统部署文档。

## 🔧 前置要求

- Python 3.11+
- uv (包管理工具)
- nox (自动化工具)
- pre-commit (Git hooks)

## 📦 安装开发依赖

```bash
# 1. 安装项目
uv sync

# 2. 安装 nox
pip install nox

# 3. 安装 pre-commit
pip install pre-commit
pre-commit install

# 4. 设置环境变量 (可选)
export CHECKPOINTS_DIR="checkpoints"
export EZTINKER_BASE_URL="http://localhost:8000"
```

## 🚀 Nox 会话命令

Nox 提供了多种自动化任务会话：

### 基础会话

```bash
# 代码格式化 (ruff format)
nox -s fmt

# lint 语法检查 (ruff check)
nox -s lint

# 自动修复 lint 问题
nox -s fix

# 类型检查 (pyright strict mode)
nox -s type-check

# 类型检查 (watch mode, live reload)
nox -s type-check-watch
```

### 测试会话

```bash
# 运行所有测试 (Python 3.11 + 3.12)
nox -s test

# 运行快速测试 (跳过慢速测试)
nox -s test-fast

# 运行性能基准测试
nox -s benchmark

# 显示覆盖率 (启动浏览器)
nox -s coverage  # 访问 http://localhost:8000
```

### 安全会话

```bash
# 运行 bandit + pip-audit
nox -s security

# 检查依赖问题
nox -s deps
```

### API 测试

```bash
# 启动 EZTinker 服务
uv run eztinker server

# 在另一个终端运行 API schema 测试
nox -s api-test
```

### 组合会话

```bash
# 完整的 CI 流水线 (recommended for pre-push)
nox -s ci

# 提交前检查 (轻量级, recommended for pre-commit)
nox -s commit

# 开发者模式：格式化 + lint + type check
nox -s dev

# 清理所有缓存和构建产物
nox -s clean
```

## 🪝 Pre-commit Hooks

Pre-commit 在每次 `git commit` 前自动触发检查：

### 安装 hooks

```bash
# 安装 hooks 到 .git/hooks/
pre-commit install

# 运行所有 hooks (所有文件)
pre-commit run --all-files

# 运行特定 hook
pre-commit run ruff --all-files
pre-commit run pyright --all-files
```

### 提交流程

```bash
# 1. 添加修改的文件
git add src/eztinker/

# 2. 提交 (自动触发 pre-commit)
git commit -m "feat: add new feature"

# 3. Push (触发 pre-push hooks: security scan)
git push origin main
```

### 跳过 hooks (不推荐)

```bash
# 强制跳过检查 (特殊情况)
git commit --no-verify
```

## 🧪 运行 Pytest 测试

除了通过 Nox，你也可以直接使用 pytest：

```bash
# 运行所有测试
pytest tests/

# 仅运行单元测试
pytest tests/unit/ -v

# 仅运行快速测试 (跳过 slow 标记)
pytest -m "not slow" -v

# 运行慢速测试
pytest -m "slow" -v

# 查看覆盖率
pytest tests/ --cov=src/eztinker --cov-report=term-missing --cov-report=html

# 开放浏览器查看详细覆盖率
open coverage_html/index.html
```

### 测试标记 (Markers)

```python
@pytest.mark.unit          # 单元测试
@pytest.mark.integration   # 集成测试
@pytest.mark.slow          # 慢速测试 (需要跳过)
@pytest.mark.benchmark     # 性能测试
@pytest.mark.regression    # 回归测试
@pytest.mark.hypothesis    # 属性测试
```

## 📊 类型检查 (Pyright)

Pyright 使用严格模式，确保 AI 生成代码的类型正确性：

```bash
# 运行 pyright
pyright --project pyrightconfig.json

# Watch mode (live updating)
pyright --project pyrightconfig.json --watch
```

### 常见类型错误

**❌ 错误示例 (AI容易犯的错):**
```python
def process(model: GPT2Model, data) -> str:
    # AI可能忘记返回类型或参数类型错误
    result = model.generate(data)  # TypeAny
    return result
```

**✅ 正确示例:**
```python
from transformers import GenerationOutput

def process(model: GPT2Model, data: InputDict) -> str:
    result: GenerationOutput = model.generate(data)
    return result.text
```

## 🔒 安全检查

### Bandit (Python 安全漏洞扫描)

```bash
# 扫描源代码
bandit -r src/eztinker -f json -o bandit-report.json

# 查看详细报告
python -c "import json; print(json.dumps(json.load(open('bandit-report.json'))))"
```

### pip-audit (依赖漏洞扫描)

```bash
# 扫描项目依赖
pip-audit --requirement pyproject.toml --format json

# 扫描当前环境依赖
pip-audit
```

## ⚡ 性能监控

### 运行基准测试

```bash
# 使用 pytest-benchmark
pytest tests/benchmarks/test_training_performance.py --benchmark-only

# 自动保存基准结果
pytest tests/benchmarks/ --benchmark-only --benchmark-autosave

# 检查性能回归
pytest tests/benchmarks/ --benchmark-compare
```

### 分析基准结果

生成的基准测试数据位于 `.benchmarks/`，可使用 Jupyter分析：

```python
import pandas as pd
import json

with open('.benchmarks/xxx.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['benchmarks'])
df.plot(x='name', y='stats'])
```

## 📋 开发工作流示例

### 发布前场景

```bash
# 1. 运行完整的质量检查
nox -s ci

# 2. 手动检查覆盖面
open coverage_html/index.html
open bandit-report.json

# 3. 如果一切正常，提交
git commit -m "feat: ready for release"
git push
```

### 日常开发场景

```bash
# 1. 修改代码
vim src/eztinker/engine/run_manager.py

# 2. 快速检查
nox -s dev  # 格式化 + lint + type check

# 3. 运行相关测试
pytest tests/unit/test_api_server.py -v

# 5. 提交
git commit -m "refactor: improve training efficiency"

# 6. 提交前会自动运行 pre-commit hooks
```

### 性能调优场景

```bash
# 1. 基准测试
nox -s benchmark

# 2. 分析结果
ls .benchmarks/Linux-CPython-3.11-64bit/
cat .benchmarks/.../xxx.json

# 3. 调整代码后重新测试
# 4. 对比性能
pytest tests/benchmarks/ --benchmark-compare
```

## 🐛 故障排除

### Pre-commit hook 失败

```bash
# 手动触发所有 hooks
pre-commit run --all-files

# 如果某个 hook 失败，可以先跳过
git commit -m "..." --no-verify

# 修复问题后再手动运行
pre-commit run ruff --all-files
```

### Pyright 报错

```python
# 如果真的有动态类型需求，使用明确的类型注解
from typing import Any

def function(x) -> Any:  # 显式声明为Any
    return x

# 或者使用 TypeGuard
from typing import TypeGuard

def is_str(x: object) -> TypeGuard[str]:
    return isinstance(x, str)
```

### Nox 会话失败

```bash
# 清理缓存
nox -s clean

# 重建虚拟环境
nox --reuse-existing-virtualenvs false -s test
```

## 🎯 推荐的 CI/CD 流程

### 本地

```bash
# 提交前 (快速, 5分钟内)
nox -s commit

# Push 前 (完整, 15分钟内)
nox -s ci
```

### CI Pipeline (可选)

添加到 `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: wntrblm/nox@2022.11.21
        with:
          sessions: [ci]
```

## 📊 可选的集成工具

以下工具未被默认包含，但可根据需要添加：

- **Django (架构)**: `import-linter` 强制模块分层
- **文档**: `Sphinx` + `sphinx-autodoc-typehints`
- **数据库**: `django-stubs` 类型检查
- **可视化**: `mypy` 插件重构建议对话框

## 🆘 遇到问题

如果遇到配置问题：

1. 检查 `noxfile.py` 中的 Python 版本
2. 确保 `uv` 已安装并配置
3. 清理缓存：`nox -s clean`
4. 阅读相关错误日志

---

✅ 这个 CI/CD 系统将确保：
- ✨ 所有代码都有类型注解
- 🧹 风格统一
- 🚫 没有安全漏洞
- 📈 测试覆盖率 > 60%
- ⚡ 性能基线可追踪
- 🤖 AI 生成代码合规