# 🚀 EZTinker 本地 CI/CD 快速启动

> ⚡ 5分钟配置完整的 AI 代码质量保障体系

## 1️⃣ 一键安装

```bash
bash setup-dev.sh
```

这个脚本会自动：
- ✅ 检查 Python 版本 (>=3.11)
- ✅ 安装项目依赖 (uv sync)
- ✅ 安装开发工具 (nox, pre-commit, pytest)
- ✅ 配置 pre-commit hooks

## 2️⃣ 查看可用命令

```bash
make help
```

你会看到所有快捷命令，包括：
- `make dev` - 完整的开发流程
- `make ci` - 完整 CI 流水线
- `make test` - 运行测试
- `make security` - 安全扫描

## 3️⃣ 开发工作流

### 方式 A: 使用 Nox (自动化，推荐)

```bash
# 代码格式化 + lint + 类型检查
nox -s dev

# 运行完整 CI 流水线 (5-15分钟)
nox -s ci

# 运行快速测试
nox -s test-fast
```

### 方式 B: 使用 Make (更快)

```bash
# 提交前快速检查
make commit

# 格式化代码
make fmt

# 类型检查
make type-check
```

### 方式 C: 手动流程 (更灵活)

```bash
ruff format src/ tests/
ruff check src/ tests/ --fix
pyright --project pyrightconfig.json
pytest tests/ -v
```

## 4️⃣ Pre-commit 自动检查

每次 `git commit` 都会自动触发检查：

```bash
git commit -m "feat: add new feature"
# ↓ 自动运行 ↓
# 1. ruff format (格式化)
# 2. ruff check (lint)
# 3. pyright (类型检查)
# 4. pydocstyle (文档检查)
# 5. bandit (安全扫描)
# 6. conventional commit (提交格式)
```

如果检查失败，提交会被阻止。修复问题后重试。

## 5️⃣ 工具速查表

| 类别 | 工具 | 作用 | 命令 |
|-----|------|------|------|
| 🧹 **格式化** | Ruff | 代码格式化 | `make fmt` |
| 🚨 **Lint** | Ruff | 语法检查 | `make lint` |
| 🧠 **类型** | Pyright | 类型检查 | `make type-check` |
| 🧪 **测试** | Pytest | 功能测试 | `make test` |
| 📈 **覆盖率** | Coverage | 测试覆盖 | `make coverage` |
| 🔒 **安全** | Bandit | 漏洞扫描 | `make security` |
| 📦 **依赖** | Pip-audit | 依赖漏洞 | `make deps` |
| ⚡ **性能** | Benchmark | 性能测试 | `make benchmark` |

## 6️⃣ 常见场景

### 🆕 开始一个新功能

```bash
# 1. 创建分支
git checkout -b feature/new-model

# 2. 修改代码
vim src/eztinker/engine/run_manager.py

# 3. 快速检查 (30秒)
make commit

# 4. 提交
git add .
git commit -m "feat: add new model support"

# 5. Push 前完整检查 (5-10分钟)
make ci
git push origin feature/new-model
```

### 🧪 运行测试

```bash
# 所有测试
pytest tests/

# 仅单元测试
pytest tests/unit/ -v

# 显示覆盖率
pytest tests/ --cov=src/eztinker --cov-report=term-missing

# 性能测试
pytest tests/benchmarks/
```

### ⚠️ 修复错误

如果你看到 "pre-commit hook failed":

```bash
# 1. 查看具体错误
pre-commit run --all-files

# 2. 自动修复 lint 问题
make fix

# 3. 如果还有类型错误，手动修复
pyright --project pyrightconfig.json

# 4. 再次提交
git commit -m "fix: resolve type issues"
```

### 📈 查看测试覆盖率

```bash
# 生成 HTML 报告
make coverage

# 在浏览器中打开
open coverage_html/index.html
```

### 🔒 检查安全漏洞

```bash
# 检查当前代码
make security

# 检查依赖漏洞
make deps

# 查看详细报告
cat bandit-report.json | jq '.'
```

## 7️⃣ VS Code 集成

安装推荐的扩展（可选）：

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",         // Python 支持
    "ms-python.vscode-pylance", // Pyright LSP
    "charliermarsh.ruff",       // Ruff Linter
    "ms-python.black-formatter" // 格式化器
  ]
}
```

## 8️⃣ AI + QA 工作流

### 场景：AI 生成代码

```python
# 你让 AI 生成代码
prompt = "写一个函数，用 Python 爬取网站标题"

# AI 编写代码
def fetch_title(url):
    import requests
    r = requests.get(url)
    return r.text.split('<title>')[1].split('</title>')[0]
```

### QA 系统自动捕获的错误：

#### 1. ⚠️ Type error (Pyright)
```
Missing type annotations
Argument 1 "url" has no type annotation
```

#### 2. 🔒 Security (Bandit)
```
[B310:urllib_request_urlopen] Use of requests.get()
without timeout protection
```

#### 3. 📝 Docstring (Pydocstyle)
```
Missing docstring in public function
```

#### 4. ✅ 修正后：

```python
from typing import Optional
import requests

def fetch_title(url: str, timeout: float = 30.0) -> Optional[str]:
    """Fetch page title from URL.

    Args:
        url: Target URL
        timeout: Request timeout in seconds

    Returns:
        Page title string or None if failed

    Raises:
        requests.RequestException: On network errors
    """
    try:
        r = requests.get(url, timeout=timeout)
        if '<title>' in r.text:
            return r.text.split('<title>')[1].split('</title>')[0]
        return None
    except Exception:
        return None
```

## 9️⃣ CI/CD vs Pre-commit

| 阶段 | 工具 | 内容 | 速度 |
|-----|------|------|------|
| **Commit前** | Pre-commit | Format + Lint + Docstring | ⚡ 快 (30s) |
| **Push前** | Nox CI | Format + Type + Test + Security | ⏳ 中 (5-10min) |
| **PR时** | GitHub Actions | Full pipeline | 🐢 慢 (20min) |

**建议流程：**
- Commit 前运行 `make commit` (轻量级)
- Push 前运行 `make ci` (完整)
- PR 时 GitHub 自动检查

## 🔟 故障排除

### "No module named 'ruff'"

```bash
# 使用 uv 安装
uv pip install ruff
uv run ruff check src/
```

### "pyright not found"

```bash
# 安装 pyright
pip install pyright
```

### "pre-commit hook failed"

```bash
# 查看详细错误
pre-commit run --all-files

# 或者跳过 (不推荐)
git commit --no-verify
```

### Commit 被阻止

通常是因为：
1. ❌ 类型错误 → 添加类型注解
2. ❌ 未使用的导入 → `ruff check --fix`
3. ❌ 缺少文档 → 添加 docstring
4. ❌ 提交信息格式错误 → 使用 `feat:` `fix:` `docs:` 等

## 📚 更多文档

- 📖 详细开发文档: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- 🧪 拒绝采样指导: [docs/REJECTION_SFT_GUIDE.md](docs/REJECTION_SFT_GUIDE.md)
- 🚀 Nox 配置: `noxfile.py`
- 🪝 提交钩子: `.pre-commit-config.yaml`

## 🎉 完成！

你已经配置了完整的 AI 代码质量保障系统：

✅ Python 3.11+
✅ 类型安全 (Pyright)
✅ 代码质量 (Ruff)
✅ 测试框架 (Pytest)
✅ 安全扫描 (Bandit)
✅ 自动化 (Nox + Pre-commit)

现在可以安全地让 AI 生成代码，QA 系统会自动捕获错误！ 🚀

---

**快速链接:**

- `make help` - 查看所有命令
- `make dev` - 开发模式
- `make ci` - CI 流水线
- `cat docs/DEVELOPMENT.md` - 详细文档