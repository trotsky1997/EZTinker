"""简化的 Nox 自动化 - 只保留核心功能：Ruff, Type Check, Test."""

import nox

nox.options.sessions = ["fmt", "lint", "type-check", "test"]
nox.options.reuse_existing_virtualenvs = True


@nox.session(name="fmt", python=["3.11"])
def format_code(session):
    """格式化代码（本地使用）."""
    session.install("ruff")
    # 使用 . 作用于整个项目，包括示例和测试
    session.run("ruff", "format", ".")


@nox.session(name="lint", python=["3.11"])
def lint_code(session):
    """Lint 代码（适用于 CI，检查核心代码）."""
    session.install("ruff")
    # CI 检查：只检查核心代码（src/）和测试（tests/），跳过示例和演示
    # 这样更严格但不会被示例文件的早期开发阶段问题阻塞
    session.run("ruff", "check", "src/", "tests/")
    session.run("ruff", "format", "src/", "tests/", "--check")


@nox.session(name="fix", python=["3.11"])
def fix_lint_issues(session):
    """自动修复代码问题（本地一键整理推荐流程）."""
    session.install("ruff")

    # 推荐本地一键整理流程：
    # 1. 修复导入排序
    session.run("ruff", "check", ".", "--fix", "--select", "I")
    # 2. 修复其他可以自动修复的 lint 问题（排除需要手动处理的错误）
    #    --exit-zero 表示即使有未修复的错误也返回 0，允许流程继续
    session.run("ruff", "check", ".", "--fix", "--exit-zero")
    # 3. 格式化代码
    session.run("ruff", "format", ".")


@nox.session(name="type-check", python=["3.11"])
def type_check(session):
    """类型检查核心代码（使用 astral-sh/ty，严格模式）."""
    # ty 需要项目依赖来解析类型，所以先安装依赖
    session.install("-e", ".", "ty")
    # ty 会自动读取 ty.toml 配置，其中已经设置了 error-on-warning = true
    # 只检查核心代码 src/，跳过示例和演示文件
    session.run("ty", "check", "src/")


@nox.session(name="test", python=["3.11"])
def test(session):
    """运行测试."""
    session.install("-e", ".", "pytest")

    session.run("pytest", "tests/", "-v")


@nox.session(name="test-fast", python=["3.11"])
def test_fast(session):
    """运行快速测试."""
    session.install("-e", ".", "pytest")

    session.run("pytest", "tests/", "-m", "not slow", "-v")


@nox.session(name="clean", python=["3.11"])
def clean(session):
    """清理所有缓存."""
    session.run("rm", "-rf", ".nox", external=True)
    session.run("rm", "-rf", ".pytest_cache", external=True)
    session.run("rm", "-rf", "*.egg-info", external=True)
    session.run("ruff", "clean", external=True)
