"""简化的 Nox 自动化 - 只保留核心功能：Ruff, Type Check, Test."""
import nox

nox.options.sessions = ["fmt", "lint", "type-check", "test"]
nox.options.reuse_existing_virtualenvs = True


@nox.session(name="fmt", python=["3.11"])
def format_code(session):
    """格式化代码."""
    session.install("ruff")
    session.run("ruff", "format", "src/")


@nox.session(name="lint", python=["3.11"])
def lint_code(session):
    """Lint 代码."""
    session.install("ruff")
    session.run("ruff", "check", "src/")


@nox.session(name="fix", python=["3.11"])
def fix_lint_issues(session):
    """自动修复代码问题."""
    session.install("ruff")
    session.run("ruff", "check", "src/", "--fix")
    session.run("ruff", "format", "src/")


@nox.session(name="type-check", python=["3.11"])
def type_check(session):
    """类型检查."""
    session.install("-e", ".", "pyright")
    session.run("pyright")


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