"""简化的 Nox 自动化 - 只保留核心功能：Ruff, Type Check, Test."""

import nox

nox.options.sessions = ["fmt", "lint", "type-check", "security", "test", "docs"]
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


@nox.session(name="security", python=["3.11"])
def security_check(session):
    """安全扫描（使用 Semgrep，检测已知安全漏洞和错误模式）."""
    session.install("semgrep")
    # 使用 auto 模式检测常见安全问题
    # --error 使发现安全问题时报错
    # 只检查核心代码 src/ 和测试 tests/，跳过示例和演示
    session.run("semgrep", "scan", "src/", "tests/", "--config", "auto", "--error")


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


@nox.session(name="docs", python=["3.11"])
def generate_docs(session):
    """生成 API 文档到 documents/ 目录（使用 pydoc）。"""
    session.install("-e", ".")
    import pathlib

    # 创建 documents 目录
    docs_dir = pathlib.Path("documents")
    docs_dir.mkdir(exist_ok=True)

    # 要检查的模块列表
    modules = [
        "eztinker",
        "eztinker.client",
        "eztinker.models",
        "eztinker.engine",
        "eztinker.core",
        "eztinker.dataset",
        "eztinker.api",
    ]

    session.log("生成 API 文档到 documents/ 目录...")
    for module in modules:
        # pydoc -w 生成 HTML 文件到当前目录
        session.run(
            "python", "-m", "pydoc", "-w", module,
            success_codes=[0, 1],
        )

        # 查找生成的HTML文件并移动到documents/
        import shutil
        import glob

        # 尝试多种可能的文件命名
        module_parts = module.split(".")

        # 检查是否有类似 "eztinker.client.html" 的文件
        html_files = [f for f in pathlib.Path(".").iterdir()
                     if f.is_file() and f.suffix == ".html"
                     and module_parts[-1] in f.name]

        if html_files:
            source_file = html_files[0]
            target_name = docs_dir / f"{module.replace('.', '_')}.html"
            shutil.move(str(source_file), str(target_name))
            session.log(f"✓ {module} -> {target_name.name}")
        else:
            session.warn(f"⚠ {module} 无文档(可能是空模块)")

    # 生成一个索引文件
    index_file = docs_dir / "README.txt"
    import datetime
    index_content = """EZTinker API 文档索引
====================

本文档目录包含使用 pydoc 自动生成的 API 文档。

文件清单:
"""

    for module in modules:
        filename = f"{module.replace('.', '_')}.txt"
        index_content += f"\n- {filename}\n  对应模块: {module}\n"

    index_content += f"\n\n生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    index_content += "\n使用指南:\n"
    index_content += "  运行 `uv run nox -s docs` 重新生成所有文档。\n"
    index_content += "  文档会随着代码注释更新自动更新。\n"

    index_file.write_text(index_content, encoding="utf-8")
    session.log(f"✓ 索引文件 -> {index_file.name}")

    session.log(f"\n文档生成完成! 共 {len(modules)} 个模块文档。")
    session.log(f"打开文档: open documents/eztinker.html")
    session.log(f"或在线查看: python -m http.server 8000 --bind 127.0.0.1")


@nox.session(name="docs-build", python=["3.11"])
def build_docs(session):
    """构建完整的 Sphinx 文档（如需详细的 HTML 文档）."""
    session.install("-e", ".", "sphinx", "sphinx-rtd-theme")
    # 如果存在 docs/ 目录，则构建
    if session.cwd.joinpath("docs").exists():
        with session.chdir("docs"):
            session.run("sphinx-build", "-b", "html", ".", "_build/html", "-W")
    else:
        session.warn("docs/ directory not found, skipping Sphinx build")


@nox.session(name="clean", python=["3.11"])
def clean(session):
    """清理所有缓存."""
    session.run("rm", "-rf", ".nox", external=True)
    session.run("rm", "-rf", ".pytest_cache", external=True)
    session.run("rm", "-rf", "*.egg-info", external=True)
    session.run("ruff", "clean", external=True)
