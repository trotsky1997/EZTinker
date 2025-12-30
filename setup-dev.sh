#!/bin/bash
# EZTinker 简化开发环境设置

set -e

echo "================================"
echo "EZTinker 开发环境设置"
echo "================================"
echo ""

# 1. 检查 Python 版本
echo "🔍 检查 Python 版本..."
if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "❌ 错误: 需要 Python >= 3.11"
    exit 1
fi
echo "✅ Python: $(python --version)"

# 2. 检查 uv
echo ""
echo "🔍 检查 uv..."
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv 未安装，正在安装..."
    pip install uv
fi
echo "✅ uv: $(uv --version)"

# 3. 安装项目依赖
echo ""
echo "📦 安装项目依赖..."
uv sync
echo "✅ 依赖安装完成"

# 4. 安装开发工具
echo ""
echo "🔧 安装开发工具..."
uv pip install --system ruff pyright pytest pre-commit
echo "✅ 开发工具安装完成"

# 5. 安装 pre-commit hooks
echo ""
echo "🪝 安装 pre-commit hooks..."
pre-commit install
echo "✅ pre-commit hooks 已安装"

echo ""
echo "================================"
echo "✅ 设置完成!"
echo "================================"
echo ""
echo "📋 下一步:"
echo ""
echo "1. 快速检查（开发常用）:"
echo "   make check    # 格式化 + lint + 类型检查"
echo ""
echo "2. 开发模式:"
echo "   make dev      # 检查 + 快速测试"
echo ""
echo "3. 运行完整测试:"
echo "   make test"
echo ""
echo "4. 提交前（会自动运行 ruff + pyright）:"
echo "   git commit -m \"feat: your message\""
echo ""
echo "📖 更多命令: make help"
echo "================================"
