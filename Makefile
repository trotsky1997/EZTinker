.PHONY: help fmt lint tc test clean dev ci check

help: ## 显示帮助信息
	@echo 'EZTinker 简化开发流程'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo '快速开始:'
	@echo '  1. make check    # 快速检查'
	@echo '  2. make dev      # 开发模式'
	@echo '  3. make test     # 运行测试'

fmt: ## 格式化代码
	ruff format src/

lint: ## Lint 代码
	ruff check src/

tc: ## 类型检查
	pyright --project pyrightconfig.json

test: ## 运行测试
	pytest tests/ -v

test-fast: ## 快速测试（跳过慢测试）
	pytest tests/ -m "not slow" -v

check: ## 快速检查（开发常用）
	ruff format src/
	ruff check src/ --fix
	pyright --project pyrightconfig.json

dev: ## 开发模式：格式化 + lint + type-check + test
	make check
	make test-fast

ci: ## CI 完整流程
	make fmt
	make lint
	make tc
	make test

clean: ## 清理缓存
	rm -rf .nox .pytest_cache *.egg-info
	ruff clean

# 快捷别名
f: fmt
l: lint
t: test
c: check