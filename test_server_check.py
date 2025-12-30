#!/usr/bin/env python3
"""快速测试服务器是否启动的脚本。"""
import sys
import requests

BASE_URL = "http://localhost:8000"

try:
    response = requests.get(f"{BASE_URL}/health", timeout=2)
    if response.status_code == 200:
        print("✅ EZTinker server is running")
        sys.exit(0)
    else:
        print("⚠️  Server responded but with error:", response.status_code)
        sys.exit(1)
except Exception as e:
    print("❌ Server not running:", str(e))
    print("\n请运行: uv run eztinker server")
    sys.exit(1)