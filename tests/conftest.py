# Pytest fixtures for EZTinker testing
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_client():
    """Test client for EZTinker API."""
    from eztinker.api.server import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def sample_run_id(test_client):
    """Create a test run for integration tests."""
    response = test_client.post(
        "/v1/runs",
        json={
            "base_model": "Qwen/Qwen2-0.5B-Instruct",
            "lora_config": {
                "r": 1,  # rank=1 for all Qwen testing
                "lora_alpha": 2,
                "lora_dropout": 0.05,
                "target_modules": "all-linear",
            },
        },
    )
    assert response.status_code == 200, f"Request failed: {response.json()}"
    return response.json()["run_id"]


@pytest.fixture(scope="session")
def sample_tokenizer():
    """Sample tokenizer for testing."""
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def sample_batch(sample_tokenizer):
    """Create a sample training batch."""
    text = "This is a test example for training"
    batch = sample_tokenizer(text, return_tensors="pt", padding=True)
    return {
        "input_ids": batch["input_ids"].tolist()[0],
        "target_ids": batch["input_ids"].tolist()[0],
    }
