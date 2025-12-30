# Pytest fixtures for EZTinker testing
import pytest
import sys
from pathlib import Path
import httpx

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
            "base_model": "gpt2",
            "lora_config": {"r": 8, "lora_alpha": 16},
        }
    )
    assert response.status_code == 200
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