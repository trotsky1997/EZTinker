"""Unit tests for EZTinker API server."""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.unit
class TestRunManagement:
    """Test run creation, listing, deletion."""

    def test_create_training_run(self, test_client):
        """Test successful training run creation."""
        response = test_client.post(
            "/v1/runs",
            json={
                "base_model": "gpt2",
                "lora_config": {"r": 8, "lora_alpha": 16},
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "run_id" in data

    def test_create_run_empty_model_fails(self, test_client):
        """Test model name validation."""
        response = test_client.post("/v1/runs", json={"base_model": ""})
        assert response.status_code == 400

    def test_list_runs_empty(self, test_client):
        """Test listing runs when none exist."""
        response = test_client.get("/v1/runs")
        assert response.status_code == 200
        assert response.json() == {"runs": []}

    def test_list_runs_with_data(self, test_client, sample_run_id):
        """Test listing runs with existing runs."""
        response = test_client.get("/v1/runs")
        assert response.status_code == 200
        runs = response.json()["runs"]
        assert len(runs) >= 1
        assert any(r["run_id"] == sample_run_id for r in runs)

    def test_delete_run(self, test_client):
        """Test deleting a run."""
        # Create a run to delete
        response = test_client.post(
            "/v1/runs",
            json={"base_model": "gpt2", "lora_config": {"r": 8}},
        )
        run_id = response.json()["run_id"]

        # Delete it
        response = test_client.delete(f"/v1/runs/{run_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


@pytest.mark.integration
class TestTrainingOperations:
    """Test forward/backward and optimizer operations."""

    def test_forward_backward(self, test_client, sample_run_id, sample_batch):
        """Test forward backward pass."""
        response = test_client.post(
            f"/v1/runs/{sample_run_id}/forward_backward",
            json=sample_batch,
        )
        assert response.status_code == 200
        result = response.json()
        assert "job_id" in result
        assert result["status"] == "completed"

    def test_optim_step(self, test_client, sample_run_id):
        """Test optimizer step."""
        response = test_client.post(
            f"/v1/runs/{sample_run_id}/optim_step",
            json={"learning_rate": 2e-4, "weight_decay": 0.01},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert "job_id" in result


@pytest.mark.integration
class TestSampling:
    """Test inference sampling."""

    def test_sample_generation(self, test_client):
        """Test text generation."""
        response = test_client.post(
            "/v1/sample",
            json={
                "prompt": "Hello world",
                "max_new_tokens": 10,
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert "job_id" in result


@pytest.mark.slow
@pytest.mark.integration
class TestCheckpointOperations:
    """Test checkpoint operations (slow)."""

    def test_save_checkpoint(self, test_client, sample_run_id):
        """Test saving a checkpoint."""
        response = test_client.post(
            f"/v1/runs/{sample_run_id}/save",
            json={"name": "test_checkpoint"},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"


@pytest.mark.integration
class TestJobPolling:
    """Test job polling."""

    def test_get_job_result(self, test_client, sample_run_id, sample_batch):
        """Test retrieving job result after forward backward."""
        # Create job
        response = test_client.post(
            f"/v1/runs/{sample_run_id}/forward_backward",
            json=sample_batch,
        )
        job_id = response.json()["job_id"]

        # Get job status
        response = test_client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == job_id
        assert result["status"] in ["queued", "completed"]