"""Unit tests for EZTinker API server."""

import pytest


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
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "lora_config": {
                    "r": 1,
                    "lora_alpha": 2,
                    "lora_dropout": 0.05,
                    "target_modules": "all-linear",
                },
            },
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
        # Clean up any existing runs first
        response = test_client.get("/v1/runs")
        runs = response.json().get("runs", [])
        for run in runs:
            test_client.delete(f"/v1/runs/{run['run_id']}")

        # Now test listing empty runs
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
            json={
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "lora_config": {
                    "r": 1,
                    "lora_alpha": 2,
                    "lora_dropout": 0.05,
                    "target_modules": "all-linear",
                },
            },
        )
        run_id = response.json()["run_id"]

        # Delete it
        response = test_client.delete(f"/v1/runs/{run_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_delete_nonexistent_run_fails(self, test_client):
        """Test deleting a non-existent run returns error."""
        response = test_client.delete("/v1/runs/nonexistent_run_123")
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_create_run_duplicate_id_fails(self, test_client):
        """Test creating a run with duplicate custom ID fails."""
        custom_id = "test_custom_run_123"

        # Create first run successfully
        response1 = test_client.post(
            "/v1/runs",
            json={
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "lora_config": {
                    "r": 1,
                    "lora_alpha": 2,
                    "lora_dropout": 0.05,
                    "target_modules": "all-linear",
                },
                "run_id": custom_id,
            },
        )
        assert response1.status_code == 200
        assert response1.json()["run_id"] == custom_id

        # Try to create run with same custom ID should fail
        response2 = test_client.post(
            "/v1/runs",
            json={
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "lora_config": {
                    "r": 1,
                    "lora_alpha": 2,
                    "lora_dropout": 0.05,
                    "target_modules": "all-linear",
                },
                "run_id": custom_id,
            },
        )
        assert response2.status_code == 400

    def test_get_nonexistent_job_fails(self, test_client):
        """Test getting non-existent job returns 404."""
        response = test_client.get("/v1/jobs/nonexistent_job_123")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_run_with_default_lora_config(self, test_client):
        """Test creating run with no LoRA config uses defaults."""
        response = test_client.post(
            "/v1/runs",
            json={"base_model": "gpt2"},
        )
        # Should succeed with default LoRA config
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "run_id" in data

    def test_create_run_with_partial_lora_config(self, test_client):
        """Test creating run with only rank specified (Qwen defaults)."""
        response = test_client.post(
            "/v1/runs",
            json={
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "lora_config": {"r": 1},
            },
        )
        # Should succeed by filling in missing LoRA defaults
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"


@pytest.mark.unit
class TestRequestValidation:
    """Test Pydantic request validation and error handling."""

    def test_create_run_invalid_lora_rank(self, test_client):
        """Test LoRA rank validation."""
        response = test_client.post(
            "/v1/runs",
            json={
                "base_model": "gpt2",
                "lora_config": {"r": -1},  # Invalid negative rank
            },
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_create_run_invalid_lora_dropout(self, test_client):
        """Test LoRA dropout validation."""
        response = test_client.post(
            "/v1/runs",
            json={
                "base_model": "gpt2",
                "lora_config": {"lora_dropout": 1.5},  # Invalid dropout > 1
            },
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_optim_step_invalid_learning_rate(self, test_client):
        """Test optimizer learning rate validation."""
        response = test_client.post(
            "/v1/runs/fake_run_id/optim_step",
            json={"learning_rate": -0.001, "weight_decay": 0.01},
        )
        assert response.status_code == 400  # ValueError from state.get_run

    def test_save_checkpoint_missing_name_parameter(self, test_client, sample_run_id):
        """Test save checkpoint without name parameter."""
        response = test_client.post(
            f"/v1/runs/{sample_run_id}/save"
            # Missing 'name' parameter
        )
        assert response.status_code == 422  # FastAPI validation error

    def test_sample_missing_prompt(self, test_client):
        """Test sampling without required prompt field."""
        response = test_client.post(
            "/v1/sample",
            json={
                "max_new_tokens": 10,
                "temperature": 0.7,
                # Missing "prompt" field
            },
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_sample_invalid_temperature(self, test_client):
        """Test sampling with invalid temperature."""
        response = test_client.post(
            "/v1/sample",
            json={
                "prompt": "Hello",
                "temperature": -0.5,  # Invalid temperature
                "max_new_tokens": 10,
            },
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_sample_invalid_max_tokens(self, test_client):
        """Test sampling with invalid max_new_tokens."""
        response = test_client.post(
            "/v1/sample",
            json={
                "prompt": "Hello",
                "temperature": 0.7,
                "max_new_tokens": -5,  # Negative tokens
            },
        )
        assert response.status_code == 422  # Pydantic validation error


@pytest.mark.unit
class TestCoreStateManagement:
    """Test ServiceState core functionality."""

    def test_state_initialization(self):
        """Test ServiceState is initialized with empty run list."""
        from eztinker.core.state import state

        runs = state.list_runs()
        assert isinstance(runs, list)
        # Should be empty or have only test run from fixture


@pytest.mark.unit
class TestTrainingRunManagement:
    """Test run operations on non-existent runs."""

    def test_qwen_model_rank1_stability(self, test_client):
        """Test Qwen/Qwen2-0.5B-Instruct with rank=1 is stable."""
        response = test_client.post(
            "/v1/runs",
            json={
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "lora_config": {
                    "r": 1,
                    "lora_alpha": 2,
                    "lora_dropout": 0.05,
                    "target_modules": "all-linear",  # Use auto-detect
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        run_id = data["run_id"]

        # Test that we can list the run
        list_response = test_client.get("/v1/runs")
        runs = list_response.json()["runs"]
        assert any(r["run_id"] == run_id for r in runs)

    def test_create_run_with_unsupported_model_fails(self, test_client):
        """Test creating run with invalid model name."""
        response = test_client.post(
            "/v1/runs",
            json={
                "base_model": "nonexistent-org/nonexistent-model-xyz",
                "lora_config": {
                    "r": 1,
                    "lora_alpha": 2,
                    "lora_dropout": 0.05,
                    "target_modules": "all-linear",
                },
            },
        )
        # Should fail gracefully when trying to download non-existent model
        assert response.status_code == 400


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
            params={"name": "test_checkpoint"},  # Send as query parameter
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
