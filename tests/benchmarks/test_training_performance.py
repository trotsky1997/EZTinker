"""Performance benchmarks for EZTinker training operations."""
import pytest
import time


@pytest.mark.benchmark
class TestTrainingPerformance:
    """Performance tests for training operations."""

    @pytest.mark.benchmark(group="forward_backward")
    def test_forward_backward_time(self, benchmark, test_client):
        """Benchmark forward + backward pass time."""
        # Setup: create run
        response = test_client.post("/v1/runs", json={
            "base_model": "gpt2",
            "lora_config": {"r": 8}
        })
        run_id = response.json()["run_id"]

        # Setup: create batch
        token_ids = list(range(10, 100))  # 90 tokens

        # Benchmark
        def api_call():
            return test_client.post(
                f"/v1/runs/{run_id}/forward_backward",
                json={"input_ids": token_ids, "target_ids": token_ids}
            )

        result = benchmark(api_call)
        assert result.status_code == 200

    @pytest.mark.benchmark(group="optim_step")
    def test_optimizer_step_time(self, benchmark, test_client):
        """Benchmark optimizer step time."""
        # Setup: create run
        response = test_client.post("/v1/runs", json={
            "base_model": "gpt2",
            "lora_config": {"r": 8}
        })
        run_id = response.json()["run_id"]

        # Benchmark
        def api_call():
            return test_client.post(
                f"/v1/runs/{run_id}/optim_step",
                json={"learning_rate": 2e-4, "weight_decay": 0.01}
            )

        result = benchmark(api_call)
        assert result.status_code == 200

    @pytest.mark.benchmark(group="sampling", warmup=True)
    def test_sampling_throughput(self, benchmark, test_client):
        """Benchmark sampling throughput."""
        def api_call():
            return test_client.post(
                "/v1/sample",
                json={
                    "prompt": "Quick brown fox jumps over the lazy dog",
                    "max_new_tokens": 50,
                    "temperature": 0.8,
                }
            )

        result = benchmark(api_call)
        assert result.status_code == 200


@pytest.mark.benchmark
class TestMemoryUsage:
    """Memory usage tests (informational)."""

    @pytest.mark.slow
    def test_memory_usage_growth(self, test_client):
        """Test if repeated operations cause memory growth."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Create run
        response = test_client.post("/v1/runs", json={
            "base_model": "gpt2",
            "lora_config": {"r": 8}
        })
        run_id = response.json()["run_id"]

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        for i in range(20):
            response = test_client.post(
                f"/v1/runs/{run_id}/forward_backward",
                json={"input_ids": list(range(50)), "target_ids": list(range(50))}
            )
            assert response.status_code == 200

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth_mb = final_memory - initial_memory

        pytest.set_trace()  # Allow investigation