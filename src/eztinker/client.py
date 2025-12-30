"""EZTinker Client API - Elegant Python API for interacting with EZTinker server."""

import time
from typing import Any

import requests

from .models.api import (
    CreateTrainingRunRequest,
    EvaluationBatch,
    EvaluationRequest,
    LoRAConfig,
    OptimParams,
    SamplingParams,
)


class EZTinkerClient:
    """High-level client for EZTinker server operations.

    This class provides an elegant Python API for interacting with EZTinker,
    abstracting away the raw HTTP API calls.

    Example:
        >>> from eztinker import EZTinkerClient
        >>> client = EZTinkerClient(base_url="http://localhost:8000")
        >>> run = client.create_run("Qwen/Qwen2-0.5B-Instruct")
        >>> print(f"Created run: {run}")
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the EZTinker client.

        Args:
            base_url: Base URL of the EZTinker server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def health(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Health information from server
        """
        response = self._session.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()

    def create_run(
        self, base_model: str, lora_rank: int = 8, lora_alpha: int = 16, run_id: str | None = None
    ) -> str:
        """Create a new training run.

        Args:
            base_model: Base model ID or path (e.g., "Qwen/Qwen2-0.5B-Instruct")
            lora_rank: LoRA rank (default: 8)
            lora_alpha: LoRA alpha (default: 16)
            run_id: Custom run ID (auto-generated if None)

        Returns:
            The created run ID

        Example:
            >>> client = EZTinkerClient()
            >>> run_id = client.create_run("Qwen/Qwen2-0.5B-Instruct", lora_rank=1)
        """
        lora_config = LoRAConfig(r=lora_rank, lora_alpha=lora_alpha)
        request = CreateTrainingRunRequest(
            base_model=base_model, lora_config=lora_config, run_id=run_id
        )

        response = self._session.post(f"{self.base_url}/v1/runs", json=request.model_dump())
        response.raise_for_status()
        result = response.json()
        return result["run_id"]

    def get_runs(self) -> list[dict[str, Any]]:
        """Get all active training runs.

        Returns:
            List of training run dictionaries

        Example:
            >>> runs = client.get_runs()
            >>> for run in runs:
            ...     print(f"Run: {run['run_id']}, Model: {run['base_model']}")
        """
        response = self._session.get(f"{self.base_url}/v1/runs")
        response.raise_for_status()
        result = response.json()
        return result.get("runs", [])

    def delete_run(self, run_id: str) -> None:
        """Delete a training run.

        Args:
            run_id: The run ID to delete

        Example:
            >>> client.delete_run("my_run")
        """
        response = self._session.delete(f"{self.base_url}/v1/runs/{run_id}")
        response.raise_for_status()

    def sample(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate (default: 100)
            temperature: Sampling temperature (default: 1.0)
            top_p: Nucleus sampling parameter (default: 1.0)
            top_k: Top-k sampling parameter (default: 50)
            do_sample: Whether to sample (default: True)

        Returns:
            Generated text

        Example:
            >>> text = client.sample("Hello!", max_new_tokens=50, temperature=0.8)
            >>> print(text)
        """
        params = SamplingParams(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        )

        response = self._session.post(f"{self.base_url}/v1/sample", json=params.model_dump())
        response.raise_for_status()
        job = response.json()
        job_id = job["job_id"]

        # Poll for completion
        result = self.wait_for_job(job_id)
        return result["result"]["generated_text"]

    def forward_backward(self, run_id: str, input_ids: list[int]) -> dict[str, Any]:
        """Perform forward and backward pass.

        Args:
            run_id: Training run ID
            input_ids: Input token IDs

        Example:
            >>> client.forward_backward(run_id, input_ids=[1, 2, 3, ...])
        """
        data = {"input_ids": input_ids, "target_ids": input_ids}
        response = self._session.post(
            f"{self.base_url}/v1/runs/{run_id}/forward_backward", json=data
        )
        response.raise_for_status()
        job = response.json()
        return self.wait_for_job(job["job_id"])

    def optim_step(
        self, run_id: str, learning_rate: float = 2e-4, weight_decay: float = 0.01
    ) -> dict[str, Any]:
        """Perform optimizer step.

        Args:
            run_id: Training run ID
            learning_rate: Learning rate for this step (default: 2e-4)
            weight_decay: Weight decay (default: 0.01)

        Example:
            >>> client.optim_step(run_id, learning_rate=2e-4)
        """
        params = OptimParams(learning_rate=learning_rate, weight_decay=weight_decay)
        response = self._session.post(
            f"{self.base_url}/v1/runs/{run_id}/optim_step", json=params.model_dump()
        )
        response.raise_for_status()
        job = response.json()
        return self.wait_for_job(job["job_id"])

    def evaluate(
        self,
        run_id: str,
        batches: list[EvaluationBatch],
        temperature: float = 1.0,
        max_new_tokens: int = 50,
    ) -> dict[str, Any]:
        """Evaluate model on batches.

        Args:
            run_id: Training run ID
            batches: List of evaluation batches
            temperature: Temperature for evaluation (default: 1.0)
            max_new_tokens: Max tokens for evaluation (default: 50)

        Returns:
            Evaluation results

        Example:
            >>> batches = [
            ...     EvaluationBatch(
            ...         input_ids=[...],
            ...         target_ids=[...],
            ...         response_ids=[...],
            ...         prompt_ids=[]
            ...     )
            ... ]
            >>> results = client.evaluate(run_id, batches)
        """
        request = EvaluationRequest(
            run_id=run_id, batches=batches, temperature=temperature, max_new_tokens=max_new_tokens
        )

        response = self._session.post(
            f"{self.base_url}/v1/runs/{run_id}/evaluate", json=request.model_dump()
        )
        response.raise_for_status()
        job = response.json()
        return self.wait_for_job(job["job_id"])

    def save_checkpoint(self, run_id: str, name: str) -> dict[str, Any]:
        """Save a checkpoint.

        Args:
            run_id: Training run ID
            name: Checkpoint name

        Example:
            >>> client.save_checkpoint(run_id, "checkpoint_001")
        """
        response = self._session.post(f"{self.base_url}/v1/runs/{run_id}/save", json={"name": name})
        response.raise_for_status()
        job = response.json()
        return self.wait_for_job(job["job_id"])

    def get_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        """Get all checkpoints for a run.

        Args:
            run_id: Training run ID

        Returns:
            List of checkpoint dictionaries

        Example:
            >>> checkpoints = client.get_checkpoints(run_id)
            >>> for ckpt in checkpoints:
            ...     print(f"Checkpoint: {ckpt['name']}")
        """
        response = self._session.get(f"{self.base_url}/v1/runs/{run_id}/checkpoints")
        response.raise_for_status()
        result = response.json()
        return result.get("checkpoints", [])

    def delete_checkpoint(self, run_id: str, checkpoint_name: str) -> None:
        """Delete a checkpoint.

        Args:
            run_id: Training run ID
            checkpoint_name: Name of checkpoint to delete
        """
        response = self._session.delete(
            f"{self.base_url}/v1/runs/{run_id}/checkpoints/{checkpoint_name}"
        )
        response.raise_for_status()

    def wait_for_job(
        self, job_id: str, timeout: float = 60.0, polling_interval: float = 0.5
    ) -> dict[str, Any]:
        """Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds (default: 60.0)
            polling_interval: Time between polling in seconds (default: 0.5)

        Returns:
            Job result dictionary

        Raises:
            TimeoutError: If job doesn't complete within timeout
            Exception: If job fails

        Example:
            >>> result = client.wait_for_job(job_id)
            >>> if result["status"] == "completed":
            ...     print(result["result"])
        """
        start_time = time.time()
        url = f"{self.base_url}/v1/jobs/{job_id}"

        while time.time() - start_time < timeout:
            response = self._session.get(url)
            response.raise_for_status()
            result = response.json()

            if result["status"] == "completed":
                return result
            elif result["status"] == "failed":
                error_msg = result.get("error", "Unknown error")
                raise Exception(f"Job failed: {error_msg}")

            time.sleep(polling_interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    def close(self):
        """Close the client and release resources."""
        if self._session:
            self._session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
