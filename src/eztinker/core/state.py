"""Global state manager for the EZTinker service."""
import threading
from typing import Dict, Optional
from ..engine.run_manager import TrainingRun
from ..engine.sampler import Sampler


class ServiceState:
    """Manages all training runs and jobs."""

    def __init__(self):
        self.runs: Dict[str, TrainingRun] = {}
        self.sampler = Sampler()
        self.lock = threading.Lock()

    def create_run(self, base_model: str, lora_config, run_id: Optional[str] = None) -> str:
        """Create a new training run."""
        import uuid
        run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"

        with self.lock:
            if run_id in self.runs:
                raise ValueError(f"Run ID {run_id} already exists")

            run = TrainingRun(
                run_id=run_id,
                base_model=base_model,
                lora_config=lora_config,
            )
            self.runs[run_id] = run

        return run_id

    def get_run(self, run_id: str) -> TrainingRun:
        """Get a training run by ID."""
        with self.lock:
            if run_id not in self.runs:
                raise ValueError(f"Run ID {run_id} not found")
            return self.runs[run_id]

    def delete_run(self, run_id: str):
        """Delete a training run."""
        with self.lock:
            if run_id not in self.runs:
                raise ValueError(f"Run ID {run_id} not found")
            del self.runs[run_id]

    def list_runs(self) -> list:
        """List all active runs."""
        with self.lock:
            return [
                {
                    "run_id": run_id,
                    "base_model": run.base_model,
                    "device": run.device,
                }
                for run_id, run in self.runs.items()
            ]


# Global state instance
state = ServiceState()