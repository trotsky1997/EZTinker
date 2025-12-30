"""Checkpoint manager."""

from datetime import datetime
from pathlib import Path


class CheckpointManager:
    """Manages checkpoints (adapter + optimizer states)."""

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self, run_id: str | None = None) -> list[dict]:
        """List all checkpoints for a run (or all runs)."""
        checkpoints = []

        if run_id:
            # List checkpoints for a specific run
            run_dir = self.base_dir / run_id
            if run_dir.exists():
                for checkpoint_file in run_dir.glob("*.adapter.pt"):
                    checkpoint_name = checkpoint_file.stem.replace(".adapter", "")
                    checkpoints.append(
                        {
                            "name": checkpoint_name,
                            "run_id": run_id,
                            "adapter_path": str(checkpoint_file),
                            "optimizer_path": str(checkpoint_file.with_suffix(".optimizer.pt"))
                            if checkpoint_file.with_suffix(".optimizer.pt").exists()
                            else None,
                            "created_at": datetime.fromtimestamp(
                                checkpoint_file.stat().st_mtime
                            ).isoformat(),
                            "size_bytes": checkpoint_file.stat().st_size,
                            "is_sampler": False,
                        }
                    )
        else:
            # List checkpoints for all runs
            for run_dir in self.base_dir.iterdir():
                if run_dir.is_dir():
                    for checkpoint_file in run_dir.glob("*.adapter.pt"):
                        checkpoint_name = checkpoint_file.stem.replace(".adapter", "")
                        checkpoints.append(
                            {
                                "name": checkpoint_name,
                                "run_id": run_dir.name,
                                "adapter_path": str(checkpoint_file),
                                "optimizer_path": str(checkpoint_file.with_suffix(".optimizer.pt"))
                                if checkpoint_file.with_suffix(".optimizer.pt").exists()
                                else None,
                                "created_at": datetime.fromtimestamp(
                                    checkpoint_file.stat().st_mtime
                                ).isoformat(),
                                "size_bytes": checkpoint_file.stat().st_size,
                                "is_sampler": False,
                            }
                        )

        return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)

    def delete_checkpoint(self, run_id: str, name: str):
        """Delete a checkpoint."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} has no checkpoints")

        checkpoint_base = run_dir / name

        # Remove adapter
        adapter_path = checkpoint_base.with_suffix(".adapter.pt")
        if adapter_path.exists():
            adapter_path.unlink()

        # Remove optimizer
        optimizer_path = checkpoint_base.with_suffix(".optimizer.pt")
        if optimizer_path.exists():
            optimizer_path.unlink()
