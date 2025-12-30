"""CLI interface for EZTinker."""
import typer
import time
import requests
import json
from typing import Optional
from rich.console import Console
from rich.table import Table


app = typer.Typer(help="EZTinker - Minimal Tinker clone for distributed training")
console = Console()


DEFAULT_BASE_URL = "http://localhost:8000"


def _get_base_url():
    import os

    return os.environ.get("EZTINKER_BASE_URL", DEFAULT_BASE_URL)


@app.command()
def server(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
):
    """Start the EZTinker API server."""
    import uvicorn
    import os

    # Set default checkpoints dir if not set
    if "CHECKPOINTS_DIR" not in os.environ:
        os.environ["CHECKPOINTS_DIR"] = "checkpoints"

    console.print(f"[green]Starting EZTinker server on {host}:{port}[/green]")
    console.print(f"[dim]Workers: {workers}, Reload: {reload}[/dim]")

    uvicorn.run(
        "eztinker.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def create(
    base_model: str = typer.Option(
        "gpt2", "--model", "-m", help="Base model ID or path"
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Custom run ID"),
):
    """Create a new training run."""
    url = f"{_get_base_url()}/v1/runs"
    data = {"base_model": base_model}
    if run_id:
        data["run_id"] = run_id

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()

        console.print(f"[green]✓ Created training run: {result['run_id']}[/green]")
    except Exception as e:
        console.print(f"[red]Error creating run: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def list_runs():
    """List active training runs."""
    url = f"{_get_base_url()}/v1/runs"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data["runs"]:
            console.print("[dim]No active training runs[/dim]")
            return

        table = Table(
            "Run ID",
            "Base Model",
        )
        for run in data["runs"]:
            table.add_row(
                run["run_id"],
                run["base_model"],
            )
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing runs: {e}[/red]")


@app.command()
def delete(run_id: str):
    """Delete a training run."""
    url = f"{_get_base_url()}/v1/runs/{run_id}"

    try:
        response = requests.delete(url)
        response.raise_for_status()
        console.print(f"[green]✓ Deleted run: {run_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error deleting run: {e}[/red]")


@app.command()
def sample(
    prompt: str,
    max_new_tokens: int = typer.Option(100, "--max-tokens", "-n"),
    temperature: float = typer.Option(1.0, "--temperature", "-t"),
):
    """Generate text from prompt."""
    url = f"{_get_base_url()}/v1/sample"
    data = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        job = response.json()

        job_id = job["job_id"]
        console.print(f"[dim]Job submitted: {job_id}[/dim]")

        # Poll for job completion
        for _ in range(10):
            result = _poll_job(job_id)
            if result["status"] == "completed":
                console.print(f"\n[green]{result['result']['generated_text']}[/green]")
                return
            elif result["status"] == "failed":
                console.print(f"[red]Error: {result['error']}[/red]")
                return
            time.sleep(0.5)

        console.print("[yellow]Job not completed yet[/yellow]")
    except Exception as e:
        console.print(f"[red]Error sampling: {e}[/red]")


@app.command()
def save(run_id: str, name: str):
    """Save checkpoint."""
    url = f"{_get_base_url()}/v1/runs/{run_id}/save"
    data = {"name": name}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        job = response.json()

        job_id = job["job_id"]
        console.print(f"[dim]Save job submitted: {job_id}[/dim]")

        # Poll for job completion
        for _ in range(20):
            result = _poll_job(job_id)
            if result["status"] == "completed":
                console.print(f"[green]✓ Checkpoint saved: {name}[/green]")
                return
            elif result["status"] == "failed":
                console.print(f"[red]Error: {result['error']}[/red]")
                return
            time.sleep(0.5)

        console.print("[yellow]Save not completed yet[/yellow]")
    except Exception as e:
        console.print(f"[red]Error saving: {e}[/red]")


def _poll_job(job_id: str):
    """Poll a job result."""
    url = f"{_get_base_url()}/v1/jobs/{job_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    app()