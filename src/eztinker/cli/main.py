"""CLI interface for EZTinker."""

import os
import sys
import time

import requests
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    help="EZTinker - Minimal Tinker clone for distributed training",
    no_args_is_help=True,
    add_completion=True,
)
console = Console()


DEFAULT_BASE_URL = "http://localhost:8000"


def _get_base_url():
    """Get base URL from environment or use default."""
    return os.environ.get("EZTINKER_BASE_URL", DEFAULT_BASE_URL)


def _check_server_health(base_url: str, timeout: int = 3) -> bool:
    """Check if server is running and healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def _wait_for_server(base_url: str, timeout: int = 30) -> bool:
    """Wait for server to become available."""
    import time

    start_time = time.time()
    while time.time() - start_time < timeout:
        if _check_server_health(base_url):
            return True
        time.sleep(0.5)
    return False


@app.command()
def server(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    checkpoints_dir: str = "checkpoints",
):
    """Start the EZTinker API server.

    Examples:
        eztinker server                          # Start on default port
        eztinker server --port 8080             # Custom port
        eztinker server --workers 4             # Multi-worker mode
        eztinker server --reload                # Auto-reload on changes
        eztinker server --checkpoints-dir data  # Custom checkpoints directory
    """
    import uvicorn

    # Set checkpoint directory
    if "CHECKPOINTS_DIR" not in os.environ:
        os.environ["CHECKPOINTS_DIR"] = checkpoints_dir

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoints_dir, exist_ok=True)

    console.print(
        Panel(
            f"[bold cyan]EZTinker Server[/bold cyan]\n"
            f"Host: {host}\nPort: {port}\nWorkers: {workers}\n"
            f"Reload: {reload}\nCheckpoints: [file]{checkpoints_dir}[/file]",
            title="Server Configuration",
        )
    )

    console.print(f"\n[green]📦 Starting EZTinker server on http://{host}:{port}[/green]\n")

    # Use rich progress for startup
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="Initializing server...")
        uvicorn.run(
            "eztinker.api.server:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
        )


@app.command()
def create(
    base_model: str = typer.Option("gpt2", "--model", "-m", help="Base model ID or path"),
    run_id: str | None = typer.Option(None, "--run-id", help="Custom run ID"),
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


@app.callback()
def main():
    """EZTinker - Minimal Tinker clone for distributed model training.

    用户在本地写训练循环/算法，服务端负责把操作可靠地跑在 GPU 集群上。
    """


@app.command()
def version():
    """Show version information."""
    from .. import __version__

    console.print(f"[bold cyan]EZTinker Version:[/bold cyan] {__version__}")
    console.print(f"[dim]Python:[/dim] {sys.version.split()[0]}")
    console.print(f"[dim]API URL:[/dim] {_get_base_url()}")


@app.command()
def health():
    """Check server health."""
    base_url = _get_base_url()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description="Checking server health...")

        if _check_server_health(base_url):
            console.print("[green]✓ Server is healthy![/green]")
            try:
                response = requests.get(f"{base_url}/health")
                info = response.json()
                console.print(f"[dim]Server time: {info.get('timestamp', 'N/A')}[/dim]")
                console.print(f"[dim]Status: {info.get('status', 'N/A')}[/dim]")
            except:
                pass
        else:
            console.print("[red]✗ Server is not responding[/red]")
            console.print(f"[dim]Attempted to reach: {base_url}/health[/dim]")
            console.print("\n[blue]Try running:[/blue] [bold]eztinker server[/bold]")
            raise typer.Exit(code=1)


@app.command()
def status():
    """Show server status and active runs."""
    base_url = _get_base_url()

    # Check health
    if not _check_server_health(base_url):
        console.print("[red]Server is not running[/red]")
        console.print("\n[blue]To start the server:[/blue]")
        console.print("  eztinker server")
        raise typer.Exit(code=1)

    # Get runs
    try:
        response = requests.get(f"{base_url}/v1/runs")
        response.raise_for_status()
        data = response.json()
        runs = data.get("runs", [])
    except:
        runs = []

    # Display status
    panel = Panel(
        f"[bold green]✓ Server is running[/bold green]\n"
        f"URL: [file]{base_url}[/file]\n"
        f"Active runs: [bold cyan]{len(runs)}[/bold cyan]",
        title="Server Status",
    )
    console.print(panel)

    if runs:
        table = Table("Run ID", "Base Model", title="Active Runs")
        for run in runs:
            table.add_row(f"[bold]{run['run_id']}[/bold]", run["base_model"])
        console.print(table)
    else:
        console.print("[dim]No active training runs[/dim]")


@app.command()
def checkpoints(run_id: str | None = None):
    """List checkpoints for a run or all runs.

    Examples:
        eztinker checkpoints              # List all checkpoints
        eztinker checkpoints --run-id abc # List checkpoints for specific run
    """
    base_url = _get_base_url()

    try:
        if run_id:
            response = requests.get(f"{base_url}/v1/runs/{run_id}/checkpoints")
            response.raise_for_status()
            data = response.json()
            checkpoints = data.get("checkpoints", [])

            table = Table("Name", "Created At", "Size", title=f"Checkpoints for {run_id}")
            for ckpt in checkpoints:
                size_mb = ckpt["size_bytes"] / (1024 * 1024)
                table.add_row(ckpt["name"], ckpt["created_at"], f"{size_mb:.1f} MB")
            console.print(table)
        else:
            response = requests.get(f"{base_url}/v1/runs")
            response.raise_for_status()
            runs = response.json().get("runs", [])

            console.print(f"Found {len(runs)} active runs with checkpoints")
            for run in runs:
                ckpt_response = requests.get(f"{base_url}/v1/runs/{run['run_id']}/checkpoints")
                if ckpt_response.status_code == 200:
                    ckpts = ckpt_response.json().get("checkpoints", [])
                    if ckpts:
                        console.print(f"\n[bold]{run['run_id']}[/bold]: {len(ckpts)} checkpoints")
                        for ckpt in ckpts[:5]:  # Show first 5
                            console.print(f"  - {ckpt['name']}")
                        if len(ckpts) > 5:
                            console.print(f"  ... and {len(ckpts) - 5} more")

    except Exception as e:
        console.print(f"[red]Error listing checkpoints: {e}[/red]")


@app.command()
def demo():
    """Run the rejection SFT demo with default parameters.

    This command starts the rejection sampling training demo using Qwen2-0.5B
    on a small sample of the GSM8K dataset.

    Examples:
        eztinker demo  # Run with default settings
    """
    console.print(Panel("[bold cyan]EZTinker Rejection SFT Demo[/bold cyan]", title="Demo"))

    if not _check_server_health(_get_base_url()):
        console.print("[yellow]⚠️  Server is not running![/yellow]")
        console.print("\nPlease start the server:\n")
        console.print("[bold]  eztinker server[/bold]\n")
        console.print("Then in another terminal:\n")
        console.print("[bold]  eztinker demo[/bold]\n")
        raise typer.Exit(code=1)

    console.print("[blue]Running demo...[/blue]\n")

    try:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "rejection_sft_demo.py",
                "--max-samples",
                "20",
                "--num-candidates",
                "3",
                "--epochs",
                "2",
            ],
            check=True,
        )
        console.print("\n[green]✓ Demo completed![/green]")

    except Exception as e:
        console.print(f"[red]Error running demo: {e}[/red]")
        raise typer.Exit(code=1)


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
