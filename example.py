#!/usr/bin/env python3
"""Example EZTinker usage - train a simple model on toy data."""
import requests
import time
from transformers import AutoTokenizer

BASE_URL = "http://localhost:8000"


def create_run():
    """Create a new training run."""
    print("Creating training run...")
    response = requests.post(
        f"{BASE_URL}/v1/runs",
        json={
            "base_model": "gpt2",
        }
    )
    run_id = response.json()["run_id"]
    print(f"✓ Created run: {run_id}")
    return run_id


def train_loop(run_id: str):
    """Simple training loop."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "Hello, this is a simple example to train"
    batch = tokenizer(text, return_tensors="pt")

    print("\nStarting training loop...")

    for i in range(5):
        print(f"\n--- Step {i+1} ---")

        # Step 1: Forward + Backward
        fb_response = requests.post(
            f"{BASE_URL}/v1/runs/{run_id}/forward_backward",
            json={
                "input_ids": batch["input_ids"].tolist()[0],
                "target_ids": batch["input_ids"].tolist()[0],
            }
        )
        job_id = fb_response.json()["job_id"]
        print(f"  Job submitted: {job_id}")

        # Step 2: Wait and check result
        for _ in range(5):
            result = requests.get(f"{BASE_URL}/v1/jobs/{job_id}").json()
            if result["status"] == "completed":
                loss = result.get("result", {}).get("loss", 0.0)
                print(f"  Loss: {loss:.4f}")
                break
            time.sleep(0.1)

        # Step 3: Optimizer step
        optim_response = requests.post(
            f"{BASE_URL}/v1/runs/{run_id}/optim_step",
            json={
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
            }
        )
        print(f"  Optimizer step: {optim_response.json()['status']}")


def sample(run_id: str):
    """Generate text samples."""
    print("\nGenerating sample...")
    response = requests.post(
        f"{BASE_URL}/v1/sample",
        json={
            "prompt": "Once upon a time,",
            "max_new_tokens": 30,
            "temperature": 0.7,
        }
    )
    job_id = response.json()["job_id"]

    # Wait for generation
    for _ in range(10):
        result = requests.get(f"{BASE_URL}/v1/jobs/{job_id}").json()
        if result["status"] == "completed":
            generated = result["result"]["generated_text"]
            print(f"\nGenerated text:")
            print(f"- {generated}")
            break
        time.sleep(0.2)


def save_checkpoint(run_id: str):
    """Save checkpoint."""
    print("\nSaving checkpoint...")
    response = requests.post(
        f"{BASE_URL}/v1/runs/{run_id}/save",
        json={"name": "checkpoint_v1"}
    )
    job_id = response.json()["job_id"]

    # Wait for save
    for _ in range(10):
        result = requests.get(f"{BASE_URL}/v1/jobs/{job_id}").json()
        if result["status"] == "completed":
            print("✓ Checkpoint saved")
            print(f"  Adapter: {result['result']['adapter_path']}")
            if result["result"]["optimizer_path"]:
                print(f"  Optimizer: {result['result']['optimizer_path']}")
            break
        time.sleep(0.2)


def main():
    """Main example workflow."""
    print("=" * 50)
    print("EZTinker Example Workflow")
    print("=" * 50)

    # Check server health
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        print("✓ Server is running")
    except requests.exceptions.RequestException:
        print("⚠ Server not found. Please run: uv run eztinker server")
        return

    # Create run
    run_id = create_run()

    # Train
    train_loop(run_id)

    # Sample
    sample(run_id)

    # Save checkpoint
    save_checkpoint(run_id)

    print("\n" + "=" * 50)
    print("Example completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()