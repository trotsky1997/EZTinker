#!/usr/bin/env python3
"""
Example usage of the EZTinker client API.

This demonstrates the elegant Python API for interacting with EZTinker.

Prerequisites:
1. Start the EZTinker server:
   $ eztinker server

Running this example:
$ uv run python examples/example_client_api.py

The script will show 3 different ways to interact with the server:
1. Greeting using context manager (cleanest code)
2. Sample generation (longest version)
3. Sample generation (compact version)
"""

import os


def example1_greeting_and_version():
    """Example 1: Check health and get version (Context Manager Style)."""
    from eztinker import EZTinkerClient

    print("\n" + "=" * 70)
    print("Example 1: Check Server Health (Context Manager Style)")
    print("=" * 70 + "\n")

    # Use context manager - automatically closes session
    try:
        with EZTinkerClient() as client:
            # Check server health
            health_info = client.health()
            print(f"✓ Server is healthy! Status: {health_info['status']}")
            print(f"  Timestamp: {health_info.get('timestamp', 'N/A')}")
    except Exception as e:
        print(f"✗ Could not connect to server: {e}")
        print("\nPlease start the server with:")
        print("  eztinker server")
        print("\nThen run this example again.\n")
        return

    print("\nCode example:")
    code = """
from eztinker import EZTinkerClient

with EZTinkerClient() as client:
    health = client.health()
    print(f"Status: {health['status']}")
"""
    print(code)


def example2_load_gsm8k_dataset():
    """Example 2: Load GSM8K dataset."""
    from eztinker import GSM8KDataset

    print("\n" + "=" * 70)
    print("Example 2: Load GSM8K Dataset")
    print("=" * 70 + "\n")

    print("Loading GSM8K dataset (small sample)...")
    dataset = GSM8KDataset(split="train", max_samples=10, use_math_verify=False)

    print(f"✓ Loaded {len(dataset)} training examples")
    print(f"  example: {dataset.get_example_question(0)[0][:60]}...")

    print("\nCode example:")
    code = """
from eztinker import GSM8KDataset

dataset = GSM8KDataset(split="train", max_samples=100)
print(f"Loaded {len(dataset)} examples")
question, prompt, answer = dataset.get_example_question(0)
"""
    print(code)


def example3_load_sharegpt_dataset():
    """Example 3: Load ShareGPT dataset with dialect detection."""
    from transformers import AutoTokenizer

    from eztinker import ShareGPTDataset

    print("\n" + "=" * 70)
    print("Example 3: Load ShareGPT Dataset (with automatic dialect detection)")
    print("=" * 70 + "\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        print("Loading ShareGPT dataset...")
        dataset = ShareGPTDataset(
            file_path="examples/sharegpt_dialect_b.json", tokenizer=tokenizer, max_samples=5
        )

        print("✓ Dataset loaded successfully!")
        print(f"  Total entries: {dataset.stats['total_loaded']}")
        print(f"  Valid conversations: {dataset.stats['valid_conversations']}")
        print("  Dialect detection:")
        for dialect, count in dataset.stats["dialect_counts"].items():
            if count > 0:
                print(f"    - {dialect}: {count}")

        # Show first conversation
        conv = dataset[0]
        print("\n  First conversation:")
        print(f"    ID: {conv['id']}")
        print(f"    Turns: {len(conv['turns'])}")
        print(f"    System: {conv['system'][:50]}...")

    except FileNotFoundError:
        print("✗ Example data not found, run this from the project root directory")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\nCode example:")
    code = """
from eztinker import ShareGPTDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
dataset = ShareGPTDataset(
    file_path="data.json",
    tokenizer=tokenizer
)
print(f"Loaded {len(dataset)} conversations")
"""
    print(code)


def example4_create_run():
    """Example 4: Create a training run."""
    from eztinker import EZTinkerClient

    print("\n" + "=" * 70)
    print("Example 4: Create Training Run")
    print("=" * 70 + "\n")

    print("Connecting to server...")
    try:
        with EZTinkerClient() as client:
            # Check server
            client.health()

            # List existing runs
            print("Current training runs:")
            runs = client.get_runs()
            if not runs:
                print("  (No active runs)")
            else:
                for run in runs:
                    print(f"  - {run['run_id']}: {run['base_model']}")

            # Create new run
            print("\nCreating new training run with Qwen2-0.5B-Instruct...")
            print("(Note: This will download the model if not cached)")

            run_id = client.create_run(
                base_model="Qwen/Qwen2-0.5B-Instruct",
                lora_rank=1,  # Very small LoRA for demonstration
                lora_alpha=2,
            )

            print(f"✓ Created training run: {run_id}")

            # List runs again
            print("\nUpdated training runs:")
            runs = client.get_runs()
            for run in runs:
                if run["run_id"] == run_id:
                    print(f"  - {run['run_id']}: {run['base_model']} (← NEW)")

    except Exception as e:
        print(f"✗ Error: {e}")

    print("\nCode example:")
    code = """
from eztinker import EZTinkerClient

client = EZTinkerClient()
run_id = client.create_run(
    base_model="Qwen/Qwen2-0.5B-Instruct",
    lora_rank=8,
    lora_alpha=16
)
print(f"Created run: {run_id}")
"""
    print(code)


def example5_generate_samples():
    """Example 5: Generate text samples."""
    from eztinker import EZTinkerClient

    print("\n" + "=" * 70)
    print("Example 5: Generate Text Samples")
    print("=" * 70 + "\n")

    if not os.environ.get("EZTINKER_BASE_URL"):
        print("Setting default to localhost:8000...")

    try:
        print("Generating sample text (this may take a few seconds)...")
        with EZTinkerClient() as client:
            # Generate with custom parameters
            text = client.sample(
                prompt="def hello_world():",
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
            )

            print("Generated text:")
            print("-" * 70)
            print(text)
            print("-" * 70)

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure the server is running:")
        print("  eztinker server")
        print("\nAnd create a run first:")
        print("  eztinker create --model Qwen/Qwen2-0.5B-Instruct")

    print("\nCode example:")
    code = """
from eztinker import EZTinkerClient

client = EZTinkerClient()
text = client.sample(
    prompt="Hello, world!",
    max_new_tokens=100,
    temperature=0.8
)
print(text)
"""
    print(code)


def example6_rejection_sampling():
    """Example 6: Rejection Sampling Workflow."""
    from eztinker import GSM8KDataset

    print("\n" + "=" * 70)
    print("Example 6: Rejection Sampling Training")
    print("=" * 70 + "\n")

    print("This example shows the rejection sampling workflow:")
    print("1. Load dataset")
    print("2. Generate multiple candidates")
    print("3. Select best candidate")
    print("4. Train on best candidate\n")

    try:
        print("Loading GSM8K dataset...")
        dataset = GSM8KDataset(split="train", max_samples=5, use_math_verify=False)
        print(f"✓ Loaded {len(dataset)} examples\n")

        question, prompt, ground_truth = dataset.get_example_question(0)
        print(f"Sample question: {question[:80]}...\n")

        print("This would typically involve:")
        print("  - Creating a run with create_training_run()")
        print("  - Generating candidates with generate_candidates()")
        print("  - Evaluating candidates with dataset.evaluate_answer()")
        print("  - Training on best candidate")
        print("\nSee rejection_sft_demo.py and rejection_sft_demo_sharegpt.py for full examples")

    except Exception as e:
        print(f"Error: {e}")

    print("\nCode example:")
    code = """
from eztinker import (
    GSM8KDataset,
    create_training_run,
    generate_candidates,
)

# Load dataset
dataset = GSM8KDataset(split="train", max_samples=100)
question, prompt, ground_truth = dataset.get_example_question(0)

# Create run
run_id = create_training_run("Qwen/Qwen2-0.5B-Instruct", lora_rank=1)

# Generate candidates
candidates = generate_candidates(
    prompt, question, run_id,
    num_candidates=4,
    temperature=0.8
)

# Evaluate and train on best
# (see rejection sampler for full implementation)
"""
    print(code)


def example7_compare_raw_api_vs_client():
    """Example 7: Compare raw API vs client API."""
    print("\n" + "=" * 70)
    print("Example 7: Raw HTTP API vs Client API Comparison")
    print("=" * 70 + "\n")

    print("OLD STYLE - Raw HTTP API calls:\n")
    raw_code = """
import requests

# Manual health check
response = requests.get("http://localhost:8000/health")
health_info = response.json()

# Manual create run
response = requests.post("http://localhost:8000/v1/runs", json={
    "base_model": "Qwen/Qwen2-0.5B-Instruct"
})
result = response.json()
run_id = result["run_id"]

# Manual sample with job polling
response = requests.post("http://localhost:8000/v1/sample", json={
    "prompt": "Hello!",
    "max_new_tokens": 100,
    "temperature": 0.8
})
job = response.json()

# Wait for job...
# (Need loop with /v1/jobs/{job_id})
"""
    print(raw_code)

    print("-" * 70)
    print("\nNEW STYLE - Elegant Client API:\n")
    new_code = """
from eztinker import EZTinkerClient

# One-liner with context manager
with EZTinkerClient() as client:
    # Simple method calls
    health = client.health()
    run_id = client.create_run("Qwen/Qwen2-0.5B-Instruct")
    text = client.sample("Hello!", max_new_tokens=100, temperature=0.8)
    print(text)
    # Cleanup happens automatically!
"""
    print(new_code)


def main():
    """Run all examples."""
    print("\n" + "🎯" * 35)
    print("EZTinker Client API Examples")
    print("🎯" * 35)

    print("\nThese examples demonstrate the elegant Python API for EZTinker.")
    print("Use the examples as a reference for building your own training workflows.")

    # Run examples
    example1_greeting_and_version()
    example2_load_gsm8k_dataset()
    example3_load_sharegpt_dataset()
    example4_create_run()
    example5_generate_samples()
    example6_rejection_sampling()
    example7_compare_raw_api_vs_client()

    # Final message
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start the server:  eztinker server")
    print("  2. Create a run:      eztinker create --model MODEL")
    print('  3. Generate text:     eztinker sample "Your prompt here"')
    print("  4. See all commands:  eztinker --help")
    print("  5. Run these examples:")
    print("       uv run python examples/example_client_api.py")

    print("\nFor more details:")
    print("  - Full documentation: README.md")
    print("  - Rejection SFT demo: rejection_sft_demo.py")
    print("  - ShareGPT demo:      rejection_sft_demo_sharegpt.py")
    print("  - Client API:         from eztinker import EZTinkerClient")
    print()


if __name__ == "__main__":
    main()
