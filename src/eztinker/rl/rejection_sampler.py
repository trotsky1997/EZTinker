"""Rejection sampling for supervised fine-tuning with Math-Verify evaluation."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from ..dataset.gsm8k import GSM8KDataset

MAX_WORKERS = 4
BASE_URL = "http://localhost:8000"


def create_training_run(base_model: str, lora_rank: int = 1) -> str:
    """Create a new training run with specified LoRA configuration.

    Args:
        base_model: HuggingFace model ID
        lora_rank: LoRA rank (1 for this demo)

    Returns:
        run_id: Training run identifier
    """
    # Try with custom run_id for easier management
    response = requests.post(
        f"{BASE_URL}/v1/runs",
        json={
            "base_model": base_model,
            "lora_config": {
                "r": lora_rank,
                "lora_alpha": lora_rank,  # Same as rank
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
    )

    if response.status_code != 200:
        raise Exception(f"Failed to create run: {response.text}")

    run_id = response.json()["run_id"]
    print(f"✓ Created training run: {run_id}")
    return run_id


def wait_for_job(job_id: str, timeout: int = 120) -> dict:
    """Wait for job completion and return result.

    Args:
        job_id: Job identifier
        timeout: Maximum wait time in seconds

    Returns:
        Job result dictionary

    Raises:
        Exception if job fails or times out
    """
    print(f"  Waiting for job {job_id[0:8]}...", end="", flush=True)
    for i in range(timeout * 10):
        response = requests.get(f"{BASE_URL}/v1/jobs/{job_id}")
        result = response.json()

        if result["status"] == "completed":
            print(" ✓")
            return result
        elif result["status"] == "failed":
            print(f" ✗\n    Error: {result['error']}")
            raise Exception(f"Job {job_id} failed: {result['error']}")

        # Show progress every 2 seconds
        if i % 20 == 0 and i > 0:
            print(".", end="", flush=True)
        import time

        time.sleep(0.1)

    raise Exception(f"Job {job_id} timed out after {timeout} seconds")


def generate_candidate_single(
    prompt: str, run_id: str, temperature: float = 0.8, max_new_tokens: int = 400
) -> str:
    """Generate a single candidate response.

    Args:
        prompt: Input prompt
        run_id: Training run ID
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    response = requests.post(
        f"{BASE_URL}/v1/sample",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "do_sample": True,
        },
    )

    job_id = response.json()["job_id"]
    result = wait_for_job(job_id)
    return result["result"]["generated_text"]


def generate_candidates(
    prompt: str, question: str, run_id: str, num_candidates: int = 8, temperature: float = 0.8
) -> list[str]:
    """Generate multiple candidate responses in parallel.

    Args:
        prompt: Formatted prompt for generation
        question: Original question (for better prompting)
        run_id: Training run ID
        num_candidates: Number of candidates to generate
        temperature: Sampling temperature

    Returns:
        List of candidate responses
    """
    with ThreadPoolExecutor(max_workers=min(num_candidates, MAX_WORKERS)) as executor:
        futures = [
            executor.submit(generate_candidate_single, prompt, run_id, temperature)
            for _ in range(num_candidates)
        ]
        candidates = [f.result() for f in futures]
    return candidates


def select_best_candidate_and_train(
    run_id: str,
    prompt: str,
    candidates: list[str],
    ground_truth: str,
    question: str,
    dataset: GSM8KDataset,
    learning_rate: float = 2e-4,
) -> dict:
    """Score candidates and train on the best one in a single step.

    This function:
    1. Scores all candidates using Math-Verify
    2. Selects the best candidate (highest confidence/correctness)
    3. Trains the model on the selected candidate

    Args:
        run_id: Training run ID
        prompt: Prompt text (without response)
        candidates: List of candidate responses
        ground_truth: Expected numerical answer
        question: Original question
        dataset: GSM8K dataset for evaluation
        learning_rate: Learning rate for training

    Returns:
        Dictionary with selection and training results
    """
    # Evaluate all candidates
    candidate_results = []
    for i, candidate in enumerate(candidates):
        eval_result = dataset.evaluate_answer(candidate, ground_truth, question)
        confidence = eval_result.get("confidence", 0.0)
        is_correct = eval_result.get("is_correct", False)
        strategy = eval_result.get("strategy", "unknown")

        # Composite metric: confidence + bonus for correctness
        metric = confidence + 0.5 * float(is_correct)

        candidate_results.append(
            {
                "index": i,
                "text": candidate,
                "confidence": confidence,
                "is_correct": is_correct,
                "strategy": strategy,
                "metric": metric,
            }
        )

    # Select best candidate
    best = max(candidate_results, key=lambda x: x["metric"])

    # Train on best candidate (but only if it's reasonably good)
    if best["is_correct"] or best["confidence"] < 0.2:
        # Encode prompt + best response
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        # For training, we want: [PROMPT] [BEST_RESPONSE] -> shift labels
        full_text = prompt + " " + best["text"]
        tokens = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)

        # Forward-backward pass
        fb_response = requests.post(
            f"{BASE_URL}/v1/runs/{run_id}/forward_backward",
            json={
                "input_ids": tokens["input_ids"].tolist()[0],
                "target_ids": tokens["input_ids"].tolist()[0],
            },
        )
        wait_for_job(fb_response.json()["job_id"])

        # Optimizer step
        optim_response = requests.post(
            f"{BASE_URL}/v1/runs/{run_id}/optim_step",
            json={
                "learning_rate": learning_rate,
                "weight_decay": 0.01,
            },
        )
        wait_for_job(optim_response.json()["job_id"])

        return {
            "selected_index": best["index"],
            "selected_score": best["metric"],
            "selected_is_correct": best["is_correct"],
            "confidence": best["confidence"],
            "selected_text": best["text"][:200] + "...",
            "trained": True,
        }
    else:
        return {
            "selected_index": best["index"],
            "selected_score": best["metric"],
            "selected_is_correct": best["is_correct"],
            "confidence": best["confidence"],
            "selected_text": best["text"][:200] + "...",
            "trained": False,
            "skip_reason": "Poor candidate quality",
        }


def save_buffer(buffer_path: Path, data: list[dict]):
    """Save rejection buffer to JSONL file.

    Args:
        buffer_path: Path to buffer file
        data: List of buffer entries
    """
    buffer_path.parent.mkdir(parents=True, exist_ok=True)

    with open(buffer_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    print(f"✓ Saved {len(data)} entries to {buffer_path}")


def load_buffer(buffer_path: Path) -> list[dict]:
    """Load rejection buffer from JSONL file.

    Args:
        buffer_path: Path to buffer file

    Returns:
        List of buffer entries
    """
    if not buffer_path.exists():
        return []

    data = []
    with open(buffer_path) as f:
        for line in f:
            data.append(json.loads(line))

    print(f"✓ Loaded {len(data)} entries from {buffer_path}")
    return data


def populate_buffer(
    dataset: GSM8KDataset,
    run_id: str,
    buffer_path: Path,
    num_examples: int = 50,
    candidates_per_example: int = 8,
    temperature: float = 0.8,
) -> list[dict]:
    """Pre-populate rejection sampling buffer with top candidates.

    Args:
        dataset: GSM8K dataset
        run_id: Training run ID for generation
        buffer_path: Path to save buffer
        num_examples: Number of examples to process
        candidates_per_example: Candidates to generate per example
        temperature: Sampling temperature

    Returns:
        List of buffer entries with best candidates
    """
    print("\n=== Populating Rejection Buffer ===")
    print(f"Examples: {num_examples}")
    print(f"Candidates per example: {candidates_per_example}")
    print(f"Buffer path: {buffer_path}")

    buffer_entries = []

    for i in range(min(num_examples, len(dataset))):
        print(f"\n[{i + 1}/{num_examples}] Processing example...")

        # Get example
        question, prompt, ground_truth = dataset.get_example_question(i)

        # Generate candidates
        print("  Generating candidates...")
        candidates = generate_candidates(
            prompt=prompt,
            question=question,
            run_id=run_id,
            num_candidates=candidates_per_example,
            temperature=temperature,
        )

        # Evaluate candidates and select best
        print("  Evaluating candidates...")
        result = select_best_candidate_and_train(
            run_id=run_id,
            prompt=prompt,
            candidates=candidates,
            ground_truth=ground_truth,
            question=question,
            dataset=dataset,
            learning_rate=2e-4,
        )

        # Compute mean confidence across candidates
        mean_confidence = result["confidence"]

        # Store in buffer
        buffer_entries.append(
            {
                "question_id": i,
                "question": question,
                "ground_truth": ground_truth,
                "prompt": prompt,
                "selected_text": result["selected_text"],
                "selected_score": result["selected_score"],
                "selected_is_correct": result["selected_is_correct"],
                "mean_confidence": mean_confidence,
                "trained": result["trained"],
                "temperature": temperature,
            }
        )

        print(
            f"  Best candidate: score={result['selected_score']:.2f}, "
            f"correct={result['selected_is_correct']}, trained={result['trained']}"
        )

    # Save buffer
    save_buffer(buffer_path, buffer_entries)

    # Compute quality statistics
    correct_count = sum(1 for x in buffer_entries if x["selected_is_correct"])
    trained_count = sum(1 for x in buffer_entries if x["trained"])

    print("\n=== Buffer Statistics ===")
    print(f"Total entries: {len(buffer_entries)}")
    print(f"Correct answers: {correct_count} ({100 * correct_count / len(buffer_entries):.1f}%)")
    print(f"Entries trained on: {trained_count} ({100 * trained_count / len(buffer_entries):.1f}%)")

    return buffer_entries


if __name__ == "__main__":
    # Quick test
    import time

    print("=== Testing Rejection Sampler ===\n")

    # Wait for server
    print("Checking server health...")
    while True:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ Server is running")
                break
        except Exception:
            print("  Server not ready, retrying...")
            time.sleep(2)

    # Create dataset
    dataset = GSM8KDataset(split="train", max_samples=5)
    print(f"Dataset size: {len(dataset)}")

    # Create training run
    run_id = create_training_run("Qwen/Qwen2-0.5B-Instruct", lora_rank=1)

    # Populate buffer with small number of examples
    buffer_path = Path("data/buffer_test.jsonl")
    buffer = populate_buffer(
        dataset=dataset,
        run_id=run_id,
        buffer_path=buffer_path,
        num_examples=2,
        candidates_per_example=2,
        temperature=0.8,
    )

    print("\n✓ Rejection sampler test completed")
