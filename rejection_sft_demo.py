#!/usr/bin/env python3
"""
Rejection SFT Demo: Train Qwen2-0.5B-Instruct on GSM8K with rejection sampling.

Architecture:
- Client-server design with EZTinker API
- Server handles model operations (forward/backward, optim, sampling)
- Client implements rejection sampling logic
- Math-Verify for GSM8K answer evaluation

Usage:
    # Terminal 1: Start server
    uv run eztinker server

    # Terminal 2: Run demo
    uv run python rejection_sft_demo.py --max-samples 50 --num-candidates 4 --epochs 3
"""

import argparse
import time
import json
from pathlib import Path
from typing import List, Dict
import requests
from transformers import AutoTokenizer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from eztinker.rl.rejection_sampler import (
    create_training_run,
    generate_candidates,
    select_best_candidate_and_train,
    wait_for_job,
    save_buffer,
    load_buffer,
)
from eztinker.dataset.gsm8k import GSM8KDataset


# Configuration
BASE_URL = "http://localhost:8000"
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DATASET_NAME = "openai/gsm8k"
DEFAULT_NUM_CANDIDATES = 4
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_SAMPLES = 100


def setup_server_connection():
    """Wait for EZTinker server to be ready."""
    print("=== Checking Server Connection ===\n")
    while True:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ EZTinker server is running\n")
                return
        except:
            print("  Server not ready, retrying in 2 seconds...")
            time.sleep(2)


def phase1_create_training_run(args) -> str:
    """Phase 1: Create training run with rank-1 LoRA.

    Returns:
        Training run ID
    """
    print("=== Phase 1: Creating Training Run ===\n")

    try:
        # Create rank-1 LoRA run
        run_id = create_training_run(
            base_model=MODEL_ID,
            lora_rank=1
        )
        print(f"✓ Training run created: {run_id}\n")
        return run_id

    except Exception as e:
        print(f"✗ Failed to create training run: {e}\n")
        raise


def phase2_populate_buffer(
    dataset: GSM8KDataset,
    run_id: str,
    args: argparse.Namespace,
    mode: str = "initial"
) -> List[Dict]:
    """Phase 2: Populate or update rejection buffer.

    Args:
        dataset: GSM8K dataset
        run_id: Training run ID
        args: Command-line arguments
        mode: 'initial' for first population, 'update' for periodic refresh

    Returns:
        List of buffer entries
    """
    print(f"\n=== Phase 2: Populating Rejection Buffer ===\n")

    # Determine how many examples to process
    num_examples = min(args.max_samples, len(dataset))

    print(f"Samples to process: {num_examples}")
    print(f"Candidates per sample: {args.num_candidates}")
    print(f"Sampling temperature: {args.temperature}\n")

    buffer_path = Path(args.output_dir) / "rejection_buffer.jsonl"
    buffer_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing buffer if updating
    buffer_entries = load_buffer(buffer_path) if mode == "update" else []

    # Clear existing buffer for initial population
    if mode == "initial":
        buffer_entries = []

    # Track new example index
    start_idx = len(buffer_entries)
    processed_count = 0

    # Process examples
    for i in range(start_idx, min(start_idx + num_examples, len(dataset))):
        processed_count += 1
        print(f"[{processed_count}/{num_examples}] Processing example {i+1}...")

        # Get example
        question, prompt, ground_truth = dataset.get_example_question(i)

        # Generate candidates
        print("  Generating candidates...")
        candidates = generate_candidates(
            prompt=prompt,
            question=question,
            run_id=run_id,
            num_candidates=args.num_candidates,
            temperature=args.temperature
        )

        # Evaluate candidates and potentially train on best
        print("  Evaluating candidates...")
        try:
            result = select_best_candidate_and_train(
                run_id=run_id,
                prompt=prompt,
                candidates=candidates,
                ground_truth=ground_truth,
                question=question,
                dataset=dataset,
                learning_rate=args.learning_rate,
            )

            # Store in buffer
            buffer_entries.append({
                "example_id": i,
                "question": question,
                "ground_truth": ground_truth,
                "prompt": prompt,
                "best_response": result.get("selected_text", "FAILED"),
                "best_score": result.get("selected_score", 0.0),
                "is_correct": result.get("selected_is_correct", False),
                "confidence": result.get("confidence", 0.0),
                "trained": result.get("trained", False),
                "temperature": args.temperature,
            })

            print(f"  Best candidate: score={result['selected_score']:.2f}, "
                  f"correct={result['selected_is_correct']}, trained={result['trained']}")

            # Save buffer periodically
            if processed_count % 10 == 0:
                save_buffer(buffer_path, buffer_entries)
                print(f"  (Saved buffer with {len(buffer_entries)} entries)")

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"  ✗ Error during processing: {e}")
            continue

    # Final save
    save_buffer(buffer_path, buffer_entries)

    # Compute statistics
    print(f"\n=== Buffer Statistics ===")
    print(f"Total entries: {len(buffer_entries)}")

    if buffer_entries:
        correct_count = sum(1 for x in buffer_entries if x.get("is_correct", False))
        trained_count = sum(1 for x in buffer_entries if x.get("trained", False))
        mean_score = sum(x.get("best_score", 0.0) for x in buffer_entries) / len(buffer_entries)

        print(f"Correct answers: {correct_count} ({100*correct_count/len(buffer_entries):.1f}%)")
        print(f"Entries trained on: {trained_count} ({100*trained_count/len(buffer_entries):.1f}%)")
        print(f"Mean score: {mean_score:.3f}")

    print()
    return buffer_entries


def phase3_training_loop(
    dataset: GSM8KDataset,
    run_id: str,
    args: argparse.Namespace
) -> Dict:
    """Phase 3: Full rejection SFT training loop.

    Args:
        dataset: GSM8K dataset
        run_id: Training run ID
        args: Command-line arguments

    Returns:
        Training metrics (losses, accuracies)
    """
    print("=== Phase 3: Rejection SFT Training Loop ===\n")

    buffer_path = Path(args.output_dir) / "rejection_buffer.jsonl"

    # Load or create buffer
    if buffer_path.exists() and not args.reset_buffer:
        buffer_entries = load_buffer(buffer_path)
        print(f"✓ Loaded {len(buffer_entries)} entries from buffer\n")
    else:
        print("Creating new buffer...\n")
        buffer_entries = phase2_populate_buffer(
            dataset=dataset,
            run_id=run_id,
            args=args,
            mode="initial"
        )

    if not buffer_entries:
        print("✗ No buffer entries to train on!\n")
        return {}

    print(f"Training for {args.epochs} epochs...")

    metrics = {
        "initial": {},
        "epoch_stats": []
    }

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---\n")

        epoch_trained = 0
        epoch_correct = 0
        epoch_total = 0

        for i, entry in enumerate(buffer_entries):
            epoch_total += 1

            # Skip entries that were already trained
            if not entry.get("trained", False):
                continue

            print(f"[{i+1}/{len(buffer_entries)}] Re-training on buffer entry...")

            question = entry["question"]
            prompt = entry["prompt"]
            ground_truth = entry["ground_truth"]
            best_response = entry["best_response"]

            # Compute confidence of current model on this response
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

                # Format for training
                full_text = prompt + " " + best_response
                tokens = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                input_ids = tokens["input_ids"].tolist()[0]

                # Forward-backward pass
                fb_response = requests.post(
                    f"{BASE_URL}/v1/runs/{run_id}/forward_backward",
                    json={
                        "input_ids": input_ids,
                        "target_ids": input_ids,
                    }
                )
                fb_result = wait_for_job(fb_response.json()["job_id"])

                # Optimizer step
                optim_response = requests.post(
                    f"{BASE_URL}/v1/runs/{run_id}/optim_step",
                    json={
                        "learning_rate": args.learning_rate,
                        "weight_decay": 0.01,
                    }
                )
                optim_result = wait_for_job(optim_response.json()["job_id"])

                epoch_trained += 1

                if i % 20 == 0:
                    print(f"  Epoch {epoch + 1}, trained on {epoch_trained}/{epoch_total} entries")

            except Exception as e:
                print(f"  ✗ Error during training: {e}")
                continue

        # Epoch statistics
        epoch_stats = {
            "epoch": epoch + 1,
            "total_entries": epoch_total,
            "trained_count": epoch_trained,
            "correct_count": epoch_correct,
        }
        metrics["epoch_stats"].append(epoch_stats)

        print(f"\n=== Epoch {epoch + 1} Complete ===")
        print(f"Trained on: {epoch_trained}/{epoch_total} entries")

        # Save checkpoint after each epoch
        if args.checkpoint:
            save_path = f"rejection_sft_epoch_{epoch+1}"
            print(f"\nSaving checkpoint: {save_path}")
            save_response = requests.post(
                f"{BASE_URL}/v1/runs/{run_id}/save",
                json={"name": save_path}
            )
            wait_for_job(save_response.json()["job_id"])
            print("✓ Checkpoint saved")

    print("\n✓ Training loop completed\n")
    return metrics


def phase4_evaluate(
    dataset: GSM8KDataset,
    run_id: str,
    args: argparse.Namespace
) -> Dict:
    """Phase 4: Final evaluation on test set.

    Args:
        dataset: GSM8K dataset (use test split)
        run_id: Training run ID
        args: Command-line arguments

    Returns:
        Evaluation metrics
    """
    print("=== Phase 4: Final Evaluation ===\n")

    # Load test split
    test_dataset = GSM8KDataset(
        split="test",
        max_samples=min(args.eval_size, 1000),
        model=None,
        temperature=0,
    )

    print(f"Evaluating on {len(test_dataset)} test examples...\n")

    correct_count = 0
    total_count = 0
    confidence_scores = []

    for i in range(len(test_dataset)):
        if total_count % 10 == 0:
            print(f"Progress: {total_count}/{len(test_dataset)}...")

        question, prompt, ground_truth = test_dataset.get_example_question(i)

        try:
            # Generate single best response
            response = requests.post(
                f"{BASE_URL}/v1/sample",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 400,
                    "temperature": 0.0,  # Deterministic for evaluation
                    "top_p": 1.0,
                    "do_sample": False
                }
            )
            job_result = wait_for_job(response.json()["job_id"])
            generated_text = job_result["result"]["generated_text"]

            # Evaluate
            eval_result = test_dataset.evaluate_answer(
                generated_text,
                ground_truth,
                question
            )

            if eval_result.get("is_correct", False):
                correct_count += 1

            confidence_scores.append(eval_result.get("confidence", 0.0))
            total_count += 1

        except Exception as e:
            print(f"  Error on example {i}: {e}")
            continue

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    metrics = {
        "total_examples": total_count,
        "correct": correct_count,
        "accuracy": accuracy,
        "mean_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    }

    print(f"\n=== Evaluation Results ===")
    print(f"Total: {total_count} examples")
    print(f"Correct: {correct_count} examples")
    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Mean confidence: {metrics['mean_confidence']:.3f}")
    print()

    return metrics


def main():
    """Main demo workflow."""
    parser = argparse.ArgumentParser(
        description="Rejection SFT Demo for GSM8K with Qwen2-0.5B"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of GSM8K training samples (default: {DEFAULT_MAX_SAMPLES})"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help=f"Candidates per example (default: {DEFAULT_NUM_CANDIDATES})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation (default: 0.8)"
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=100,
        help="Number of test examples for evaluation (default: 100)"
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Save checkpoints after each epoch"
    )
    parser.add_argument(
        "--reset-buffer",
        action="store_true",
        help="Reset rejection buffer and start fresh"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for logs and checkpoints (default: data)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Rejection SFT Demo: GSM8K with Qwen2-0.5B-Instruct + Rank-1 LoRA")
    print("=" * 70 + "\n")

    print("Configuration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Candidates per sample: {args.num_candidates}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Eval size: {args.eval_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Temperature: {args.temperature}")
    print()

    # Step 0: Check server
    setup_server_connection()

    # Phase 1: Create training run
    run_id = phase1_create_training_run(args)

    # Load train dataset
    print("=== Loading GSM8K Dataset ===\n")
    train_dataset = GSM8KDataset(
        split="train",
        max_samples=args.max_samples,
        model=None,
        temperature=0,
    )
    print(f"✓ Loaded {len(train_dataset)} training examples\n")

    # Phase 2: Populate rejection buffer
    buffer_entries = phase2_populate_buffer(
        dataset=train_dataset,
        run_id=run_id,
        args=args,
        mode="initial"
    )

    # Phase 3: Rejection SFT training loop
    metrics = phase3_training_loop(
        dataset=train_dataset,
        run_id=run_id,
        args=args
    )

    # Phase 4: Evaluation
    eval_metrics = phase4_evaluate(
        dataset=GSM8KDataset,
        run_id=run_id,
        args=args
    )

    # Save results
    results_path = Path(args.output_dir) / "rejection_sft_results.json"
    results = {
        "run_id": run_id,
        "config": vars(args),
        "training_metrics": metrics,
        "evaluation_metrics": eval_metrics,
        "buffer_size": len(buffer_entries) if buffer_entries else 0,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Rejection SFT Demo Completed!")
    print("=" * 70)
    print(f"\nFinal test accuracy: {100 * eval_metrics['accuracy']:.2f}%")
    print(f"Results saved to: {results_path}")
    print()


if __name__ == "__main__":
    main()