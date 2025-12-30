#!/usr/bin/env python3
"""
Rejection SFT Demo with ShareGPT Dataset Support.

Supports both GSM8K and ShareGPT datasets with the same rejection sampling training loop.

Usage:
    # Terminal 1: Start server
    uv run eztinker server

    # Terminal 2: Run demo with ShareGPT data
    uv run python rejection_sft_demo_sharegpt.py --dataset-type sharegpt --data-path examples/sharegpt_dialect_b.json --max-samples 50 --epochs 3

    # Or with GSM8K
    uv run python rejection_sft_demo_sharegpt.py --dataset-type gsm8k --max-samples 50 --epochs 3
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
from eztinker.dataset.sharegpt import ShareGPTDataset


# Configuration
BASE_URL = "http://localhost:8000"
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DATASET_GSM8K = "openai/gsm8k"
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


def _evaluate_candidates_gsm8k(candidates, ground_truth, question, dataset):
    """Evaluate candidates for GSM8K dataset."""
    return dataset.evaluate_answer(candidates, ground_truth, question)


def _evaluate_candidate_sharegpt(candidate_text, ground_truth_response):
    """Evaluate candidate for ShareGPT dataset (simple text similarity).

    For ShareGPT, we use a simple overlap-based scoring since we don't have
    programmatic answer extraction like GSM8K. In production, you'd use
    a more sophisticated semantic similarity metric.

    Returns:
        Dict with score, is_correct, confidence
    """
    # Simple token overlap scoring
    candidate_tokens = set(candidate_text.lower().split())
    ground_tokens = set(ground_truth_response.lower().split())

    if len(ground_tokens) == 0:
        return {
            "score": 0.0,
            "is_correct": False,
            "confidence": 0.0
        }

    # Calculate Jaccard similarity
    intersection = len(candidate_tokens & ground_tokens)
    union = len(candidate_tokens | ground_tokens)
    jaccard_score = intersection / union if union > 0 else 0.0

    # Simple threshold for correctness
    is_correct = jaccard_score > 0.3

    # Confidence based on overlap
    confidence = min(jaccard_score * 2, 1.0)  # Scale to [0, 1]

    return {
        "score": jaccard_score,
        "is_correct": is_correct,
        "confidence": confidence
    }


def _extract_last_assistant_response(conversation_turns):
    """Extract the last assistant response from conversation turns."""
    # Find the last assistant turn
    for role, content in reversed(conversation_turns):
        if role == "assistant":
            return content
    return ""


def phase2_populate_buffer(
    dataset,
    dataset_type: str,
    run_id: str,
    args: argparse.Namespace,
    mode: str = "initial"
) -> List[Dict]:
    """Phase 2: Populate or update rejection buffer.

    Args:
        dataset: Dataset object (GSM8K or ShareGPT)
        dataset_type: 'gsm8k' or 'sharegpt'
        run_id: Training run ID
        args: Command-line arguments
        mode: 'initial' for first population, 'update' for periodic refresh

    Returns:
        List of buffer entries
    """
    print(f"\n=== Phase 2: Populating Rejection Buffer ===\n")

    # Determine how many examples to process
    num_examples = min(args.max_samples, len(dataset))

    print(f"Dataset type: {dataset_type}")
    print(f"Samples to process: {num_examples}")
    print(f"Candidates per sample: {args.num_candidates}")
    print(f"Sampling temperature: {args.temperature}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_path = output_dir / "rejection_buffer.jsonl"

    # Load existing buffer if updating
    buffer_entries = load_buffer(buffer_path) if mode == "update" and buffer_path.exists() else []

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

        try:
            if dataset_type == "gsm8k":
                # GSM8K format
                question, prompt, ground_truth = dataset.get_example_question(i)
                full_prompt = prompt
            else:
                # ShareGPT format
                conv_id, formatted_text, num_turns = dataset.get_conversation_turns(i)
                full_prompt = formatted_text
                # For ShareGPT, ground truth is the last assistant response
                conversation = dataset[i]
                ground_truth = _extract_last_assistant_response(conversation['turns'])
                question = conv_id  # Use conversation ID as identifier

            # Print sample
            print(f"  Sample prompt preview: {full_prompt[:100]}...")
            if args.verbose:
                print(f"\n  Full prompt:\n{full_prompt}\n")
                if dataset_type == "sharegpt":
                    print(f"  Ground truth response: {ground_truth[:100]}...\n")

            # Generate candidates
            print("  Generating candidates...")

            # For ShareGPT, we need to use a prompt that doesn't include the assistant response
            if dataset_type == "sharegpt":
                # Build prompt without the assistant response
                conversation = dataset[i]
                turns_to_assistant = conversation['turns'][:-1]  # All turns except last
                prompt_conv = {
                    "id": conversation['id'],
                    "system": conversation['system'],
                    "turns": turns_to_assistant
                }
                prompt_for_generation = dataset.format_conversation_qwen2(prompt_conv)
            else:
                prompt_for_generation = full_prompt

            candidates = generate_candidates(
                prompt=prompt_for_generation,
                question=question,
                run_id=run_id,
                num_candidates=args.num_candidates,
                temperature=args.temperature
            )

            print(f"  Generated {len(candidates)} candidates")

            # Select best candidate
            print("  Evaluating candidates...")

            best_candidate_idx = 0
            best_score = -1
            best_candidate_text = ""

            for idx, cand in enumerate(candidates):
                if dataset_type == "gsm8k":
                    eval_result = _evaluate_candidates_gsm8k(cand, ground_truth, question, dataset)
                    score = eval_result.get("score", 0.0)
                else:
                    eval_result = _evaluate_candidate_sharegpt(cand, ground_truth)
                    score = eval_result.get("score", 0.0)

                if score > best_score:
                    best_score = score
                    best_candidate_idx = idx
                    best_candidate_text = cand

            # For ShareGPT, we train on the full conversation with the best candidate
            if dataset_type == "sharegpt":
                # Create full conversation with best candidate
                conversation = dataset[i]
                # Replace last turn with best candidate
                modified_turns = conversation['turns'][:-1] + [("assistant", best_candidate_text)]
                modified_conv = {
                    "id": conversation['id'],
                    "system": conversation['system'],
                    "turns": modified_turns
                }
                full_text = dataset.format_conversation_qwen2(modified_conv)

                # Tokenize for training
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                tokens = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length
                )
                input_ids = tokens["input_ids"].tolist()[0]

                # Train if best candidate meets threshold
                trained = best_score > 0.3

                if trained:
                    print(f"  Training on best candidate (score={best_score:.3f})...")
                    fb_response = requests.post(
                        f"{BASE_URL}/v1/runs/{run_id}/forward_backward",
                        json={"input_ids": input_ids, "target_ids": input_ids}
                    )
                    fb_result = wait_for_job(fb_response.json()["job_id"])

                    optim_response = requests.post(
                        f"{BASE_URL}/v1/runs/{run_id}/optim_step",
                        json={"learning_rate": args.learning_rate, "weight_decay": 0.01}
                    )
                    optim_result = wait_for_job(optim_response.json()["job_id"])
                else:
                    print(f"  Skipping training (score={best_score:.3f} < threshold)")

            else:
                # GSM8K - use existing logic
                result = select_best_candidate_and_train(
                    run_id=run_id,
                    prompt=full_prompt,
                    candidates=candidates,
                    ground_truth=ground_truth,
                    question=question,
                    dataset=dataset,
                    learning_rate=args.learning_rate,
                )
                best_candidate_text = result.get("selected_text", "")
                best_score = result.get("selected_score", 0.0)
                trained = result.get("trained", False)

            # Store in buffer
            buffer_entries.append({
                "example_id": i,
                "question": question,
                "ground_truth": ground_truth,
                "prompt": full_prompt,
                "best_response": best_candidate_text,
                "best_score": best_score,
                "is_correct": best_score > 0.3,  # Simple threshold
                "confidence": best_score,
                "trained": trained,
                "temperature": args.temperature,
                "dataset_type": dataset_type
            })

            print(f"  Best candidate: score={best_score:.3f}, correct={best_score > 0.3}, trained={trained}")

            # Save buffer periodically
            if processed_count % 10 == 0:
                save_buffer(buffer_path, buffer_entries)
                print(f"  (Saved buffer with {len(buffer_entries)} entries)")

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"  ✗ Error during processing: {e}")
            import traceback
            traceback.print_exc()
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
    dataset,
    dataset_type: str,
    run_id: str,
    args: argparse.Namespace
) -> Dict:
    """Phase 3: Full rejection SFT training loop.

    Args:
        dataset: Dataset object
        dataset_type: 'gsm8k' or 'sharegpt'
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
            dataset_type=dataset_type,
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

            # For ShareGPT, we need to reconstruct the full text
            if dataset_type == "sharegpt":
                # Use the stored best response to create training example
                full_text = prompt  # Prompt already includes the full conversation if GSM8K format
            else:
                # For GSM8K, concatenate prompt and response
                full_text = prompt + best_response

            # Tokenize for training
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
                tokens = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length
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

                if i % 20 == 0 or i >= len(buffer_entries) - 5:
                    print(f"  Epoch {epoch + 1}, trained on {epoch_trained}/{epoch_total} entries")

            except Exception as e:
                print(f"  ✗ Error during training: {e}")
                import traceback
                traceback.print_exc()
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
    dataset,
    dataset_type: str,
    run_id: str,
    args: argparse.Namespace
) -> Dict:
    """Phase 4: Final evaluation on test set.

    Args:
        dataset: Dataset object
        dataset_type: 'gsm8k' or 'sharegpt'
        run_id: Training run ID
        args: Command-line arguments

    Returns:
        Evaluation metrics
    """
    print("=== Phase 4: Final Evaluation ===\n")

    if dataset_type == "gsm8k":
        # Load test split
        test_dataset = GSM8KDataset(
            split="test",
            max_samples=min(args.eval_size, 1000),
            model=None,
            temperature=0,
        )
    else:
        # For ShareGPT, use train split for evaluation (in real setup, should have separate test split)
        print("⚠️  Warning: Using training split for evaluation (no test split in example data)")
        test_dataset = dataset

    print(f"Evaluating on {len(test_dataset)} test examples...\n")

    correct_count = 0
    total_count = 0
    confidence_scores = []

    for i in range(min(len(test_dataset), args.eval_size)):
        if total_count % 10 == 0:
            print(f"Progress: {total_count}/{min(len(test_dataset), args.eval_size)}...")

        try:
            if dataset_type == "gsm8k":
                question, prompt, ground_truth = test_dataset.get_example_question(i)
                full_prompt = prompt
            else:
                # ShareGPT format
                conv_id, formatted_text, num_turns = test_dataset.get_conversation_turns(i)
                full_prompt = test_dataset.format_conversation_qwen2(test_dataset[i])

                # Remove last assistant response for generation
                conversation = test_dataset[i]
                last_assistant_idx = None
                for idx, (role, _) in enumerate(reversed(conversation['turns'])):
                    if role == "assistant":
                        last_assistant_idx = len(conversation['turns']) - 1 - idx
                        break

                if last_assistant_idx is not None:
                    ground_truth = conversation['turns'][last_assistant_idx][1]
                    # Create prompt up to assistant
                    turns_to_assistant = conversation['turns'][:last_assistant_idx]
                    prompt_conv = {
                        "id": conversation['id'],
                        "system": conversation['system'],
                        "turns": turns_to_assistant
                    }
                    full_prompt = test_dataset.format_conversation_qwen2(prompt_conv)
                else:
                    ground_truth = ""
                    print("  Warning: No assistant response found in conversation")

            # Generate single best response
            response = requests.post(
                f"{BASE_URL}/v1/sample",
                json={
                    "prompt": full_prompt,
                    "max_new_tokens": 400,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "do_sample": False
                }
            )
            job_result = wait_for_job(response.json()["job_id"])
            generated_text = job_result["result"]["generated_text"]

            # Evaluate
            if dataset_type == "gsm8k":
                eval_result = test_dataset.evaluate_answer(
                    generated_text,
                    ground_truth,
                    question
                )
            else:
                eval_result = _evaluate_candidate_sharegpt(generated_text, ground_truth)

            if eval_result.get("is_correct", False):
                correct_count += 1

            confidence_scores.append(eval_result.get("confidence", 0.0))
            total_count += 1

        except Exception as e:
            print(f"  Error on example {i}: {e}")
            import traceback
            traceback.print_exc()
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
        description="Rejection SFT Demo for GSM8K and ShareGPT with Qwen2-0.5B"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["gsm8k", "sharegpt"],
        default="gsm8k",
        help="Dataset type to use (default: gsm8k)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to ShareGPT JSON/JSONL file (required for sharegpt)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of training samples (default: {DEFAULT_MAX_SAMPLES})"
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
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum token length for truncation (default: 1024)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.dataset_type == "sharegpt" and not args.data_path:
        parser.error("--data-path is required when dataset-type is sharegpt")

    print("\n" + "=" * 70)
    print(f"Rejection SFT Demo: {args.dataset_type.upper()} with Qwen2-0.5B-Instruct + Rank-1 LoRA")
    print("=" * 70 + "\n")

    print("Configuration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Dataset type: {args.dataset_type}")
    if args.dataset_type == "sharegpt":
        print(f"  Data path: {args.data_path}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Candidates per sample: {args.num_candidates}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Eval size: {args.eval_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max length: {args.max_length}")
    print()

    # Step 0: Check server
    setup_server_connection()

    # Phase 1: Create training run
    run_id = phase1_create_training_run(args)

    # Load dataset
    print("=== Loading Dataset ===\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if args.dataset_type == "gsm8k":
        train_dataset = GSM8KDataset(
            split="train",
            max_samples=args.max_samples,
            model=None,
            temperature=0,
        )
    else:
        # ShareGPT dataset
        train_dataset = ShareGPTDataset(
            file_path=args.data_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_samples=args.max_samples,
            strict=True,
            shuffle=True,
            seed=42
        )

    print(f"✓ Loaded {len(train_dataset)} training examples\n")

    # Phase 2: Populate rejection buffer
    buffer_entries = phase2_populate_buffer(
        dataset=train_dataset,
        dataset_type=args.dataset_type,
        run_id=run_id,
        args=args,
        mode="initial"
    )

    # Phase 3: Rejection SFT training loop
    metrics = phase3_training_loop(
        dataset=train_dataset,
        dataset_type=args.dataset_type,
        run_id=run_id,
        args=args
    )

    # Phase 4: Evaluation
    eval_metrics = phase4_evaluate(
        dataset=train_dataset,
        dataset_type=args.dataset_type,
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
    print(f"\nFinal accuracy: {100 * eval_metrics['accuracy']:.2f}%")
    print(f"Results saved to: {results_path}")
    print()


if __name__ == "__main__":
    main()