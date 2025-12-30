#!/usr/bin/env python3
"""
Rejection Sampling RLHF Demo using EZTinker + TRL

Demonstrates rejection sampling fine-tuning workflow:
1. Generate multiple responses from the policy model
2. Score responses using a reward model
3. Select best response using rejection sampling
4. Fine-tune on accepted responses (ranked pairs)

Concept:
Rejection sampling finetuning (RSF) is a form of reinforcement learning that:
- Generates multiple candidate responses for each prompt
- Uses a reward model to score all responses
- Selects the best response (or top-K) based on reward scores
- Fine-tunes the model on selected high-quality responses
- This teaches the model to generate better quality outputs

Requirements:
    pip install trl transformers torch

Advantages:
    - More stable than PPO
    - No value head needed
    - Easier to debug and understand
    - Works well with LLMs

Note: This demo shows the full workflow. For production use with EZTinker,
you would typically use proper RL techniques like PPO with value models.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check if required libraries are installed
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠ transformers not installed. Install: pip install transformers torch")

# =============================================================================
# Configuration
# =============================================================================

class RSFConfig:
    """Configuration for Rejection Sampling Fine-tuning."""

    # Model configuration
    POLICY_MODEL = "Qwen/Qwen2-0.5B-Instruct"
    REWARD_MODEL = None  # Using synthetic rewards for demo

    # Rejection sampling settings
    NUM_CANDIDATES = 4  # Generate 4 responses per prompt
    TOP_K_RATIO = 0.5   # Select top 50% (2 out of 4)
    TEMPERATURE = 0.7   # Generation temperature

    # Training settings
    NUM_PROMPTS = 10    # Number of training prompts
    NUM_EPOCHS = 1
    LEARNING_RATE = 5e-5

    # LoRA configuration
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1

    # Generation settings
    MAX_NEW_TOKENS = 100

    @classmethod
    def print_config(cls):
        print("\n" + "="*80)
        print("Rejection Sampling Fine-tuning Configuration")
        print("="*80)
        print(f"Policy model: {cls.POLICY_MODEL}")
        print(f"Reward model: {cls.REWARD_MODEL or '(Synthetic)'}")
        print(f"Candidates per prompt: {cls.NUM_CANDIDATES}")
        print(f"Acceptance ratio: {cls.TOP_K_RATIO} (top {int(cls.NUM_CANDIDATES * cls.TOP_K_RATIO)}/{cls.NUM_CANDIDATES})")
        print(f"Training prompts: {cls.NUM_PROMPTS}")
        print(f"LoRA rank: {cls.LORA_R}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print("="*80)


# =============================================================================
# Helper Function: Synthetic Reward Model
# =============================================================================

def synthetic_reward_score(text: str) -> float:
    """Compute synthetic reward score for demo purposes.

    In production, you would use a real reward model like:
    - Anthropic's helpfulness/harmlessness reward models
    - Constitutional AI reward models
    - Custom trained reward model on preference data

    This demo uses simple heuristics:
    - Length (longer responses get slightly higher scores)
    - Contains reasoning keywords (explain, because, therefore)
    - Contains step-by-step structure
    - Avoids repetition

    Args:
        text: Response text to score

    Returns:
        Reward score (higher is better)
    """
    text_lower = text.lower()

    # Base score
    score = 0.5

    # Length bonus
    length_bonus = min(len(text) / 200, 0.3)
    score += length_bonus

    # Reasoning keywords
    reasoning_keywords = ['explain', 'because', 'therefore', 'since', 'thus', 'hence', 'so']
    for keyword in reasoning_keywords:
        if keyword in text_lower:
            score += 0.1

    # Step-by-step structure
    step_keywords = ['first', 'second', 'third', 'step', '1.', '2.', '3.']
    for keyword in step_keywords:
        if keyword in text_lower:
            score += 0.05

    # Question asking (good for clarification)
    if '?' in text:
        score += 0.05

    # Penalize repetition
    words = text.split()
    if len(words) > 5:
        # Check for repeated words
        unique_words = set(w.lower() for w in words)
        repetition_ratio = len(unique_words) / len(words)
        if repetition_ratio < 0.7:
            score -= 0.2

    # Penalize very short responses (less helpful)
    if len(words) < 10:
        score -= 0.2

    # Clip to plausible reward range
    return max(0.0, min(1.0, score))


# =============================================================================
# Rejection Sampling Functions
# =============================================================================

def generate_candidates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_candidates: int
) -> List[str]:
    """Generate multiple candidate responses for a prompt.

    Args:
        model: Language model to generate from
        tokenizer: Tokenizer for processing
        prompt: Input prompt
        num_candidates: Number of candidates to generate

    Returns:
        List of response strings
    """
    candidates = []
    temperature = RSFConfig.TEMPERATURE
    max_new_tokens = RSFConfig.MAX_NEW_TOKENS

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for _ in range(num_candidates):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0])
        candidates.append(response)

    return candidates


def score_and_rank_candidates(
    candidates: List[str],
    reward_func
) -> Tuple[List[Dict], float, float]:
    """Score candidates and return ranked list with statistics.

    Args:
        candidates: List of response strings
        reward_func: Function to score responses

    Returns:
        Tuple of (ranked_candidates, avg_score, best_score)
    """
    # Score all candidates
    scored = []
    for cand in candidates:
        score = reward_func(cand)
        scored.append({"response": cand, "score": score, "reward": score})

    # Rank by score (descending)
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)

    # Compute statistics
    avg_score = sum(x["score"] for x in ranked) / len(ranked)
    best_score = ranked[0]["score"]

    return ranked, avg_score, best_score


def select_best_responses(
    ranked_candidates: List[Dict],
    top_k_ratio: float
) -> List[Dict]:
    """Select top-K responses from ranked list.

    Args:
        ranked_candidates: List of scored and ranked candidates
        top_k_ratio: Ratio of top candidates to select

    Returns:
        List of selected high-quality responses
    """
    num_select = max(1, int(len(ranked_candidates) * top_k_ratio))
    selected = ranked_candidates[:num_select]

    return selected


# =============================================================================
# Training Loop
# =============================================================================

def create_rlhf_preference_dataset(scenarios: List[str], model, tokenizer, reward_func):
    """Create preference dataset using rejection sampling.

    Args:
        scenarios: List of prompt scenarios
        model: Policy model for generation
        tokenizer: Tokenizer
        reward_func: Reward function

    Returns:
        Dataset with (prompt, chosen_response, rejected_response) triplets
    """
    preference_data = []

    print(f"\n{'='*80}")
    print("Generating Preference Data with Rejection Sampling")
    print(f"{'='*80}\n")

    for i, scenario in enumerate(scenarios):
        print(f"Scenario {i+1}/{len(scenarios)}: {scenario[:50]}...")

        # Generate multiple candidates
        candidates = generate_candidates(
            model,
            tokenizer,
            scenario,
            RSFConfig.NUM_CANDIDATES
        )

        # Score and rank
        ranked, avg_score, best_score = score_and_rank_candidates(
            candidates,
            reward_func
        )

        # Select best responses
        selected = select_best_responses(
            ranked,
            RSFConfig.TOP_K_RATIO
        )

        # Create preference pairs: chosen is the best, rejected is the rest
        chosen = selected[0]

        for candidate in ranked[1:]:
            if candidate["score"] < chosen["score"]:
                preference_data.append({
                    "prompt": scenario,
                    "chosen": chosen["response"],
                    "rejected": candidate["response"],
                    "chosen_score": chosen["score"],
                    "rejected_score": candidate["score"],
                })

        print(f"  Generated: {len(candidates)} candidates")
        print(f"  Accepted: {len(selected)} (avg score: {avg_score:.3f}, "
              f"best: {best_score:.3f})")
        print()

    print(f"✓ Created {len(preference_data)} preference pairs")
    print(f"{'='*80}\n")

    return preference_data


def train_with_trl_ppo(demonstration_data: List[Dict], reward_scores: Dict[str, float]):
    """Demonstrate training with TRL PPO.

    Note: This is a demonstration of the concept. In production with EZTinker,
    you would integrate this with the EZTinker training framework.

    Args:
        demonstration_data: Data with prompts and responses
        reward_scores: Dictionary of response -> reward score

    Returns:
        None
    """
    print("\n" + "="*80)
    print("Rejection Sampling Fine-tuning (Conceptual)")
    print("="*80)
    print("\nNote: Full TRL PPO integration with EZTinker is a major feature.")
    print("This demo shows the rejection sampling workflow using local transformers.\n")

    print("Key Steps:")
    print("1. Generate multiple responses for each prompt")
    print("2. Score all responses with reward model")
    print("3. Select top-K responses based on scores")
    print("4. Create preference pairs (chosen vs rejected)")
    print("5. Fine-tune on accepted responses with KL penalty")

    print("\n" + "-"*80)
    print("Would integrate with TRL's PPOTrainer:")
    print("-"*80)

    print("""
    from trl import PPOTrainer, PPOConfig

    # Configure PPO training
    config = PPOConfig(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        learning_rate=5e-5,
        batch_size=4,
        target_kl=0.1,
        ppo_epochs=4,
    )

    # Initialize PPO trainer with:
    # - Policy model: Base model to optimize
    # - Reward model: Scorer for response quality
    # - Value model: Critic for advantage estimation
    - Reference model: For KL penalty (original model)

    # Training loop:
    # 1. Generate responses from policy model
    # 2. Compute rewards from reward model
    # 3. Compute advantages using value model
    # 4. Update policy model with PPO loss
    # 5. Update value model to match rewards
    """)

    print("\n" + "="*80)


# =============================================================================
# Main Demo Execution
# =============================================================================

def run_rejection_sampling_demo():
    """Main execution of rejection sampling demo."""

    # Print configuration
    RSFConfig.print_config()

    # Check dependencies
    if not HAS_TRANSFORMERS:
        print("\n⚠ Required libraries not installed.")
        print("  Install: pip install transformers torch trl")
        return

    # Define training scenarios
    scenarios = [
        "Explain how to calculate the area of a circle.",
        "What is machine learning and how does it work?",
        "Write a short poem about technology.",
        "How do you solve a quadratic equation?",
        "Explain the concept of recursion in programming.",
        "What are the main differences between Python and JavaScript?",
        "Describe how photosynthesis works in plants.",
        "How would you approach debugging a complex software issue?",
        "Explain the benefits of version control systems like Git.",
        "What is the difference between HTTP and HTTPS?",
    ]

    # Download and load model
    print("\n" + "="*80)
    print("Loading Policy Model")
    print("="*80)
    print(f"\nModel: {RSFConfig.POLICY_MODEL}")
    print("This may take a few minutes on first run...\n")

    try:
        # Load policy model (the model we want to improve)
        model = AutoModelForCausalLM.from_pretrained(
            RSFConfig.POLICY_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            RSFConfig.POLICY_MODEL,
            trust_remote_code=True,
        )

        print("✓ Model and tokenizer loaded successfully\n")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nYou may need to:")
        print("  1. Install transformers: pip install transformers")
        print("  2. Download model first (may take a few minutes)")
        return

    # Create preference dataset using rejection sampling
    preference_data = create_rlhf_preference_dataset(
        scenarios,
        model,
        tokenizer,
        synthetic_reward_score  # Use synthetic rewards for demo
    )

    # Save preference data
    output_file = "rejection_sampling_preferences.json"
    with open(output_file, 'w') as f:
        json.dump(preference_data, f, indent=2)
    print(f"✓ Saved preference data to: {output_file}")

    # Print sample preference pair
    if len(preference_data) > 0:
        print("\n" + "="*80)
        print("Sample Preference Pair")
        print("="*80)

        sample = preference_data[0]
        print(f"\nPrompt:\n{sample['prompt'][:150]}...")
        print(f"\nCHOSEN (score: {sample['chosen_score']:.3f}):")
        print(f"{sample['chosen'][:150]}...")
        print(f"\nREJECTED (score: {sample['rejected_score']:.3f}):")
        print(f"{sample['rejected'][:150]}...")

    # Demonstrate training approach
    reward_scores = {}
    for item in preference_data:
        reward_scores[item['chosen']] = item['chosen_score']
        reward_scores[item['rejected']] = item['rejected_score']

    train_with_trl_ppo(preference_data, reward_scores)
    print("="*80)

    # Summary
    print("\n" + "="*80)
    print("Rejection Sampling Demo Complete!")
    print("="*80)
    print(f"\nGenerated {len(preference_data)} preference pairs")
    print(f"From {len(scenarios)} prompts")
    print(f"Using {RSFConfig.NUM_CANDIDATES} candidates per prompt")
    print(f"\nOutput file: {output_file}")
    print("="*80)

    # Explain integration
    print("\n" + "="*80)
    print("Next Steps: EZTinker Integration")
    print("="*80)
    print("""
To integrate this into EZTinker, you would:

1. Add reward modeling support to EZTinker server:
   - RewardModel class
   - Forward pass that returns scalar rewards
   - Support for sequence classification models

2. Add RL training APIs:
   - /v1/runs with reward_model option
   - /v1/generate endpoint for sampling
   - /v1/rl_step for PPO updates

3. Implement PPOTrainer:
   - Value model (critic)
   - Advantage computation
   - PPO loss with KL penalty
   - Clipped policy gradients

4. Support KL penalty:
   - Store reference model (original policy)
   - Compute KL divergence in loss

5. Add sequence batching for RL:
   - Variable-length responses
   - Masking for padding
   - Value head outputs

This requires significant EZTinker core development but is fully feasible!
    """)
    print("="*80)


if __name__ == "__main__":
    run_rejection_sampling_demo()