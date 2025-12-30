"""
Example: Using the new standardized LossFunction interface
==========================================================

This example demonstrates how to use the new standardized loss function interface
with proper type annotations and callable signatures.

The LossFunction Protocol defines a fixed signature that all loss functions must follow:
- Input: logits, labels, optional weights
- Output: scalar loss tensor
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from typing import Protocol

# Import from EZTinker
from eztinker.engine import (
    LossFunction,
    get_loss_function,
    list_loss_functions,
    register_loss_function,
)


# =============================================================================
# Example 1: Using built-in loss functions
# =============================================================================

print("Available loss functions:", list_loss_functions())

# Get the standard cross-entropy loss
cross_entropy_fn = get_loss_function("cross_entropy")

# Get focal loss with custom parameters
focal_loss_fn = get_loss_function("focal_loss")
# Or call it directly with keyword arguments
from eztinker.engine import focal_loss

# Create dummy data
batch_size, seq_len, vocab_size = 2, 16, 50257
logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

# Compute loss
loss = cross_entropy_fn(logits, labels)
print(f"Cross-entropy loss: {loss.item():.4f}")

# Compute loss with weights
weights = torch.ones(batch_size, seq_len)
loss_weighted = cross_entropy_fn(logits, labels, weights=weights)
print(f"Weighted cross-entropy loss: {loss_weighted.item():.4f}")


# =============================================================================
# Example 2: Creating custom loss functions
# =============================================================================

# Custom loss function must follow the LossFunction signature
# def custom_loss_name(
#     logits: torch.Tensor,
#     labels: torch.Tensor,
#     weights: torch.Tensor | None = None,
#     **kwargs,  # Optional configuration parameters
# ) -> torch.Tensor:

def my_label_smoothing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    smoothing: float = 0.1,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """Custom cross-entropy with label smoothing.

    Args:
        logits: Model output [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        weights: Optional token weights [batch_size, seq_len]
        smoothing: Label smoothing factor (default: 0.1)
        ignore_index: Index to ignore in loss computation
        **kwargs: Additional parameters (ignored)

    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Create smoothed labels
    num_classes = shift_logits.size(-1)
    confidence = 1.0 - smoothing
    smooth_labels = torch.zeros_like(shift_logits).scatter_(
        1, shift_labels.unsqueeze(1), confidence
    )
    smooth_labels += smoothing / num_classes

    # Ignore specific indices
    valid = shift_labels != ignore_index

    # Compute KL divergence
    log_probs = F.log_softmax(shift_logits[valid], dim=-1)
    loss = (-smooth_labels[valid] * log_probs).sum(dim=-1).mean()

    return loss


def my_temperature_scaled_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    temperature: float = 2.0,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """Cross-entropy with temperature scaling.

    Higher temperature = softer probability distribution.

    Args:
        logits: Model output [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        weights: Optional token weights [batch_size, seq_len]
        temperature: Temperature parameter (> 1.0 = softer)
        ignore_index: Index to ignore in loss computation
        **kwargs: Additional parameters (ignored)

    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Apply temperature scaling
    scaled_logits = shift_logits / temperature

    # Compute cross-entropy
    loss = F.cross_entropy(
        scaled_logits,
        shift_labels,
        ignore_index=ignore_index,
        reduction="none",
    )

    # Apply weights if provided
    if weights is not None:
        shift_weights = weights[:, 1:].contiguous().view(-1)
        valid = shift_labels != ignore_index
        loss = loss * shift_weights

        if loss[valid].numel() > 0:
            loss = loss[valid].mean()
    else:
        valid = shift_labels != ignore_index
        if loss[valid].numel() > 0:
            loss = loss[valid].mean()

    return loss


# =============================================================================
# Example 3: Registering custom loss functions
# =============================================================================

# Register the custom loss functions
register_loss_function("label_smoothing", my_label_smoothing_loss)
register_loss_function("temperature_scaled", my_temperature_scaled_loss)

# Now use them by name
print("\nAvailable loss functions after registration:", list_loss_functions())

# Use custom loss functions
label_smooth_fn = get_loss_function("label_smoothing")
temp_scaled_fn = get_loss_function("temperature_scaled")

# Compute losses
loss1 = label_smooth_fn(logits, labels, smoothing=0.2)
loss2 = temp_scaled_fn(logits, labels, temperature=1.5, weights=weights, ignore_index=-100)

print(f"Label smoothing loss: {loss1.item():.4f}")
print(f"Temperature scaled loss: {loss2.item():.4f}")


# =============================================================================
# Example 4: Using loss functions in training loop
# =============================================================================

# Define your loss function
loss_fn = get_loss_function("cross_entropy")

# In your training loop:
#
# for batch in dataloader:
#     input_ids = batch["input_ids"].to(device)
#     labels = batch["labels"].to(device)
#     weights = batch.get("weights", None)
#     if weights is not None:
#         weights = weights.to(device)
#
#     # Forward pass
#     outputs = model(input_ids=input_ids, use_cache=False)
#     logits = outputs.logits
#
#     # Compute loss using standardized interface
#     loss = loss_fn(logits, labels, weights=weights)
#
#     # Backward
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()


# =============================================================================
# Example 5: Advanced composition - Weighted combination of multiple losses
# =============================================================================

def multi_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    cross_entropy_weight: float = 0.8,
    entropy_weight: float = 0.2,
    **kwargs,
) -> torch.Tensor:
    """Combined cross-entropy + entropy regularization.

    Args:
        logits: Model output [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        weights: Optional token weights [batch_size, seq_len]
        cross_entropy_weight: Weight for cross-entropy term
        entropy_weight: Weight for entropy regularization term
        **kwargs: Additional parameters

    Returns:
        Combined loss tensor
    """
    # Get standard cross-entropy
    ce_term = cross_entropy_fn(logits, labels, weights=weights)

    # Add entropy regularization
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    valid = shift_labels != kwargs.get("ignore_index", -100)
    valid_logits = shift_logits[valid]

    log_probs = F.log_softmax(valid_logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

    combined_loss = cross_entropy_weight * ce_term - entropy_weight * entropy

    return combined_loss


# Register it
register_loss_function("multi_loss", multi_loss)

# Use it
multi_loss_fn = get_loss_function("multi_loss")
combined_loss = multi_loss_fn(
    logits,
    labels,
    weights=weights,
    cross_entropy_weight=0.7,
    entropy_weight=0.3,
)
print(f"Combined loss (CE + entropy): {combined_loss.item():.4f}")


# =============================================================================
# Type safety
# =============================================================================

# IDE will enforce correct types
# For example, this will show type errors in editors with LSP:
# ❌ cross_entropy_fn(logits, labels, wrong_param=1.0)  # Unsupported keyword argument
# ❌ cross_entropy_fn(logits)  # Missing required argument 'labels'

# ✅ Correct usage:
# cross_entropy_fn(logits, labels)
# cross_entropy_fn(logits, labels, weights=weights)
# cross_entropy_fn(logits, labels, ignore_index=-100)
# cross_entropy_fn(logits, labels, weights=weights, reduction="mean")

print("\n" + "=" * 70)
print("✅ New standardized LossFunction interface demonstrated successfully!")
print("=" * 70)