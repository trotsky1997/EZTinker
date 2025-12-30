"""New loss function interface with proper typing and callable signature."""

from typing import Protocol

import torch
import torch.nn.functional as F


class LossFunction(Protocol):
    """Protocol for loss functions with standardized signature.

    All loss functions must implement this interface:
    - Input: logits, labels, optional weights
    - Output: scalar loss tensor
    """

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss from model outputs and targets.

        Args:
            logits: Model output [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            weights: Optional token-level weights [batch_size, seq_len]
            **kwargs: Additional configuration (ignore_index, reduction, etc.)

        Returns:
            Scalar loss tensor
        """
        ...


# =============================================================================
# Built-in Loss Functions (all follow LossFunction protocol)
# =============================================================================


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Standard cross-entropy loss for language modeling.

    Args:
        logits: Model output [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        weights: Optional token weights [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_weights = weights[:, 1:].contiguous().view(-1) if weights is not None else None

    # Compute loss
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,
        reduction="none" if shift_weights is not None else reduction,
        label_smoothing=label_smoothing,
    )

    # Apply weights if provided
    if shift_weights is not None:
        loss = loss * shift_weights

        # Mask out ignore_index
        valid = shift_labels != ignore_index
        valid_loss = loss[valid]

        if reduction == "mean":
            return valid_loss.sum() / valid.float().sum()
        elif reduction == "sum":
            return valid_loss.sum()
        else:
            # 'none' not supported with weights, fall back to mean
            return valid_loss.mean()

    return loss


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Weighted cross-entropy loss with token-level weights.

    Requires weights to be provided, unlike cross_entropy_loss.

    Args:
        logits: Model output [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        weights: REQUIRED token weights [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss tensor

    Raises:
        ValueError: If weights are not provided
    """
    if weights is None:
        raise ValueError("weights are required for weighted_cross_entropy")

    return cross_entropy_loss(
        logits=logits,
        labels=labels,
        weights=weights,
        ignore_index=ignore_index,
        reduction=reduction,
    )


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Focal loss for classification tasks with class imbalance.

    Args:
        logits: Model output [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        weights: Optional token weights [batch_size, seq_len]
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma (focusing) parameter
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute cross-entropy
    ce_loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        reduction="none",
        ignore_index=ignore_index,
    )

    # Get probabilities
    probs = F.softmax(shift_logits, dim=-1)
    pt = probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    # Focal loss weight
    focal_weight = (1 - pt) ** gamma

    # Apply focal weight
    focal_loss = ce_loss * focal_weight

    # Apply token weights if provided
    if weights is not None:
        shift_weights = weights[:, 1:].contiguous().view(-1) if weights is not None else None
        if shift_weights is not None:
            focal_loss = focal_loss * shift_weights

    # Filter out ignore_index
    valid = shift_labels != ignore_index
    valid_loss = focal_loss[valid]

    if reduction == "mean":
        return valid_loss.sum() / valid.float().sum()
    elif reduction == "sum":
        return valid_loss.sum()
    else:
        return valid_loss.mean()


def smooth_l1_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Smooth L1 loss for token prediction (alternative to cross-entropy).

    Useful for regression-type tasks or continuous embeddings.

    Args:
        logits: Model output [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        weights: Optional token weights [batch_size, seq_len]
        beta: Smooth L1 beta parameter
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Convert labels to float
    target = shift_labels.float()

    # Compute smooth L1 loss
    loss = F.smooth_l1_loss(
        shift_logits,
        target.unsqueeze(1).expand(-1, shift_logits.size(-1)),
        reduction="none",
        beta=beta,
    ).mean(dim=-1)

    # Apply weights if provided
    if weights is not None:
        shift_weights = weights[:, 1:].contiguous().view(-1)
        loss = loss * shift_weights

    # Filter out ignore_index
    valid = shift_labels != ignore_index
    valid_loss = loss[valid]

    if reduction == "mean":
        return valid_loss.sum() / valid.float().sum()
    elif reduction == "sum":
        return valid_loss.sum()
    else:
        return valid_loss.mean()


def contrastive_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    entropy_weight: float = 0.01,
    ignore_index: int = -100,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Contrastive loss with entropy regularization.

    Combines cross-entropy with entropy regularization to encourage
    model confidence while maintaining diversity.

    Args:
        logits: Model output [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        weights: Optional token weights [batch_size, seq_len]
        entropy_weight: Weight for entropy regularization term
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss tensor
    """
    # Standard cross-entropy
    ce = cross_entropy_loss(
        logits=logits,
        labels=labels,
        weights=weights,
        ignore_index=ignore_index,
        reduction=reduction,
    )

    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute entropy regularization
    valid = shift_labels != ignore_index
    valid_logits = shift_logits[valid]

    log_probs = F.log_softmax(valid_logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

    # Combined loss
    loss = ce - entropy_weight * entropy

    return loss


# =============================================================================
# Loss Function Registry
# =============================================================================

LOSS_REGISTRY: dict[str, LossFunction] = {
    "cross_entropy": cross_entropy_loss,
    "weighted_cross_entropy": weighted_cross_entropy,
    "focal_loss": focal_loss,
    "smooth_l1": smooth_l1_loss,
    "contrastive_loss": contrastive_loss,
}


def get_loss_function(name_or_config: str) -> LossFunction:
    """Get loss function by name or config.

    Args:
        name_or_config: Name of the loss function (str) or LossFunctionConfig object

    Returns:
        Loss function callable

    Raises:
        ValueError: If loss function name is not in registry
        TypeError: If name_or_config has invalid type
    """
    # Import here to avoid circular imports
    from ..models.api import LossFunctionConfig

    # Support both string name and LossFunctionConfig
    if isinstance(name_or_config, LossFunctionConfig):
        name = name_or_config.loss_type
    elif isinstance(name_or_config, str):
        name = name_or_config
    else:
        raise TypeError(f"Expected str or LossFunctionConfig, got {type(name_or_config).__name__}")

    if name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise ValueError(f"Unknown loss function: {name}. Available: {available}")

    loss_fn = LOSS_REGISTRY[name]

    # Wrap with kwargs extraction if using config object
    if isinstance(name_or_config, LossFunctionConfig):
        return lambda logits, labels, weights=None, **kwargs: loss_fn(
            logits,
            labels,
            weights,
            ignore_index=name_or_config.ignore_index,
            reduction=name_or_config.reduction,
            **kwargs,
        )

    return loss_fn


def register_loss_function(name: str, func: LossFunction) -> None:
    """Register a custom loss function.

    Args:
        name: Name to register the loss function under
        func: Loss function callable following LossFunction protocol

    Examples:
        >>> def my_custom_loss(logits, labels, weights=None, **kwargs):
        ...     # Your implementation here
        ...     return loss
        ...
        >>> register_loss_function("my_loss", my_custom_loss)
        >>> loss_fn = get_loss_function("my_loss")
    """
    # Verify function signature
    import inspect

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    required = ["logits", "labels", "weights"]
    for req in required:
        if req not in params[:3]:
            raise ValueError(
                f"Loss function must accept 'logits', 'labels', and 'weights' "
                f"in that order. Got: {params}"
            )

    LOSS_REGISTRY[name] = func


def list_loss_functions() -> list[str]:
    """List all available loss functions."""
    return sorted(LOSS_REGISTRY.keys())
