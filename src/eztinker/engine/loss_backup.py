"""Loss function registry and implementations for EZTinker."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.api import LossFunctionConfig


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss with optional token-level weights."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index  # type: ignore

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            weights: Optional token-level weights [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        if weights is not None:
            shift_weights = weights[:, 1:].contiguous()
            # Flatten
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)

            # Compute loss
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
            # Apply weights
            loss = loss * shift_weights
            # Mask out ignore_index
            valid = shift_labels != self.ignore_index
            loss = loss[valid].sum() / valid.float().sum()
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.ignore_index,
            )

        return loss


class FocalLoss(nn.Module):
    """Focal loss for classification tasks with class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha  # type: ignore
        self.gamma = gamma  # type: ignore
        self.ignore_index = ignore_index  # type: ignore

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            weights: Optional token-level weights [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            shift_logits, shift_labels, reduction="none", ignore_index=self.ignore_index
        )

        # Get probabilities
        probs = F.softmax(shift_logits, dim=-1)
        pt = probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

        # Focal loss weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = ce_loss * focal_weight

        # Apply token weights if provided
        if weights is not None:
            shift_weights = weights[:, 1:].contiguous().view(-1)
            focal_loss = focal_loss * shift_weights

        # Mask out ignore_index
        valid = shift_labels != self.ignore_index
        loss = focal_loss[valid].sum() / valid.float().sum()

        return loss


class CustomLossFunction(nn.Module):
    """Dynamic custom loss function from code string."""

    def __init__(self, code: str, globals_dict: dict | None = None):
        super().__init__()
        self.code = code  # type: ignore
        self.globals_dict = globals_dict or {}  # type: ignore

        # Add standard imports to globals
        self.globals_dict["torch"] = torch
        self.globals_dict["F"] = F
        self.globals_dict["nn"] = nn

        # Compile the code
        try:
            self.compiled_code = compile(code, "<string>", "exec")  # type: ignore
        except SyntaxError as e:
            raise ValueError(f"Invalid custom loss code syntax: {e}") from e

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Execute custom loss function.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            weights: Optional token-level weights [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        # Create local variables for the function
        locals_dict = {
            "logits": logits,
            "labels": labels,
            "weights": weights,
            "shift_logits": logits[:, :-1, :].contiguous(),
            "shift_labels": labels[:, 1:].contiguous(),
            "loss": None,  # To be set by the custom code
        }

        # Execute the custom code
        try:
            exec(self.compiled_code, self.globals_dict, locals_dict)
            loss = locals_dict.get("loss")

            if loss is None:
                raise ValueError("Custom loss code must set 'loss' variable")

            if not isinstance(loss, torch.Tensor):
                raise ValueError("Custom loss code must return torch.Tensor")

            return loss

        except Exception as e:
            raise RuntimeError(f"Custom loss function execution error: {e}") from e


def get_loss_function(config: LossFunctionConfig) -> Callable:
    """Factory function to get loss function based on config.

    Args:
        config: Loss function configuration

    Returns:
        Loss function callable
    """
    if config.loss_type == "cross_entropy":

        def cross_entropy_loss(logits, labels, weights=None, reduction=config.reduction):
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=config.ignore_index,
                reduction=reduction,
            )

        return cross_entropy_loss

    elif config.loss_type == "weighted_cross_entropy":
        return WeightedCrossEntropyLoss(ignore_index=config.ignore_index)

    elif config.loss_type == "focal_loss":
        return FocalLoss(
            alpha=config.focal_alpha, gamma=config.focal_gamma, ignore_index=config.ignore_index
        )

    elif config.loss_type == "custom":
        if not config.custom_code:
            raise ValueError("Custom loss code must be provided for custom loss_type")

        return CustomLossFunction(code=config.custom_code)

    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")


# Built-in loss function examples
CUSTOM_LOSS_EXAMPLES = {
    "smooth_l1": """
# Smooth L1 loss for token predictions
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

shift_logits = shift_logits.view(-1, shift_logits.size(-1))
shift_labels = shift_labels.view(-1)

# Only compute on non-ignored tokens
valid = shift_labels != -100
loss = F.smooth_l1_loss(shift_logits[valid], shift_labels[valid].float())
""",
    "label_smoothing": """
# Cross-entropy with label smoothing
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

shift_logits = shift_logits.view(-1, shift_logits.size(-1))
shift_labels = shift_labels.view(-1)

# Create smoothed labels
num_classes = shift_logits.size(-1)
smoothing = 0.1
confidence = 1.0 - smoothing
smooth_labels = torch.zeros_like(shift_logits).scatter_(1, shift_labels.unsqueeze(1), confidence)
smooth_labels += smoothing / num_classes

# Compute KL divergence
log_probs = F.log_softmax(shift_logits, dim=-1)
loss = (-smooth_labels * log_probs).sum(dim=-1).mean()
""",
    "contrastive_loss": """
# Simple contrastive loss for language modeling
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

# Standard cross-entropy
ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

# Add regularization term
shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
shift_labels_flat = shift_labels.view(-1)

valid = shift_labels_flat != -100
valid_logits = shift_logits_flat[valid]
valid_labels = shift_labels_flat[valid]

# Compute similarity regularization
log_probs = F.log_softmax(valid_logits, dim=-1)
entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

loss = ce_loss - 0.01 * entropy
""",
}
