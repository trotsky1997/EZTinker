"""Utilities for LoRA."""


def get_peft_model_state_dict(model):
    """Get only trainable parameters (LoRA weights)."""
    from peft import get_peft_model_state_dict

    return get_peft_model_state_dict(model)
