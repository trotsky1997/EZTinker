"""Dataset loaders for EZTinker."""

from .gsm8k import GSM8KDataset
from .sharegpt import ShareGPTDataset

__all__ = ["GSM8KDataset", "ShareGPTDataset"]