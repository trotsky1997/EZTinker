"""EZTinker - Minimal Tinker clone for distributed model training.

用户在本地写训练循环/算法，服务端负责把操作可靠地跑在 GPU 集群上。

Features:
- High-level Python API for training workflows
- Server-client architecture
- LoRA fine-tuning support
- Multiple dataset formats (GSM8K, ShareGPT)
- Simple CLI for server management

Example:
    # Start server with CLI
    $ eztinker server

    # Use Python client API
    >>> from eztinker import EZTinkerClient
    >>>
    >>> # Create client
    >>> client = EZTinkerClient(base_url="http://localhost:8000")
    >>>
    >>> # Create training run
    >>> run_id = client.create_run("Qwen/Qwen2-0.5B-Instruct", lora_rank=1)
    >>> print(f"Started run: {run_id}")
    >>>
    >>> # Generate samples
    >>> text = client.sample("Hello!", max_new_tokens=50, temperature=0.8)
    >>> print(text)
    >>>
    >>> # Close client
    >>> client.close()

    # Or use context manager
    >>> with EZTinkerClient() as client:
    ...     run_id = client.create_run("Qwen/Qwen2-0.5B-Instruct")
    ...     text = client.sample("Test prompt")
    ...     print(text)

Datasets:
    >>> from eztinker import GSM8KDataset, ShareGPTDataset
    >>>
    >>> # Load GSM8K
    >>> gsm8k = GSM8KDataset(split="train", max_samples=100)
    >>> print(len(gsm8k))
    >>>
    >>> # Load ShareGPT
    >>> sharegpt = ShareGPTDataset(
    ...     file_path="data.json",
    ...     tokenizer=tokenizer
    ... )
    >>> print(sharegpt.stats)

Rejection Sampling:
    >>> from eztinker import create_training_run, generate_candidates
    >>> # See modules for detailed examples
"""

# Client API
from .client import EZTinkerClient

# Dataset loaders
from .dataset import GSM8KDataset, ShareGPTDataset

# Models
from .models.api import (
    CreateTrainingRunRequest,
    EvaluationBatch,
    LoRAConfig,
    OptimParams,
    SamplingParams,
    ShareGPTConversation,
    ShareGPTMessage,
)

# Rejection sampling utilities
from .rl.rejection_sampler import (
    create_training_run,
    generate_candidates,
    load_buffer,
    save_buffer,
    select_best_candidate_and_train,
    wait_for_job,
)

__version__ = "0.1.0"
__author__ = "EZTinker Team"

__all__ = [
    # Main client API
    "EZTinkerClient",
    # Dataset loaders
    "GSM8KDataset",
    "ShareGPTDataset",
    # Rejection sampling
    "create_training_run",
    "generate_candidates",
    "select_best_candidate_and_train",
    "wait_for_job",
    "save_buffer",
    "load_buffer",
    # Models
    "CreateTrainingRunRequest",
    "LoRAConfig",
    "OptimParams",
    "SamplingParams",
    "EvaluationBatch",
    "ShareGPTMessage",
    "ShareGPTConversation",
]
