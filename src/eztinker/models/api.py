"""API schemas for EZTinker."""

from typing import Any
from pydantic import BaseModel, Field


# =============================================================================
# Environments Hub Models
# =============================================================================

class EnvironmentTemplate(BaseModel):
    """Environment configuration template."""

    template_id: str = Field(description="Unique template ID")
    name: str = Field(description="Environment template display name")
    description: str = Field("", description="Template description")
    model: str = Field(description="Model name (e.g., Qwen/Qwen2-0.5B-Instruct)")

    # LoRA Configuration
    lora_rank: int = Field(1, description="LoRA rank")
    lora_alpha: int = Field(2, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout rate")
    target_modules: str = Field("all-linear", description="LoRA target modules")

    # Training configuration
    learning_rate: float = Field(2e-4, description="Optimizer learning rate")
    weight_decay: float = Field(0.0, description="Weight decay")
    max_length: int = Field(2048, description="Max sequence length")
    batch_size: int = Field(1, description="Training batch size")

    # Device config
    device: str = Field("cuda" if True else "cpu", description="Compute device")  # Simplified
    dtype: str = Field("bfloat16", description="Model precision")

    # Template tags for filtering
    tags: list[str] = Field(default_factory=list, description="Template categories/tags")

    # Usage statistics
    usage_count: int = Field(0, description="Creation count")


class RuntimeEnvironment(BaseModel):
    """Runtime environment state."""

    env_id: str = Field(description="Unique environment identifier")
    template_id: str = Field(description="Environment template reference")
    run_id: str = Field(description="Associated training run")
    created_at: str = Field(description="Environment creation timestamp")
    status: str = Field("active", description="Environment status")
    running_tasks: list[str] = Field(default_factory=list, description="Active tasks/training")
    error_message: str | None = Field(None, description="Last error message")


class BatchInput(BaseModel):
    """Training batch input (text-only for MVP)."""

    input_ids: list[int] = Field(description="Input token IDs")
    target_ids: list[int] = Field(description="Target token IDs (for loss)")
    weights: list[float] | None = Field(None, description="Token-level weights (optional)")


class LoRAConfig(BaseModel):
    """LoRA configuration."""

    r: int = Field(8, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")
    target_modules: str | list[str] = Field("all-linear", description="Target modules for LoRA (Peft auto-detect)")


class CreateTrainingRunRequest(BaseModel):
    """Request to create a training run."""

    base_model: str = Field(description="HF model ID or path")
    lora_config: LoRAConfig | None = Field(default_factory=LoRAConfig)
    run_id: str | None = Field(None, description="Custom run ID (auto-generated if None)")


class CreateTrainingRunResponse(BaseModel):
    """Response for training run creation."""

    run_id: str
    status: str = Field("created")
    message: str


class OptimParams(BaseModel):
    """Optimizer parameters."""

    learning_rate: float = Field(2e-4)
    weight_decay: float = Field(0.0)
    betas: tuple = Field((0.9, 0.999))
    eps: float = Field(1e-8)


class SamplingParams(BaseModel):
    """Sampling parameters."""

    prompt: str
    max_new_tokens: int = Field(100)
    temperature: float = Field(1.0)
    top_p: float = Field(1.0)
    top_k: int = Field(50)
    do_sample: bool = Field(True)


class JobResponse(BaseModel):
    """Standard job response with future handle."""

    job_id: str
    status: str = Field("queued")


class JobResult(BaseModel):
    """Job result for polling."""

    job_id: str
    status: str = Field(description="queued | running | completed | failed")
    result: dict[str, Any] | None = Field(None)
    error: str | None = Field(None)


class CheckpointInfo(BaseModel):
    """Checkpoint information."""

    name: str
    run_id: str
    created_at: str
    size_bytes: int
    is_sampler: bool = Field(False, description="Sampler-optimized weights")


class EvaluationBatch(BaseModel):
    """Single batch for evaluation."""

    input_ids: list[int] = Field(description="Full sequence input IDs (prompt + response)")
    target_ids: list[int] = Field(description="Full sequence target IDs (shifted inputs)")
    response_ids: list[int] = Field(description="Response portion token IDs only")
    prompt_ids: list[int] = Field(description="Prompt portion token IDs only")


class EvaluationRequest(BaseModel):
    """Request to evaluate multiple responses and return scores."""

    run_id: str = Field(description="Training run ID to use for evaluation")
    batches: list[EvaluationBatch] = Field(description="Evaluation batches")
    temperature: float = Field(1.0, description="Temperature for generation (not used now)")
    max_new_tokens: int = Field(50, description="Max tokens (not used now)")


# =============================================================================
# ShareGPT Data Models
# =============================================================================


class ShareGPTMessage(BaseModel):
    """Single message in a conversation (replaces ConversationTurn)."""

    # Support both dialects
    from_role: str | None = Field(None, alias="from", description="Speaker role (human/gpt)")
    role: str | None = Field(None, description="Speaker role (user/assistant)")
    value: str | None = Field(None, description="Message content (dialect A)")
    content: str | None = Field(None, description="Message content (dialect B)")


class ShareGPTConversation(BaseModel):
    """ShareGPT conversation format."""

    id: str = Field(description="Unique conversation ID")
    conversations: list[ShareGPTMessage] | None = Field(
        None, description="Conversation turns (dialect A)"
    )
    messages: list[ShareGPTMessage] | None = Field(None, description="Message list (dialect B)")
    system: str | None = Field(None, description="System prompt")
    dataset: str | None = Field(None, description="Dataset source")

    model_config = {"populate_by_name": True}  # Allow alias "from"


class ShareGPTBatchInput(BaseModel):
    """Batch input for ShareGPT format training."""

    conversations: list[ShareGPTConversation] = Field(description="List of ShareGPT conversations")
    tokenizer_name: str = Field(description="Tokenizer model name for formatting")
    max_length: int = Field(2048, description="Maximum sequence length")
    system_prompt: str | None = Field(None, description="System prompt template")
