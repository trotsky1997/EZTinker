"""API schemas for EZTinker."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class BatchInput(BaseModel):
    """Training batch input (text-only for MVP)."""
    input_ids: List[int] = Field(description="Input token IDs")
    target_ids: List[int] = Field(description="Target token IDs (for loss)")
    weights: Optional[List[float]] = Field(
        None, description="Token-level weights (optional)"
    )


class LoRAConfig(BaseModel):
    """LoRA configuration."""
    r: int = Field(8, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")
    target_modules: List[str] = Field(
        ["q_proj", "v_proj"], description="Target modules for LoRA"
    )


class CreateTrainingRunRequest(BaseModel):
    """Request to create a training run."""
    base_model: str = Field(description="HF model ID or path")
    lora_config: Optional[LoRAConfig] = Field(default_factory=LoRAConfig)
    run_id: Optional[str] = Field(
        None, description="Custom run ID (auto-generated if None)"
    )


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
    result: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)


class CheckpointInfo(BaseModel):
    """Checkpoint information."""
    name: str
    run_id: str
    created_at: str
    size_bytes: int
    is_sampler: bool = Field(False, description="Sampler-optimized weights")


class EvaluationBatch(BaseModel):
    """Single batch for evaluation."""
    input_ids: List[int] = Field(description="Full sequence input IDs (prompt + response)")
    target_ids: List[int] = Field(description="Full sequence target IDs (shifted inputs)")
    response_ids: List[int] = Field(description="Response portion token IDs only")
    prompt_ids: List[int] = Field(description="Prompt portion token IDs only")


class EvaluationRequest(BaseModel):
    """Request to evaluate multiple responses and return scores."""
    run_id: str = Field(description="Training run ID to use for evaluation")
    batches: List[EvaluationBatch] = Field(description="Evaluation batches")
    temperature: float = Field(1.0, description="Temperature for generation (not used now)")
    max_new_tokens: int = Field(50, description="Max tokens (not used now)")


# =============================================================================
# ShareGPT Data Models
# =============================================================================

class ShareGPTMessage(BaseModel):
    """Single message in a conversation (replaces ConversationTurn)."""
    # Support both dialects
    from_role: Optional[str] = Field(None, alias="from", description="Speaker role (human/gpt)")
    role: Optional[str] = Field(None, description="Speaker role (user/assistant)")
    value: Optional[str] = Field(None, description="Message content (dialect A)")
    content: Optional[str] = Field(None, description="Message content (dialect B)")


class ShareGPTConversation(BaseModel):
    """ShareGPT conversation format."""
    id: str = Field(description="Unique conversation ID")
    conversations: Optional[List[ShareGPTMessage]] = Field(None, description="Conversation turns (dialect A)")
    messages: Optional[List[ShareGPTMessage]] = Field(None, description="Message list (dialect B)")
    system: Optional[str] = Field(None, description="System prompt")
    dataset: Optional[str] = Field(None, description="Dataset source")

    model_config = {"populate_by_name": True}  # Allow alias "from"


class ShareGPTBatchInput(BaseModel):
    """Batch input for ShareGPT format training."""
    conversations: List[ShareGPTConversation] = Field(description="List of ShareGPT conversations")
    tokenizer_name: str = Field(description="Tokenizer model name for formatting")
    max_length: int = Field(2048, description="Maximum sequence length")
    system_prompt: Optional[str] = Field(None, description="System prompt template")