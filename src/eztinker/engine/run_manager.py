"""Training run manager - holds state for each active run."""

import threading
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.api import BatchInput, EvaluationBatch, LoRAConfig, LossFunctionConfig, OptimParams
from .loss import get_loss_function


class TrainingDataset(Dataset):
    """In-memory dataset for training batches."""

    def __init__(self, batches: list[dict]):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore
        batch = self.batches[idx]
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch.get("target_ids", batch["input_ids"]), dtype=torch.long),
        }


class TrainingRun:
    """Represents a single training run."""

    def __init__(
        self,
        run_id: str,
        base_model: str,
        lora_config: LoRAConfig,
        loss_config: LossFunctionConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.run_id = run_id
        self.base_model = base_model
        self.device = device
        self.lock = threading.Lock()
        self.batches: list[dict] = []

        # Initialize loss function
        if loss_config is None:
            from ..models.api import LossFunctionConfig

            loss_config = LossFunctionConfig()
        self.loss_function = get_loss_function(loss_config)

        # Initialize model and optimizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )

        # Apply LoRA
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model_obj, peft_config)
        self.model = self.model.to(device)

        # Optimizer only for trainable params
        self.optimizer = None  # Initialized on first optim_step
        self.optim_params = None

        # Accumulated gradients
        self.accumulated_step = 0

    def add_batch(self, batch: BatchInput):
        """Add a batch for training (accumulation)."""
        with self.lock:
            self.batches.append(
                {
                    "input_ids": batch.input_ids,
                    "target_ids": batch.target_ids,
                    "weights": batch.weights,  # Store weights if provided
                }
            )

    def forward_backward(self, accumulation_steps: int = 1) -> dict:
        """Perform forward and backward pass on accumulated batches."""
        with self.lock:
            if not self.batches:
                return {"loss": 0.0, "batches": 0}

            # Create dataloader
            dataset = TrainingDataset(self.batches)
            dataloader = DataLoader(
                dataset,
                batch_size=min(len(dataset), 4),
                shuffle=False,
                pin_memory=True,
            )

            total_loss = 0.0
            step_count = 0

            self.model.train()
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass - get logits instead of computing loss
                outputs = self.model(input_ids=input_ids, labels=labels, use_cache=False)
                logits = outputs.logits

                # Compute custom loss
                weights = batch.get("weights")
                if weights is not None:
                    weights = weights.to(self.device)

                loss = self.loss_function(logits, labels, weights=weights) / accumulation_steps
                total_loss += loss.item() * accumulation_steps

                # Backward
                loss.backward()
                step_count += 1

            # Clear batches after processing
            self.batches.clear()

            return {
                "loss": total_loss / step_count if step_count > 0 else 0.0,
                "batches": step_count,
            }

    def optim_step(self, optim_params: OptimParams) -> dict:
        """Perform optimizer step."""
        with self.lock:
            if self.optimizer is None or self.optim_params != optim_params:
                self.optim_params = optim_params
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=optim_params.learning_rate,
                    weight_decay=optim_params.weight_decay,
                    betas=optim_params.betas,
                    eps=optim_params.eps,
                )

            # Clip and step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            return {"status": "optimizer_step_completed"}

    def save_checkpoint(
        self, checkpoint_dir: str, name: str, sampler_optimized: bool = False
    ) -> dict:
        """Save checkpoint (adapter + optimizer)."""
        with self.lock:
            run_dir = Path(checkpoint_dir) / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            base_path = run_dir / name

            # Save adapter
            if sampler_optimized:
                # Save model adapter in safetensors (sampler-optimized)
                adapter_path = base_path.with_suffix(".adapter.safetensors")
                from safetensors.numpy import save_file

                # Get adapter weights
                state_dict = {
                    k: v.clone().detach().contiguous()
                    for k, v in get_peft_model_state_dict(self.model).items()
                }

                # Convert to numpy
                np_state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
                save_file(np_state_dict, adapter_path)
            else:
                # Save model adapter in torch format
                adapter_path = base_path.with_suffix(".adapter.pt")
                adapter_state = self.model.state_dict()
                torch.save(adapter_state, adapter_path)

            # Save optimizer state
            opt_path = base_path.with_suffix(".optimizer.pt")
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), opt_path)

            return {
                "adapter_path": str(adapter_path),
                "optimizer_path": str(opt_path) if self.optimizer is not None else None,
            }

    def evaluate_responses(self, batches: list[EvaluationBatch]) -> dict:
        """Evaluate multiple responses using loss as scoring metric.

        Args:
            batches: List of evaluation batches containing prompt and response

        Returns:
            Dict containing:
                - scores: List[float] - Loss values (lower is better)
                - log_probs: List[float] - Average log probabilities
        """
        self.model.eval()
        scores = []
        log_probs = []

        with torch.no_grad():
            for batch in batches:
                # Convert to tensors
                input_ids = (
                    torch.tensor(batch.input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                )
                target_ids = (
                    torch.tensor(batch.target_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                )

                # Compute loss for this response
                inputs = {"input_ids": input_ids, "labels": target_ids}
                outputs = self.model(**inputs)

                # Loss indicates quality (lower is better)
                loss = outputs.loss.item()

                # Sometimes we want to track what log prob looks like
                # Extract logits and compute log probabilities
                logits = outputs.logits
                log_probs_for_tokens = torch.log_softmax(logits, dim=-1)
                # Get log probs for actual tokens (shift by 1 for next token prediction)
                target_tokens = target_ids[:, 1:]  # Remove first token (no prediction before it)
                token_log_probs = torch.gather(
                    log_probs_for_tokens[:, :-1, :], dim=2, index=target_tokens.unsqueeze(-1)
                )
                avg_log_prob = token_log_probs.mean().item()

                scores.append(loss)
                log_probs.append(avg_log_prob)

        # Invert scores so higher = better (but keep as loss for client)
        return {
            "scores": scores,  # Lower is better
            "log_probs": log_probs,  # Higher is better
        }
