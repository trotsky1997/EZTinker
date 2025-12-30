"""Sampling engine for inference and evaluation."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class Sampler:
    """Handles sampling from models (base or adapter)."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.models = {}
        self.tokenizers = {}

    def load_model(
        self,
        model_id: str,
        model_key: str | None = None,
        adapter_path: str | None = None,
    ):
        """Load a model (possibly with adapter)."""
        model_key = model_key or model_id

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

        # Load adapter if provided
        if adapter_path:
            if adapter_path.endswith(".safetensors"):
                # Load safetensors format
                from safetensors.numpy import load_file

                torch_weights = load_file(adapter_path)
                # Convert to tensor
                adapter_state_dict = {k: torch.from_numpy(v) for k, v in torch_weights.items()}

                # Create dummy LoRA config and load into model
                peft_config = LoraConfig(
                    r=8,  # Just a guess
                    lora_alpha=16,
                    lora_dropout=0.0,
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_config)
                model.load_state_dict(adapter_state_dict, strict=False)
            else:
                # Load adapter from checkpoint format
                model = PeftModel.from_pretrained(model, adapter_path)
                model = model.merge_and_unload()

        model = model.to(self.device)
        model.eval()

        # Cache models
        self.models[model_key] = model
        self.tokenizers[model_key] = tokenizer

        return model_key

    def sample(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate text from prompt."""
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded")

        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return result
