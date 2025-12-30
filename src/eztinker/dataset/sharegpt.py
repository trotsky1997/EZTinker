"""ShareGPT dataset loader with support for multiple dialects.

Supports:
- Dialect A: from/value (original ShareGPT)
- Dialect B: role/content (OpenAI style)
- Mixed formats and automatic dialect detection
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Literal
import warnings

try:
    from pydantic import ValidationError
except ImportError:
    ValidationError = Exception

# Note: BaseDataset import removed (not needed for this implementation)
from ..models.api import ShareGPTConversation, ShareGPTMessage


class ShareGPTDataset:
    """Dataset loader for ShareGPT formatted conversation data.

    Features:
    - Automatic dialect detection (from/value vs role/content)
    - Support for JSON and JSONL files
    - Automatic validation and normalization
    - Tokenization with Qwen2-style templates
    - Conversation filtering and truncation
    """

    VALID_FROM_VALUES = {"human", "gpt"}
    VALID_ROLE_VALUES = {"user", "assistant"}
    ROLE_MAPPING = {
        "human": "user",
        "gpt": "assistant",
        "user": "user",
        "assistant": "assistant",
    }

    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        max_length: int = 2048,
        min_length: int = 10,
        system_prompt: Optional[str] = None,
        strict: bool = True,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ):
        """Initialize ShareGPT dataset.

        Args:
            file_path: Path to ShareGPT JSON/JSONL file
            tokenizer: HuggingFace tokenizer (e.g., Qwen2Tokenizer)
            max_length: Maximum token length (for truncation)
            min_length: Minimum conversation length (filter)
            system_prompt: Optional system prompt template
            strict: If True, raise errors on validation failures
            max_samples: Maximum number of samples to load (None for all)
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.strict = strict
        self.max_samples = max_samples
        self.shuffle = shuffle

        if shuffle:
            random.seed(seed)

        # Conversation storage (normalized format)
        self.conversations: List[Dict] = []

        # Statistics
        self.stats = {
            "total_loaded": 0,
            "valid_conversations": 0,
            "invalid_conversations": 0,
            "total_turns": 0,
            "total_tokens": 0,
            "dialect_counts": {"from_value": 0, "role_content": 0, "mixed": 0}
        }

        self._load_and_prepare()

    def _load_file(self) -> List[dict]:
        """Load JSON or JSONL file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            if self.file_path.suffix == '.jsonl':
                return [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                return data if isinstance(data, list) else [data]

    def _normalize_turn(self, turn: dict) -> Optional[Tuple[str, str]]:
        """Normalize a conversation turn to (role, content).

        Supports both dialects:
        - from/value: {"from": "human/gpt", "value": "..."}
        - role/content: {"role": "user/assistant", "content": "..."}

        Returns:
            (role, content) tuple or None if invalid
        """
        # Try dialect A: from/value
        if "from" in turn:
            role_raw = turn.get("from")
            content = turn.get("value", turn.get("content"))

            if role_raw in self.VALID_FROM_VALUES:
                return self.ROLE_MAPPING[role_raw], content

        # Try dialect B: role/content
        if "role" in turn:
            role_raw = turn.get("role")
            content = turn.get("content", turn.get("value"))

            if role_raw in self.VALID_ROLE_VALUES:
                return role_raw, content

        # Invalid turn
        return None

    def _validate_conversation(self, turns: List[Tuple[str, str]]) -> bool:
        """Validate conversation structure.

        Rules:
        - Must start with user
        - Must end with assistant
        - Must alternate roles
        - Must have non-empty content
        """
        if not turns:
            return False

        # Check start/end
        if turns[0][0] != "user" or turns[-1][0] != "assistant":
            return False

        # Check alternation
        for i in range(len(turns) - 1):
            if turns[i][0] == turns[i+1][0]:
                return False

        # Check content
        for role, content in turns:
            if not content or not isinstance(content, str):
                return False

        return True

    def _detect_dialect(self, raw_conversation: dict) -> Literal["from_value", "role_content", "mixed"]:
        """Detect which dialect this conversation uses."""
        messages = raw_conversation.get("messages") or raw_conversation.get("conversations", [])

        if not messages:
            return "mixed"

        has_from = any("from" in msg for msg in messages)
        has_role = any("role" in msg for msg in messages)

        if has_from and not has_role:
            return "from_value"
        elif has_role and not has_from:
            return "role_content"
        else:
            return "mixed"

    def _load_and_prepare(self):
        """Load data and prepare conversations."""
        print(f"Loading ShareGPT data from {self.file_path}...")

        # Load raw data
        raw_data = self._load_file()
        print(f"  Found {len(raw_data)} entries in file")

        if self.max_samples:
            raw_data = raw_data[:self.max_samples]
            print(f"  Limited to {self.max_samples} samples")

        # Process each conversation
        for idx, raw_conv in enumerate(raw_data):
            try:
                # Detect dialect
                dialect = self._detect_dialect(raw_conv)
                self.stats["dialect_counts"][dialect] += 1

                # Get messages (try both field names)
                messages = raw_conv.get("messages") or raw_conv.get("conversations", [])

                if not messages:
                    if self.strict:
                        raise ValueError(f"Conversation {idx} has no messages")
                    self.stats["invalid_conversations"] += 1
                    continue

                # Normalize turns
                normalized_turns = []
                for turn in messages:
                    normalized = self._normalize_turn(turn)
                    if normalized:
                        normalized_turns.append(normalized)
                    elif self.strict:
                        raise ValueError(f"Invalid turn in conversation {idx}: {turn}")

                # Validate conversation
                if not self._validate_conversation(normalized_turns):
                    self.stats["invalid_conversations"] += 1
                    if self.strict:
                        raise ValueError(f"Invalid conversation structure at {idx}")
                    continue

                # Add to conversations
                self.conversations.append({
                    "id": raw_conv.get("id", f"conv_{idx}"),
                    "turns": normalized_turns,
                    "system": raw_conv.get("system", self.system_prompt),
                    "dataset": raw_conv.get("dataset", "unknown")
                })

                self.stats["valid_conversations"] += 1
                self.stats["total_turns"] += len(normalized_turns)

            except Exception as e:
                self.stats["invalid_conversations"] += 1
                if self.strict:
                    raise
                else:
                    warnings.warn(f"Skipping conversation {idx}: {e}")

        self.stats["total_loaded"] = len(raw_data)

        if self.shuffle:
            random.shuffle(self.conversations)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print loading summary."""
        print(f"\n=== ShareGPT Dataset Summary ===")
        print(f"Total entries loaded: {self.stats['total_loaded']}")
        print(f"Valid conversations: {self.stats['valid_conversations']}")
        print(f"Invalid conversations: {self.stats['invalid_conversations']}")
        print(f"Total turns: {self.stats['total_turns']}")

        if self.stats['valid_conversations'] > 0:
            print(f"Avg turns/conversation: {self.stats['total_turns'] / self.stats['valid_conversations']:.1f}")

        print(f"\nDialects detected:")
        for dialect, count in self.stats["dialect_counts"].items():
            if count > 0:
                print(f"  {dialect}: {count}")
        print()

    def format_conversation_qwen2(self, conversation: dict) -> str:
        """Format conversation using Qwen2 chat template.

        Template:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
        """
        parts = []

        # Add system prompt
        system = conversation.get("system", self.system_prompt)
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")

        # Add conversation turns
        for role, content in conversation["turns"]:
            if role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

        return "".join(parts)

    def get_conversation_turns(self, idx: int) -> Tuple[str, str, str]:
        """Get conversation by index.

        Returns:
            (id, formatted_text, num_turns)
        """
        conv = self.conversations[idx]
        formatted = self.format_conversation_qwen2(conv)
        return conv["id"], formatted, len(conv["turns"])

    def get_training_turn(self, idx: int, turn_idx: int) -> Optional[Dict]:
        """Get a specific turn from a conversation for training.

        Returns dict with:
        - prompt: Text up to this turn
        - response: The assistant response
        - full_text: Complete formatted conversation
        """
        conv = self.conversations[idx]

        if turn_idx >= len(conv["turns"]) or turn_idx % 2 == 0:
            return None  # Invalid turn (not an assistant turn)

        # Build prompt up to this turn
        prompt_turns = conv["turns"][:turn_idx]
        response_role, response_content = conv["turns"][turn_idx]

        # Format prompt
        prompt_conv = {
            "id": conv["id"],
            "system": conv["system"],
            "turns": prompt_turns + [(response_role, response_content)]
        }

        # Format full text
        full_text = self.format_conversation_qwen2(conv)

        # Get prompt text (everything before assistant response)
        prompt_conv_for_prompt = {
            "id": conv["id"],
            "system": conv["system"],
            "turns": prompt_turns
        }
        prompt_text = self.format_conversation_qwen2(prompt_conv_for_prompt)

        return {
            "prompt": prompt_text,
            "response": response_content,
            "full_text": full_text,
            "conversation_id": conv["id"],
            "turn_index": turn_idx
        }

    def tokenize_for_training(self, idx: int, turn_idx: int = -1) -> Dict:
        """Tokenize a conversation for training.

        Args:
            idx: Conversation index
            turn_idx: Which assistant turn to train on (-1 for full conversation)

        Returns:
            Dict with tokenized inputs
        """
        conv = self.conversations[idx]
        formatted = self.format_conversation_qwen2(conv)

        # Tokenize
        tokens = self.tokenizer(
            formatted,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0).tolist(),
            "attention_mask": tokens["attention_mask"].squeeze(0).tolist(),
            "conversation_id": conv["id"],
            "num_turns": len(conv["turns"])
        }

    def export_to_jsonl(self, output_path: Union[str, Path]):
        """Export normalized conversations to JSONL format."""
        output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in self.conversations:
                # Convert to ShareGPT format (role/content dialect)
                messages = [
                    {"role": role, "content": content}
                    for role, content in conv["turns"]
                ]

                export_item = {
                    "id": conv["id"],
                    "messages": messages,
                    "system": conv["system"],
                    "dataset": conv["dataset"]
                }

                f.write(json.dumps(export_item, ensure_ascii=False) + '\n')

        print(f"✅ Exported {len(self.conversations)} conversations to {output_path}")

    def __len__(self) -> int:
        """Number of conversations."""
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict:
        """Get conversation by index."""
        return self.conversations[idx]