# EZTinker Examples

This directory contains example scripts demonstrating how to use EZTinker for various training scenarios.

## Available Examples

### GSM8K SFT Training

**File**: `gsm8k_sft.py`

Demonstrates supervised fine-tuning (SFT) of Qwen/Qwen2-0.5B-Instruct model on GSM8K math problems using rank-1 LoRA.

#### Features
- 🧮 **Real Math Problems**: Uses 1000 authentic GSM8K training examples
- 🎯 **Rank-1 LoRA**: Efficient parameter tuning (only 550K trainable parameters)
- 📊 **Proper Tokenization**: Qwen2 tokenizer with conversation formatting
- 📈 **Loss Monitoring**: Tracks training convergence
- 🔄 **Zero Dummy Data**: All real tokenized examples

#### Configuration
```python
Model: Qwen/Qwen2-0.5B-Instruct
LoRA: Rank 1, Alpha 2, Dropout 0.05
Dataset: GSM8K (1000 math problems)
Training steps: 100
Learning rate: 5e-5
Parameter efficiency: 0.11% trainable
```

#### Quick Start

1. **Start EZTinker server**:
```bash
uvicorn src.eztinker.api.server:app --host 0.0.0.0 --port 8000
```

2. **Run training** (from root directory):
```bash
python examples/gsm8k_sft.py
```

#### Expected Results

The training should converge smoothly with excellent loss reduction:

| Metric | Initial | Final | Change |
|--------|---------|-------|--------|
| Loss | ~8.2 | ~0.2-0.3 | **~97% reduction** |

**Loss Progression**:
- Steps 1-10: 8.2 → 6.1 (warmup)
- Steps 11-30: 6.1 → 1.7 (rapid descent)
- Steps 31-50: 1.7 → 0.4 (continued convergence)
- Steps 51-100: 0.4 → 0.2 (stable low loss)

#### Output Files

After training, you'll find:
- `loss_data_{run_id}.json` - Complete training statistics
- Checkpoint files in `checkpoints/` directory

#### Code Structure

The example demonstrates:
1. **Proper dataset loading** with GSM8KDataset
2. **Conversation formatting** for Qwen2 chat models
3. **Tokenization** with proper padding and truncation
4. **Training run creation** with rank-1 LoRA config
5. **Training loop** with forward/backward passes
6. **Optimizer steps** with learning rate and weight decay
7. **Checkpoint saving** for model persistence
8. **Loss analysis** with trend monitoring

#### Advanced Usage

Modify the script for different scenarios:

- **More samples**: Change `NUM_SAMPLES` to 1000-5000
- **Higher LoRA rank**: Change `r` to 8 or 16 for more expressiveness
- **Different learning rate**: Adjust `LEARNING_RATE` to 1e-5 or 1e-4
- **More steps**: Increase `NUM_TRAINING_STEPS` for longer training

#### Success Criteria

✅ **Training passes if**:
- Loss converges from ~8 to <1.0
- No divergence (max loss stays below 10.0)
- Consistent downward trend
- At least 90% steps completed

#### Dependencies

```bash
# Core requirements
pip install eztinker transformers datasets torch

# For visualization (optional)
pip install matplotlib

# Model download (first time)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct')"
```

#### Troubleshooting

**Server not reachable**:
- Make sure EZTinker server is running on port 8000
- Check if base_url matches your setup

**Out of memory**:
- Reduce batch size in server configuration
- Use smaller LoRA rank (r=1 is already minimal)

**Tokenizer issues**:
- Install transformers: `pip install transformers`
- Download model first: May take a few minutes on first run

**Loss not converging**:
- Check if training data is properly formatted
- Ensure tokenizer is loaded correctly
- Verify server is processing requests successfully

---

**Next Steps**: Try the training yourself and experiment with different configurations!