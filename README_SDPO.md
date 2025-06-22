# SDPO: Selective Direct Preference Optimization

## Overview

SDPO (Selective Direct Preference Optimization) is an extension of the standard DPO algorithm that incorporates **importance-based token masking**. Unlike traditional DPO which considers all tokens equally, SDPO selectively focuses on the most important tokens during loss computation, leading to more efficient and effective preference optimization.

## Key Features

- **Token Importance Scoring**: Uses the difference between policy and reference model log probabilities to identify important tokens
- **Quantile-based Masking**: Filters out unimportant tokens based on a configurable threshold
- **Seamless Integration**: Works as a drop-in replacement for standard DPO loss
- **Memory Efficient**: Reduces computation by focusing only on relevant tokens

## How SDPO Works

1. **Importance Calculation**: For each token, compute importance score as:
   ```
   importance_chosen = ref_chosen_logp - policy_chosen_logp
   importance_rejected = policy_rejected_logp - ref_rejected_logp
   ```

2. **Token Filtering**: Keep only tokens with importance scores above the specified quantile threshold

3. **Loss Computation**: Apply standard DPO loss only on the filtered important tokens

## Implementation

### Configuration

SDPO is implemented as a new loss type in the DPO trainer. Key configuration options:

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="sdpo",           # Use SDPO loss
    sdpo_threshold=0.6,         # Keep top 60% of important tokens
    # ... other DPO config options
)
```

### Key Parameters

- `loss_type`: Set to `"sdpo"` to enable Selective DPO
- `sdpo_threshold`: Quantile threshold for importance masking (default: 0.6)
  - 0.6 means keep top 40% most important tokens
  - Higher values = more selective (fewer tokens)
  - Lower values = less selective (more tokens)

### Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Prepare your preference dataset
train_dataset = Dataset.from_list([
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "The capital of France is London."
    },
    # ... more examples
])

# Configure SDPO
config = DPOConfig(
    output_dir="./sdpo_output",
    loss_type="sdpo",
    sdpo_threshold=0.6,
    per_device_train_batch_size=4,
    learning_rate=1e-6,
    num_train_epochs=3,
)

# Create trainer
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# Train with SDPO
trainer.train()
```

## Technical Details

### Core Function: `mask_unimportant_tokens`

The heart of SDPO is the token masking function:

```python
@staticmethod
def mask_unimportant_tokens(
    chosen_per_token_logps: torch.FloatTensor,
    rejected_per_token_logps: torch.FloatTensor,
    ref_chosen_per_token_logps: torch.FloatTensor,
    ref_rejected_per_token_logps: torch.FloatTensor,
    threshold: float = 0.6,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
```

This function:
1. Calculates importance scores for each token
2. Determines quantile thresholds for each sequence
3. Masks unimportant tokens (sets them to 0)
4. Returns summed log probabilities for important tokens only

### Integration Points

SDPO modifies three key components:

1. **`concatenated_forward`**: Stores per-token log probabilities when `loss_type="sdpo"`
2. **`compute_ref_log_probs`**: Returns per-token logps for reference model in SDPO mode
3. **`get_batch_loss_metrics`**: Applies importance masking before calling `dpo_loss`

## Advantages

1. **Improved Efficiency**: Focuses computation on tokens that matter most
2. **Better Signal-to-Noise**: Reduces impact of less relevant tokens
3. **Flexible Control**: Adjustable threshold allows fine-tuning of selectivity
4. **Backward Compatible**: Can fall back to standard DPO when threshold=0

## Configuration Guidelines

### Choosing `sdpo_threshold`

- **0.5**: Medium selectivity (top 50% tokens)
- **0.6**: Higher selectivity (top 40% tokens) - **recommended default**
- **0.7**: High selectivity (top 30% tokens)
- **0.8**: Very high selectivity (top 20% tokens)

Start with 0.6 and adjust based on your specific task and dataset characteristics.

### When to Use SDPO

SDPO is particularly beneficial when:
- Working with long sequences where many tokens are less important
- Training data contains noisy or less relevant tokens
- Computational resources are limited
- Standard DPO shows slow convergence

## Monitoring Training

SDPO provides the same metrics as standard DPO:
- `rewards/chosen`: Mean rewards for chosen responses
- `rewards/rejected`: Mean rewards for rejected responses  
- `rewards/accuracies`: Accuracy of preference predictions
- `rewards/margins`: Difference between chosen and rejected rewards

## Files Modified

The SDPO implementation required changes to:

1. **`trl/trainer/dpo_config.py`**:
   - Added `sdpo_threshold` parameter
   - Updated loss_type documentation

2. **`trl/trainer/dpo_trainer.py`**:
   - Implemented `mask_unimportant_tokens` static method
   - Modified `concatenated_forward` to store per-token logps
   - Updated `compute_ref_log_probs` for per-token support
   - Enhanced `get_batch_loss_metrics` with SDPO logic
   - Added SDPO case to `dpo_loss` method

## Testing

Run the test script to verify SDPO functionality:

```bash
python test_sdpo.py
```

This will test:
- Basic SDPO configuration
- Token masking function correctness
- Training step execution

## Citation

If you use SDPO in your research, please cite:

```bibtex
@article{sdpo2024,
  title={Selective Direct Preference Optimization: Token-Level Importance for Efficient Preference Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Contributing

To contribute to SDPO development:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

## License

This implementation follows the same license as the TRL library.
