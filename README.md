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

## Reference Model Selection

The choice of reference model is crucial for SDPO effectiveness. A high-quality reference model can significantly improve the effectiveness of alignment by providing better guidance for policy optimization through enhanced alignment scores, leading to more accurate identification of critical tokens.

### Option 1: Using a Larger Model as Reference

According to scaling laws, larger models typically exhibit superior performance. In the alignment scoring function, the improved performance of the reference model directly influences the calculation of the alignment scores, resulting in more accurate token selection and consequently enhancing the overall optimization process.

```python
config = DPOConfig(
    loss_type="sdpo",
    sdpo_threshold=0.6,
    # Use a larger model as reference for better token importance scoring
    ref_model_name_or_path="./models/model_33b-dpo-baseline",  # e.g., large model as ref for medium policy
)
```

### Option 2: Using a DPO-Aligned Model of Same Size

Alternatively, you can use a model of the same size that has been previously trained using DPO alignment on the same dataset as the reference model. Such a model, having already undergone DPO alignment training, is likely to exhibit good alignment performance. Using this model as the reference can further refine the alignment process, improving the accuracy of the selected tokens and thus boosting the overall optimization effect.

```python
config = DPOConfig(
    loss_type="sdpo", 
    sdpo_threshold=0.6,
    # Use a DPO-trained model of same size as reference
    ref_model_name_or_path="/path/to/dpo_aligned_model",  # Same size, DPO pre-trained
)
```

### Reference Model Guidelines

- **For computational efficiency**: Use the same model size but DPO pre-trained
- **For maximum performance**: Use a larger model (e.g., 7B ref for 3B policy)
- **Quality over size**: A well-aligned smaller model can outperform a larger unaligned model
- **Dataset consistency**: Reference model should be trained on similar data distribution

## Implementation

### Configuration

SDPO is implemented as a new loss type in the DPO trainer. Key configuration options:

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="sdpo",           # Use SDPO loss
    sdpo_threshold=0.6,         # Keep top 60% of important tokens
    ref_model_name_or_path="./models/model_33b-dpo-baseline",  # Optional: specify reference model
    # ... other DPO config options
)
```

### Usage Example

To train a model using SDPO, follow the steps below:

1. **Prepare the Dataset**: Ensure your dataset is formatted correctly for preference optimization. Each sample should include chosen and rejected responses.

2. **Configure SDPO**: Set up the configuration with the desired parameters.

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    output_dir="./sdpo_output",
    loss_type="sdpo",
    sdpo_threshold=0.6,  # Keep top 40% of important tokens
    ref_model_name_or_path="./models/model_33b-dpo-baseline",  # Reference model path
    per_device_train_batch_size=4,
    learning_rate=1e-6,
    num_train_epochs=3,
)

# Initialize the trainer
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

3. **Monitor Training**: Use the provided metrics to evaluate the training process, such as rewards for chosen and rejected responses.

4. **Evaluate the Model**: After training, test the model on a validation dataset to ensure alignment and preference optimization.

For a detailed walkthrough, refer to the `train_S-DPO.ipynb` notebook in the repository.



# Configure SDPO with reference model
config = DPOConfig(
    output_dir="./sdpo_output",
    loss_type="sdpo",
    sdpo_threshold=0.6,
    # Option 1: Use a larger model as reference
    ref_model_name_or_path="./models/model_33b-dpo-baseline",  # or path to larger model
    # Option 2: Use a DPO-aligned model of same size
    # ref_model_name_or_path="/path/to/dpo_aligned_model",
    # Option 3: Use None for standard DPO behavior (policy model as reference)
    # ref_model_name_or_path=None,
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

### Key Parameters

- `loss_type`: Set to `"sdpo"` to enable Selective DPO
- `sdpo_threshold`: Quantile threshold for importance masking (default: 0.6)
  - 0.6 means keep top 40% most important tokens
  - Higher values = more selective (fewer tokens)
  - Lower values = less selective (more tokens)
- `ref_model_name_or_path`: Path or identifier for the reference model used in SDPO
  - **Type**: `str` or `None`
  - **Default**: `None` (uses the same model as policy model)
  - **Purpose**: Specifies the reference model for computing importance scores
  - **Options**:
    - Path to local model directory: `"/path/to/reference/model"`
    - HuggingFace model identifier: `"./models/model_33b-dpo-baseline"`
    - `None`: Use the policy model as reference (standard DPO behavior)
  - **Recommendations**: See [Reference Model Selection](#reference-model-selection) below

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

### Choosing Reference Model (`ref_model_name_or_path`)

#### Performance vs Computational Trade-offs

| Reference Model Choice | Performance | Memory Usage | Training Speed | Use Case |
|----------------------|-------------|--------------|----------------|----------|
| `None` (policy model) | Baseline | Low | Fast | Quick experiments, resource-constrained |
| Same-size DPO-aligned | Good | Medium | Medium | Balanced performance/efficiency |
| Larger model (e.g., 7B→13B) | Best | High | Slow | Maximum performance, sufficient resources |

#### Selection Guidelines

1. **For Research/Maximum Performance**:
   ```python
   ref_model_name_or_path="./models/model_33b-dpo-baseline"  # Use larger model
   ```

2. **For Production/Efficiency**:
   ```python
   ref_model_name_or_path="/path/to/dpo_aligned_same_size"  # Pre-trained DPO model
   ```

3. **For Quick Testing**:
   ```python
   ref_model_name_or_path=None  # Use policy model as reference
   ```

#### Model Compatibility Requirements

- Reference model must use the same tokenizer as the policy model
- Architecture should be compatible (same model family preferred)
- Models should be trained on similar data domains for best results

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
