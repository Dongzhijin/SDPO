# Not All Preferences are What You Need for Post-Training: Selective Alignment Strategy for Preference Optimization

## Overview

This project implements a selective alignment strategy for preference optimization that addresses the critical challenge that **not all tokens contribute equally to model performance** in post-training alignment of large language models (LLMs). Our approach leverages **token-level log-probability differences** between the current policy and a reference model to identify and optimize **high-impact tokens** within preference pairs, leading to more efficient and effective preference optimization.

For more details, refer to the [academic paper](https://arxiv.org/abs/2025.12345).

## Key Features

- **Alignment Score Computation**: Uses the difference between policy and reference model log probabilities to identify high-impact tokens
- **Quantile-based Selection**: Filters tokens based on alignment scores using a configurable threshold
- **Seamless Integration**: Works as a drop-in replacement for standard DPO loss
- **Computational Efficiency**: Reduces computation by focusing only on tokens that matter most for alignment

## How It Works

1. **Alignment Score Calculation**: For each token, compute alignment scores as:
   ```
   alignment_chosen = ref_chosen_logp - policy_chosen_logp
   alignment_rejected = policy_rejected_logp - ref_rejected_logp
   ```

2. **Token Selection**: Keep only tokens with alignment scores above the specified quantile threshold

3. **Loss Computation**: Apply standard DPO loss only on the selected high-impact tokens

## Workflow Overview

The following diagram illustrates the complete workflow of our selective alignment strategy:

```text
┌─────────────────────────────────────────────────────────────────┐
│ Input: Preference Dataset, Policy Model, Reference Model        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────▼──────────────────┐
          │ Step 1: Compute Alignment Scores │
          └───────────────┬──────────────────┘
                          │
     ┌────────────────────▼─────────────────────┐
     │ Step 2: Select Top k% High-Impact Tokens │
     └────────────────────┬─────────────────────┘
                          │
     ┌────────────────────▼─────────────────────────────┐
     │ Step 3: Optimize Policy Using Selective-DPO Loss │
     └────────────────────┬─────────────────────────────┘
                          │
          ┌───────────────▼────────────────┐
          │ Output: Optimized Policy Model │
          └────────────────────────────────┘
```

**Figure 1.** Workflow of the Selective Alignment Strategy for Preference Optimization. The method focuses on high-impact tokens within preference pairs, leveraging token-level log-probability differences to score and select the most informative tokens for optimization. The process consists of three main steps: computing alignment scores, selecting top k% high-impact tokens, and optimizing the policy using selective-DPO loss.

## Reference Model Selection

The choice of reference model is crucial for the effectiveness of our selective alignment strategy. A high-quality reference model provides better guidance for policy optimization through more accurate alignment score computation, leading to improved identification of high-impact tokens and enhanced overall optimization performance.

### Option 1: Using a Larger Model as Reference

According to scaling laws, larger models typically exhibit superior performance. In the alignment scoring function, the improved performance of the reference model directly influences the calculation of alignment scores, resulting in more accurate high-impact token selection and consequently enhancing the overall optimization process.

```python
config = DPOConfig(
    loss_type="sdpo",
    sdpo_threshold=0.6,
    # Use a larger model as reference for better alignment score computation
    ref_model_name_or_path="./models/model_33b-dpo-baseline",  # e.g., large model as ref for medium policy
)
```

### Option 2: Using a DPO-Aligned Model of Same Size

Alternatively, you can use a model of the same size that has been previously trained using DPO alignment on the same dataset as the reference model. Such a model, having already undergone DPO alignment training, is likely to exhibit good alignment performance. Using this model as the reference can further refine the alignment process through knowledge distillation, improving the accuracy of high-impact token selection and thus boosting the overall optimization effect.

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
- **For maximum performance**: Use a larger model (e.g., 33B ref for 3B policy)
- **Quality over size**: A well-aligned smaller model can outperform a larger unaligned model
- **Dataset consistency**: Reference model should be trained on similar data distribution

## Implementation

### Configuration

Our selective alignment strategy is implemented as a new loss type in the DPO trainer. Key configuration options:

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="sdpo",           # Use selective alignment strategy
    sdpo_threshold=0.6,         # Keep top 40% highest-impact tokens
    ref_model_name_or_path="./models/model_33b-dpo-baseline",  # Optional: specify reference model
    # ... other DPO config options
)
```

### Usage Example

To train a model using our selective alignment strategy, follow the steps below:

1. **Prepare the Dataset**: Ensure your dataset is formatted correctly for preference optimization. Each sample should include chosen and rejected responses.

2. **Configure the Strategy**: Set up the configuration with the desired parameters.

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    output_dir="./selective_alignment_output",
    loss_type="sdpo",
    sdpo_threshold=0.6,  # Keep top 40% highest-impact tokens
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

1. **Monitor Training**: Use the provided metrics to evaluate the training process, such as rewards for chosen and rejected responses.

2. **Evaluate the Model**: After training, test the model on a validation dataset to ensure alignment and preference optimization.

For a detailed walkthrough, refer to the `train_S-DPO.ipynb` notebook in the repository.

### Key Parameters

- `loss_type`: Set to `"sdpo"` to enable selective alignment strategy
- `sdpo_threshold`: Quantile threshold for high-impact token selection (default: 0.6)
  - 0.6 means keep top 40% highest-impact tokens
  - Higher values = more selective (fewer tokens)
  - Lower values = less selective (more tokens)
- `ref_model_name_or_path`: Path or identifier for the reference model used in alignment scoring
  - **Type**: `str` or `None`
  - **Default**: `None` (uses the same model as policy model)
  - **Purpose**: Specifies the reference model for computing alignment scores
  - **Options**:
    - Path to local model directory: `"/path/to/reference/model"`
    - HuggingFace model identifier: `"./models/model_33b-dpo-baseline"`
    - `None`: Use the policy model as reference (standard DPO behavior)
  - **Recommendations**: See [Reference Model Selection](#reference-model-selection) below

## Technical Details

### Core Function: `mask_unimportant_tokens`

The core of our selective alignment strategy is the token selection function:

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

1. Calculates alignment scores for each token
2. Determines quantile thresholds for each sequence
3. Selects high-impact tokens (sets others to 0)
4. Returns summed log probabilities for high-impact tokens only

### Integration Points

Our selective alignment strategy modifies three key components:

1. **`concatenated_forward`**: Stores per-token log probabilities when `loss_type="sdpo"`
2. **`compute_ref_log_probs`**: Returns per-token logps for reference model in selective alignment mode
3. **`get_batch_loss_metrics`**: Applies high-impact token selection before calling `dpo_loss`

## Advantages

1. **Enhanced Optimization Efficiency**: Focuses computation on high-impact tokens that matter most for alignment
2. **Improved Signal Quality**: Reduces noise from less relevant tokens in the preference optimization process
3. **Flexible Control**: Adjustable threshold allows fine-tuning of token selectivity
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
| Same-size DPO-aligned | Good | Medium | Medium | Balanced performance/efficiency |
| Larger model | Best | High | Slow | Maximum performance, sufficient resources |

#### Model Compatibility Requirements

- Reference model must use the same tokenizer as the policy model
- Architecture should be compatible (same model family preferred)
- Models should be trained on similar data domains for best results

### When to Use This Approach

Our selective alignment strategy is particularly beneficial when:

- Working with long sequences where many tokens are less critical for alignment
- Training data contains noisy or less relevant tokens
- Computational resources are limited
- Standard DPO shows slow convergence or suboptimal performance

## Monitoring Training

Our selective alignment strategy provides the same metrics as standard DPO:

- `rewards/chosen`: Mean rewards for chosen responses
- `rewards/rejected`: Mean rewards for rejected responses  
- `rewards/accuracies`: Accuracy of preference predictions
- `rewards/margins`: Difference between chosen and rejected rewards

## Files Modified

The selective alignment strategy implementation required changes to:

1. **`trl/trainer/dpo_config.py`**:
   - Added `sdpo_threshold` parameter
   - Updated loss_type documentation

2. **`trl/trainer/dpo_trainer.py`**:
   - Implemented `mask_unimportant_tokens` static method for high-impact token selection
   - Modified `concatenated_forward` to store per-token logps
   - Updated `compute_ref_log_probs` for per-token support
   - Enhanced `get_batch_loss_metrics` with selective alignment logic
   - Added selective alignment case to `dpo_loss` method

## Citation

If you use this selective alignment strategy in your research, please cite:

```bibtex
@article{selective_alignment2025,
  title={Not All Preferences are What You Need for Post-Training: Selective Alignment Strategy for Preference Optimization},
  author={Zhijin Dong},
  journal={arXiv preprint},
  year={2025}
}
```

## Contributing

To contribute to this selective alignment strategy development:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

## License

This implementation follows the same license as the TRL library.
