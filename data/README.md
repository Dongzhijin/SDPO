# Data Directory

This directory is intended for storing dataset files used with SDPO training.

## Usage

Place your preference dataset files here before running SDPO training:

- `train_dataset.json` - Training preference data
- `eval_dataset.json` - Evaluation preference data
- `test_dataset.json` - Test preference data

## Dataset Format

Datasets should follow the standard DPO format with `prompt`, `chosen`, and `rejected` fields:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "The capital of France is London."
}
```

## Examples

Refer to the examples in `trl/examples/datasets/` for dataset preparation scripts.
