# GR Rebuild

A project for fine-tuning models with LoRAs using Unsloth, transformers, and Dstack.

## Features

- **Unsloth Integration**: Efficient LoRA fine-tuning with Unsloth
- **CLI Interface**: Easy-to-use command-line tools
- **Dstack Integration**: Cloud compute with RunPod backend
- **Slack-style Training**: Optimized for conversational AI training
- **Modular Design**: Clean, extensible codebase

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Usage

### CLI Commands

```bash
# Check project status
grr status

# Train with sample data (for testing)
grr train --model unsloth/llama-3.2-1b-bnb-4bit --dataset sample

# Train with custom dataset
grr train --dataset custom --data-path your_data.json --epochs 5

# Create sample dataset file
grr create-sample --output my_data.json

# Evaluate a model (coming soon)
grr evaluate --model-path /path/to/model
```

### Python API

```python
from grr import UnslothFineTuner, TrainingConfig

# Create configuration
config = TrainingConfig(
    model_name="unsloth/llama-3.2-1b-bnb-4bit",
    max_seq_length=2048,
    num_train_epochs=3,
    output_dir="./output"
)

# Create fine-tuner
fine_tuner = UnslothFineTuner(config)

# Prepare training data
training_data = [
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    },
    # ... more examples
]

# Train the model
fine_tuner.train(training_data)
```

### Dataset Format

Your training data should be in JSON format:

```json
[
    {
        "instruction": "User question or instruction",
        "input": "Optional additional context",
        "output": "Expected model response"
    }
]
```

For evaluation data, you can provide separate train/eval sets:

```json
{
    "train": [...],
    "eval": [...]
}
```

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Build package
uv run hatch build
```

## Project Structure

```
gr-rebuild/
├── src/
│   └── grr/
│       ├── __init__.py
│       ├── cli.py
│       └── finetune.py          # Unsloth fine-tuning module
├── examples/
│   └── finetune_example.py      # Example usage
├── tests/
│   ├── __init__.py
│   └── test_cli.py
├── pyproject.toml
└── README.md
```

## Dstack Integration

To use with Dstack cloud compute:

1. **Configure RunPod backend** in `~/.dstack/server/config.yml`
2. **Start Dstack server**
 ```bash
 uv run dstack server
 ```
3. **Start development environment in a new terminal**:
   ```bash

   uv run dstack apply -f dev.dstack.yml
   ```
4. **Connect via SSH or Cursor**:
   ```bash
   ssh fudgin
   # or use the Cursor link
   ```
5. **Install dependencies in the cloud environment**:
   ```bash
   uv sync
   ```
6. **Run training**:
   ```bash
   grr train --dataset sample
   ```

## Build Datasets

```bash
# Build dataset from Discord export (chat format)
uv run grr build-dataset \
  --source discord \
  --input data.json \
  --target-user duffdogg \
  --format chat \
  --output discord_chat.json

# Build dataset from Slack export (alpaca format)
uv run grr build-dataset \
  --source slack \
  --input slack_messages.jsonl \
  --target-user U06BOT \
  --format alpaca \
  --output slack_alpaca.json
```

## Training Configuration

The `TrainingConfig` class supports various parameters:

- **Model**: `unsloth/llama-3.2-1b-bnb-4bit` (default)
- **LoRA**: r=16, alpha=32, dropout=0.1
- **Training**: 3 epochs, batch size 2, learning rate 2e-4
- **Sequence Length**: 2048 tokens

## License

MIT License 