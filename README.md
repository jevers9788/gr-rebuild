# GR Rebuild

TODO 

## Features

TODO

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Usage

```bash
# Check project status
gr-rebuild status

# Train a model (coming soon)
gr-rebuild train --model gpt2 --dataset squad

# Evaluate a model (coming soon)
gr-rebuild evaluate --model-path /path/to/model
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
│       └── cli.py
├── tests/
│   ├── __init__.py
│   └── test_cli.py
├── pyproject.toml
└── README.md
```

## License

MIT License 