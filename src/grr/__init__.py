"""GR Rebuild - A project for fine-tuning models with LoRAs using transformers and Dstack."""

__version__ = "0.1.0"
__author__ = "Gustavo Rezende"  # noice, keeping this
__email__ = "gustavo.rezende@gmail.com"

# Import fine-tuning components if unsloth is there
# ruff: noqa: F401 # dont freak out the linter if no
try:
    from .finetune import (
        TrainingConfig,
        UnslothFineTuner,
        create_sample_data,
    )

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "UNSLOTH_AVAILABLE",
]

if UNSLOTH_AVAILABLE:
    __all__.extend(
        [
            "UnslothFineTuner",
            "TrainingConfig",
            "create_sample_data",
        ]
    )
