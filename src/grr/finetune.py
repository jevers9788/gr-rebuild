"""Fine-tuning module using Unsloth for LoRA training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments

# Try to import Unsloth (optional)
try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""

    model_name: str = "unsloth/llama-3.2-1b-bnb-4bit"
    max_seq_length: int = 2048
    dtype: str = "bfloat16"
    load_in_4bit: bool = True

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    # Training configuration
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    output_dir: str = "./output"

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


class UnslothFineTuner:
    """Fine-tuner using Unsloth for efficient LoRA training."""

    def __init__(self, config: TrainingConfig):
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is not available. Install with 'uv sync --extra gpu' "
                "in a GPU environment (like RunPod via dstack)."
            )

        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Load the model and tokenizer using Unsloth."""
        print(f"Loading model: {self.config.model_name}")

        # Load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )

        print("Model loaded successfully!")

    def setup_lora(self):
        """Configure LoRA for the model."""
        print("Setting up LoRA configuration...")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        """Prepare dataset for training.
        Args:
            data: List of dictionaries with 'instruction', 'input', and 'output' keys
        Returns:
            Tokenized dataset ready for training
        """
        print(f"Preparing dataset with {len(data)} examples...")

        def formatting_prompts_func(examples):
            """Format prompts for conversational training."""
            convos = examples["conversations"]
            texts = []

            for convo in convos:
                text = self.tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)

            return {"text": texts}

        # Convert to dataset format
        dataset_data = []
        for item in data:
            # Convert to conversation format
            conversation = []
            if "instruction" in item:
                conversation.append({"role": "user", "content": item["instruction"]})
            if "input" in item and item["input"]:
                conversation.append({"role": "user", "content": item["input"]})
            if "output" in item:
                conversation.append({"role": "assistant", "content": item["output"]})

            dataset_data.append({"conversations": conversation})

        dataset = Dataset.from_list(dataset_data)

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            formatting_prompts_func, batched=True, remove_columns=dataset.column_names
        ).map(
            tokenize_function,
            batched=True,
            remove_columns=tokenized_dataset.column_names,  # noqa: F821
        )

        return tokenized_dataset

    def setup_trainer(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ):
        """Set up the trainer with training arguments."""
        print("Setting up trainer...")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=None,  # Unsloth handles this
        )

    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]] = None,
    ):
        """Train the model with the provided data."""
        print("Starting training process...")

        # Load model and setup LoRA
        self.load_model()
        self.setup_lora()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        eval_dataset = None
        if eval_data:
            eval_dataset = self.prepare_dataset(eval_data)

        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)

        # Start training
        print("Training started!")
        self.trainer.train()

        # Save the model
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)

        self.trainer.save_model(str(output_path / "final_model"))
        self.tokenizer.save_pretrained(str(output_path / "final_model"))

        print(f"Training completed! Model saved to {output_path / 'final_model'}")

    def save_config(self, output_path: str):
        """Save the training configuration."""
        config_path = Path(output_path) / "training_config.json"
        config_dict = {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "num_train_epochs": self.config.num_train_epochs,
            "learning_rate": self.config.learning_rate,
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {config_path}")


def create_sample_data() -> List[Dict[str, str]]:
    """Create sample training data for testing."""
    return [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris.",
        },
        {
            "instruction": "Explain photosynthesis",
            "input": "",
            "output": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        },
        {
            "instruction": "Write a short poem about coding",
            "input": "",
            "output": "Lines of logic flow like streams,\nFunctions dance in digital dreams,\nBugs may hide in shadows deep,\nBut clean code helps us sleep.",
        },
    ]


def main():
    """Main function for testing the fine-tuner."""
    if not UNSLOTH_AVAILABLE:
        print(
            "Unsloth not available. Install with 'uv sync --extra gpu' in a GPU environment."
        )
        return

    # Create configuration
    config = TrainingConfig(
        model_name="unsloth/llama-3.2-1b-bnb-4bit",
        max_seq_length=2048,
        num_train_epochs=1,  # Short for testing
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        output_dir="./output/test_training",
    )

    # Create fine-tuner
    fine_tuner = UnslothFineTuner(config)

    # Create sample data
    sample_data = create_sample_data()

    # Train the model
    fine_tuner.train(sample_data)

    # Save configuration
    fine_tuner.save_config(config.output_dir)


if __name__ == "__main__":
    main()
