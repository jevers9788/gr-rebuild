"""Command-line interface for GR Rebuild."""

import json

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option()
def main():
    """GR Rebuild - Fine-tune models with LoRAs using transformers and Dstack."""
    pass


@main.command()
@click.option(
    "--model", default="unsloth/llama-3.2-1b-bnb-4bit", help="Model to fine-tune"
)
@click.option("--dataset", default="sample", help="Dataset to use (sample, custom)")
@click.option("--data-path", help="Path to custom dataset JSON file")
@click.option(
    "--output-dir", default="./output", help="Output directory for trained model"
)
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--batch-size", default=2, help="Training batch size")
@click.option("--learning-rate", default=2e-4, help="Learning rate")
@click.option("--max-length", default=2048, help="Maximum sequence length")
def train(
    model: str,
    dataset: str,
    data_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
):
    """Train a model with LoRA fine-tuning using Unsloth."""
    console.print(Panel(f"Training {model} with Unsloth", title="🚀 Training"))

    try:
        from .finetune import TrainingConfig, UnslothFineTuner, create_sample_data

        # Create training configuration
        config = TrainingConfig(
            model_name=model,
            max_seq_length=max_length,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
        )

        # Load training data
        if dataset == "sample":
            console.print("Using sample dataset for testing...")
            train_data = create_sample_data()
            eval_data = None
        elif dataset == "custom":
            if not data_path:
                console.print(
                    "[red]Error: --data-path is required for custom dataset[/red]"
                )
                return

            console.print(f"Loading custom dataset from {data_path}...")
            try:
                with open(data_path) as f:
                    data = json.load(f)
                train_data = data.get("train", data)
                eval_data = data.get("eval", None)
            except Exception as e:
                console.print(f"[red]Error loading dataset: {e}[/red]")
                return
        else:
            console.print(f"[red]Unknown dataset: {dataset}[/red]")
            return

        # Create fine-tuner and train
        fine_tuner = UnslothFineTuner(config)
        fine_tuner.train(train_data, eval_data)
        fine_tuner.save_config(output_dir)

        console.print(f"[green]Training completed! Model saved to {output_dir}[/green]")
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(
            "[yellow]Make sure you're running this in a GPU environment (like RunPod via dstack)[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")


@main.command()
@click.option("--model-path", required=True, help="Path to trained model")
def evaluate(model_path: str):
    """Evaluate a trained model."""
    console.print(Panel(f"Evaluating model at {model_path}", title="📊 Evaluation"))
    # TODO: Implement evaluation logic
    console.print("Evaluation functionality coming soon!")


@main.command()
def status():
    """Check project status."""
    console.print(Panel("GR Rebuild Status", title="✅ Status"))
    console.print("• Transformers and LoRA support configured")
    console.print("• Unsloth fine-tuning module available")
    console.print("• Dstack compute integration ready")
    console.print("• CLI interface available")
    console.print(
        "\n[yellow]Note: Training requires GPU environment (RunPod via dstack)[/yellow]"
    )


@main.command()
@click.option(
    "--output", default="sample_data.json", help="Output file for sample data"
)
def create_sample(output: str):
    """Create a sample dataset file for testing."""
    console.print(Panel("Creating sample dataset", title="📝 Sample Data"))

    try:
        from .finetune import create_sample_data

        sample_data = create_sample_data()

        with open(output, "w") as f:
            json.dump(sample_data, f, indent=2)

        console.print(f"[green]Sample dataset created: {output}[/green]")
        console.print(
            "You can use this file with: grr train --dataset custom --data-path sample_data.json"
        )
    except ImportError:
        console.print("[red]Error: Could not import fine-tuning module[/red]")


if __name__ == "__main__":
    main()
