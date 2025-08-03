"""Command-line interface for GR Rebuild."""

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
@click.option("--model", default="gpt2", help="Model to fine-tune")
@click.option("--dataset", default="squad", help="Dataset to use")
def train(model: str, dataset: str):
    """Train a model with LoRA fine-tuning."""
    console.print(Panel(f"Training {model} on {dataset} dataset", title="🚀 Training"))
    # TODO: Implement training logic
    console.print("Training functionality coming soon!")


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
    console.print(Panel("GR Rebuild is ready for development!", title="✅ Status"))
    console.print("• Transformers and LoRA support configured")
    console.print("• Dstack compute integration ready")
    console.print("• CLI interface available")


if __name__ == "__main__":
    main()
