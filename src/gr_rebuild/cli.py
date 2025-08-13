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
def generate_sample_slack():
    """Generate a sample Slack conversation dataset."""
    console.print(
        Panel("Generating sample Slack conversation dataset", title="🚀 Generating")
    )
    from scripts.generate_sample_slack import generate_sample_slack

    generate_sample_slack()
    console.print("Sample Slack conversation dataset generated successfully!")


@main.command()
@click.option(
    "--dir", default="data", help="Directory containing Slack conversation data"
)
@click.option("--user-id", default="U06BOT", help="User ID to filter by")
@click.option(
    "--filename",
    default="slack_messages_sample.jsonl",
    help="Filename to save the dataset",
)
def build_conversation_dataset(dir: str, user_id: str, filename: str):
    """Build a conversation dataset from Slack conversation data."""
    console.print(Panel("Building conversation dataset", title="🚀 Building"))
    from gr_rebuild.datasets import (
        load_slack_messages,
        preprocess_slack_messages,
        tokenize_dataset,
    )

    ds = load_slack_messages(dir, user_id, filename)
    ds = preprocess_slack_messages(ds, user_id)
    ds = tokenize_dataset(ds)

    console.print("Sample Slack conversation dataset generated successfully!")


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
