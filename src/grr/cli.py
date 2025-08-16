"""Command-line interface for GR Rebuild."""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option()
def main():
    """GR Rebuild - Fine-tune models with LoRAs using transformers and Dstack."""
    pass


# ---------------------------
# Utilities
# ---------------------------

def _read_any_json(path: str):
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with p.open() as f:
            return json.load(f)

def _save_json(obj, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------
# Sample Slack data generator (unchanged)
# ---------------------------

@main.command()
def generate_sample_slack():
    """Generate a sample Slack conversation dataset."""
    console.print(Panel("Generating sample Slack conversation dataset", title="🚀 Generating"))
    from scripts.generate_sample_slack import generate_sample_slack
    generate_sample_slack()
    console.print("Sample Slack conversation dataset generated successfully!")


# ---------------------------
# New: build dataset (Slack/Discord → chat or Alpaca)
# ---------------------------

@main.command(name="build-dataset")
@click.option("--source", type=click.Choice(["slack", "discord"], case_sensitive=False), required=True,
              help="Source platform.")
@click.option("--input", "input_path", required=True, help="Path to input file (.jsonl for Slack, .json for Discord).")
@click.option("--target-user", required=True, help="Persona user id or name to treat as assistant.")
@click.option("--format", "out_format", type=click.Choice(["chat", "alpaca"], case_sensitive=False), default="chat",
              help="Output format.")
@click.option("--output", "output_path", required=True, help="Where to save the built dataset (.json).")
@click.option("--history-max-turns", default=6, show_default=True, help="Max prior turns before assistant reply.")
@click.option("--min-turns", default=1, show_default=True, help="Min context turns required to emit an example.")
@click.option("--eval-split", default=0.1, show_default=True, help="Eval fraction.")
def build_dataset_cmd(source, input_path, target_user, out_format, output_path, history_max_turns, min_turns, eval_split):
    """Build a persona dataset from Slack or Discord exports."""
    console.print(Panel(f"Building {out_format} dataset from {source}", title="🚀 Building"))

    # import our new builders
    from grr.datasets import (
        WindowConfig,
        build_alpaca_dataset,
        build_chat_dataset,
    )

    win = WindowConfig(history_max_turns=history_max_turns, min_turns_required=min_turns)

    if out_format == "chat":
        ds = build_chat_dataset(source, input_path, target_user, win)
    else:
        ds = build_alpaca_dataset(source, input_path, target_user, win)

    # save to a single JSON with {"train":[...], "test":[...]}
    out_obj = {
        "train": list(ds["train"]),
        "eval": list(ds["test"]) if "test" in ds else []
    }
    _save_json(out_obj, output_path)
    console.print(f"[green]Saved {out_format} dataset → {output_path}[/green]")


# ---------------------------
# (Legacy) Slack-only builder – kept for compatibility
# ---------------------------

@main.command()
@click.option("--dir", default="data", help="Directory containing Slack conversation data (ignored; deprecated).")
@click.option("--user-id", default="U06BOT", help="User ID to filter by")
@click.option("--filename", default="slack_messages_sample.jsonl", help="Slack JSONL file")
@click.option("--output", default="slack_chat.json", help="Output JSON (chat format)")
def build_conversation_dataset(dir: str, user_id: str, filename: str, output: str):
    """[Deprecated] Build a conversation dataset from Slack JSONL."""
    console.print(Panel("Building conversation dataset (Slack)", title="🚀 Building"))
    from grr.datasets import WindowConfig, build_chat_dataset
    ds = build_chat_dataset("slack", filename, user_id, WindowConfig())
    out_obj = {"train": list(ds["train"]), "eval": list(ds["test"]) if "test" in ds else []}
    _save_json(out_obj, output)
    console.print(f"[green]Saved chat dataset → {output}[/green]")


# ---------------------------
# Train
# ---------------------------

@main.command()
@click.option("--model", default="unsloth/llama-3.2-1b-bnb-4bit", help="Model to fine-tune")
@click.option("--dataset", default="sample", help="Dataset to use (sample, custom)")
@click.option("--data-path", help="Path to custom dataset (.json or .jsonl)")
@click.option("--data-format", type=click.Choice(["chat", "alpaca"], case_sensitive=False), default="chat",
              help="Structure of records inside data-path.")
@click.option("--output-dir", default="./output", help="Output directory for trained model")
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--batch-size", default=2, help="Training batch size")
@click.option("--learning-rate", default=2e-4, help="Learning rate")
@click.option("--max-length", default=2048, help="Maximum sequence length")
def train(
    model: str,
    dataset: str,
    data_path: str,
    data_format: str,
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
                console.print("[red]Error: --data-path is required for custom dataset[/red]")
                return

            console.print(f"Loading custom dataset from {data_path} ...")
            try:
                data = _read_any_json(data_path)

                # Accept either {"train":[...], "eval":[...]} or a flat list
                if isinstance(data, dict) and "train" in data:
                    train_data = data["train"]
                    eval_data = data.get("eval")
                elif isinstance(data, list):
                    train_data = data
                    eval_data = None
                else:
                    console.print("[red]Unrecognized data structure. Expect list or {'train':..., 'eval':...}[/red]")
                    return

                # Quick schema check
                if data_format == "chat":
                    # expect dicts with 'messages' (+ optional 'system')
                    if not train_data or "messages" not in train_data[0]:
                        console.print("[red]data-format=chat expects records with 'messages'[/red]")
                        return
                else:
                    # Alpaca expects instruction/input/output
                    if not train_data or "instruction" not in train_data[0] or "output" not in train_data[0]:
                        console.print("[red]data-format=alpaca expects 'instruction'/'output' fields[/red]")
                        return

            except Exception as e:
                console.print(f"[red]Error loading dataset: {e}[/red]")
                return

        else:
            console.print(f"[red]Unknown dataset: {dataset}[/red]")
            return

        # Create fine-tuner and train
        fine_tuner = UnslothFineTuner(config, data_format=data_format)  # pass format so collator knows
        fine_tuner.train(train_data, eval_data)
        fine_tuner.save_config(output_dir)

        console.print(f"[green]Training completed! Model saved to {output_dir}[/green]")

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure you're running this in a GPU environment (like RunPod via dstack)[/yellow]")
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")


# ---------------------------
# Evaluate (stub)
# ---------------------------

@main.command()
@click.option("--model-path", required=True, help="Path to trained model")
def evaluate(model_path: str):
    """Evaluate a trained model."""
    console.print(Panel(f"Evaluating model at {model_path}", title="📊 Evaluation"))
    console.print("Evaluation functionality coming soon!")


# ---------------------------
# Sample data (unchanged)
# ---------------------------

@main.command()
@click.option("--output", default="sample_data.json", help="Output file for sample data")
def create_sample(output: str):
    """Create a sample dataset file for testing."""
    console.print(Panel("Creating sample dataset", title="📝 Sample Data"))
    try:
        from .finetune import create_sample_data
        sample_data = create_sample_data()
        _save_json(sample_data, output)
        console.print(f"[green]Sample dataset created: {output}[/green]")
        console.print("Use with: grr train --dataset custom --data-path sample_data.json")
    except ImportError:
        console.print("[red]Error: Could not import fine-tuning module[/red]")


if __name__ == "__main__":
    main()
