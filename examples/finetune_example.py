#!/usr/bin/env python3
"""Example script demonstrating how to use the Unsloth fine-tuning module."""

import json
from pathlib import Path

from grr import TrainingConfig, UnslothFineTuner, create_sample_data


def main():
    """Example of fine-tuning with Unsloth."""
    print("🚀 GR Rebuild - Unsloth Fine-tuning Example")
    print("=" * 50)

    # Create a simple configuration for testing
    config = TrainingConfig(
        model_name="unsloth/llama-3.2-1b-bnb-4bit",
        max_seq_length=2048,
        num_train_epochs=1,  # Short for testing
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        output_dir="./output/example_training",
    )

    print(f"Model: {config.model_name}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Training epochs: {config.num_train_epochs}")
    print(f"Output directory: {config.output_dir}")
    print()

    # Create sample training data
    print("📝 Creating sample training data...")
    sample_data = create_sample_data()

    # Add some Slack-style conversations
    slack_data = [
        {
            "instruction": "How do I deploy to production?",
            "input": "",
            "output": "To deploy to production, first run the tests with `npm test`, then use `npm run deploy` to push to the staging environment. Once verified, merge to main branch which auto-deploys to production.",
        },
        {
            "instruction": "What's the status of the API?",
            "input": "",
            "output": "The API is currently running with 99.9% uptime. Last deployment was 2 hours ago. All endpoints are responding normally.",
        },
        {
            "instruction": "Can you help debug this error?",
            "input": "TypeError: Cannot read property 'name' of undefined",
            "output": "This error occurs when trying to access a property on an undefined object. Check if your variable is properly initialized before accessing its properties. Use optional chaining (`?.`) or add a null check.",
        },
    ]

    # Combine sample data
    training_data = sample_data + slack_data

    print(f"Training data prepared: {len(training_data)} examples")
    print()

    # Create fine-tuner
    print("🔧 Setting up fine-tuner...")
    fine_tuner = UnslothFineTuner(config)

    # Train the model
    print("🎯 Starting training...")
    try:
        fine_tuner.train(training_data)
        fine_tuner.save_config(config.output_dir)

        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: {config.output_dir}/final_model")
        print(f"📋 Config saved to: {config.output_dir}/training_config.json")

        # Show the saved configuration
        config_path = Path(config.output_dir) / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)
            print("\n📋 Training Configuration:")
            for key, value in saved_config.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install unsloth bitsandbytes flash-attn")


if __name__ == "__main__":
    main()
