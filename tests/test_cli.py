"""Tests for CLI functionality."""

import pytest
from click.testing import CliRunner

from grr.cli import main


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


def test_status_command(runner):
    """Test the status command."""
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "✅" in result.output


def test_train_command_help(runner):
    """Test the train command help."""
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "Train a model with LoRA fine-tuning" in result.output


def test_evaluate_command_help(runner):
    """Test the evaluate command help."""
    result = runner.invoke(main, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Evaluate a trained model" in result.output
