#!/bin/bash
# Setup script for ShallowFaker project
# Creates and updates the virtual environment using uv

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "Setting up ShallowFaker virtual environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    uv venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Install dependencies from pyproject.toml (preferred) or requirements.txt
if [ -f "${SCRIPT_DIR}/pyproject.toml" ]; then
    echo "Installing dependencies from pyproject.toml..."
    # uv sync automatically uses .venv if it exists
    cd "$SCRIPT_DIR"
    uv sync
elif [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    uv pip install --python "$VENV_DIR/bin/python" -r "${SCRIPT_DIR}/requirements.txt"
else
    echo "Error: Neither pyproject.toml nor requirements.txt found."
    exit 1
fi

echo ""
echo "Setup complete! Virtual environment is ready at $VENV_DIR"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or use uv to run commands directly:"
echo "  uv run <command>"

