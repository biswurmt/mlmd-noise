#!/usr/bin/env bash
set -euo pipefail

# Find the lilaw-poc directory (look for pyproject.toml relative to repo root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "lilaw-poc/pyproject.toml" ]; then
    PROJECT_DIR="$(pwd)/lilaw-poc"
elif [ -f "pyproject.toml" ]; then
    PROJECT_DIR="$(pwd)"
else
    echo "Error: Cannot find pyproject.toml. Run this script from the repo root or lilaw-poc/ directory."
    exit 1
fi

cd "$PROJECT_DIR"
echo "Working directory: $PROJECT_DIR"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install deps
echo "Creating virtual environment and installing dependencies..."
uv venv .venv
uv pip install -e ".[dev]"

# Install ty
echo "Installing ty type checker..."
uv pip install ty

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source .venv/bin/activate"
echo "Run tests:      pytest"
echo "Lint:           ruff check src/ tests/"
echo "Type check:     ty check src/"
