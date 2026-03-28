# Task 1: Project Scaffolding

**Files:**
- Create: `lilaw-poc/pyproject.toml`
- Create: `lilaw-poc/setup.sh`
- Create: `lilaw-poc/src/lilaw_poc/__init__.py`
- Create: `lilaw-poc/src/lilaw_poc/data/__init__.py`
- Create: `lilaw-poc/.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "lilaw-poc"
version = "0.1.0"
description = "LiLAW proof-of-concept for MLMD noise robustness"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "matplotlib>=3.8",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.11",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "ANN", "B", "SIM"]
ignore = ["ANN101", "ANN102"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create `setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create venv and install deps
echo "Creating virtual environment and installing dependencies..."
uv venv .venv
uv pip install -e ".[dev]"

# Install ty
echo "Installing ty type checker..."
uv pip install ty

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Run tests:  pytest"
echo "Lint:       ruff check src/ tests/"
echo "Type check: ty check src/"
```

- [ ] **Step 3: Create `__init__.py` files and `.gitignore`**

`src/lilaw_poc/__init__.py` — empty.
`src/lilaw_poc/data/__init__.py` — empty.

`.gitignore`:
```
.venv/
__pycache__/
*.egg-info/
results/
.ruff_cache/
```

- [ ] **Step 4: Run `setup.sh` and verify**

Run: `cd lilaw-poc && bash setup.sh`
Expected: venv created, all deps installed, no errors.

- [ ] **Step 5: Verify toolchain**

Run: `source .venv/bin/activate && python -c "import torch; print(torch.__version__)" && ruff check src/ && pytest --co`
Expected: torch version printed, ruff reports no issues, pytest collects 0 tests.

- [ ] **Step 6: Commit**

```bash
git add lilaw-poc/pyproject.toml lilaw-poc/setup.sh lilaw-poc/src/ lilaw-poc/.gitignore
git commit -m "feat(lilaw-poc): scaffold project with uv, ruff, ty, pytest"
```
