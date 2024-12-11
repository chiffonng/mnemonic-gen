#!/bin/bash
# https://docs.astral.sh/uv/getting-started/features/

# Catch --optional flag
if [ "$1" == "--optional" ]; then
    echo "Optional dependencies will be installed."
    OPTIONAL=true
else
    OPTIONAL=false
fi

# If uv is available, use it to install the project
if command -v uv &> /dev/null
then
    uv venv
    # shellcheck source=.venv/bin/activate
    source .venv/bin/activate
    if [ "$OPTIONAL" = true ]; then
        uv pip install -r pyproject.toml -e . --all-extras
    else
        uv pip install -r pyproject.toml -e .
    fi

else
    python3 -m venv venv
    # shellcheck source=.venv/bin/activate
    source venv/bin/activate
    if [ "$OPTIONAL" = true ]; then
        pip install -e .[all]
    else
        pip install -e .
    fi
fi

# Upgrade
# uv tool upgrade --all

# Run
# uv run main.py

# Copy .env.template to .env
if [ ! -f .env ]; then
    cp .env.template .env
fi

# Create project directories
mkdir -p logs
