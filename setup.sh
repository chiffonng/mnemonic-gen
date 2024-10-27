#!/bin/bash
# https://docs.astral.sh/uv/getting-started/features/

# If uv is available, use it to install the project
if command -v uv &> /dev/null
then
    uv venv
    source .venv/bin/activate
    uv pip install -r pyproject.toml -e .
    uv sync
else
    python3.12 -m venv venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Upgrade
# uv tool upgrade --all

# Run
# uv run main.py

# Copy .env.template to .env
if [ ! -f .env ]; then
    cp .env.template .env
fi
