#!/bin/bash

echo "Adding dev dependencies..."
source .venv/bin/activate
uv pip install --upgrade pip
uv sync # install dev dependencies by default

echo "Update pre-commit hooks..."
pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type pre-merge-commit
pre-commit autoupdate
