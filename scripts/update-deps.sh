#!/bin/bash

# Update dependencies for the project
echo "Updating dependencies..."
uv pip compile pyproject.toml -o requirements.txt
