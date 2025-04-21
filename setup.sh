#!/bin/bash

chmod +x scripts/*.sh

mkdir -p logs
mkdir -p data

# Clone .env file
copy .env.template .env

# Create conda environment
conda env create -n mnemonic-gen python=3.11 torch==2.6.0
conda activate mnemonic-gen
uv pip install -r pyproject.toml -e .
