# LINKS: Generate linguistically grounded mnemonic devices for vocabulary learning with reasoning, multilingual LLMs

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/collections/chiffonng/mnemonic-generation-67563a0a1ab91e84e9827579)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/) [![wakatime](https://wakatime.com/badge/user/8256474a-d9a4-40f0-8879-659cd7b79a98/project/8890bf24-8c9d-4cb7-a5d1-bd438039c365.svg)](https://wakatime.com/badge/user/8256474a-d9a4-40f0-8879-659cd7b79a98/project/8890bf24-8c9d-4cb7-a5d1-bd438039c365)

Mnemonic devices (memory aids) are powerful tools to help individuals remember information more effectively, such as acquiring new, abstract vocabulary fast. This project proposes to explore the potential of using large language models (LLMs) to generate linguisticaly-grounded mnemonics, with the goal of aiding vocabulary acquisition and retention. The system currently works for English-English mnemonics.

Main steps:

1. Generate synthetic dataset simulating traces of reasoning through linguistic features and grounding creative writing to arrive at a mnemonic device, using LLMs + chain-of-thought rationales, and few- or many-shot in-context learning, and
2. Distill linguistic reasoning to a smaller language model, so it can generate mnemonics for English vocabulary words, and
3. Evaluate the model's performance on a test set of vocabulary words, using human evaluation and LLM-as-a-judge approach.

## Setup

Requirements: Linux, Python >=3.10 (>=3.11 recommended), PyTorch >=2.6, and a GPU with at least 16GB of VRAM.

### Installation

Prerequisites: Have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), [uv](https://docs.astral.sh/uv/), and git installed GLOBALLY (root user).

Here is the suggested installation process (after cloning the repo):

```bash
conda env create -n mnemonic-gen python=3.11 torch==2.6.0
conda activate mnemonic-gen
uv pip install -r pyproject.toml -e .
```

`requirements.txt` is also available, and is kept up to date with `pyproject.toml` using a pre-commit hook. The requirements don't include `jupyter` or `ipykernel`, but you can install them with:

```bash
uv pip install -r pyproject.toml -e .[dev]
```

### Secrets

Create a `.env` by cloning `.env.template`. You will need:

- Hugging Face Access Token (see the [doc](https://huggingface.co/docs/hub/en/security-tokens)). You can get it from [here](https://huggingface.co/settings/token).
- Wandb API key (optional: for logging experiments). You can get it from [here](https://wandb.ai/authorize).
- DeepSeek API keys. You can get keys from [DeepSeek](https://platform.deepseek.com/api_keys).

## Scripts

To train the model with GRPO on LINKS dataset:

```bash
python scripts/train.py
```

## Development

Install and update development dependencies:

```bash
bash scripts/setup-dev.sh
bash scripts/update-deps.sh
```

Run pre-commit hooks to ensure code quality (formatting, linting, checking type hints, etc.):

```bash
pre-commit run --all-files
```

Dealing with dependencies:

1. Add new dependencies to `pyproject.toml` or use `uv add <package>`. To remove a package, use `uv remove <package>`.
2. Compile to `requirements.txt` with `uv pip compile pyproject.toml -o requirements.txt`
3. Sync the environment with uv.lock and install dev dependencies: `uv sync`.
4. Upgrade the environment with `uv lock --upgrade`.

## Personal motivation

I'm a language learner myself, and I've always been fascinated by the power of mnemonics in enhancing memory retention. I've used mnemonics in learning Chinese characters and English terms and I've seen how effective they can be in helping me remember the characters and their meanings. I believe that mnemonics can be a powerful tool for language learners, and I'm excited to explore how they can be used to enhance vocabulary acquisition in English.

On the technical side, it's also a great opportunity for me to fuse some LLMOps practices with research in NLP & computational linguistics. I have a few technical goals:

- Increase reproducibility of this project, by resolving the dependencies and environment issues, and tracking experiments and hyperparameter search
- Increase manageability of the project, by using a modular structure and clear documentation.
- Have a template for future projects that involve post-training models on custom datasets.
- Learn mathematical and technical details of SOTA techniques, such as LoRA (QLoRA, rank-stablized LoRA), instruction tuning, reinforcement learning.
