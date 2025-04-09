# Mining LLMs for Mnemonic Devices to aid Vocabulary Acquisition

[![wakatime](https://wakatime.com/badge/user/8256474a-d9a4-40f0-8879-659cd7b79a98/project/8890bf24-8c9d-4cb7-a5d1-bd438039c365.svg)](https://wakatime.com/badge/user/8256474a-d9a4-40f0-8879-659cd7b79a98/project/8890bf24-8c9d-4cb7-a5d1-bd438039c365)

[Hugging Face Artifacts](https://huggingface.co/collections/chiffonng/mnemonic-generation-67563a0a1ab91e84e9827579)

Mnemonic devices (memory aids) are powerful tools to help individuals remember information more effectively, such as acquiring new, abstract vocabulary fast. This project proposes to explore the potential of using large language models (LLMs) to generate linguisticaly-grounded mnemonics, with the goal of aiding vocabulary acquisition and retention. The system currently works for English-English mnemonics.

Main steps:

1. Generate synthetic dataset simulating traces of reasoning through linguistic features and grounding creative writing to arrive at a mnemonic device, using LLMs + chain-of-thought rationales, and few- or many-shot in-context learning, and
2. Fine-tune a smaller language model to generate mnemonics for English vocabulary words, and
3. Evaluate the model's performance on a test set of vocabulary words, using human evaluation and LLM-as-a-judge approach.
4. Deploy the fine-tuned model in a local web UI, allowing users to input vocabulary words and receive mnemonics as output.

## Setup

Requirements: Linux, Python >=3.10.

### Installation (In development)

Prerequisites: Have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), [uv](https://docs.astral.sh/uv/), and git installed GLOBALLY (root user).

If you have `conda`, you can create a new environment with the following command:

```bash
conda env create -n mnemonic-gen python=3.10 torch==2.4.0
conda activate mnemonic-gen
uv pip install -r pyproject.toml -e .
```

### Secrets

Create a `.env` by cloning `.env.template`. You will need:

- Hugging Face Access Token (see the [doc](https://huggingface.co/docs/hub/en/security-tokens)). You can get it from [here](https://huggingface.co/settings/token).
- Wandb API key (optional: for logging experiments). You can get it from [here](https://wandb.ai/authorize).
- OpenAI API & DeepSeek API keys. You can get keys from [OpenAI](https://platform.openai.com/account/api-keys) and [DeepSeek](https://platform.deepseek.com/api_keys).

## Development

Install development dependencies:

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
- Have a template for future projects that involve fine-tuning models on custom datasets, and deploying them in a web interface.
- Learn mathematical and technical details of SOTA techniques, such as LoRA (QLoRA, rank-stablized LoRA), instruction tuning.
