# Mining LLMs for Mnemonic Devices to aid Vocabulary Acquisition

[Hugging Face Artifacts](https://huggingface.co/collections/chiffonng/mnemonic-generation-67563a0a1ab91e84e9827579)

Can we use LLMs to generate **diverse** mnemonic devices for vocabulary acquisition with **linguistic reasoning**?

---

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

- Hugging Face Access Token. You will need at least `read` access token to load the dataset and model from Hugging Face (see the [doc](https://huggingface.co/docs/hub/en/security-tokens)). You can get it from [here](https://huggingface.co/settings/token).
- Wandb API key (optional: for logging experiments). You can get it from [here](https://wandb.ai/authorize).
- OpenAI API key (optional: for `data_prep` module). You can get keys from [OpenAI](https://platform.openai.com/account/api-keys) and [Anthropic](https://anthropic.com/).

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

## Personal motivation

I'm a language learner myself, and I've always been fascinated by the power of mnemonics in enhancing memory retention. I've used mnemonics in learning Chinese characters and English terms and I've seen how effective they can be in helping me remember the characters and their meanings. I believe that mnemonics can be a powerful tool for language learners, and I'm excited to explore how they can be used to enhance vocabulary acquisition in English.

On the technical side, it's also a great opportunity for me to fuse some LLMOps practices with research in NLP & computational linguistics. I have a few technical goals:

- Increase reproducibility of this project, by resolving the dependencies and environment issues, and tracking experiments and hyperparameter search
- Increase manageability of the project, by using a modular structure and clear documentation.
- Have a template for future projects that involve fine-tuning models on custom datasets, and deploying them in a web interface.
- Learn mathematical and technical details of SOTA techniques, such as LoRA (QLoRA, rank-stablized LoRA), instruction tuning.
