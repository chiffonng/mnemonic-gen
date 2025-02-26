# Mnemonic Generation for English Words

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg)](https://huggingface.co/chiffonng/gemma2-9b-it-mnemonics) [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/chiffonng/en-vocab-mnemonics) [![Space on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/chiffonng/en-mnemonic-gen-demo)

Vocabulary acquisition poses a significant challenge for language learners, particularly at medium and advanced levels, where the complexity and volume of new words can hinder retention. One promising solution is mnemonics, which leverage associations between new vocabulary and memorable cues to enhance recall. Previous efforts to automate generating these mnemonics often focus primarily on mnemonics generated by **keyword method**, which is a form of _shallow encoding_ that relies on simple associations between the new word and a familiar cue (e.g., a homophonic or rhyming word). For more abstract or complex words frequent at higher-level or academic level of English usage, these methods may not be as effective.

This project explores an alternative approach by instruction tuning the Gemma 2 (9B) language model on a manually curated dataset of over 1,000 examples. Unlike prior methods, this dataset includes more _deep-encoding mnemonics_ and those generated by the same method used in language education. By fine-tuning the model on this diverse dataset, we aim to improve the quality and variety of mnemonics generated by the model, and improve the retention of new vocabulary for language learners.

| **Shallow-Encoding Mnemonics**                                                      | **Deep-Encoding Mnemonics**                                                                                  |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Homophonic:** olfactory sounds like "old factory."                                | **Etymology**: preposterous - pre (before) + post (after) + erous, which implies absurd.                     |
| **Chunking:** obsequious sounds like "ob-se-ki-ass. Obedient servants kiss your ass | **Morphology:** Suffixes "ate" are usually verbs. Prefix "ab" means from, away.                              |
| **Keyword:** Phony sounds like “phone-y,” which means fraudulent (phone calls).     | **Context/Story:** His actions of pouring acid on the environment are detrimental                            |
| **Rhyming:** wistful/longing for the past but wishful for the future.               | **Synonym/Antonym** "benevolent" ~ "kind" or "generous," and "malevolent" is its antonym.                    |
|                                                                                     | **Image Association:** exuberant - The ex-Uber driver never ranted; he always seems ebullient and lively.    |
|                                                                                     | **Related words**: Phantasm relates to an illusion or ghostly figure, closely tied to the synonym “phantom.” |

---

## Project components

- [ ] A web interface (using Gradio) for the tuned model.
- [x] A dataset of 1200 examples of English words with mnemonics.
- [ ] This documented codebase.

## Setup

### Installation (In development)

Currently `conda` is the recommended way to install the dependencies:

```bash
conda env create -n mnemonic-gen python=3.10 -f environment.yaml
conda activate mnemonic-gen
```

Otherwise, you can try the setup script:

```bash
bash setup.sh
```

It attempts to install with [uv](https://docs.astral.sh/uv/) (a fast, Rust-based Python package and project manager) using `pyproject.toml` file. This is the recommended way to manage the project, since its dependency resolver is faster and more reliable than `pip`. Otherwise, it falls back to `pip` installation.

### Secrets

Create a `.env` by cloning `.env.template`. You will need:

- Hugging Face Access Token. You will need at least `read` access token to load the dataset and model from Hugging Face (see the [doc](https://huggingface.co/docs/hub/en/security-tokens)). You can get it from [here](https://huggingface.co/settings/token).
- Wandb API key (optional: for logging experiments). You can get it from [here](https://wandb.ai/authorize).
- OpenAI API key (optional: for `data_prep` module). You can get it from [here](https://platform.openai.com/account/api-keys).

## Development

Run pre-commit hooks to ensure code quality (formatting, linting, checking type hints, etc.):

```bash
pre-commit install
pre-commit run --all-files
```

## Personal motivation

I'm a language learner myself, and I've always been fascinated by the power of mnemonics in enhancing memory retention. I've used mnemonics in learning Chinese characters and English terms and I've seen how effective they can be in helping me remember the characters and their meanings. I believe that mnemonics can be a powerful tool for language learners, and I'm excited to explore how they can be used to enhance vocabulary acquisition in English.

On the technical side, it's also a great opportunity for me to fuse some LLMOps practices with research in NLP & computational linguistics. I have a few technical goals:

- Increase reproducibility of this project, by resolving the dependencies and environment issues, and tracking experiments and hyperparameter search
- Increase manageability of the project, by using a modular structure and clear documentation.
- Have a template for future projects that involve fine-tuning models on custom datasets, and deploying them in a web interface.
- Learn mathematical and technical details of SOTA techniques, such as LoRA (QLoRA, rank-stablized LoRA), instruction tuning (self-instruct pipeline), and preference learning (ORPO).
