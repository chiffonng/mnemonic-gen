# Keyword Mnemonic Generation for English Words

Vocabulary acquisition poses a significant challenge for language learners, particularly at medium and advanced levels, where the complexity and volume of new words can hinder retention. One promising solution is mnemonics, which leverage associations between new vocabulary and memorable cues to enhance recall. Previous efforts to automate generating these mnemonics often focus primarily on _shallow-encoding mnemonics_ (spelling or phonological features of a word) and are limited in their ability to generate diverse and contextually relevant mnemonics.

This project explores an alternative approach by instruction tuning the LLaMA 3 (8B) language model on a manually curated dataset of over 1,000 examples. Unlike prior methods, this dataset includes more _deep-encoding mnemonics_ (such as morphology and etymology, associations with synonyms, antonyms, or related words and concepts).

The fine-tuned model will generate diverse, contextually relevant and coherent mnemonics.

# Project goals

- [ ] Research: Compare performance between tuned and untuned models.
- [ ] Gradio: Create a web interface for the model.

# Setup

Python >= 3.10 and `requirements.txt`.
