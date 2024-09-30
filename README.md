# Keyword Mnemonic Generation for English Words

Vocabulary acquisition poses a significant challenge for language learners, particularly at medium and advanced levels, where the complexity and volume of new words can hinder retention. One promising solution is keyword mnemonics, which leverage associations between new vocabulary and memorable cues to enhance recall. Previous efforts to automate generating these mnemonics often lack diversity and structure in resulting mnemonics.

This project explores an alternative approach by fine-tuning the LLaMA 3 (8B) language model using instruction tuning on a manually curated dataset of over 1,000 examples. Unlike prior methods that primarily focus on syllabic and phonetic mnemonics, this dataset is more representative of mnemonic types, including more etymological mnemonics, which research shows can deepen understanding and retention by linking new vocabulary to their roots and origins. The fine-tuned model will generate diverse, contextually relevant and coherent mnemonics.

# Setup

Python >= 3.10 and `requirements.txt`.
