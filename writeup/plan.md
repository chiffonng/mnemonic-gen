Executive summary

Abstract

Generated linguistically grounded mnemonic devices (i.e. memory aids) to aid advanced English vocabulary acquisition

<>

Possible applications

Reusable prompts and guidelines to prompt LLMs for effective mnemonic devices in English-English or other pairs of languages

Reproducible pipeline to generate large-scale mnemonic devices and examples for vocabulary

Dataset of (vocabulary, linguistic analysis, mnemonic, example) exported to Spaced Repetition System (SRS) like Anki for English vocabulary learning

Distilled

Note: Due to limited computing, some experiments conducted are small-scale and need more data for robust validation and conclusion. However, the codebase is reproducible and scalable when there is more computing.

Tags: computational linguistics, natural language processing, large language model, language education, English as a foreign language, vocabulary acquisition, synthetic data generation.

Introduction

literature review

Figure 1 Highlight the difference between keyword "mnemonic" and "linguistically grounded mnemonics" with an example

Caption: An example pair of keyword mnemonic and etymology. Linguistically grounded mnemonics leverage linguistic features. The keyword method is a common way to construct linguistically grounded mnemonics, but it's a shallow-encoding type. All linguistic features considered for mnemonic generation are listed in Table

Contributions:

Generate linguistically-rich mnemonics to learn and memorize the meaning and writing of English vocabulary, through prompting LLMs. Our method overcomes LLMs pre-training biases, e.g. associated with the word “mnemonic” by using synthetic chain-of-thought reasoning on linguistic features of the vocabulary and few-shot structured examples.

Impart the linguistic reasoning and creative writing style to smaller models like Gemma 3 through fine-tuning on synthetic CoT data in the first step.

2. Background

2.1. Mnemonics

Keyword method works by

Breaking the target word into smaller phonetic chunks

Finding words or phrases that sound similar to those chunks

Creating a sentence that incorporates these similar-sounding words

Making the sentence relate to the actual meaning of the target word

Keyword method is handy for a language learner when their target language is vastly different from their native language (e.g. English vs Mandarin) and when their vocabulary level is from upper-basic to intermediate.

Figure 1. Characteristics of good and bad mnemonics (with examples)

A good mnemonic should have the following:

Components: vocab -> association/explanation -> mnemonic

Vocabulary is used correctly in mnemonic

A clear explanation linking the vocabulary and the mnemonics

Strong association between the vocabulary and the mnemonic

Mnemonic is easy to understand (use similar or lower vocabulary than the term)

Mnemonic is memorable (by linking to animate/concrete phrases, useful contexts, or perceived threat)

Bad mnemonics:

Incorrect definition/usage of vocabulary:

Circular association:

Semantically close association (the guide words are more complex than the vocabulary):

Weak or unclear association:

Very hard, abstract mnemonic:

Usage of offensive language:

Table 1: Types of linguistic features to ground mnemonics

feature

description

example

phonetics

vocab’s sound patterns

aberrant sounds like 'a bare Asian'. You will never see a bare Asian (without shoes). It's atypical.

orthography

vocab’s written/spelling patterns

abet looks like 'a' + 'bet'. Visualize a bet: gamblers often abet each other in making more bets.

morphology

vocab structure in modern English, including free morphemes and bound morphemes (affixes)

aggrandize = a- + grand + -ize, to mean 'make something grander'.

etymology

vocab origin

adumbrate comes from Latin ad- (to, on) + umbra (shade) + ate, to mean foreshadow or outline.

semantics

vocab meaning (denotation, connotation) and relationships (e.g. synonyms, antonyms, polysemy, and related words)

confound has similar meaning and history with 'confuse'.

We consider all branches of linguistics on the word level

2.2. Language Models (LMs)

Review important facts about language models here, including their usage as knowledge bases for synthetic data generation, reasoning, and creative writing

surface the usage of LMs

3 In-context learning performance

Prompting LLMs is the easiest way to automate the generation of vocabulary mnemonics.

LLMs are sizable knowledge bases, although they can be unreliable and tend to hallucinate <>.

For linguistics, LLMs are shown to know linguistic rules and patterns on languages they are trained on <> <>, including grammar and morphology (structure of words), but fall short in phonology (sound of words) <PhonologyBench>

--

Since the quality of explanation connecting the vocabulary and the actual mnemonic is important, we use a reasoning model

Fig 2: Prompting methods.

Axes: % of linguistically-grounded mnemonics generated out of 50 requests sent for each prompt type

Observation: word importance / pre-training bias: mnemonic is associated with initialisms or keyword method breakdown. How do I know? Prompt the model with different prompts, and every prompt is accompanied by good mnemonic characteristics in Table 1

CAP-33

Vanilla: Generate a mnemonic to help learn and remember the meaning of English vocabulary and how it is written: {term}.

Vanilla-Alternative: Generate a memory cue to help learn and remember the meaning of English vocabulary and how it is written: {term}.

The difference is between mnemonic and memory aid, illustrating the word's importance

Structured: Generate a linguistically grounded mnemonic to help me learn and remember the meaning of English vocabulary and how it is written: ephemeral. Only explore 1-2 alternatives and stop when you have a good linguistic connection. You have to use that linguistic feature to form a mnemonic for the word

10-shot CoT (human-written):

4 Knowledge & reasoning distillation

Figure 3: Process / Explanation diagram

Generate chain-of-thought reasoning traces of mixed lengths from a teacher LLM, DeepSeek-R1

Teach a student model, Gemma3-1b-it, to adopt the reasoning process and generate mnemonics with reinforcement learning

Perform reinforcement learning,

Data Construction

The vocabulary is scraped from four different sources, preferring American English spellings: English as a foreign language tests (EFL) (IELTS, TOEFL), broad-based standardized tests (SAT, GRE), English profile CEFR levels C1 & C2, and the Oxford dictionary of philosophy to represent the arts & humanities domain. All datasets are combined, deduplicated, and decontaminated; 5000 distinct vocabulary words were chosen

System prompt: encourage concise reasoning (since LLMs like DeepSeek-R1 tend to overthink)

User prompt: generic template

Table 2. A summary of data construction pipeline components

Reasoning & Style distillation

For the student model, we chose Google's open-weight model Gemma3-1b-it for its instruction-following abilities, creative writing, and multi-linguality

Training methods: GRPO and LORA (refer to appendix ?? for training details)

Reward functions (relate to effective mnemonic requirements above)

Evaluation

Generate mnemonics for 200 example vocabulary in test set, using both

Results

Fig 4 Report pairwise reference for models: Gemma3-1b-it (base) vs Gemma3-1b-vmm

Conclusion

Limitations

LLM-as-a-judge

While using gpt4o-mini, adopt best practices for robust evaluation: 1) do not evaluate GPT-generated outputs to avoid self-enhancement bias (Panickssery et al., 2024); 2) ensuring GPT4 has high agreement with humans on held-out mnemonic comparisons (Zheng et al., 2024); and 3) shuffle order of mnemonic pairs on instance-level to reduce position bias (Wang et al., 2024).

There is likely a learnability gap between the teacher model DeepSeek-R1-670B and the student model Gemma3-1B-it <>. The next project iteration will incorporate synthetic data by another reasoning model, like QwQ-32B, to diversify reasoning and text generation style.

Appendix

Appendix / Optional: Distribution of synthetic data (optional)

Technical preliminaries

Talk about GRPO math and how it works here

Training details

Model choice gemma-3-4b-it:

Limited domain knowledge of small models may hinder their learning from strong reasoning teachers.

it is more challenging for small base models to effectively learn from long CoT data or large teacher CoT, compared to its instruction-following variant

Small enough to be downloaded and serve locally, such as on consumer GPUs and on Apple's MLX GPU

Annotation details

Table ?? Blinded dataset 20 examples (shuffled row-wise)

Table ?? Non-blinded dataset 20 examples (different from above)

Tech stack

Cost

Financial costs

Environmental costs

Human labor costs

Documentation of previous attempts
