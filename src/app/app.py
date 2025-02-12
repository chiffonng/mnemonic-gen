"""Chat interface demo for Google Gemma 2 9B IT model.

Cloned and adapted from the demo: https://huggingface.co/spaces/huggingface-projects/gemma-2-9b-it/tree/main/app.py
"""

import os
from threading import Thread
from typing import Iterator

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from unsloth import FastLanguageModel

DESCRIPTION = """
This is a demo for the Google Gemma 2 9B IT model. Use it to generate mnemonics for English words you want to learn and remember.
Input your instructions or start with one of the examples provided. The input supports a subset of markdown formatting such as bold, italics, code, tables.
"""

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_id = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.sliding_window = 4096
model.eval()


@spaces.GPU(duration=90)
def generate(
    message: str,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    """Generate a response to a message using the model.

    Args:
        message: The message to respond to.
        chat_history: The conversation history.
        max_new_tokens: The maximum number of tokens to generate.
        temperature: The temperature for sampling.
        top_p: The top-p value for nucleus sampling.
        top_k: The top-k value for sampling.
        repetition_penalty: The repetition penalty.

    Yields:
        Iterator[str]: The generated response.
    """
    conversation = chat_history.copy()
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(
            f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens."
        )
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=True,
    examples=[
        [
            "Produce a cue to help me learn and retrieve the meaning of this word whenever I look at it (and nothing else): preposterous"
        ],
        [
            "Create a cue that elicits vivid mental image for the word 'observient' so I could remember its meaning."
        ],
        [
            "I need a mnemonic for 'dilapidated' to learn its meaning and contextual usage."
        ],
        [
            "Help me remember the meaning of 'encapsulate' by connecting it to its etymology or related words."
        ],
    ],
    cache_examples=False,
    type="messages",
)

with gr.Blocks(css_paths="style.css", fill_height=True) as demo:
    gr.Markdown(DESCRIPTION)
    (chat_interface.render(),)
    gr.ClearButton(elem_id="clear-button")


if __name__ == "__main__":
    demo.queue(max_size=20).launch(sharer=True)
