"""Gradio interface for generating mnemonics from instructions.

TODO: Combine this interface with the chatbot interface in app.py.
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "chiffonng/gemma2-9b-it-mnemonics"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_text(instruction: str) -> str:
    """Generate mnemonic from user input/instruction.

    Args:
        instruction (str): User instructions to generate mnemonic.

    Returns:
        str: Generated mnemonic text.
    """
    inputs = tokenizer.encode(instruction, return_tensors="pt")
    outputs = model.generate(inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Create simple Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(label="Instruction"),
    outputs=gr.Textbox(label="Output"),
    title="Mnemonic Generation",
    description="Enter an instruction to generate mnemonic text.",
)


def chatbot_response(message: str, history: list) -> list:
    """Generates a response from the chatbot based on the input message and updates the conversation history.

    Args:
        message (str): The input message from the user.
        history (list): The conversation history, a list of tuples where each tuple contains a user message and a chatbot response.

    Returns:
        list: The updated conversation history with the new message and response appended.
    """
    inputs = tokenizer.encode(message, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    history.append((message, response))
    return history


# Create Gradio ChatInterface
chatbot = gr.ChatInterface(
    fn=chatbot_response,
    title="Mnemonic Generation Chatbot",
    description="Chat with the model to generate mnemonics.",
    retry_btn=True,
    undo_btn=True,
    clear_btn=True,
)


# Launch the interface
demo.launch()
# chatbot.launch()
