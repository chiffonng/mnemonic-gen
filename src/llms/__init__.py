import os

import litellm
from dotenv import load_dotenv

load_dotenv()

litellm.openai_key = os.getenv("OPENAI_API_KEY")
litellm.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
litellm.huggingface_key = os.getenv("HF_TOKEN")
