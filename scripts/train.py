"""Script for training Gemma-3 models with GRPO and LoRA adapters."""

import os

from unsloth import FastModel  # isort: skip

import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from src import constants as const
from src.train.reward_functions import (
    check_essential_format,
    contains_linguistic_feature,
    mnemonic_contains_term_no_acronyms,
)
from src.utils.hf_utils import get_hf_token, login_hf_hub
from structlog import getLogger
from trl import GRPOConfig, GRPOTrainer

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = getLogger(__name__)

# LoRA and model config
hf_username = "chiffonng"
hf_base_model = "unsloth/gemma-3-1b-it"
max_seq_length = 1024
lora_rank = 8
per_device_batch_size = 16

load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(
    project=f"{hf_base_model}-links-grpo",
    job_type="training",
    anonymous="allow",
)
logger.info("WandB initialized")
login_hf_hub()

# Load datasets
required_columns = ["term", "prompt", "completion"]
logger.info("Loading datasets")
train_dataset = load_dataset(const.HF_CONST.RL_DATASET_NAME, split="train")
val_dataset = load_dataset(const.HF_CONST.RL_DATASET_NAME, split="val")
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in required_columns]
)
val_dataset = val_dataset.remove_columns(
    [col for col in val_dataset.column_names if col not in required_columns]
)

# Setup model with LoRA
logger.info(f"Loading base model: {hf_base_model}")
model, tokenizer = FastModel.from_pretrained(
    model_name=hf_base_model,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=lora_rank,
    lora_alpha=2 * lora_rank,
    lora_dropout=0,
    bias="none",
    random_state=42,
    use_rslora=True,
)

# GRPO training config
reward_funcs = [
    check_essential_format,
    contains_linguistic_feature,
    mnemonic_contains_term_no_acronyms,
]
reward_weights = [1.0, 1.0, 1.5]

training_args = GRPOConfig(
    learning_rate=3e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=1,
    num_generations=2,
    max_prompt_length=800,
    max_completion_length=600,
    num_train_epochs=1,
    max_grad_norm=0.1,
    gradient_checkpointing=True,
    bf16=True,
    # Logging
    report_to="wandb",
    logging_steps=20,
    logging_dir="logs",
    # Save strategy
    output_dir="outputs",
    run_name=run.id,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    # Evaluation strategy
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    eval_steps=200,
)

# Create and run the GRPO trainer
logger.info("Setting up GRPO trainer")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    reward_weights=reward_weights,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
)

logger.info("Starting training")
trainer.train()
logger.info("Training complete")

# Push models to HuggingFace Hub for vllm
logger.info("Pushing models to HuggingFace Hub")
model.push_to_hub_merged(
    f"{hf_username}/{hf_base_model}-links-grpo",
    tokenizer,
    save_method="merged_4bit",
    hf_token=get_hf_token(),
)

# Push GGUF model
logger.info("Pushing GGUF model")
model.push_to_hub_gguf(
    f"{hf_username}/{hf_base_model}-links-grpo",
    tokenizer,
    quantization_method=["q4_k_m", "f16"],
    token=get_hf_token(),
)

logger.info("All done!")
wandb.finish()
