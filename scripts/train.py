"""Script for training Gemma-3 models with GRPO and LoRA adapters."""

import os

from unsloth import FastModel # noqa: F401

import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from src import constants as const
from src.train.reward_functions import (
    check_essential_format,
    contains_linguistic_feature,
    mnemonic_contains_term_no_acronyms,
)
from src.utils.hf_utils import login_hf_hub
from structlog import getLogger
from trl import GRPOConfig, GRPOTrainer


logger = getLogger(__name__)

# LoRA and model config
max_seq_length = 2048
lora_rank = 16
base_model = "unsloth/gemma-3-4b-it"
per_device_batch_size = 8

load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(
    project="gemma-3-4b-it-vmm",
    job_type="training",
    anonymous="allow",
)
logger.info("WandB initialized")
login_hf_hub()

# Load datasets
logger.info("Loading datasets")
train_dataset= load_dataset(
    const.HF_CONST.RL_DATASET_NAME, split="train"
)
val_dataset = load_dataset(
    const.HF_CONST.RL_DATASET_NAME, split="val"
)

# Setup model with LoRA
logger.info(f"Loading base model: {base_model}")
model, tokenizer = FastModel.from_pretrained(
    model_name=base_model,
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
    gradient_accumulation_steps=2,
    num_generations=2,
    max_prompt_length=1000,
    max_completion_length=800,
    num_train_epochs=3,
    max_grad_norm=0.1,

    # Logging
    report_to="wandb",
    logging_steps=10,
    logging_dir="logs",

    # Save strategy
    output_dir="outputs",
    run_name=run.id,
    save_strategy="steps",
    save_total_limit=5,
    load_best_model_at_end=True,

    # Evaluation strategy
    per_device_eval_batch_size=per_device_batch_size,
    eval_strategy="steps",
    eval_steps=50,
    gradient_checkpointing=True,
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
model.push_to_hub_merged("gemma3-grpo", tokenizer, save_method="merged_4bit")
model.push_to_hub_merged("gemma3-grpo", tokenizer, save_method="merged_16bit")

# Push GGUF model
logger.info("Pushing GGUF model")
model.push_to_hub_gguf(
    "chiffonng/gemma3-vmm-grpo", tokenizer, quantization_method=["q4_k_m", "f16"]
)

logger.info("All done!")
