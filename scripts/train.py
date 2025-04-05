"""Script for training Gemma-3 models with GRPO and LoRA adapters."""

import os

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
from unsloth import FastModel

logger = getLogger(__name__)

# LoRA and model config
max_seq_length = 2048
lora_rank = 16
base_model = "unsloth/gemma-3-4b-it"

load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(
    project="ft-gemma-3-4b-it-en-mnemonics-reason",
    job_type="training",
    anonymous="allow",
)
logger.info("WandB initialized")
login_hf_hub()

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

# Load datasets
logger.info("Loading datasets")
train_dataset, val_dataset = load_dataset(
    const.HF_CONST.RL_DATASET_NAME, split="train+val"
)

# GRPO training config
num_gens = 3
reward_funcs = [
    check_essential_format,
    contains_linguistic_feature,
    mnemonic_contains_term_no_acronyms,
]
reward_weights = [1.0, 1.0, 1.5]

training_args = GRPOConfig(
    learning_rate=2e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=10,
    per_device_train_batch_size=num_gens,
    gradient_accumulation_steps=4,
    num_generations=num_gens,
    max_prompt_length=1000,
    max_completion_length=800,
    num_train_epochs=3,
    save_steps=50,
    max_grad_norm=0.1,
    report_to="wandb",
    output_dir="outputs",
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
