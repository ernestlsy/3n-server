from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, DatasetDict
import torch
from preprocessor import Preprocessor
from trl import SFTTrainer, SFTConfig
import os
import sys

model, tokenizer = FastModel.from_pretrained(
    # model_name needs to be an absolute path WITH "gemma-3n" in its name
    model_name = "../gemma-3n-e2b-it",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 32,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "gemma-3",
# )

def formatting_prompts_func(example):
    preprocessor = Preprocessor(module_name="incident")
    texts = preprocessor.preprocess(example)
    return { "text" : texts, }

# dataset = load_dataset("csv", data_files=dataset_path, split = "train[:20%]")
dataset = load_dataset("csv", data_files="/mnt/c/Users/UserAdmin/Documents/demo2/server/data/small_dataset.csv", split = "train[:20%]")
dataset = dataset.map(formatting_prompts_func, batched = False)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

# only train on the assistant outputs and ignore the loss on the user's inputs
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>assistant",
)

trainer_stats = trainer.train()

print(trainer_stats)

# save model weights as safetensors
model.save_pretrained("/mnt/c/Users/UserAdmin/Documents/demo2/server/unsloth/out")
