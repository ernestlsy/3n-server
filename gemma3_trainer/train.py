import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import datasets
import torch
from datasets import DatasetDict
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
from preprocessor import Preprocessor

# Trainer for first round of fine-tuning
class InitialTrainer:
    # Initialize model, tokenizer and config settings
    def __init__(
        self, 
        checkpoint_path,
        lora_path,
        output_path,
        epoch_path,
        dataset_path,
        learning_rate,
        weight_decay,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        lora_dims,
        lora_alpha,
        lora_dropout
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path, attn_implementation='eager')
        self.dataset_path = dataset_path
        module_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.preprocessor = Preprocessor(self.tokenizer, module_name)
        self.lora_path = lora_path
        self.output_path = output_path
        self.training_args = TrainingArguments(
            output_dir=epoch_path,
            eval_strategy="epoch",
            learning_rate=float(learning_rate),
            weight_decay=weight_decay,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            save_strategy="epoch",
            logging_dir="./logs",
            report_to="none",  # disable wandb/logging
            fp16=True,  # if using GPU with half precision
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False  # Causal LM = not masked language modeling
        )
        # create LoRA configuration object
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, # type of task to train on
            inference_mode=False, # set to False for training
            r=lora_dims, # dimension of the smaller matrices
            lora_alpha=lora_alpha, # scaling factor
            lora_dropout=lora_dropout, # dropout of LoRA layers
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(self.model, self.lora_config)

    # Merge LoRA adapters and base model weights to obtain merged weights for inference
    def merge_and_export(self):
        config = PeftConfig.from_pretrained(self.lora_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        lora_model = PeftModel.from_pretrained(model, self.lora_path)
        lora_model = lora_model.merge_and_unload()
        if torch.cuda.is_available():
            print("Halving")
            lora_model = lora_model.half()
        lora_model.save_pretrained(self.output_path)

    def train_and_save(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("Model device:", next(self.model.parameters()).device)
        dataset = datasets.load_dataset("csv", data_files=self.dataset_path)
        split_dataset = dataset['train'].train_test_split(test_size=0.1)
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
        tokenized_dataset = dataset.map(self.preprocessor.preprocess, batched=False)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        trainer.train()
        trainer.save_model(self.lora_path)

        self.merge_and_export()

# Trainer for subsequent rounds of fine-tuning (i.e. when LoRA adapters already exist)
class DefaultTrainer:
    # Initialize model, tokenizer and config settings
    def __init__(
        self,
        checkpoint_path,
        lora_path,
        output_path,
        epoch_path,
        dataset_path,
        learning_rate,
        weight_decay,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        lora_dims,
        lora_alpha,
        lora_dropout
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path, attn_implementation='eager')
        self.dataset_path = dataset_path
        module_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.preprocessor = Preprocessor(self.tokenizer, module_name)
        self.lora_path = lora_path
        self.output_path = output_path
        self.training_args = TrainingArguments(
            output_dir=epoch_path,
            eval_strategy="epoch",
            learning_rate=float(learning_rate),
            weight_decay=weight_decay,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            save_strategy="epoch",
            logging_dir="./logs",
            report_to="none",  # disable wandb/logging
            fp16=True,  # if using GPU with half precision
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False  # Causal LM = not masked language modeling
        )
        self.model = PeftModel.from_pretrained(self.model, self.lora_path, is_trainable=True)

    # Merge LoRA adapters and base model weights to obtain merged weights for inference
    def merge_and_export(self):
        print("Trainer: Merging")
        config = PeftConfig.from_pretrained(self.lora_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        lora_model = PeftModel.from_pretrained(model, self.lora_path)
        lora_model = lora_model.merge_and_unload()
        lora_model = lora_model.half()
        lora_model.save_pretrained(self.output_path)

    
    def train_and_save(self):
        print("Trainer: Training")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("Model device:", next(self.model.parameters()).device)
        dataset = datasets.load_dataset("csv", data_files=self.dataset_path)
        split_dataset = dataset['train'].train_test_split(test_size=0.1)
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
        tokenized_dataset = dataset.map(self.preprocessor.preprocess, batched=False)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        trainer.train()
        trainer.save_model(self.lora_path)

        self.merge_and_export()






