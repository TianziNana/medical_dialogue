#!/usr/bin/env python3

import os
import torch
import pandas as pd
import pickle
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType

from src.models.dataset import MedicalSummarizationDataset

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Load data
    train_data = pd.read_pickle('data/processed/train_tensor_data.pkl')
    val_data = pd.read_pickle('data/processed/val_tensor_data.pkl')
    
    # Initialize tokenizer and model
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=12,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q', 'v', 'k', 'o', 'wi_0', 'wi_1', 'wo']
    )
    
    model = get_peft_model(model, lora_config)
    
    # Create datasets
    train_dataset = MedicalSummarizationDataset(train_data, tokenizer)
    val_dataset = MedicalSummarizationDataset(val_data, tokenizer)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir='results/baseline_training',
        num_train_epochs=4,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        learning_rate=8e-4,
        weight_decay=0.01,
        eval_steps=382,
        save_steps=764,
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        warmup_ratio=0.1,
        logging_steps=38,
        save_total_limit=2,
        report_to=[]
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    trainer.train()
    
    # Save model
    os.makedirs('results/models/baseline', exist_ok=True)
    model.save_pretrained('results/models/baseline')
    tokenizer.save_pretrained('results/models/baseline')
    
    print("Baseline model training completed")

if __name__ == "__main__":
    main()
