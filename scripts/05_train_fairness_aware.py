#!/usr/bin/env python3

import os
import pandas as pd
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType

from src.models.dataset import MedicalSummarizationDataset
from src.models.fairness_aware import FairnessAwareTrainer
from src.utils.helpers import save_json

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Load augmented data
    augmented_train_data = pd.read_pickle('data/processed/augmented_train_data.pkl')
    val_data = pd.read_pickle('data/processed/val_tensor_data.pkl')
    
    # Initialize model from baseline
    tokenizer = T5Tokenizer.from_pretrained('results/models/baseline')
    model = T5ForConditionalGeneration.from_pretrained('results/models/baseline')
    
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
    
    # Prepare fairness config
    fairness_config = {
        'counterfactual_weight': 1.3,
        'group_adaptive_weights': True
    }
    
    # Create datasets
    train_dataset = MedicalSummarizationDataset(
        augmented_train_data[['input_text', 'target_summary']],
        tokenizer
    )
    val_dataset = MedicalSummarizationDataset(val_data, tokenizer)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir='results/fairness_training',
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
    
    # Create fairness-aware trainer
    trainer = FairnessAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        fairness_config=fairness_config
    )
    
    # Train model
    trainer.train()
    
    # Save model
    os.makedirs('results/models/fairness_aware', exist_ok=True)
    model.save_pretrained('results/models/fairness_aware')
    tokenizer.save_pretrained('results/models/fairness_aware')
    
    # Save training history
    save_json(trainer.training_history, 'results/training/fairness_training_history.json')
    
    print("Fairness-aware model training completed")

if __name__ == "__main__":
    main()
