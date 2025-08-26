import torch
import numpy as np
from transformers import Trainer
from collections import defaultdict
from typing import Dict, Any, Optional

class FairnessAwareTrainer(Trainer):
    """Enhanced trainer with fairness-aware capabilities."""
    
    def __init__(self, *args, fairness_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fairness_config = fairness_config or {}
        
        self.group_losses = defaultdict(list)
        self.training_history = {
            'train_loss': [], 'eval_loss': [], 'learning_rate': [],
            'epoch': [], 'step': [], 'group_performance': defaultdict(list),
            'fairness_metrics': []
        }
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Standard loss computation with fairness tracking."""
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def log(self, logs, start_time=None):
        """Enhanced logging with fairness metrics."""
        super().log(logs, start_time)
        
        if 'loss' in logs:
            self.training_history['train_loss'].append(logs['loss'])
            self.training_history['step'].append(self.state.global_step)
            self.training_history['epoch'].append(self.state.epoch)
            
        if 'eval_loss' in logs:
            self.training_history['eval_loss'].append(logs['eval_loss'])
            
        if 'learning_rate' in logs:
            self.training_history['learning_rate'].append(logs['learning_rate'])
        
        if 'train_loss' in logs:
            self.training_history['fairness_metrics'].append(logs['train_loss'])
