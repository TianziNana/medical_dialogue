import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class ResultsVisualizer:
    """Visualization utilities for fairness evaluation results."""
    
    def __init__(self):
        self.colors = {
            'primary_blue': '#499BC0',
            'light_blue': '#8FDEE3', 
            'yellow': '#FDD786',
            'orange': '#FAAF7F',
            'red': '#F78779'
        }
        
        plt.style.use('default')
    
    def plot_group_performance(self, group_metrics: Dict[str, Dict], 
                             metric: str = 'rougeL', 
                             title: str = 'Performance by Demographic Group',
                             save_path: str = None):
        """Plot performance metrics by demographic group."""
        groups = list(group_metrics.keys())
        values = [group_metrics[g][metric] for g in groups]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(groups, values, color=self.colors['primary_blue'], alpha=0.8)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Demographic Groups')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fairness_comparison(self, baseline_gaps: Dict[str, float], 
                               fairness_gaps: Dict[str, float],
                               save_path: str = None):
        """Compare fairness gaps between baseline and fairness-aware models."""
        metrics = ['rouge1_gap', 'rouge2_gap', 'rougeL_gap', 'entity_coverage_gap']
        baseline_values = [baseline_gaps.get(m, 0) for m in metrics]
        fairness_values = [fairness_gaps.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, baseline_values, width, label='Baseline', 
                color=self.colors['red'], alpha=0.8)
        plt.bar(x + width/2, fairness_values, width, label='Fairness-Aware', 
                color=self.colors['primary_blue'], alpha=0.8)
        
        plt.title('Fairness Gap Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Performance Gap')
        plt.xticks(x, [m.replace('_gap', '').upper() for m in metrics])
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, history: Dict[str, List], save_path: str = None):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training and validation loss
        if history['step'] and history['train_loss']:
            axes[0].plot(history['step'], history['train_loss'], 
                        label='Training Loss', color=self.colors['primary_blue'])
        
        if history['eval_loss'] and len(history['eval_loss']) > 0:
            eval_steps = history['step'][::max(1, len(history['step'])//len(history['eval_loss']))][:len(history['eval_loss'])]
            axes[0].plot(eval_steps, history['eval_loss'], 
                        label='Validation Loss', color=self.colors['red'])
        
        axes[0].set_title('Training Progress')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Learning rate
        if history['learning_rate']:
            lr_steps = history['step'][:len(history['learning_rate'])]
            axes[1].plot(lr_steps, history['learning_rate'], 
                        color=self.colors['orange'])
            axes[1].set_title('Learning Rate Schedule')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
