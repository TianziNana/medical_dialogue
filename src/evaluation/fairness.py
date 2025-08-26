import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

class FairnessEvaluator:
    """Comprehensive fairness evaluation framework."""
    
    def __init__(self):
        self.demographic_groups = [
            'male_adult', 'male_pediatric', 'male_elderly',
            'female_adult', 'female_pediatric', 'female_elderly'
        ]
    
    def compute_group_disparities(self, group_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Compute disparities between demographic groups."""
        disparities = {}
        
        for metric in ['rouge1', 'rouge2', 'rougeL', 'entity_coverage']:
            if all(metric in group_metrics[g] for g in group_metrics):
                values = [group_metrics[g][metric] for g in group_metrics]
                disparities[f'{metric}_gap'] = max(values) - min(values)
                disparities[f'{metric}_std'] = np.std(values)
                disparities[f'{metric}_coefficient_of_variation'] = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        return disparities
    
    def compute_fairness_improvements(self, baseline_metrics: Dict, 
                                    fairness_metrics: Dict) -> Dict[str, float]:
        """Compare fairness between baseline and fairness-aware models."""
        improvements = {}
        
        for metric in ['rouge1', 'rouge2', 'rougeL', 'entity_coverage']:
            baseline_gap = baseline_metrics['fairness_gaps'].get(f'{metric}_gap', 0)
            fairness_gap = fairness_metrics['fairness_gaps'].get(f'{metric}_gap', 0)
            
            if baseline_gap > 0:
                improvement = (baseline_gap - fairness_gap) / baseline_gap * 100
                improvements[f'{metric}_gap_reduction'] = improvement
        
        return improvements
    
    def analyze_intersectional_fairness(self, predictions: List[str], 
                                      references: List[str], 
                                      genders: List[str], 
                                      ages: List[str]) -> Dict[str, Any]:
        """Analyze fairness across intersectional demographic categories."""
        from src.evaluation.metrics import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator()
        
        intersectional_groups = defaultdict(lambda: {'preds': [], 'refs': []})
        
        for pred, ref, gender, age in zip(predictions, references, genders, ages):
            group_key = f"{gender}_{age}"
            intersectional_groups[group_key]['preds'].append(pred)
            intersectional_groups[group_key]['refs'].append(ref)
        
        intersectional_metrics = {}
        for group, data in intersectional_groups.items():
            if len(data['preds']) > 0:
                rouge_scores = evaluator.compute_rouge_scores(data['preds'], data['refs'])
                entity_coverage, _, _ = evaluator.compute_entity_coverage(data['preds'], data['refs'])
                
                intersectional_metrics[group] = {
                    'sample_count': len(data['preds']),
                    'rougeL': rouge_scores['rougeL'],
                    'entity_coverage': entity_coverage
                }
        
        return intersectional_metrics
