#!/usr/bin/env python3

import os
import pandas as pd
import pickle

from src.models.baseline import BaselineModel
from src.evaluation.metrics import ComprehensiveEvaluator
from src.evaluation.counterfactual import CounterfactualEvaluator
from src.evaluation.fairness import FairnessEvaluator
from src.utils.helpers import save_json, compute_improvement_percentage

def evaluate_model(model_path: str, model_name: str, test_data: pd.DataFrame, 
                  demographic_mapping: dict) -> dict:
    """Evaluate a single model."""
    model = BaselineModel()
    model.load_model()
    # Load trained weights here
    
    evaluator = ComprehensiveEvaluator()
    cf_evaluator = CounterfactualEvaluator()
    
    # Generate predictions
    predictions = []
    references = []
    groups = []
    
    for idx in range(len(test_data)):
        item = test_data.iloc[idx]
        pred = model.generate(item['input_text'])
        
        predictions.append(pred)
        references.append(item['target_summary'])
        
        if idx in demographic_mapping['test']:
            groups.append(demographic_mapping['test'][idx]['demo_label'])
        else:
            groups.append('unknown')
    
    # Compute standard metrics
    overall_metrics = evaluator.compute_rouge_scores(predictions, references)
    entity_coverage, _, entity_details = evaluator.compute_entity_coverage(predictions, references)
    overall_metrics['entity_coverage'] = entity_coverage
    
    group_metrics = evaluator.compute_group_metrics(predictions, references, groups)
    fairness_gaps = evaluator.compute_fairness_gaps(group_metrics)
    
    # Compute counterfactual fairness
    cf_results = cf_evaluator.evaluate_counterfactual_fairness(
        model.model, model.tokenizer, test_data
    )
    
    return {
        'overall_metrics': overall_metrics,
        'group_metrics': group_metrics,
        'fairness_gaps': fairness_gaps,
        'counterfactual_results': cf_results
    }

def main():
    # Load test data
    test_data = pd.read_pickle('data/processed/test_tensor_data.pkl')
    with open('data/processed/demographic_mapping.pkl', 'rb') as f:
        demographic_mapping = pickle.load(f)
    
    # Evaluate both models
    baseline_results = evaluate_model('results/models/baseline', 'Baseline', 
                                    test_data, demographic_mapping)
    fairness_results = evaluate_model('results/models/fairness_aware', 'Fairness-Aware', 
                                    test_data, demographic_mapping)
    
    # Compute improvements
    fairness_evaluator = FairnessEvaluator()
    improvements = fairness_evaluator.compute_fairness_improvements(
        baseline_results, fairness_results
    )
    
    # Performance comparison
    performance_comparison = {}
    for metric in ['rouge1', 'rouge2', 'rougeL', 'entity_coverage']:
        baseline_val = baseline_results['overall_metrics'][metric]
        fairness_val = fairness_results['overall_metrics'][metric]
        improvement = compute_improvement_percentage(baseline_val, fairness_val)
        performance_comparison[f'{metric}_improvement'] = improvement
    
    # Counterfactual comparison
    cf_comparison = {}
    for metric in ['gender_cf_similarity', 'age_cf_similarity']:
        baseline_val = baseline_results['counterfactual_results'][metric]
        fairness_val = fairness_results['counterfactual_results'][metric]
        improvement = compute_improvement_percentage(baseline_val, fairness_val)
        cf_comparison[f'{metric}_improvement'] = improvement
    
    # Compile results
    comparative_results = {
        'baseline_results': baseline_results,
        'fairness_results': fairness_results,
        'improvements': improvements,
        'performance_comparison': performance_comparison,
        'counterfactual_comparison': cf_comparison
    }
    
    # Save results
    os.makedirs('results/evaluation', exist_ok=True)
    save_json(comparative_results, 'results/evaluation/comparative_results.json')
    
    # Print summary
    print("Comparative Evaluation Results:")
    print(f"Overall Performance Change (ROUGE-L): {performance_comparison.get('rougeL_improvement', 0):.2f}%")
    print(f"Age Counterfactual Improvement: {cf_comparison.get('age_cf_similarity_improvement', 0):.2f}%")
    print(f"Gender Counterfactual Change: {cf_comparison.get('gender_cf_similarity_improvement', 0):.2f}%")

if __name__ == "__main__":
    main()
