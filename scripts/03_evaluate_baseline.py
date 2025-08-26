#!/usr/bin/env python3

import os
import pandas as pd
import pickle
from src.models.baseline import BaselineModel
from src.evaluation.metrics import ComprehensiveEvaluator
from src.utils.helpers import save_json, print_results_summary

def main():
    # Load test data
    test_data = pd.read_pickle('data/processed/test_tensor_data.pkl')
    with open('data/processed/demographic_mapping.pkl', 'rb') as f:
        demographic_mapping = pickle.load(f)
    
    # Load model
    model = BaselineModel()
    model.load_model()
    model.model.load_state_dict(torch.load('results/models/baseline/pytorch_model.bin'))
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
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
    
    # Compute metrics
    overall_metrics = evaluator.compute_rouge_scores(predictions, references)
    entity_coverage, _, entity_details = evaluator.compute_entity_coverage(predictions, references)
    overall_metrics['entity_coverage'] = entity_coverage
    
    group_metrics = evaluator.compute_group_metrics(predictions, references, groups)
    fairness_gaps = evaluator.compute_fairness_gaps(group_metrics)
    
    # Save results
    results = {
        'overall_metrics': overall_metrics,
        'group_metrics': group_metrics,
        'fairness_gaps': fairness_gaps,
        'predictions': predictions[:20],  # Save sample predictions
        'references': references[:20]
    }
    
    os.makedirs('results/evaluation', exist_ok=True)
    save_json(results, 'results/evaluation/baseline_results.json')
    
    # Print summary
    print_results_summary(results)

if __name__ == "__main__":
    main()
