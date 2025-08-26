#!/usr/bin/env python3

import json
from src.utils.visualization import ResultsVisualizer
from src.utils.helpers import load_json

def main():
    # Load results
    try:
        baseline_results = load_json('results/evaluation/baseline_results.json')
        fairness_results = load_json('results/evaluation/fairness_results.json')  
        comparative_results = load_json('results/evaluation/comparative_results.json')
    except FileNotFoundError as e:
        print(f"Results file not found: {e}")
        return
    
    # Initialize visualizer
    visualizer = ResultsVisualizer()
    
    # Plot group performance comparison
    visualizer.plot_group_performance(
        baseline_results['group_metrics'],
        metric='rougeL',
        title='Baseline Model: ROUGE-L by Demographic Group',
        save_path='results/plots/baseline_group_performance.png'
    )
    
    visualizer.plot_group_performance(
        fairness_results['group_metrics'],
        metric='rougeL', 
        title='Fairness-Aware Model: ROUGE-L by Demographic Group',
        save_path='results/plots/fairness_group_performance.png'
    )
    
    # Plot fairness gap comparison
    visualizer.plot_fairness_comparison(
        baseline_results['fairness_gaps'],
        fairness_results['fairness_gaps'],
        save_path='results/plots/fairness_comparison.png'
    )
    
    # Plot training curves if available
    try:
        training_history = load_json('results/training/fairness_training_history.json')
        visualizer.plot_training_curves(
            training_history,
            save_path='results/plots/training_curves.png'
        )
    except FileNotFoundError:
        print("Training history not found, skipping training curves")
    
    print("Visualization completed. Results saved to results/plots/")

if __name__ == "__main__":
    import os
    os.makedirs('results/plots', exist_ok=True)
    main()
