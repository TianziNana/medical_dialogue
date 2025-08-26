#!/usr/bin/env python3

import pandas as pd
import pickle
from tqdm import tqdm
from collections import defaultdict

from src.data.augmentation import CounterfactualAugmenter
from src.utils.helpers import save_pickle, load_pickle

def main():
    # Load training data
    train_data = pd.read_pickle('data/processed/train_tensor_data.pkl')
    with open('data/processed/demographic_mapping.pkl', 'rb') as f:
        demographic_mapping = pickle.load(f)
    
    # Load baseline evaluation to identify underperforming groups
    try:
        with open('results/evaluation/baseline_results.json', 'r') as f:
            import json
            baseline_eval = json.load(f)
        
        # Identify underperforming groups (ROUGE-L < 0.55)
        group_performance = []
        for group, metrics in baseline_eval['group_metrics'].items():
            group_performance.append((group, metrics['rougeL']))
        
        group_performance.sort(key=lambda x: x[1])
        underperforming_groups = [g[0] for g in group_performance[:3]]
    except:
        # Default underperforming groups
        underperforming_groups = ['male_adult', 'male_pediatric', 'female_pediatric']
    
    print(f"Targeting underperforming groups: {underperforming_groups}")
    
    # Initialize augmenter
    augmenter = CounterfactualAugmenter()
    
    # Set augmentation targets
    augmentation_targets = {}
    for group in underperforming_groups:
        augmentation_targets[group] = 250
    
    all_groups = ['male_adult', 'female_adult', 'male_elderly', 'female_elderly', 
                  'male_pediatric', 'female_pediatric']
    for group in all_groups:
        if group not in augmentation_targets:
            augmentation_targets[group] = 50
    
    # Generate augmented data
    augmented_samples = []
    group_counts = defaultdict(int)
    
    for idx in tqdm(range(len(train_data)), desc="Generating counterfactuals"):
        row = train_data.iloc[idx]
        
        # Get demographic info
        demo_label = 'unknown'
        if idx in demographic_mapping.get('train', {}):
            demo_label = demographic_mapping['train'][idx].get('demo_label', 'unknown')
        
        # Decide whether to augment
        remaining_needed = augmentation_targets.get(demo_label, 0) - group_counts[demo_label]
        if remaining_needed > 0:
            if demo_label in underperforming_groups:
                prob = min(0.95, remaining_needed / 100)
            else:
                prob = min(0.6, remaining_needed / 100)
            
            if np.random.random() < prob:
                counterfactuals = augmenter.generate_counterfactual(row['input_text'])
                
                if counterfactuals:
                    # Take best counterfactual
                    best_cf = counterfactuals[0]
                    for cf_type, cf_text in counterfactuals:
                        if 'syntactic' in cf_type:
                            best_cf = (cf_type, cf_text)
                            break
                    
                    cf_type, cf_text = best_cf
                    augmented_samples.append({
                        'input_text': cf_text,
                        'target_summary': row['target_summary'],
                        'is_counterfactual': True,
                        'cf_type': cf_type,
                        'original_idx': idx,
                        'original_group': demo_label
                    })
                    group_counts[demo_label] += 1
    
    # Create final augmented dataset
    augmented_df = pd.DataFrame(augmented_samples)
    original_df = train_data.copy()
    original_df['is_counterfactual'] = False
    original_df['cf_type'] = 'original'
    original_df['original_idx'] = range(len(original_df))
    original_df['original_group'] = [
        demographic_mapping.get('train', {}).get(idx, {}).get('demo_label', 'unknown')
        for idx in range(len(original_df))
    ]
    
    combined_df = pd.concat([original_df, augmented_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save augmented data
    os.makedirs('data/processed', exist_ok=True)
    save_pickle(combined_df, 'data/processed/augmented_train_data.pkl')
    
    print(f"Generated {len(augmented_samples)} counterfactual samples")
    print(f"Total augmented dataset: {len(combined_df)} samples")
    print(f"Counterfactual ratio: {len(augmented_samples)/len(combined_df)*100:.1f}%")

if __name__ == "__main__":
    import numpy as np
    main()
