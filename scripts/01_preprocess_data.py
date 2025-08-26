#!/usr/bin/env python3

import os
import pandas as pd
import pickle
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.demographic_extraction import DemographicExtractor
from src.data.preprocessing import clean_summary, select_input_text

def main():
    # Load dataset
    dataset = load_dataset("AGBonnet/augmented-clinical-notes")
    df = pd.DataFrame(dataset['train'])
    
    # Extract demographics
    df['combined_text'] = df['note'].fillna('') + ' ' + df['conversation'].fillna('')
    extractor = DemographicExtractor()
    df = extractor.process_dataframe(df)
    
    # Filter and clean data
    df_filtered = df[(df['gender'] != 'unknown') & (df['age_group'] != 'unknown')].copy()
    df_filtered['summary_clean'] = df_filtered['summary'].apply(clean_summary)
    df_filtered['input_text'] = df_filtered.apply(select_input_text, axis=1)
    df_filtered = df_filtered[df_filtered['input_text'].str.len() > 100]
    
    # Balance dataset
    group_counts = df_filtered.groupby(['gender', 'age_group']).size()
    min_group_size = group_counts.min()
    target_size_per_group = max(min_group_size, 1500)
    
    balanced_samples = []
    for (gender, age_group), group_df in df_filtered.groupby(['gender', 'age_group']):
        current_size = len(group_df)
        if current_size >= target_size_per_group:
            sampled = group_df.sample(n=target_size_per_group, random_state=42)
        else:
            sampled = group_df.copy()
        balanced_samples.append(sampled)
    
    df_balanced = pd.concat(balanced_samples, ignore_index=True)
    df_balanced['demo_label'] = df_balanced['gender'] + '_' + df_balanced['age_group']
    
    # Create splits
    train_df, temp_df = train_test_split(
        df_balanced, test_size=0.3, stratify=df_balanced['demo_label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['demo_label'], random_state=42
    )
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    
    final_columns = ['input_text', 'summary_clean', 'gender', 'age_group', 'demo_label']
    train_final = train_df[final_columns].copy().rename(columns={'summary_clean': 'target_summary'})
    val_final = val_df[final_columns].copy().rename(columns={'summary_clean': 'target_summary'})
    test_final = test_df[final_columns].copy().rename(columns={'summary_clean': 'target_summary'})
    
    # Save tensor data
    train_final[['input_text', 'target_summary']].to_pickle('data/processed/train_tensor_data.pkl')
    val_final[['input_text', 'target_summary']].to_pickle('data/processed/val_tensor_data.pkl')
    test_final[['input_text', 'target_summary']].to_pickle('data/processed/test_tensor_data.pkl')
    
    # Save demographic mapping
    demographic_mapping = {
        'train': {idx: {'demo_label': row['demo_label'], 'gender': row['gender'], 'age_group': row['age_group']} 
                 for idx, (_, row) in enumerate(train_final.iterrows())},
        'val': {idx: {'demo_label': row['demo_label'], 'gender': row['gender'], 'age_group': row['age_group']} 
               for idx, (_, row) in enumerate(val_final.iterrows())},
        'test': {idx: {'demo_label': row['demo_label'], 'gender': row['gender'], 'age_group': row['age_group']} 
                for idx, (_, row) in enumerate(test_final.iterrows())}
    }
    
    with open('data/processed/demographic_mapping.pkl', 'wb') as f:
        pickle.dump(demographic_mapping, f)
    
    # Save metadata
    metadata = {
        'total_samples': len(df_balanced),
        'train_samples': len(train_final),
        'val_samples': len(val_final),
        'test_samples': len(test_final),
        'demographic_groups': sorted(df_balanced['demo_label'].unique().tolist()),
        'preprocessing_version': '1.0'
    }
    
    with open('data/processed/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data preprocessing completed:")
    print(f"  Train: {len(train_final)} samples")
    print(f"  Validation: {len(val_final)} samples")
    print(f"  Test: {len(test_final)} samples")

if __name__ == "__main__":
    main()
