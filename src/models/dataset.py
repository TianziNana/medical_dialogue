import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd
from typing import Dict, Optional, List

class MedicalSummarizationDataset(Dataset):
    """Dataset class for medical text summarization with fairness capabilities."""
    
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 128,
        demographic_mapping: Optional[Dict] = None,
        split_name: str = 'train'
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.demographic_mapping = demographic_mapping
        self.split_name = split_name
        
        if demographic_mapping:
            self._process_demographics()
        else:
            self.demographics = ['unknown'] * len(self.data)
            self.is_counterfactual = [False] * len(self.data)
    
    def _process_demographics(self):
        """Extract demographic info for each sample."""
        self.demographics = []
        self.is_counterfactual = []
        
        for idx in range(len(self.data)):
            demo_label = 'unknown'
            if hasattr(self.data.iloc[idx], 'original_group'):
                demo_label = self.data.iloc[idx]['original_group']
            elif idx in self.demographic_mapping.get(self.split_name, {}):
                demo_label = self.demographic_mapping[self.split_name][idx].get('demo_label', 'unknown')
            
            self.demographics.append(demo_label)
            
            is_cf = False
            if hasattr(self.data.iloc[idx], 'is_counterfactual'):
                is_cf = self.data.iloc[idx]['is_counterfactual']
            
            self.is_counterfactual.append(is_cf)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        input_text = f"summarize: {item['input_text']}"
        target_text = str(item['target_summary'])
        
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_output_length,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels
        }
