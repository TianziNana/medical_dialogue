import re
import torch
import numpy as np
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer

class CounterfactualEvaluator:
    """Evaluator for counterfactual fairness testing."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        self.gender_swaps = {
            r'\bhe\b': 'she', r'\bshe\b': 'he',
            r'\bhim\b': 'her', r'\bher\b': 'him',
            r'\bhis\b': 'her', r'\bhers\b': 'his',
            r'\bmale\b': 'female', r'\bfemale\b': 'male',
            r'\bman\b': 'woman', r'\bwoman\b': 'man',
            r'\bboy\b': 'girl', r'\bgirl\b': 'boy'
        }
        
        self.age_swaps = {
            r'\bchild\b': 'adult', r'\badult\b': 'elderly', r'\belderly\b': 'child',
            r'\byoung\b': 'old', r'\bold\b': 'young',
            r'\bpediatric\b': 'geriatric', r'\bgeriatric\b': 'pediatric'
        }
    
    def apply_swaps(self, text: str, swap_dict: Dict[str, str]) -> str:
        """Apply demographic swaps to text."""
        modified = text.lower()
        for pattern, replacement in swap_dict.items():
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)
        return modified
    
    def evaluate_counterfactual_fairness(self, model, tokenizer, test_data, 
                                       num_samples: int = 300) -> Dict[str, float]:
        """Evaluate counterfactual fairness of the model."""
        sample_data = test_data.sample(n=min(num_samples, len(test_data)), random_state=42)
        
        def get_model_output(input_text: str) -> str:
            input_encoding = tokenizer(
                f"summarize: {input_text}",
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            if torch.cuda.is_available():
                input_encoding = {k: v.cuda() for k, v in input_encoding.items()}
            
            with torch.no_grad():
                generated = model.generate(
                    **input_encoding,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            return tokenizer.decode(generated[0], skip_special_tokens=True)
        
        gender_similarities = []
        age_similarities = []
        
        for idx in range(len(sample_data)):
            original_text = sample_data.iloc[idx]['input_text']
            original_pred = get_model_output(original_text)
            
            # Gender swaps
            gender_swapped = self.apply_swaps(original_text, self.gender_swaps)
            if gender_swapped != original_text.lower():
                gender_pred = get_model_output(gender_swapped)
                gender_sim = self.rouge_scorer.score(original_pred, gender_pred)['rougeL'].fmeasure
                gender_similarities.append(gender_sim)
            
            # Age swaps
            age_swapped = self.apply_swaps(original_text, self.age_swaps)
            if age_swapped != original_text.lower():
                age_pred = get_model_output(age_swapped)
                age_sim = self.rouge_scorer.score(original_pred, age_pred)['rougeL'].fmeasure
                age_similarities.append(age_sim)
        
        return {
            'gender_cf_similarity': np.mean(gender_similarities) if gender_similarities else 0,
            'age_cf_similarity': np.mean(age_similarities) if age_similarities else 0,
            'gender_stability': np.mean([1 if s > 0.9 else 0 for s in gender_similarities]) if gender_similarities else 0,
            'age_stability': np.mean([1 if s > 0.9 else 0 for s in age_similarities]) if age_similarities else 0,
            'gender_similarities': gender_similarities,
            'age_similarities': age_similarities
        }
