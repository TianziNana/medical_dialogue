import re
import numpy as np
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
from collections import defaultdict

class MedicalEntityExtractor:
    """Extract medical entities from text."""
    
    def __init__(self):
        self.medical_patterns = {
            'symptoms': r'\b(pain|ache|discomfort|swelling|inflammation|fever|nausea|headache|fatigue|dizzy|dizziness|weakness|shortness|breathing|cough|rash|itching|burning|tingling|numbness|soreness|stiffness)\b',
            'body_parts': r'\b(head|neck|back|chest|abdomen|stomach|arm|leg|hand|foot|shoulder|knee|ankle|wrist|elbow|hip|spine|throat|eye|ear|nose|mouth|tooth|teeth|finger|toe|joint|muscle)\b',
            'conditions': r'\b(infection|fracture|strain|sprain|diabetes|hypertension|asthma|pneumonia|bronchitis|arthritis|migraine|depression|anxiety|allergy|cancer|tumor|disease|syndrome|disorder)\b',
        }
    
    def extract_entities(self, text: str) -> set:
        """Extract medical entities from text."""
        entities = set()
        text_lower = text.lower()
        
        for category, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, text_lower)
            entities.update(matches)
        
        return entities

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for medical summarization."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.entity_extractor = MedicalEntityExtractor()
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'rouge1_std': np.std(rouge1_scores),
            'rouge2_std': np.std(rouge2_scores),
            'rougeL_std': np.std(rougeL_scores),
            'rouge1_scores': rouge1_scores,
            'rouge2_scores': rouge2_scores,
            'rougeL_scores': rougeL_scores
        }
    
    def compute_entity_coverage(self, predictions: List[str], references: List[str]) -> Tuple[float, List[float], Dict]:
        """Compute entity coverage metrics."""
        coverages = []
        entity_details = {
            'total_reference_entities': 0,
            'total_predicted_entities': 0,
            'total_matched_entities': 0
        }
        
        for pred, ref in zip(predictions, references):
            ref_entities = self.entity_extractor.extract_entities(ref)
            pred_entities = self.entity_extractor.extract_entities(pred)
            
            entity_details['total_reference_entities'] += len(ref_entities)
            entity_details['total_predicted_entities'] += len(pred_entities)
            
            if len(ref_entities) == 0:
                coverage = 1.0 if len(pred_entities) == 0 else 0.5
            else:
                covered = len(ref_entities.intersection(pred_entities))
                entity_details['total_matched_entities'] += covered
                coverage = covered / len(ref_entities)
            
            coverages.append(coverage)
        
        return np.mean(coverages), coverages, entity_details
    
    def compute_group_metrics(self, predictions: List[str], references: List[str], 
                            groups: List[str]) -> Dict[str, Dict]:
        """Compute metrics for each demographic group."""
        group_predictions = defaultdict(list)
        group_references = defaultdict(list)
        
        for pred, ref, group in zip(predictions, references, groups):
            group_predictions[group].append(pred)
            group_references[group].append(ref)
        
        group_metrics = {}
        for group_name in group_predictions:
            if len(group_predictions[group_name]) > 0:
                group_preds = group_predictions[group_name]
                group_refs = group_references[group_name]
                
                group_rouge = self.compute_rouge_scores(group_preds, group_refs)
                group_entity, group_entity_scores, _ = self.compute_entity_coverage(group_preds, group_refs)
                
                group_metrics[group_name] = {
                    'sample_count': len(group_preds),
                    'rouge1': group_rouge['rouge1'],
                    'rouge2': group_rouge['rouge2'],
                    'rougeL': group_rouge['rougeL'],
                    'rouge1_std': group_rouge['rouge1_std'],
                    'rouge2_std': group_rouge['rouge2_std'],
                    'rougeL_std': group_rouge['rougeL_std'],
                    'entity_coverage': group_entity,
                    'entity_scores': group_entity_scores
                }
        
        return group_metrics
    
    def compute_fairness_gaps(self, group_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Compute fairness gaps across groups."""
        fairness_gaps = {}
        for metric in ['rouge1', 'rouge2', 'rougeL', 'entity_coverage']:
            values = [group_metrics[g][metric] for g in group_metrics if metric in group_metrics[g]]
            if values:
                fairness_gaps[f'{metric}_gap'] = max(values) - min(values)
                fairness_gaps[f'{metric}_std'] = np.std(values)
                fairness_gaps[f'{metric}_max'] = max(values)
                fairness_gaps[f'{metric}_min'] = min(values)
                fairness_gaps[f'{metric}_mean'] = np.mean(values)
        
        return fairness_gaps
