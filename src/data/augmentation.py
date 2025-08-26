import re
import random
import spacy
import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict

class CounterfactualAugmenter:
    """Advanced counterfactual data augmentation using syntactic analysis."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            raise RuntimeError("Please install spaCy English model: python -m spacy download en_core_web_sm")
            
        self.gender_mappings = {
            'he': 'she', 'she': 'he', 'him': 'her', 'her': 'him',
            'his': 'her', 'hers': 'his', 'mr': 'ms', 'ms': 'mr', 'mrs': 'mr',
            'man': 'woman', 'woman': 'man', 'male': 'female', 'female': 'male',
            'boy': 'girl', 'girl': 'boy', 'gentleman': 'lady', 'lady': 'gentleman',
            'father': 'mother', 'mother': 'father', 'son': 'daughter', 'daughter': 'son'
        }
        
        self.age_mappings = {
            'child': 'adult', 'adult': 'elderly', 'elderly': 'child',
            'pediatric': 'geriatric', 'geriatric': 'pediatric',
            'young': 'elderly', 'old': 'young', 'infant': 'senior', 'senior': 'infant'
        }
    
    def analyze_syntax(self, text: str) -> Dict:
        """Analyze sentence structure using spaCy."""
        doc = self.nlp(text)
        
        analysis = {
            'subjects': [], 'objects': [], 'entities': [],
            'demographic_tokens': [], 'sentences': []
        }
        
        for sent in doc.sents:
            sent_info = {'text': sent.text, 'tokens': [], 'subjects': [], 'objects': []}
            
            for token in sent:
                token_info = {
                    'text': token.text.lower(), 'lemma': token.lemma_.lower(),
                    'pos': token.pos_, 'dep': token.dep_, 'head': token.head.text
                }
                
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    sent_info['subjects'].append(token_info)
                    analysis['subjects'].append(token_info)
                
                if token.dep_ in ['dobj', 'pobj', 'iobj']:
                    sent_info['objects'].append(token_info)
                    analysis['objects'].append(token_info)
                
                if (token.text.lower() in self.gender_mappings or 
                    token.text.lower() in self.age_mappings):
                    analysis['demographic_tokens'].append(token_info)
                
                sent_info['tokens'].append(token_info)
            
            analysis['sentences'].append(sent_info)
        
        return analysis
    
    def generate_gender_counterfactual(self, text: str, syntax_analysis: Dict) -> Tuple[str, List[str]]:
        """Generate gender counterfactual based on syntactic analysis."""
        modified_text = text.lower()
        changes_made = []
        
        for token_info in syntax_analysis['demographic_tokens']:
            original = token_info['text']
            replacement = self.gender_mappings.get(original)
            
            if replacement:
                pattern = r'\b' + re.escape(original) + r'\b'
                if re.search(pattern, modified_text):
                    modified_text = re.sub(pattern, replacement, modified_text, count=1)
                    changes_made.append(f"{original} -> {replacement}")
        
        return modified_text, changes_made
    
    def generate_age_counterfactual(self, text: str, syntax_analysis: Dict) -> Tuple[str, List[str]]:
        """Generate age counterfactual based on syntactic analysis."""
        modified_text = text.lower()
        changes_made = []
        
        for token_info in syntax_analysis['demographic_tokens']:
            original = token_info['text']
            replacement = self.age_mappings.get(original)
            
            if replacement:
                pattern = r'\b' + re.escape(original) + r'\b'
                if re.search(pattern, modified_text):
                    modified_text = re.sub(pattern, replacement, modified_text, count=1)
                    changes_made.append(f"{original} -> {replacement}")
        
        age_pattern = r'\b(\d+)[-\s]?year[-\s]?old\b'
        age_matches = list(re.finditer(age_pattern, modified_text))
        
        for match in age_matches:
            original_age = int(re.findall(r'\d+', match.group(0))[0])
            
            if original_age < 18:
                new_age = random.choice([30, 45, 70])
            elif original_age < 65:
                new_age = random.choice([8, 12, 75, 82])
            else:
                new_age = random.choice([10, 15, 25, 40])
            
            new_text = f"{new_age}-year-old"
            modified_text = modified_text[:match.start()] + new_text + modified_text[match.end():]
            changes_made.append(f"{match.group(0)} -> {new_text}")
            break
        
        return modified_text, changes_made
    
    def generate_counterfactual(self, text: str, cf_type: str = 'both') -> List[Tuple[str, str]]:
        """Generate high-quality counterfactuals using syntactic analysis."""
        if len(text.strip()) < 20:
            return []
        
        counterfactuals = []
        
        try:
            syntax_analysis = self.analyze_syntax(text)
            
            if cf_type in ['gender', 'both']:
                gender_cf, gender_changes = self.generate_gender_counterfactual(text, syntax_analysis)
                if gender_changes:
                    counterfactuals.append(('gender_syntactic', gender_cf))
            
            if cf_type in ['age', 'both']:
                age_cf, age_changes = self.generate_age_counterfactual(text, syntax_analysis)
                if age_changes:
                    counterfactuals.append(('age_syntactic', age_cf))
        
        except Exception:
            sentences = text.lower().split('.')
            if len(sentences) > 1 and sentences[0].strip():
                demographic = random.choice(['young', 'elderly', 'male', 'female', '25-year-old', '70-year-old'])
                modified = f"{demographic} {sentences[0].strip()}. {'. '.join(sentences[1:])}"
                counterfactuals.append(('fallback_insertion', modified))
        
        return counterfactuals
