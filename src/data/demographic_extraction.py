import re
import pandas as pd
from typing import Dict, List, Tuple

class DemographicExtractor:
    """Extract demographic information from medical text."""
    
    def __init__(self):
        self.gender_patterns = {
            'male': r'\b(he|him|his|male|man|boy|gentleman|mr\.?|father|son|brother|husband|boyfriend)\b',
            'female': r'\b(she|her|hers|female|woman|girl|lady|ms\.?|mrs\.?|mother|daughter|sister|wife|girlfriend)\b'
        }
        
    def extract_gender_simple(self, text: str) -> str:
        """Extract gender from text using pattern matching."""
        text = str(text).lower()
        male_count = len(re.findall(self.gender_patterns['male'], text))
        female_count = len(re.findall(self.gender_patterns['female'], text))
        
        if male_count > female_count:
            return 'male'
        elif female_count > male_count:
            return 'female'
        else:
            return 'unknown'
    
    def extract_age_group_simple(self, text: str) -> str:
        """Extract age group from text using pattern matching."""
        text = str(text).lower()
        ages = []
        matches = re.findall(r'\b(\d+)\s*-?\s*(year|yr|yo)\s*-?\s*old\b', text)
        for match in matches:
            try:
                age = int(match[0])
                if 0 <= age <= 120:
                    ages.append(age)
            except:
                continue
        
        if not ages:
            return 'unknown'
        
        age = max(set(ages), key=ages.count)
        if age < 18:
            return 'pediatric'
        elif age < 65:
            return 'adult'
        else:
            return 'elderly'
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'combined_text') -> pd.DataFrame:
        """Process entire dataframe to extract demographics."""
        df = df.copy()
        df['gender'] = df[text_column].apply(self.extract_gender_simple)
        df['age_group'] = df[text_column].apply(self.extract_age_group_simple)
        return df
