import json
import pandas as pd
from typing import Dict, Any

def clean_summary(summary_text: str) -> str:
    """Clean and standardize summary format."""
    try:
        summary_text = str(summary_text).strip()
        if not summary_text.startswith('{'):
            summary_text = '{' + summary_text
        if not summary_text.endswith('}'):
            summary_text = summary_text + '}'
        
        summary_dict = json.loads(summary_text)
        
        visit_motivation = summary_dict.get('visit motivation', '').strip()
        patient_summary = summary_dict.get('patient summary', '').strip()
        
        admission_info = summary_dict.get('admission', [])
        admission_text = ''
        if isinstance(admission_info, list) and len(admission_info) > 0:
            first_admission = admission_info[0]
            if isinstance(first_admission, dict):
                reason = first_admission.get('reason', 'None')
                admission_text = f"Admission reason: {reason}"
        
        standardized = f"Visit motivation: {visit_motivation}\n"
        if patient_summary:
            standardized += f"Patient summary: {patient_summary}\n"
        if admission_text and 'None' not in admission_text:
            standardized += f"{admission_text}\n"
        
        return standardized.strip()
        
    except Exception:
        return str(summary_text)[:200].strip()

def select_input_text(row: pd.Series) -> str:
    """Select best input text from available fields."""
    if pd.notna(row['conversation']) and len(str(row['conversation']).strip()) > 100:
        return str(row['conversation']).strip()
    elif pd.notna(row['full_note']) and len(str(row['full_note']).strip()) > 100:
        return str(row['full_note']).strip()
    elif pd.notna(row['note']) and len(str(row['note']).strip()) > 100:
        return str(row['note']).strip()
    else:
        return ''
