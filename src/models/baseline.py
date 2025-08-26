import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any

class BaselineModel:
    """Baseline medical summarization model using Flan-T5 with LoRA."""
    
    def __init__(self, model_name: str = "google/flan-t5-base", config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Load and setup the model with LoRA configuration."""
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map={"": 0} if torch.cuda.is_available() else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=self.config.get('r', 12),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            target_modules=self.config.get('target_modules', 
                          ['q', 'v', 'k', 'o', 'wi_0', 'wi_1', 'wo'])
        )
        
        self.model = get_peft_model(self.model, lora_config)
    
    def generate(self, input_text: str, max_length: int = 128, num_beams: int = 4) -> str:
        """Generate summary for input text."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        input_encoding = self.tokenizer(
            f"summarize: {input_text}",
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            input_encoding = {k: v.cuda() for k, v in input_encoding.items()}
        
        with torch.no_grad():
            generated = self.model.generate(
                **input_encoding,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False
            )
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
