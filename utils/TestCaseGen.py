# utils/TestCaseGen.py

import torch
import re
import json
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.ModelManager import ModelManager

class TestCaseOutput:
    def __init__(self, test_cases: Optional[List[Dict]] = None):
        self.test_cases = test_cases or []

class TestCaseGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", use_quantization=True):
         self.manager = ModelManager()
         self.model, self.tokenizer = self.manager.get_model(model_name, use_quantization)
    
    def generate_tests(self, question):
        """Generate test cases for the given problem"""
        system_prompt = """
You are a test case generator. Generate comprehensive test cases for the given problem.

Return ONLY JSON with this structure: {"test_cases": [{"input": "...", "expected": "..."}]}

Rules:
1. Include edge cases, normal cases, and error cases
2. Keep test cases simple but thorough
3. Generate 3-5 test cases
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse JSON
        try:
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return TestCaseOutput(test_cases=result.get("test_cases", []))
        except:
            pass
        
        return TestCaseOutput(test_cases=[])