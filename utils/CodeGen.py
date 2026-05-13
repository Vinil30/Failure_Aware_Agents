import os
import torch
import re
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.ModelManager import ModelManager
import json

class Output:
    def __init__(self, code: Optional[str] = None, reasoning: Optional[str] = None):
        self.code = code
        self.reasoning = reasoning

class CodeGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", use_quantization=True):
        """
        Initialize Qwen code generator
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        
        # Get shared model instance (LOADED ONLY ONCE)
        self.manager = ModelManager()
        self.model, self.tokenizer = self.manager.get_model(model_name, use_quantization)
        
        # Store generation history for confidence calibration
        self.confidence_history = []
            
        self.system_prompt = """
You are an expert competitive programmer and software engineer.
Your task is to generate the most optimal, correct, and production-ready code for the given problem.

Strict Rules:
1. Return ONLY valid JSON with exactly this structure: {"code": "...", "reasoning": "..."}
2. Do NOT include markdown formatting.
3. Do NOT include explanations outside the JSON.
4. Do NOT include example usage.
5. Do NOT include test cases.
6. Code must be syntactically correct and directly executable.
7. Prefer time and space optimal solutions.
8. If assumptions are required, clearly state them inside the "reasoning" field.
"""
    
    def generate_code(self, question, max_new_tokens=512, temperature=0.7):
        """
        Generate code for the given question
        
        Returns:
            Output object with code and reasoning
        """
        # Format messages for chat template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with logprobs for confidence calculation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True  # Important for confidence
            )
        
        # Extract generated text
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Calculate raw confidence (mean token probability)
        # MATCHING DATASET GENERATION: confidence = exp(mean_logprob)
        logprobs_list = []
        probs_list = []

        for i, score in enumerate(outputs.scores):
            if i >= len(generated_ids):
                break
            
            # Get log probs for confidence calculation
            log_probs = torch.log_softmax(score, dim=-1)
            token_log_prob = log_probs[0, generated_ids[i]].item()
            logprobs_list.append(token_log_prob)
            
            # Also get raw probabilities for reference
            probs = torch.softmax(score, dim=-1)
            token_prob = probs[0, generated_ids[i]].item()
            probs_list.append(token_prob)

        if logprobs_list:
            # EXACT MATCH to dataset generation
            mean_logprob = float(np.mean(logprobs_list))
            raw_confidence = float(np.clip(np.exp(mean_logprob), 0.0, 1.0))
        else:
            raw_confidence = 0.5
        
        # Store both raw and normalized for different uses
        self.last_logprob = mean_logprob if logprobs_list else -100.0
        self.last_raw_confidence = raw_confidence
        self.last_normalized_confidence = self.normalize_confidence(raw_confidence)
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                code = result.get("code", "")
                reasoning = result.get("reasoning", "")
            else:
                # Fallback: treat entire response as code
                code = self._extract_code(generated_text)
                reasoning = "Code extracted from response"
        except:
            code = self._extract_code(generated_text)
            reasoning = "Failed to parse JSON response"
        
        return Output(code=code, reasoning=reasoning)
    
    def normalize_confidence(self, raw_conf):
        """
        Normalize confidence for compatibility with existing systems.
        For risk estimation, use raw_confidence (get_last_raw_confidence).
        """
        raw_conf = float(np.clip(raw_conf, 0.0, 1.0))
        # This is for other systems that expect 0.85-0.97 range
        return 0.85 + (raw_conf * (0.97 - 0.85))
    
    def get_last_confidence(self):
        """Get NORMALIZED confidence (for backward compatibility)"""
        return getattr(self, 'last_normalized_confidence', 0.89)
    
    def get_last_raw_confidence(self):
        """Get RAW confidence matching dataset generation"""
        return getattr(self, 'last_raw_confidence', 0.89)
    
    def get_last_logprob(self):
        """Get raw logprob for exact dataset matching"""
        return getattr(self, 'last_logprob', -100.0)
    
    def _extract_code(self, text):
        """Extract Python code from text"""
        # Remove markdown code blocks
        text = re.sub(r'```python\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        
        # Try to find function definition
        match = re.search(r'def\s+\w+\(.*?\):\s*(?:.*?\n)+?(?=\n\S|\Z)', text, re.DOTALL)
        if match:
            return match.group(0)
        
        # If no function found, return cleaned text
        return text.strip()
