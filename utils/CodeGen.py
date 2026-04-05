import os
import torch
import re
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
        
        Args:
            model_name: Hugging Face model name
            use_quantization: Use 4-bit quantization to reduce memory
        """
        """
        Initialize Qwen code generator using shared model
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        
        # Get shared model instance (LOADED ONLY ONCE)
        self.manager = ModelManager()
        self.model, self.tokenizer = self.manager.get_model(model_name, use_quantization)
            
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
        
        # Calculate confidence (mean log probability)
        probs_list = []

        for i, score in enumerate(outputs.scores):
            if i >= len(generated_ids):
                break

            probs = torch.softmax(score, dim=-1)
            token_prob = probs[0, generated_ids[i]].item()
            probs_list.append(token_prob)

        if probs_list:
            confidence = float(np.percentile(probs_list, 25))  # ✅ robust (better than mean)
        else:
            confidence = 0.5

        # safety clamp
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # normalize to training distribution
        confidence = self.normalize_confidence(confidence)
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
        
        # Store confidence for later use
        self.last_confidence = confidence
        
        return Output(code=code, reasoning=reasoning)
    def normalize_confidence(self, raw_conf):
        raw_conf = float(np.clip(raw_conf, 0.0, 1.0))
        return 0.85 + (raw_conf * (0.97 - 0.85))
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
    
    def get_last_confidence(self):
        """Get confidence from last generation"""
        return getattr(self, 'last_confidence', -100.0)

# Optional: Add NP import at top
import numpy as np