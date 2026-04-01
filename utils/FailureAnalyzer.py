# utils/FailureAnalyzer.py

import torch
import re
import json
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class FailureOutput:
    def __init__(self, failure_reason: Optional[str] = None, severity: Optional[str] = None):
        self.failure_reason = failure_reason
        self.severity = severity

class FailureAnalyzer:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", use_quantization=True):
        """
        Initialize Qwen failure analyzer
        
        Args:
            model_name: Hugging Face model name
            use_quantization: Use 4-bit quantization to reduce memory
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen failure analyzer on {self.device}...")
        
        # Configure quantization if requested
        if use_quantization and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            quantization_config = bnb_config
        else:
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.system_prompt = """
You are an expert debugging and program analysis system.

Analyze why the generated code failed.

You will receive:
- Original problem
- Generated code
- Test cases
- Execution results

Strict Rules:
1. Return ONLY valid JSON with exactly this structure: {"failure_reason": "...", "severity": "..."}
2. Do NOT include markdown formatting.
3. Do NOT include explanations outside the JSON.
4. Provide concise root cause reasoning.
5. Classify severity as one of: LOW, MEDIUM, HIGH
"""

    def analyze(self, question, code, test_cases, execution_result):
        """
        Analyze failure and return reason and severity
        
        Returns:
            FailureOutput object with failure_reason and severity
        """
        # Format the payload
        payload = f"""
Problem:
{question}

Code:
{code}

Test Cases:
{test_cases}

Execution Result:
{execution_result}
"""
        
        # Format messages for chat template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": payload}
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
        
        # Generate analysis
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,  # Lower temperature for more focused analysis
                do_sample=False,  # Deterministic for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated text
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                failure_reason = result.get("failure_reason", "Unknown failure")
                severity = result.get("severity", "MEDIUM")
            else:
                # Fallback
                failure_reason = generated_text.strip()[:500]  # Truncate if too long
                severity = "MEDIUM"
        except Exception as e:
            failure_reason = f"Failed to parse analysis: {str(e)}"
            severity = "MEDIUM"
        
        # Validate severity
        if severity not in ["LOW", "MEDIUM", "HIGH"]:
            severity = "MEDIUM"
        
        return FailureOutput(
            failure_reason=failure_reason,
            severity=severity
        )