# utils/ModelManager.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional

class ModelManager:
    """Singleton manager for shared Qwen model instance"""
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", 
                  use_quantization=True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Get or load the model (loaded only once)"""
        
        if self._model is None:
            print(f"🔄 Loading Qwen model for the first time...")
            self._load_model(model_name, use_quantization)
        else:
            print(f"✅ Reusing already loaded Qwen model")
            
        return self._model, self._tokenizer
    
    def _load_model(self, model_name, use_quantization):
        """Private method to load model once"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure quantization
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
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token if missing
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def clear_model(self):
        """Clear model from memory (useful for cleanup)"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            torch.cuda.empty_cache()
            self._model = None
            self._tokenizer = None
            print("🧹 Model cleared from memory")