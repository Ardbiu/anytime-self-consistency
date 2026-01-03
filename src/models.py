import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import setup_logging

logger = setup_logging(__name__)

class ModelRunner:
    def __init__(self, model_name: str, device: str = None, dtype: str = "auto", max_new_tokens: int = 512, trust_remote_code: bool = False):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu" and torch.backends.mps.is_available():
                self.device = "mps"
        else:
            self.device = device
            
        logger.info(f"Loading model {model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
                device_map="auto" if self.device != "cpu" else None, # MPS/CPU often handled better manually or with specific calls
                trust_remote_code=trust_remote_code
            )
            if self.device == "cpu" or self.device == "mps":
                self.model.to(self.device)
            
            # Ensure pad_token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def generate(self, prompt: str, temperature: float = 0.7, top_p: float = 1.0, top_k: int = 50, do_sample: bool = True, stop_tokens: list = None, seed: int = None) -> dict:
        """
        Generates text and returns stats.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        start_time = time.time()
        
        # Construct explicit generation args
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k
        else:
            # Greedy
            gen_kwargs["num_return_sequences"] = 1
            # Explicitly disable sampling params to avoid warnings
            # gen_kwargs["temperature"] = None # Transformers might complain if passed None
            pass
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
        end_time = time.time()
        
        total_len = outputs.shape[1]
        completion_tokens = total_len - input_len
        
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        return {
            "text": text,
            "prompt_tokens": input_len,
            "completion_tokens": completion_tokens,
            "total_tokens": total_len,
            "time_s": end_time - start_time
        }
