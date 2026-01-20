import torch
import time
from typing import Optional, Dict
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig
from .utils import setup_logging

logger = setup_logging(__name__)

class ModelRunner:
    def __init__(
        self,
        model_name: str,
        device: str = None,
        dtype: str = "auto",
        max_new_tokens: int = 512,
        trust_remote_code: bool = False,
        use_flash_attention: bool = False,
        use_compile: bool = False,
        task: str = "generate",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.task = (task or "generate").lower()
        self.is_reward_model = self.task in {"reward", "sequence_classification", "classifier"}
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu" and torch.backends.mps.is_available():
                self.device = "mps"
        else:
            self.device = device
            
        logger.info(f"Loading model {model_name} on {self.device}...")
        
        # Determine optimal dtype for A100
        if dtype == "auto" and self.device == "cuda":
            # A100 has excellent bfloat16 support
            if torch.cuda.is_bf16_supported():
                dtype = "bfloat16"
                logger.info("Using bfloat16 for A100 optimization")
            else:
                dtype = "float16"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            
            # Build model kwargs
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if dtype == "bfloat16" else (
                    torch.float16 if dtype == "float16" else "auto"
                ),
                "device_map": "auto" if self.device != "cpu" else None,
                "trust_remote_code": trust_remote_code,
            }
            
            # Enable Flash Attention 2 if requested and available
            if use_flash_attention and self.device == "cuda":
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Enabled Flash Attention 2")
                except Exception as e:
                    logger.warning(f"Flash Attention 2 not available: {e}")
            
            if self.is_reward_model:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            if self.device == "cpu" or self.device == "mps":
                self.model.to(self.device)
            
            # Apply torch.compile for additional speedup (PyTorch 2.0+)
            if use_compile and self.device == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Applied torch.compile for optimization")
                except Exception as e:
                    logger.warning(f"torch.compile not available: {e}")
            
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
        if self.is_reward_model:
            raise ValueError("generate() is not supported for reward/classifier models.")
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        start_time = time.time()
        
        # Construct explicit generation args based on strict valid flags
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
            # Greedy: strictly NO sampling params
            gen_kwargs["num_return_sequences"] = 1
            # Ensure no temperature/top_p/top_k are passed
            gen_config = getattr(self.model, "generation_config", None)
            if gen_config is not None:
                if hasattr(gen_config, "clone"):
                    gen_config = gen_config.clone()
                elif hasattr(gen_config, "copy"):
                    gen_config = gen_config.copy()
                else:
                    gen_config = GenerationConfig.from_dict(gen_config.to_dict())

                gen_config.do_sample = False
                if hasattr(gen_config, "temperature"):
                    gen_config.temperature = 1.0
                if hasattr(gen_config, "top_p"):
                    gen_config.top_p = 1.0
                if hasattr(gen_config, "top_k"):
                    gen_config.top_k = 50
                if hasattr(gen_config, "typical_p"):
                    gen_config.typical_p = 1.0
                if hasattr(gen_config, "min_p"):
                    gen_config.min_p = None
                if hasattr(gen_config, "epsilon_cutoff"):
                    gen_config.epsilon_cutoff = 0.0
                if hasattr(gen_config, "eta_cutoff"):
                    gen_config.eta_cutoff = 0.0
                if hasattr(gen_config, "penalty_alpha"):
                    gen_config.penalty_alpha = None
                gen_kwargs["generation_config"] = gen_config
        
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

    def generate_speculative(
        self,
        prompt: str,
        draft_model: "ModelRunner",
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
        seed: int = None,
    ) -> dict:
        """
        Speculative decoding using a draft model if supported by Transformers.
        Falls back to standard generation when unsupported.
        """
        if self.is_reward_model:
            raise ValueError("generate_speculative() is not supported for reward/classifier models.")
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        start_time = time.time()
        fallback = False
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    assistant_model=draft_model.model,
                    **gen_kwargs,
                )
        except TypeError:
            fallback = True
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
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
            "time_s": end_time - start_time,
            "speculative_fallback": fallback,
        }

    def generate_medusa(
        self,
        prompt: str,
        medusa_heads: int = 4,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
        seed: int = None,
    ) -> dict:
        """
        Medusa decoding if supported by the underlying model implementation.
        Falls back to standard generation when unsupported.
        """
        if self.is_reward_model:
            raise ValueError("generate_medusa() is not supported for reward/classifier models.")
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        start_time = time.time()
        fallback = False
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    medusa_heads=int(medusa_heads),
                    **gen_kwargs,
                )
        except TypeError:
            fallback = True
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
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
            "time_s": end_time - start_time,
            "medusa_fallback": fallback,
        }

    def get_prompt_context(self, prompt: str, use_hidden_state: bool = False) -> Dict[str, Optional[float]]:
        """
        Returns lightweight prompt context features.
        - prompt_tokens: tokenized prompt length
        - embedding_norm: L2 norm of pooled last hidden state (optional)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs.input_ids.shape[1])
        embedding_norm = None
        if use_hidden_state:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                last_hidden = hidden_states[-1]
                pooled = last_hidden.mean(dim=1)
                embedding_norm = torch.norm(pooled, dim=-1).item()
        return {"prompt_tokens": prompt_tokens, "embedding_norm": embedding_norm}

    def generate_batch(
        self,
        prompts: list,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        seeds: list = None,
    ) -> list:
        """
        Batched generation for multiple prompts.
        
        Uses left-padding for proper batched generation with causal LMs.
        Returns list of result dicts (same format as generate()).
        
        Note: For reproducibility with seeds, we generate sequentially within
        the batch since torch doesn't support per-sequence seeds in batched mode.
        For maximum throughput without exact reproducibility, pass seeds=None.
        """
        if self.is_reward_model:
            raise ValueError("generate_batch() is not supported for reward/classifier models.")
        if not prompts:
            return []
        
        # If seeds are provided, fall back to sequential for reproducibility
        if seeds is not None and len(seeds) == len(prompts):
            results = []
            for prompt, seed in zip(prompts, seeds):
                results.append(self.generate(
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    seed=seed,
                ))
            return results
        
        # Save original padding side and switch to left for batched generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        try:
            # Tokenize all prompts with padding
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            
            # Track input lengths for each sequence (excluding padding)
            attention_mask = inputs.attention_mask
            input_lens = attention_mask.sum(dim=1).tolist()
            
            start_time = time.time()
            
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "do_sample": do_sample,
            }
            
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                gen_kwargs["top_k"] = top_k
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            
            end_time = time.time()
            total_time = end_time - start_time
            time_per_sample = total_time / len(prompts)
            
            # Decode each output
            results = []
            batch_input_len = inputs.input_ids.shape[1]  # Padded length
            
            for i, (output, orig_input_len) in enumerate(zip(outputs, input_lens)):
                # The output includes the padded input, so we slice from batch_input_len
                generated_tokens = output[batch_input_len:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                completion_tokens = len(generated_tokens)
                total_tokens = orig_input_len + completion_tokens
                
                results.append({
                    "text": text,
                    "prompt_tokens": orig_input_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "time_s": time_per_sample,  # Amortized time
                    "batch_size": len(prompts),
                    "batch_total_time_s": total_time,
                })
            
            return results
            
        finally:
            # Restore original padding side
            self.tokenizer.padding_side = original_padding_side

    def score_yes_no(self, prompt: str, yes_token: str = " yes", no_token: str = " no") -> float:
        """
        Returns P(yes) based on next-token logits for a yes/no verifier prompt.
        Uses first token of each label if multi-token.
        """
        if self.is_reward_model:
            return self.score_reward(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
        yes_ids = self.tokenizer(yes_token, add_special_tokens=False).input_ids
        no_ids = self.tokenizer(no_token, add_special_tokens=False).input_ids
        if not yes_ids or not no_ids:
            return 0.5
        yes_id = yes_ids[0]
        no_id = no_ids[0]
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[0, yes_id].item()
        p_no = probs[0, no_id].item()
        denom = p_yes + p_no
        return p_yes / denom if denom > 0 else 0.5

    def score_reward(self, prompt: str) -> float:
        """
        Returns a scalar reward/logit for a reward or classifier model.
        Higher is better.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            return logits[0, 0].item()
        return logits[0, -1].item()

    def score_candidate(self, prompt: str) -> float:
        """
        Unified scorer for verifier models (yes/no or reward).
        """
        if self.is_reward_model:
            return self.score_reward(prompt)
        return self.score_yes_no(prompt)
