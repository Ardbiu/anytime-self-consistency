from dataclasses import dataclass
import math
import os
from .utils import load_prompt_template

@dataclass
class Policy:
    name: str
    prompt_template_path: str
    temperature: float = 0.7
    top_p: float = 1.0
    do_sample: bool = True
    max_new_tokens: int = 512

    def __post_init__(self):
        # Cache the template content
        if os.path.exists(self.prompt_template_path):
            self.template_content = load_prompt_template(self.prompt_template_path)
        else:
            self.template_content = "{question}" # Fallback

class BwKShadowPricePolicy:
    """
    Primal-dual shadow pricing policy for Bandits with Knapsacks (BwK).
    Based on Agrawal & Devanur (2014).
    """

    def __init__(self, lambda_init: float = 0.01, eta: float = 0.01) -> None:
        self.lambda_price = float(lambda_init)
        self.eta = float(eta)

    def update_price(self, consumed_tokens: float, target_tokens: float) -> float:
        """Update shadow price lambda based on resource consumption."""
        if target_tokens <= 0:
            return self.lambda_price
        delta = float(consumed_tokens) - float(target_tokens)
        self.lambda_price *= math.exp(self.eta * delta)
        return self.lambda_price

    def score(self, p_correct: float, normalized_cost: float) -> float:
        """BwK index: P(correct) - lambda * normalized_cost."""
        return float(p_correct) - self.lambda_price * float(normalized_cost)

def make_prompt(policy: Policy, question: str) -> str:
    """Formats the prompt for the given question."""
    return policy.template_content.format(question=question)

def load_policies_from_config(config: dict, root_dir: str = ".") -> list[Policy]:
    """Helper to instantiate policies from config names."""
    # This is a bit ad-hoc, but we define standard map based on specifications
    policies = []
    # If config explicitly lists policies by details, easy. 
    # If it lists just names like ["direct", "cot"], we map them to defaults.
    
    known_policies = {
        "direct": Policy("direct", f"{root_dir}/prompts/gsm8k_direct.txt", temperature=0.2), # Low temp for direct
        "cot": Policy("cot", f"{root_dir}/prompts/gsm8k_cot.txt", temperature=0.7),
        "decompose": Policy("decompose", f"{root_dir}/prompts/gsm8k_decompose.txt", temperature=0.7),
    }
    
    # Check if config has 'policies' list
    names = config.get("policies", [])
    if not names and "policy" in config:
        names = [config["policy"]] # Handle single policy case
        
    for name in names:
        if name in known_policies:
            policies.append(known_policies[name])
        else:
            # Maybe default fallback?
            pass
            
    return policies
