from dataclasses import dataclass
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
