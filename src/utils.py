import random
import numpy as np
import torch
import logging
import os
import json

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(name: str = "anytime-sc"):
    """Configures a basic logger."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    return logging.getLogger(name)

def ensure_dir(path: str):
    """Ensures directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_prompt_template(path: str) -> str:
    """Loads a text file as a string."""
    with open(path, 'r') as f:
        return f.read().strip()
