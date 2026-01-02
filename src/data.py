import re
import datasets
from typing import List, Dict, Optional, Any
from .utils import setup_logging

logger = setup_logging(__name__)

def parse_gsm8k_gold(answer_text: str) -> Optional[float]:
    """
    Extracts the numeric value after '####' in GSM8K gold answers.
    Example: "The answer is #### 42" -> 42.0
    """
    if not isinstance(answer_text, str):
        return None
    parts = answer_text.split("####")
    if len(parts) < 2:
        return None
    raw_val = parts[-1].strip()
    try:
        # Remove commas, e.g. 1,000 -> 1000
        clean_val = raw_val.replace(',', '')
        return float(clean_val)
    except ValueError:
        return None

def load_gsm8k(split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Loads GSM8K dataset from HuggingFace.
    Returns list of dicts: {'id': ..., 'question': ..., 'answer': ..., 'gold': float}
    """
    logger.info(f"Loading gsm8k[{split}] limit={limit}...")
    try:
        ds = datasets.load_dataset("gsm8k", "main", split=split)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return []

    if limit is not None:
        # Shuffle deterministically before slicing if we want random subset,
        # or just take first N. For research stability, usually first N or fixed seed shuffle.
        # Here we do a seeded shuffle to get a representative sample if Limit is small.
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    
    examples = []
    for i, item in enumerate(ds):
        gold_val = parse_gsm8k_gold(item['answer'])
        examples.append({
            "id": f"gsm8k_{split}_{i}",
            "question": item['question'],
            "answer": item['answer'], # full reasoning + gold
            "gold": gold_val
        })
    
    logger.info(f"Loaded {len(examples)} examples.")
    return examples
