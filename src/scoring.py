import re
from typing import Optional, Union

def extract_final_answer(text: str) -> Optional[str]:
    """
    Extracts the final answer from the model output.
    Priority 1: 'Final: <val>' line (enforced by some prompts)
    Priority 2: The last numeric value (common fallback for GSM8K)
    Priority 3: The whole text (if short)
    """
    # Check for specific "Final:" marker if used in prompt
    # Case insensitive
    match = re.search(r"Final:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for the last number in the text (often works for GSM8K CoT)
    # This regex looks for number patterns including commas, decimals.
    # We ignore numbers that might be just bullet points (e.g. "1.") if possible, but 
    # strictly grabbing the *last* number is the standard GSM8K baseline heuristic.
    # Patterns: -123, 123.45, 1,234
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    
    return None

def normalize_numeric_answer(ans: str) -> Optional[float]:
    """
    Converts logical string answer to float.
    Removes commas, handles simple formatting.
    """
    if not ans:
        return None
    try:
        # Clean string
        clean = ans.replace(',', '').strip()
        # Remove trailing period if present
        if clean.endswith('.'):
            clean = clean[:-1]
        return float(clean)
    except ValueError:
        return None

def correctness_gsm8k(pred_text: str, gold_val: Union[float, int, None]) -> bool:
    """
    Checks if prediction matches gold value (allows small float tolerance).
    """
    if gold_val is None:
        return False # Cannot evaluate
    
    extracted = extract_final_answer(pred_text)
    if extracted is None:
        return False
        
    pred_val = normalize_numeric_answer(extracted)
    if pred_val is None:
        return False
        
    # Check equality with tolerance
    return abs(pred_val - gold_val) < 1e-6

def score_candidate(pred_text: str, gold_val: Optional[float] = None) -> float:
    """
    Heuristic score for Best-of-N.
    If gold is provided (oracle reranking), returns 1.0/0.0
    Otherwise returns a heuristic quality score [0, 1].
    """
    # If we have gold (e.g. for Oracle / Upper Bound analysis), use it? 
    # Usually "Best of N" implies using a learned verifier or heuristic *without* gold.
    # The prompt asks for a "scoring function to pick best candidate".
    
    extracted = extract_final_answer(pred_text)
    val = normalize_numeric_answer(extracted)
    
    score = 0.0
    
    # 1. Parsability
    if val is not None:
        score += 0.5
    
    # 2. "Final:" presence (indicates instruction following)
    if "Final:" in pred_text:
        score += 0.3
        
    # 3. Conciseness or specific length bonus?
    # Maybe penalize extremely short or extremely long nonsense?
    # For now, just add small noise or length penalty to break ties?
    # Let's just keep it simple.
    
    return score
