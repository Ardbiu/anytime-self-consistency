import re
from typing import Optional, Union

def _find_last_boxed(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    token = "\\boxed{"
    last = None
    idx = 0
    while True:
        start = text.find(token, idx)
        if start == -1:
            break
        brace_start = start + len(token)
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[brace_start:i - 1]
            idx = i
        else:
            break
    return last

def _strip_latex_wrappers(text: str) -> str:
    text = text.strip()
    if text.startswith("$") and text.endswith("$") and len(text) > 1:
        text = text[1:-1]
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text)
    if boxed:
        text = boxed.group(1)
    return text.strip()

def _normalize_math_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = _strip_latex_wrappers(text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"[{}]", "", text)
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    text = text.strip().strip(".")
    return text if text else None

def _parse_numeric(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None
    try:
        return float(text)
    except ValueError:
        pass
    frac_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", text)
    if frac_match:
        num, denom = text.split("/")
        denom_val = float(denom)
        if denom_val == 0:
            return None
        return float(num) / denom_val
    return None

def extract_final_answer(text: str) -> Optional[str]:
    """
    Extracts the final answer from the model output.
    Priority 1: 'Final: <val>' line (enforced by some prompts)
    Priority 2: The last numeric value (common fallback for GSM8K)
    Priority 3: The whole text (if short)
    """
    if not isinstance(text, str):
        return None

    boxed = _find_last_boxed(text)
    if boxed:
        return boxed.strip()

    match = re.search(r"Final:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    math_matches = re.findall(r"\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", text, flags=re.DOTALL)
    if math_matches:
        flat = [m for group in math_matches for m in group if m]
        if flat:
            return flat[-1].strip()
    
    # Fallback: look for the last number in the text (often works for GSM8K CoT)
    # This regex looks for number patterns including commas, decimals.
    # We ignore numbers that might be just bullet points (e.g. "1.") if possible, but 
    # strictly grabbing the *last* number is the standard GSM8K baseline heuristic.
    # Patterns: -123, 123.45, 1,234
    fractions = re.findall(r'-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?', text)
    if fractions:
        return fractions[-1].replace(" ", "")

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

def normalize_answer_for_candidates(ans: str) -> Optional[Union[float, str]]:
    normalized = _normalize_math_text(ans)
    if normalized is None:
        return None
    num = _parse_numeric(normalized)
    if num is not None:
        return num
    return normalized

def correctness_from_target(pred_text: str, target: Optional[str]) -> bool:
    if target is None:
        return False
    extracted = extract_final_answer(pred_text)
    if extracted is None:
        return False
    pred_norm = _normalize_math_text(extracted)
    target_norm = _normalize_math_text(str(target))
    if pred_norm is None or target_norm is None:
        return False
    pred_num = _parse_numeric(pred_norm)
    target_num = _parse_numeric(target_norm)
    if pred_num is not None and target_num is not None:
        return abs(pred_num - target_num) < 1e-6
    return pred_norm == target_norm

def compare_answer_values(pred_val: Optional[Union[float, str]], target: Optional[str]) -> bool:
    if pred_val is None or target is None:
        return False
    target_norm = normalize_answer_for_candidates(str(target))
    if target_norm is None:
        return False
    if isinstance(pred_val, (int, float)) and isinstance(target_norm, (int, float)):
        return abs(float(pred_val) - float(target_norm)) < 1e-6
    return str(pred_val) == str(target_norm)

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
    val = normalize_answer_for_candidates(extracted)
    
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
