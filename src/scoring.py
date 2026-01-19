import ast
import multiprocessing as mp
import re
from typing import Optional, Union, List, Dict

ANSWER_TYPE_NUMERIC = "numeric"
ANSWER_TYPE_MULTIPLE_CHOICE = "multiple_choice"
ANSWER_TYPE_CODE = "code"
ANSWER_TYPE_TEXT = "text"

def get_answer_type(example: Optional[Dict[str, object]], default: str = ANSWER_TYPE_NUMERIC) -> str:
    if not example:
        return default
    return str(example.get("answer_type") or default)

def _normalize_text_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _normalize_choice_label(label: Union[str, int, None]) -> Optional[str]:
    if label is None:
        return None
    if isinstance(label, int):
        if label < 0:
            return None
        return chr(ord("A") + label)
    label_str = str(label).strip()
    if not label_str:
        return None
    if label_str.isdigit():
        idx = int(label_str) - 1
        if idx < 0:
            return None
        return chr(ord("A") + idx)
    return label_str.upper()[0]

def _choice_labels_from_example(
    example: Optional[Dict[str, object]],
    choices: Optional[List[str]],
    choice_labels: Optional[List[str]],
) -> List[str]:
    if choice_labels:
        labels = []
        for c in choice_labels:
            norm = _normalize_choice_label(c)
            if norm is not None:
                labels.append(norm)
        if labels:
            return labels
    if example and "choice_labels" in example:
        labels = []
        for c in example.get("choice_labels", []):
            norm = _normalize_choice_label(c)
            if norm is not None:
                labels.append(norm)
        if labels:
            return labels
    if choices:
        return [chr(ord("A") + i) for i in range(len(choices))]
    return ["A", "B", "C", "D"]

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

def extract_multiple_choice_answer(
    text: str,
    choices: Optional[List[str]] = None,
    choice_labels: Optional[List[str]] = None,
) -> Optional[str]:
    if not isinstance(text, str):
        return None
    labels = _choice_labels_from_example(None, choices, choice_labels)
    label_set = set(labels)
    if not label_set:
        return None

    stripped = text.strip()
    if len(stripped) == 1:
        letter = stripped.upper()
        if letter in label_set:
            return letter

    match = re.search(r"Final:\s*([A-Za-z])", stripped, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in label_set:
            return letter

    pattern = r"\\b([" + re.escape("".join(label_set)) + r"])\\b"
    matches = re.findall(pattern, stripped.upper())
    for letter in reversed(matches):
        if letter in label_set:
            return letter

    digit_matches = re.findall(r"\\b(\\d+)\\b", stripped)
    if digit_matches:
        idx = int(digit_matches[-1]) - 1
        if 0 <= idx < len(labels):
            return labels[idx]

    if choices:
        normalized_answer = _normalize_text_for_match(stripped)
        for label, choice in zip(labels, choices):
            if not choice:
                continue
            normalized_choice = _normalize_text_for_match(str(choice))
            if not normalized_choice:
                continue
            if normalized_choice in normalized_answer or normalized_answer in normalized_choice:
                return label
    return None

def normalize_multiple_choice_answer(
    ans: Union[str, int, None],
    choices: Optional[List[str]] = None,
    choice_labels: Optional[List[str]] = None,
) -> Optional[str]:
    if ans is None:
        return None
    if isinstance(ans, int):
        return _normalize_choice_label(ans)
    if isinstance(ans, str) and ans.strip().isdigit():
        return _normalize_choice_label(int(ans.strip()))
    return extract_multiple_choice_answer(str(ans), choices=choices, choice_labels=choice_labels)

def normalize_code_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    if not isinstance(ans, str):
        ans = str(ans)
    lines = [line.rstrip() for line in ans.splitlines()]
    normalized = "\n".join(lines).strip()
    return normalized if normalized else None

def extract_candidate_answer(
    text: str,
    answer_type: str = ANSWER_TYPE_NUMERIC,
    choices: Optional[List[str]] = None,
    choice_labels: Optional[List[str]] = None,
) -> Optional[str]:
    if answer_type == ANSWER_TYPE_CODE:
        return text
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        extracted = extract_multiple_choice_answer(text, choices=choices, choice_labels=choice_labels)
        if extracted is not None:
            return extracted
    return extract_final_answer(text)

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

def normalize_answer_for_candidates(
    ans: Optional[str],
    answer_type: str = ANSWER_TYPE_NUMERIC,
    choices: Optional[List[str]] = None,
    choice_labels: Optional[List[str]] = None,
) -> Optional[Union[float, str]]:
    if answer_type == ANSWER_TYPE_CODE:
        return normalize_code_answer(ans)
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        return normalize_multiple_choice_answer(ans, choices=choices, choice_labels=choice_labels)
    if answer_type == ANSWER_TYPE_TEXT:
        if ans is None:
            return None
        return str(ans).strip()
    normalized = _normalize_math_text(ans) if ans is not None else None
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

def compare_answer_values(
    pred_val: Optional[Union[float, str]],
    target: Optional[Union[str, int]],
    answer_type: str = ANSWER_TYPE_NUMERIC,
    choices: Optional[List[str]] = None,
    choice_labels: Optional[List[str]] = None,
) -> bool:
    if pred_val is None or target is None:
        return False
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        pred_label = normalize_multiple_choice_answer(pred_val, choices=choices, choice_labels=choice_labels)
        target_label = normalize_multiple_choice_answer(target, choices=choices, choice_labels=choice_labels)
        return pred_label is not None and pred_label == target_label
    if answer_type == ANSWER_TYPE_CODE:
        return False
    target_norm = normalize_answer_for_candidates(str(target), answer_type=answer_type)
    if target_norm is None:
        return False
    if isinstance(pred_val, (int, float)) and isinstance(target_norm, (int, float)):
        return abs(float(pred_val) - float(target_norm)) < 1e-6
    return str(pred_val) == str(target_norm)

def _code_exec_worker(program: str, test_code: str, entry_point: Optional[str], queue: mp.Queue) -> None:
    try:
        env: Dict[str, object] = {}
        exec(program, env)
        if test_code:
            exec(test_code, env)
        if entry_point and "check" in env:
            env["check"](env[entry_point])
        queue.put(True)
    except Exception:
        queue.put(False)

def _run_code_tests(program: str, test_code: str, entry_point: Optional[str], timeout_s: float) -> bool:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_code_exec_worker, args=(program, test_code, entry_point, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    if queue.empty():
        return False
    return bool(queue.get())

def _should_prefix_prompt(example: Dict[str, object], prompt: str) -> bool:
    code_task = example.get("code_task")
    if code_task == "humaneval":
        return True
    if code_task == "mbpp":
        return False
    stripped = prompt.lstrip()
    return stripped.startswith(("def ", "class ", "import ", "from "))

def evaluate_code_prediction(pred_text: str, example: Dict[str, object], timeout_s: float = 5.0) -> bool:
    if not pred_text:
        return False
    prompt = str(example.get("question") or "")
    program = f"{prompt}{pred_text}" if _should_prefix_prompt(example, prompt) else pred_text
    test_list = example.get("test_list") or []
    if isinstance(test_list, str):
        test_list = [test_list]
    setup_code = example.get("test_setup_code") or example.get("setup_code") or ""
    if test_list:
        test_lines = [setup_code] if setup_code else []
        test_lines.extend(test_list)
        test_code = "\n".join(line for line in test_lines if line)
        entry_point = None
    else:
        test_code = str(example.get("test") or "")
        entry_point = example.get("entry_point")
    return _run_code_tests(program, test_code, entry_point, timeout_s)

def evaluate_prediction(
    pred_text: str,
    pred_val: Optional[Union[float, str]],
    example: Dict[str, object],
    code_timeout_s: float = 5.0,
) -> bool:
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    if answer_type == ANSWER_TYPE_CODE:
        return evaluate_code_prediction(pred_text, example, timeout_s=code_timeout_s)
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        pred_label = normalize_multiple_choice_answer(pred_val, choices=choices, choice_labels=choice_labels)
        if pred_label is None:
            pred_label = normalize_multiple_choice_answer(pred_text, choices=choices, choice_labels=choice_labels)
        target_label = normalize_multiple_choice_answer(example.get("target"), choices=choices, choice_labels=choice_labels)
        return pred_label is not None and pred_label == target_label
    if pred_val is not None:
        return compare_answer_values(pred_val, example.get("target"), answer_type=answer_type)
    return correctness_from_target(pred_text, example.get("target"))

def score_candidate(
    pred_text: str,
    gold_val: Optional[float] = None,
    answer_type: str = ANSWER_TYPE_NUMERIC,
    example: Optional[Dict[str, object]] = None,
    choices: Optional[List[str]] = None,
    choice_labels: Optional[List[str]] = None,
) -> float:
    """
    Heuristic score for Best-of-N.
    If gold is provided (oracle reranking), returns 1.0/0.0
    Otherwise returns a heuristic quality score [0, 1].
    """
    if answer_type == ANSWER_TYPE_CODE:
        prompt = ""
        if example:
            prompt = str(example.get("question") or "")
        program = f"{prompt}{pred_text}" if example and _should_prefix_prompt(example, prompt) else pred_text
        try:
            ast.parse(program)
            return 1.0
        except Exception:
            return 0.0
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        letter = normalize_multiple_choice_answer(pred_text, choices=choices, choice_labels=choice_labels)
        return 1.0 if letter is not None else 0.0

    extracted = extract_final_answer(pred_text)
    val = normalize_answer_for_candidates(extracted, answer_type=answer_type)

    score = 0.0
    if val is not None:
        score += 0.5
    if "Final:" in pred_text:
        score += 0.3
    return score

def build_verifier_prompt(question: str, candidate_text: str, final_answer: Optional[str]) -> str:
    final_line = final_answer if final_answer is not None else "unknown"
    return (
        "You are a strict math verifier.\n"
        "Answer with a single word: yes or no.\n\n"
        f"Question: {question}\n"
        f"Candidate solution: {candidate_text}\n"
        f"Candidate final answer: {final_line}\n"
        "Is the final answer correct?"
    )
