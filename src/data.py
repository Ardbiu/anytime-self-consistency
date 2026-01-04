import re
import datasets
from typing import List, Dict, Optional, Any
from .utils import setup_logging

logger = setup_logging(__name__)

_HENDRYCKS_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

def _apply_limit(ds, limit: Optional[int], seed: int):
    if limit is None:
        return ds
    return ds.shuffle(seed=seed).select(range(min(limit, len(ds))))

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

def parse_gsm8k_target(answer_text: str) -> Optional[str]:
    if not isinstance(answer_text, str):
        return None
    parts = answer_text.split("####")
    if len(parts) >= 2:
        return parts[-1].strip()
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', answer_text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None

def parse_gsm_plus_target(item: Dict[str, Any]) -> Optional[str]:
    ans = item.get("answer")
    if isinstance(ans, str) and ans.strip():
        return ans.strip()
    solution = item.get("solution", "")
    return parse_gsm8k_target(solution)

def parse_hendrycks_target(solution_text: str) -> Optional[str]:
    if not isinstance(solution_text, str):
        return None
    boxed = _find_last_boxed(solution_text)
    if boxed:
        return boxed.strip()
    math_matches = re.findall(r"\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", solution_text, flags=re.DOTALL)
    if math_matches:
        flat = [m for group in math_matches for m in group if m]
        if flat:
            return flat[-1].strip()
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', solution_text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None

def load_gsm8k(split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info(f"Loading gsm8k[{split}] limit={limit}...")
    try:
        ds = datasets.load_dataset("openai/gsm8k", "main", split=split)
    except Exception as e:
        logger.error(f"Failed to load gsm8k: {e}")
        return []
    ds = _apply_limit(ds, limit, seed)

    examples = []
    parse_errors = 0
    for i, item in enumerate(ds):
        target = parse_gsm8k_target(item.get("answer", ""))
        parse_error = target is None
        if parse_error:
            if parse_errors < 3:
                logger.warning(f"gsm8k parse_error at idx={i}")
            elif parse_errors == 3:
                logger.warning("gsm8k parse_error warnings suppressed after 3 occurrences.")
            parse_errors += 1
        examples.append({
            "id": f"gsm8k_{split}_{i}",
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "target": target,
            "parse_error": parse_error,
        })
    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_gsm_plus(split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info(f"Loading gsm_plus[{split}] limit={limit}...")
    try:
        ds = datasets.load_dataset("qintongli/GSM-Plus", split=split)
    except Exception as e:
        logger.error(f"Failed to load gsm_plus: {e}")
        return []
    ds = _apply_limit(ds, limit, seed)

    examples = []
    parse_errors = 0
    for i, item in enumerate(ds):
        target = parse_gsm_plus_target(item)
        parse_error = target is None
        if parse_error:
            if parse_errors < 3:
                logger.warning(f"gsm_plus parse_error at idx={i}")
            elif parse_errors == 3:
                logger.warning("gsm_plus parse_error warnings suppressed after 3 occurrences.")
            parse_errors += 1
        examples.append({
            "id": f"gsm_plus_{split}_{i}",
            "question": item.get("question", ""),
            "answer": item.get("solution", ""),
            "target": target,
            "parse_error": parse_error,
        })
    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_hendrycks_math(split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info(f"Loading hendrycks_math[{split}] limit={limit}...")
    datasets_list = []
    for cfg in _HENDRYCKS_CONFIGS:
        try:
            ds = datasets.load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
            ds = ds.add_column("subject", [cfg] * len(ds))
            datasets_list.append(ds)
        except Exception as e:
            logger.warning(f"Failed to load hendrycks_math/{cfg}: {e}")
    if not datasets_list:
        logger.error("Failed to load any hendrycks_math configs.")
        return []
    ds_all = datasets.concatenate_datasets(datasets_list)
    ds_all = _apply_limit(ds_all, limit, seed)

    examples = []
    parse_errors = 0
    for i, item in enumerate(ds_all):
        target = parse_hendrycks_target(item.get("solution", ""))
        parse_error = target is None
        if parse_error:
            if parse_errors < 3:
                logger.warning(f"hendrycks_math parse_error at idx={i}")
            elif parse_errors == 3:
                logger.warning("hendrycks_math parse_error warnings suppressed after 3 occurrences.")
            parse_errors += 1
        examples.append({
            "id": f"hendrycks_math_{split}_{i}",
            "question": item.get("problem", ""),
            "answer": item.get("solution", ""),
            "target": target,
            "parse_error": parse_error,
            "subject": item.get("subject", ""),
        })
    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_dataset_records(dataset_name: str, split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    name = dataset_name.lower()
    if name in {"gsm8k", "openai/gsm8k"}:
        return load_gsm8k(split=split, limit=limit, seed=seed)
    if name in {"gsm_plus", "gsm-plus", "qintongli/gsm-plus"}:
        return load_gsm_plus(split=split, limit=limit, seed=seed)
    if name in {"hendrycks_math", "math", "eleutherai/hendrycks_math"}:
        return load_hendrycks_math(split=split, limit=limit, seed=seed)
    logger.error(f"Unknown dataset: {dataset_name}")
    return []
