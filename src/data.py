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

def _load_with_fallbacks(dataset_name: str, configs: List[Optional[str]], split: str):
    last_error = None
    for cfg in configs:
        try:
            if cfg:
                return datasets.load_dataset(dataset_name, cfg, split=split)
            return datasets.load_dataset(dataset_name, split=split)
        except Exception as e:
            last_error = e
            continue
    if last_error is not None:
        logger.error(f"Failed to load {dataset_name} with configs {configs}: {last_error}")
    return None

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

def _normalize_choice_label(label: Any) -> Optional[str]:
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

def _format_multiple_choice_question(question: str, choices: List[str], labels: List[str]) -> str:
    lines = [question.strip()]
    for label, text in zip(labels, choices):
        lines.append(f"{label}. {text}")
    return "\n".join(lines)

def _infer_function_name_from_code(code: str) -> Optional[str]:
    if not isinstance(code, str):
        return None
    # Use a standard regex and guard against malformed patterns in older envs.
    try:
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    except re.error:
        return None
    if match:
        return match.group(1)
    return None

def _infer_function_name_from_tests(test_list: List[str]) -> Optional[str]:
    for test in test_list or []:
        try:
            match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(test))
        except re.error:
            continue
        if match:
            return match.group(1)
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
            "answer_type": "numeric",
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
            "answer_type": "numeric",
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
            "answer_type": "numeric",
        })
    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_arc_challenge(split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info(f"Loading arc_challenge[{split}] limit={limit}...")
    try:
        ds = datasets.load_dataset("ai2_arc", "ARC-Challenge", split=split)
    except Exception as e:
        logger.error(f"Failed to load arc_challenge: {e}")
        return []
    ds = _apply_limit(ds, limit, seed)

    examples = []
    for i, item in enumerate(ds):
        question_obj = item.get("question") or {}
        question_text = question_obj.get("stem") if isinstance(question_obj, dict) else str(question_obj)
        choices_raw = item.get("choices") or question_obj.get("choices") or []
        choice_labels: List[str] = []
        choice_texts: List[str] = []

        if isinstance(choices_raw, dict):
            labels = choices_raw.get("label") or choices_raw.get("labels") or []
            texts = choices_raw.get("text") or choices_raw.get("texts") or []
            for label, text in zip(labels, texts):
                norm_label = _normalize_choice_label(label)
                if norm_label is None:
                    continue
                choice_labels.append(norm_label)
                choice_texts.append(str(text))
        elif isinstance(choices_raw, list):
            for choice in choices_raw:
                if not isinstance(choice, dict):
                    continue
                norm_label = _normalize_choice_label(choice.get("label"))
                if norm_label is None:
                    continue
                choice_labels.append(norm_label)
                choice_texts.append(str(choice.get("text", "")))

        if not choice_labels and choice_texts:
            choice_labels = [chr(ord("A") + idx) for idx in range(len(choice_texts))]

        formatted_question = _format_multiple_choice_question(question_text or "", choice_texts, choice_labels)
        target = _normalize_choice_label(item.get("answerKey") or item.get("answer"))

        examples.append({
            "id": f"arc_challenge_{split}_{i}",
            "question": formatted_question,
            "answer_type": "multiple_choice",
            "choices": choice_texts,
            "choice_labels": choice_labels,
            "target": target,
        })
    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_mmlu(split: str = "test", limit: Optional[int] = None, seed: int = 42, subject: Optional[str] = None) -> List[Dict[str, Any]]:
    subject = subject or "abstract_algebra"
    logger.info(f"Loading mmlu/{subject}[{split}] limit={limit}...")
    try:
        ds = datasets.load_dataset("cais/mmlu", subject, split=split)
    except Exception as e:
        logger.error(f"Failed to load mmlu/{subject}: {e}")
        return []
    ds = _apply_limit(ds, limit, seed)

    examples = []
    for i, item in enumerate(ds):
        question_text = str(item.get("question", ""))
        choices = item.get("choices") or []
        choices = [str(c) for c in choices]
        labels = [chr(ord("A") + idx) for idx in range(len(choices))]
        formatted_question = _format_multiple_choice_question(question_text, choices, labels)
        target = _normalize_choice_label(item.get("answer"))

        examples.append({
            "id": f"mmlu_{subject}_{split}_{i}",
            "question": formatted_question,
            "answer_type": "multiple_choice",
            "choices": choices,
            "choice_labels": labels,
            "target": target,
            "subject": subject,
        })
    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_gpqa(split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info(f"Loading gpqa[{split}] limit={limit}...")
    ds = _load_with_fallbacks("gpqa", ["gpqa_main", "main", None], split)
    if ds is None:
        return []
    ds = _apply_limit(ds, limit, seed)

    examples = []
    for i, item in enumerate(ds):
        question_text = str(item.get("question") or item.get("query") or item.get("stem") or item.get("prompt") or "")
        choice_labels: List[str] = []
        choice_texts: List[str] = []
        target = None

        if not choice_texts:
            choices_raw = item.get("choices") or item.get("options") or item.get("answers") or []
            if isinstance(choices_raw, dict):
                labels = choices_raw.get("label") or choices_raw.get("labels") or []
                texts = choices_raw.get("text") or choices_raw.get("texts") or []
                for label, text in zip(labels, texts):
                    norm_label = _normalize_choice_label(label)
                    if norm_label is None:
                        continue
                    choice_labels.append(norm_label)
                    choice_texts.append(str(text))
            elif isinstance(choices_raw, list):
                for choice in choices_raw:
                    if isinstance(choice, dict):
                        norm_label = _normalize_choice_label(choice.get("label") or choice.get("key"))
                        text = choice.get("text") or choice.get("content") or choice.get("value")
                        if text is None:
                            continue
                        if norm_label is None:
                            norm_label = _normalize_choice_label(len(choice_labels))
                        choice_labels.append(norm_label)
                        choice_texts.append(str(text))
                    else:
                        choice_texts.append(str(choice))

        incorrect = item.get("incorrect_answers") or item.get("incorrect") or []
        correct_text = item.get("correct_answer") or item.get("answer") or item.get("gold")
        if not choice_texts and incorrect and correct_text:
            choice_texts = [str(x) for x in incorrect] + [str(correct_text)]

        if choice_texts and not choice_labels:
            choice_labels = [chr(ord("A") + idx) for idx in range(len(choice_texts))]

        if target is None:
            raw_target = item.get("answer") or item.get("correct_answer") or item.get("answerKey")
            target = _normalize_choice_label(raw_target)

        if target is None and correct_text and choice_texts:
            for label, text in zip(choice_labels, choice_texts):
                if str(text).strip() == str(correct_text).strip():
                    target = label
                    break

        if choice_texts:
            formatted_question = _format_multiple_choice_question(question_text, choice_texts, choice_labels)
            answer_type = "multiple_choice"
        else:
            formatted_question = question_text
            answer_type = "text"

        examples.append({
            "id": f"gpqa_{split}_{i}",
            "question": formatted_question,
            "answer_type": answer_type,
            "choices": choice_texts,
            "choice_labels": choice_labels,
            "target": target if target is not None else (str(correct_text) if correct_text is not None else None),
        })

    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_bbh(
    split: str = "test",
    limit: Optional[int] = None,
    seed: int = 42,
    task: Optional[str] = None,
) -> List[Dict[str, Any]]:
    logger.info(f"Loading bbh[{split}] limit={limit} task={task}...")
    ds = _load_with_fallbacks("lukaemon/bbh", [task, None], split)
    if ds is None:
        ds = _load_with_fallbacks("bigbench", ["bbh", task], split)
    if ds is None:
        return []
    ds = _apply_limit(ds, limit, seed)

    examples = []
    for i, item in enumerate(ds):
        base_prompt = str(item.get("input") or item.get("question") or item.get("prompt") or "")
        few_shot = item.get("few_shot") or item.get("cot_prompt") or item.get("prefix")
        if few_shot:
            prompt = f"{few_shot}\n{base_prompt}"
        else:
            prompt = base_prompt

        choice_labels: List[str] = []
        choice_texts: List[str] = []
        choices_raw = item.get("choices") or item.get("options") or []
        if isinstance(choices_raw, list):
            for choice in choices_raw:
                if isinstance(choice, dict):
                    norm_label = _normalize_choice_label(choice.get("label") or choice.get("key"))
                    text = choice.get("text") or choice.get("content") or choice.get("value")
                    if text is None:
                        continue
                    if norm_label is None:
                        norm_label = _normalize_choice_label(len(choice_labels))
                    choice_labels.append(norm_label)
                    choice_texts.append(str(text))
                else:
                    choice_texts.append(str(choice))

        if choice_texts and not choice_labels:
            choice_labels = [chr(ord("A") + idx) for idx in range(len(choice_texts))]

        target = item.get("target") or item.get("answer") or item.get("label")
        if choice_texts:
            formatted_question = _format_multiple_choice_question(prompt, choice_texts, choice_labels)
            answer_type = "multiple_choice"
            target = _normalize_choice_label(target) if target is not None else None
        else:
            formatted_question = prompt
            answer_type = "text"

        examples.append({
            "id": f"bbh_{split}_{i}",
            "question": formatted_question,
            "answer_type": answer_type,
            "choices": choice_texts,
            "choice_labels": choice_labels,
            "target": str(target) if target is not None else None,
            "subject": item.get("task") or task or item.get("category"),
        })

    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_humaneval(split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info(f"Loading humaneval[{split}] limit={limit}...")
    try:
        ds = datasets.load_dataset("openai_humaneval", split=split)
    except Exception as e:
        logger.error(f"Failed to load humaneval: {e}")
        return []
    ds = _apply_limit(ds, limit, seed)

    examples = []
    for i, item in enumerate(ds):
        examples.append({
            "id": f"humaneval_{split}_{i}",
            "question": item.get("prompt", ""),
            "answer_type": "code",
            "code_task": "humaneval",
            "entry_point": item.get("entry_point"),
            "test": item.get("test"),
            "task_id": item.get("task_id"),
        })
    logger.info(f"Loaded {len(examples)} examples.")
    return examples

def load_mbpp(split: str = "test", limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info(f"Loading mbpp[{split}] limit={limit}...")
    try:
        ds = datasets.load_dataset("mbpp", split=split)
    except Exception as e:
        logger.error(f"Failed to load mbpp: {e}")
        return []
    ds = _apply_limit(ds, limit, seed)

    examples = []
    for i, item in enumerate(ds):
        prompt = item.get("prompt")
        if not prompt:
            prompt = str(item.get("text", ""))
            func_name = _infer_function_name_from_code(item.get("code", ""))
            if func_name is None:
                func_name = _infer_function_name_from_tests(item.get("test_list", []))
            if func_name:
                prompt = f"{prompt}\\nWrite a Python function named `{func_name}`."

        examples.append({
            "id": f"mbpp_{split}_{i}",
            "question": prompt,
            "answer_type": "code",
            "code_task": "mbpp",
            "test_list": item.get("test_list") or [],
            "test_setup_code": item.get("test_setup_code") or item.get("setup_code"),
            "task_id": item.get("task_id"),
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
    if name in {"arc_challenge", "arc", "ai2_arc/arc-challenge"}:
        return load_arc_challenge(split=split, limit=limit, seed=seed)
    if name.startswith("mmlu"):
        subject = None
        if ":" in dataset_name:
            subject = dataset_name.split(":", 1)[1]
        return load_mmlu(split=split, limit=limit, seed=seed, subject=subject)
    if name in {"gpqa", "gpqa_main", "gpqa-main"}:
        return load_gpqa(split=split, limit=limit, seed=seed)
    if name.startswith("bbh"):
        task = None
        if ":" in dataset_name:
            task = dataset_name.split(":", 1)[1]
        return load_bbh(split=split, limit=limit, seed=seed, task=task)
    if name in {"humaneval", "human_eval", "openai_humaneval", "openai/humaneval"}:
        return load_humaneval(split=split, limit=limit, seed=seed)
    if name in {"mbpp", "google/mbpp"}:
        return load_mbpp(split=split, limit=limit, seed=seed)
    logger.error(f"Unknown dataset: {dataset_name}")
    return []
