import collections
from typing import Optional
from .models import ModelRunner
from .policies import Policy, make_prompt
from .profiler import LatencyProfiler
from .scoring import (
    get_answer_type,
    extract_candidate_answer,
    normalize_answer_for_candidates,
    score_candidate,
    build_verifier_prompt,
    evaluate_prediction,
)

def run_greedy(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    seed: int = 42,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    """Run single greedy sample (or near greedy)."""
    if policy is None:
        prompt = example['question']
        policy_name = "raw"
    else:
        prompt = make_prompt(policy, example['question'])
        policy_name = policy.name
    
    if profiler:
        with profiler.track("sampling", use_cuda=True):
            res = model.generate(
                prompt,
                do_sample=False,
                seed=seed
            )
    else:
        res = model.generate(
            prompt,
            do_sample=False,
            seed=seed
        )
    
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    pred_text = res["text"]
    if profiler:
        with profiler.track("scoring"):
            candidate_text = extract_candidate_answer(
                pred_text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            pred_val = normalize_answer_for_candidates(
                candidate_text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            is_correct = evaluate_prediction(pred_text, pred_val, example)
    else:
        candidate_text = extract_candidate_answer(
            pred_text,
            answer_type=answer_type,
            choices=choices,
            choice_labels=choice_labels,
        )
        pred_val = normalize_answer_for_candidates(
            candidate_text,
            answer_type=answer_type,
            choices=choices,
            choice_labels=choice_labels,
        )
        is_correct = evaluate_prediction(pred_text, pred_val, example)
    
    return {
        "example_id": example['id'],
        "method": "greedy",
        "policy": policy_name,
        "n": 1,
        "pred": candidate_text, # Raw string extracted (or code completion)
        "pred_val": pred_val, # Normalized value
        "is_correct": is_correct,
        "prompt_tokens": res['prompt_tokens'],
        "completion_tokens": res['completion_tokens'],
        "total_tokens": res['total_tokens'],
        "time_s": res['time_s'],
        "extra": {
            "full_text": pred_text,
            "latency": profiler.summary() if profiler else None,
        }
    }

_DEFAULT_SELF_CORRECTION_PROMPT = (
    "{question}\n\n"
    "Initial answer:\n{draft}\n\n"
    "Review the solution for mistakes and provide a corrected final answer only.\n"
    "Final:"
)

def run_self_correction(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    seed: int = 42,
    correction_prompt: Optional[str] = None,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    prompt = make_prompt(policy, example["question"])
    if profiler:
        with profiler.track("sampling", use_cuda=True):
            draft_res = model.generate(
                prompt,
                temperature=getattr(policy, "temperature", 0.7),
                top_p=getattr(policy, "top_p", 1.0),
                top_k=getattr(policy, "top_k", 50),
                do_sample=True,
                seed=seed,
            )
    else:
        draft_res = model.generate(
            prompt,
            temperature=getattr(policy, "temperature", 0.7),
            top_p=getattr(policy, "top_p", 1.0),
            top_k=getattr(policy, "top_k", 50),
            do_sample=True,
            seed=seed,
        )

    question_text = example.get("question", "")
    template = correction_prompt or _DEFAULT_SELF_CORRECTION_PROMPT
    correction_text = template.format(question=question_text, draft=draft_res["text"])
    if profiler:
        with profiler.track("sampling", use_cuda=True):
            corrected_res = model.generate(
                correction_text,
                do_sample=False,
                seed=seed + 1,
            )
    else:
        corrected_res = model.generate(
            correction_text,
            do_sample=False,
            seed=seed + 1,
        )

    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    if profiler:
        with profiler.track("scoring"):
            candidate_text = extract_candidate_answer(
                corrected_res["text"],
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            pred_val = normalize_answer_for_candidates(
                candidate_text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            is_correct = evaluate_prediction(corrected_res["text"], pred_val, example)
    else:
        candidate_text = extract_candidate_answer(
            corrected_res["text"],
            answer_type=answer_type,
            choices=choices,
            choice_labels=choice_labels,
        )
        pred_val = normalize_answer_for_candidates(
            candidate_text,
            answer_type=answer_type,
            choices=choices,
            choice_labels=choice_labels,
        )
        is_correct = evaluate_prediction(corrected_res["text"], pred_val, example)

    return {
        "example_id": example["id"],
        "method": "self_correction",
        "policy": policy.name,
        "n": 2,
        "pred": pred_val,
        "is_correct": is_correct,
        "prompt_tokens": draft_res["prompt_tokens"] + corrected_res["prompt_tokens"],
        "completion_tokens": draft_res["completion_tokens"] + corrected_res["completion_tokens"],
        "total_tokens": draft_res["total_tokens"] + corrected_res["total_tokens"],
        "time_s": draft_res["time_s"] + corrected_res["time_s"],
        "extra": {
            "draft_text": draft_res["text"],
            "correction_text": corrected_res["text"],
            "latency": profiler.summary() if profiler else None,
        },
    }

def run_speculative_decoding(
    model: ModelRunner,
    draft_model: ModelRunner,
    policy: Policy,
    example: dict,
    seed: int = 42,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    prompt = make_prompt(policy, example["question"])
    if profiler:
        with profiler.track("sampling", use_cuda=True):
            res = model.generate_speculative(
                prompt,
                draft_model,
                temperature=getattr(policy, "temperature", 0.7),
                top_p=getattr(policy, "top_p", 1.0),
                top_k=getattr(policy, "top_k", 50),
                do_sample=False,
                seed=seed,
            )
    else:
        res = model.generate_speculative(
            prompt,
            draft_model,
            temperature=getattr(policy, "temperature", 0.7),
            top_p=getattr(policy, "top_p", 1.0),
            top_k=getattr(policy, "top_k", 50),
            do_sample=False,
            seed=seed,
        )

    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    if profiler:
        with profiler.track("scoring"):
            candidate_text = extract_candidate_answer(
                res["text"],
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            pred_val = normalize_answer_for_candidates(
                candidate_text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            is_correct = evaluate_prediction(res["text"], pred_val, example)
    else:
        candidate_text = extract_candidate_answer(
            res["text"],
            answer_type=answer_type,
            choices=choices,
            choice_labels=choice_labels,
        )
        pred_val = normalize_answer_for_candidates(
            candidate_text,
            answer_type=answer_type,
            choices=choices,
            choice_labels=choice_labels,
        )
        is_correct = evaluate_prediction(res["text"], pred_val, example)

    return {
        "example_id": example["id"],
        "method": "speculative_decoding",
        "policy": policy.name,
        "n": 1,
        "pred": pred_val,
        "is_correct": is_correct,
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"],
        "total_tokens": res["total_tokens"],
        "time_s": res["time_s"],
        "extra": {
            "speculative_fallback": res.get("speculative_fallback", False),
            "draft_model": draft_model.model_name,
            "latency": profiler.summary() if profiler else None,
        },
    }

def run_medusa(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    seed: int = 42,
    medusa_heads: int = 4,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    prompt = make_prompt(policy, example["question"])
    if profiler:
        with profiler.track("sampling", use_cuda=True):
            res = model.generate_medusa(
                prompt,
                medusa_heads=medusa_heads,
                temperature=getattr(policy, "temperature", 0.7),
                top_p=getattr(policy, "top_p", 1.0),
                top_k=getattr(policy, "top_k", 50),
                do_sample=False,
                seed=seed,
            )
    else:
        res = model.generate_medusa(
            prompt,
            medusa_heads=medusa_heads,
            temperature=getattr(policy, "temperature", 0.7),
            top_p=getattr(policy, "top_p", 1.0),
            top_k=getattr(policy, "top_k", 50),
            do_sample=False,
            seed=seed,
        )

    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    if profiler:
        with profiler.track("scoring"):
            candidate_text = extract_candidate_answer(
                res["text"],
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            pred_val = normalize_answer_for_candidates(
                candidate_text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            is_correct = evaluate_prediction(res["text"], pred_val, example)
    else:
        candidate_text = extract_candidate_answer(
            res["text"],
            answer_type=answer_type,
            choices=choices,
            choice_labels=choice_labels,
        )
        pred_val = normalize_answer_for_candidates(
            candidate_text,
            answer_type=answer_type,
            choices=choices,
            choice_labels=choice_labels,
        )
        is_correct = evaluate_prediction(res["text"], pred_val, example)

    return {
        "example_id": example["id"],
        "method": "medusa",
        "policy": policy.name,
        "n": 1,
        "pred": pred_val,
        "is_correct": is_correct,
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"],
        "total_tokens": res["total_tokens"],
        "time_s": res["time_s"],
        "extra": {
            "medusa_heads": medusa_heads,
            "medusa_fallback": res.get("medusa_fallback", False),
            "latency": profiler.summary() if profiler else None,
        },
    }

def run_self_consistency(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    n: int,
    seed: int = 42,
    batched: bool = False,
    batched_seeded: bool = False,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    """Run Self-Consistency with Majority Vote.
    
    Args:
        batched: If True, use batched inference for faster generation.
                 Note: batched mode may not be exactly reproducible with seeds.
    """
    prompt = make_prompt(policy, example['question'])
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    
    candidates = []
    candidate_texts = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    
    if batched and hasattr(model, 'generate_batch'):
        # Batched generation for throughput
        prompts = [prompt] * n
        seeds = [seed + i for i in range(n)] if batched_seeded else None
        if profiler:
            with profiler.track("sampling", use_cuda=True):
                results = model.generate_batch(
                    prompts,
                    temperature=getattr(policy, 'temperature', 0.7),
                    top_p=getattr(policy, 'top_p', 1.0),
                    top_k=getattr(policy, 'top_k', 50),
                    do_sample=True,
                    seeds=seeds,  # None enables true batching
                )
        else:
            results = model.generate_batch(
                prompts,
                temperature=getattr(policy, 'temperature', 0.7),
                top_p=getattr(policy, 'top_p', 1.0),
                top_k=getattr(policy, 'top_k', 50),
                do_sample=True,
                seeds=seeds,  # None enables true batching
            )
        for res in results:
            total_prompt_tokens += res['prompt_tokens']
            total_completion_tokens += res['completion_tokens']
            total_time += res['time_s']
            
            if profiler:
                with profiler.track("scoring"):
                    candidate_text = extract_candidate_answer(
                        res["text"],
                        answer_type=answer_type,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
                    ans_val = normalize_answer_for_candidates(
                        candidate_text,
                        answer_type=answer_type,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
            else:
                candidate_text = extract_candidate_answer(
                    res["text"],
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                ans_val = normalize_answer_for_candidates(
                    candidate_text,
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
            candidates.append(ans_val)
            candidate_texts.append(candidate_text)
    else:
        # Serial generation (original behavior)
        for i in range(n):
            if profiler:
                with profiler.track("sampling", use_cuda=True):
                    res = model.generate(
                        prompt,
                        temperature=getattr(policy, 'temperature', 0.7),
                        top_p=getattr(policy, 'top_p', 1.0),
                        top_k=getattr(policy, 'top_k', 50),
                        do_sample=True,
                        seed=seed + i
                    )
            else:
                res = model.generate(
                    prompt,
                    temperature=getattr(policy, 'temperature', 0.7),
                    top_p=getattr(policy, 'top_p', 1.0),
                    top_k=getattr(policy, 'top_k', 50),
                    do_sample=True,
                    seed=seed + i
                )
            total_prompt_tokens += res['prompt_tokens']
            total_completion_tokens += res['completion_tokens']
            total_time += res['time_s']
            
            if profiler:
                with profiler.track("scoring"):
                    candidate_text = extract_candidate_answer(
                        res["text"],
                        answer_type=answer_type,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
                    ans_val = normalize_answer_for_candidates(
                        candidate_text,
                        answer_type=answer_type,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
            else:
                candidate_text = extract_candidate_answer(
                    res["text"],
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                ans_val = normalize_answer_for_candidates(
                    candidate_text,
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
            candidates.append(ans_val)
            candidate_texts.append(candidate_text)
        
    # Vote
    # Filter Nones if possible, but SC should count them maybe as "Error" class?
    valid_candidates = [c for c in candidates if c is not None]
    
    unique_candidates = set(valid_candidates)
    num_unique = len(unique_candidates)
    unique_frac = num_unique / len(candidates) if candidates else 0.0
    
    if not valid_candidates:
        top_ans = None
    else:
        counter = collections.Counter(valid_candidates)
        top_ans, count = counter.most_common(1)[0]
    
    is_correct = False
    pred_text = None
    if top_ans is not None:
        for idx, val in enumerate(candidates):
            if val == top_ans:
                pred_text = candidate_texts[idx]
                break
        if pred_text is None and candidate_texts:
            pred_text = candidate_texts[0]
        if profiler:
            with profiler.track("scoring"):
                is_correct = evaluate_prediction(pred_text or "", top_ans, example)
        else:
            is_correct = evaluate_prediction(pred_text or "", top_ans, example)

    return {
        "example_id": example['id'],
        "method": "self_consistency",
        "policy": policy.name,
        "n": n,
        "pred": top_ans,
        "is_correct": is_correct,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "time_s": total_time,
        "extra": {
            "candidates": candidates,
            "unique_candidate_frac": unique_frac,
            "num_candidates": len(candidates),
            "latency": profiler.summary() if profiler else None,
        }
    }

def run_best_of_n(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    n: int,
    seed: int = 42,
    batched: bool = False,
    batched_seeded: bool = False,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    """Run Best-of-N using a heuristic scorer.
    
    Args:
        batched: If True, use batched inference for faster generation.
    """
    prompt = make_prompt(policy, example['question'])
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    
    best_score = -1.0
    best_res = None
    best_ans_val = None
    best_text = None
    candidates = []
    all_texts = []
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    
    if batched and hasattr(model, 'generate_batch'):
        # Batched generation
        prompts = [prompt] * n
        seeds = [seed + i for i in range(n)] if batched_seeded else None
        if profiler:
            with profiler.track("sampling", use_cuda=True):
                results = model.generate_batch(
                    prompts,
                    temperature=getattr(policy, 'temperature', 0.7),
                    top_p=getattr(policy, 'top_p', 1.0),
                    top_k=getattr(policy, 'top_k', 50),
                    do_sample=True,
                    seeds=seeds,
                )
        else:
            results = model.generate_batch(
                prompts,
                temperature=getattr(policy, 'temperature', 0.7),
                top_p=getattr(policy, 'top_p', 1.0),
                top_k=getattr(policy, 'top_k', 50),
                do_sample=True,
                seeds=seeds,
            )
        for res in results:
            total_prompt_tokens += res['prompt_tokens']
            total_completion_tokens += res['completion_tokens']
            total_time += res['time_s']
            
            if profiler:
                with profiler.track("scoring"):
                    candidate_text = extract_candidate_answer(
                        res["text"],
                        answer_type=answer_type,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
                    sc = score_candidate(
                        res["text"],
                        answer_type=answer_type,
                        example=example,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
                    ans_val = normalize_answer_for_candidates(
                        candidate_text,
                        answer_type=answer_type,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
            else:
                candidate_text = extract_candidate_answer(
                    res["text"],
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                sc = score_candidate(
                    res["text"],
                    answer_type=answer_type,
                    example=example,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                ans_val = normalize_answer_for_candidates(
                    candidate_text,
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
            candidates.append(ans_val)
            all_texts.append(res["text"])

            if sc > best_score:
                best_score = sc
                best_res = res
                best_ans_val = ans_val
                best_text = res["text"]
    else:
        # Serial generation (original behavior)
        for i in range(n):
            if profiler:
                with profiler.track("sampling", use_cuda=True):
                    res = model.generate(
                        prompt,
                        temperature=getattr(policy, 'temperature', 0.7),
                        top_p=getattr(policy, 'top_p', 1.0),
                        top_k=getattr(policy, 'top_k', 50),
                        do_sample=True,
                        seed=seed + i
                    )
            else:
                res = model.generate(
                    prompt,
                    temperature=getattr(policy, 'temperature', 0.7),
                    top_p=getattr(policy, 'top_p', 1.0),
                    top_k=getattr(policy, 'top_k', 50),
                    do_sample=True,
                    seed=seed + i
                )
            total_prompt_tokens += res['prompt_tokens']
            total_completion_tokens += res['completion_tokens']
            total_time += res['time_s']
            
            if profiler:
                with profiler.track("scoring"):
                    candidate_text = extract_candidate_answer(
                        res["text"],
                        answer_type=answer_type,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
                    sc = score_candidate(
                        res["text"],
                        answer_type=answer_type,
                        example=example,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
                    ans_val = normalize_answer_for_candidates(
                        candidate_text,
                        answer_type=answer_type,
                        choices=choices,
                        choice_labels=choice_labels,
                    )
            else:
                candidate_text = extract_candidate_answer(
                    res["text"],
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                sc = score_candidate(
                    res["text"],
                    answer_type=answer_type,
                    example=example,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                ans_val = normalize_answer_for_candidates(
                    candidate_text,
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
            candidates.append(ans_val)

            if sc > best_score:
                best_score = sc
                best_res = res
                best_ans_val = ans_val
                best_text = res["text"]
    
    unique_candidates = set([c for c in candidates if c is not None])
    unique_frac = len(unique_candidates) / len(candidates) if candidates else 0.0
    
    # Check correctness of chosen one
    is_correct = False
    if best_ans_val is not None:
        pred_text = best_text or ""
        if profiler:
            with profiler.track("scoring"):
                is_correct = evaluate_prediction(pred_text, best_ans_val, example)
        else:
            is_correct = evaluate_prediction(pred_text, best_ans_val, example)

    return {
        "example_id": example['id'],
        "method": "best_of_n",
        "policy": policy.name,
        "n": n,
        "pred": best_ans_val,
        "is_correct": is_correct,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "time_s": total_time,
        "extra": {
            "best_score": best_score,
            "unique_candidate_frac": unique_frac,
            "num_candidates": len(candidates),
            "latency": profiler.summary() if profiler else None,
        }
    }

def run_best_of_n_verifier(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    n: int,
    verifier: ModelRunner,
    seed: int = 42,
    batched: bool = False,
    batched_seeded: bool = False,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    """Run Best-of-N using a learned verifier score (yes/no)."""
    prompt = make_prompt(policy, example["question"])
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")

    candidates = []
    candidate_texts = []
    verifier_scores = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0

    if batched and hasattr(model, "generate_batch"):
        prompts = [prompt] * n
        seeds = [seed + i for i in range(n)] if batched_seeded else None
        if profiler:
            with profiler.track("sampling", use_cuda=True):
                results = model.generate_batch(
                    prompts,
                    temperature=getattr(policy, "temperature", 0.7),
                    top_p=getattr(policy, "top_p", 1.0),
                    top_k=getattr(policy, "top_k", 50),
                    do_sample=True,
                    seeds=seeds,
                )
        else:
            results = model.generate_batch(
                prompts,
                temperature=getattr(policy, "temperature", 0.7),
                top_p=getattr(policy, "top_p", 1.0),
                top_k=getattr(policy, "top_k", 50),
                do_sample=True,
                seeds=seeds,
            )
        for res in results:
            total_prompt_tokens += res["prompt_tokens"]
            total_completion_tokens += res["completion_tokens"]
            total_time += res["time_s"]
            candidate_texts.append(res["text"])
    else:
        for i in range(n):
            if profiler:
                with profiler.track("sampling", use_cuda=True):
                    res = model.generate(
                        prompt,
                        temperature=getattr(policy, "temperature", 0.7),
                        top_p=getattr(policy, "top_p", 1.0),
                        top_k=getattr(policy, "top_k", 50),
                        do_sample=True,
                        seed=seed + i
                    )
            else:
                res = model.generate(
                    prompt,
                    temperature=getattr(policy, "temperature", 0.7),
                    top_p=getattr(policy, "top_p", 1.0),
                    top_k=getattr(policy, "top_k", 50),
                    do_sample=True,
                    seed=seed + i
                )
            total_prompt_tokens += res["prompt_tokens"]
            total_completion_tokens += res["completion_tokens"]
            total_time += res["time_s"]
            candidate_texts.append(res["text"])

    for text in candidate_texts:
        if profiler:
            with profiler.track("scoring"):
                candidate_text = extract_candidate_answer(
                    text,
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                ans_val = normalize_answer_for_candidates(
                    candidate_text,
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
            verifier_prompt = build_verifier_prompt(example["question"], text, candidate_text)
            with profiler.track("verification", use_cuda=True):
                score = verifier.score_candidate(verifier_prompt)
        else:
            candidate_text = extract_candidate_answer(
                text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            ans_val = normalize_answer_for_candidates(
                candidate_text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            verifier_prompt = build_verifier_prompt(example["question"], text, candidate_text)
            score = verifier.score_candidate(verifier_prompt)
        candidates.append(ans_val)
        verifier_scores.append(score)

    best_idx = int(max(range(len(verifier_scores)), key=lambda i: verifier_scores[i]))
    best_ans_val = candidates[best_idx]

    unique_candidates = set([c for c in candidates if c is not None])
    unique_frac = len(unique_candidates) / len(candidates) if candidates else 0.0

    is_correct = False
    if best_ans_val is not None:
        pred_text = candidate_texts[best_idx] if candidate_texts else ""
        if profiler:
            with profiler.track("scoring"):
                is_correct = evaluate_prediction(pred_text, best_ans_val, example)
        else:
            is_correct = evaluate_prediction(pred_text, best_ans_val, example)

    return {
        "example_id": example["id"],
        "method": "best_of_n_verifier",
        "policy": policy.name,
        "n": n,
        "pred": best_ans_val,
        "is_correct": is_correct,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "time_s": total_time,
        "extra": {
            "verifier_scores": verifier_scores,
            "unique_candidate_frac": unique_frac,
            "num_candidates": len(candidates),
            "latency": profiler.summary() if profiler else None,
        }
    }

def run_self_consistency_early_stop(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    max_n: int,
    stop_ratio: float = 0.6,
    min_samples: int = 2,
    stop_count: int = None,
    seed: int = 42,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    prompt = make_prompt(policy, example['question'])
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")

    candidates = []
    candidate_texts = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    counts = collections.Counter()
    stop_reason = "max_n"

    for i in range(max_n):
        if profiler:
            with profiler.track("sampling", use_cuda=True):
                res = model.generate(
                    prompt,
                    temperature=getattr(policy, 'temperature', 0.7),
                    top_p=getattr(policy, 'top_p', 1.0),
                    top_k=getattr(policy, 'top_k', 50),
                    do_sample=True,
                    seed=seed + i
                )
        else:
            res = model.generate(
                prompt,
                temperature=getattr(policy, 'temperature', 0.7),
                top_p=getattr(policy, 'top_p', 1.0),
                top_k=getattr(policy, 'top_k', 50),
                do_sample=True,
                seed=seed + i
            )
        total_prompt_tokens += res['prompt_tokens']
        total_completion_tokens += res['completion_tokens']
        total_time += res['time_s']

        if profiler:
            with profiler.track("scoring"):
                candidate_text = extract_candidate_answer(
                    res["text"],
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                ans_val = normalize_answer_for_candidates(
                    candidate_text,
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
        else:
            candidate_text = extract_candidate_answer(
                res["text"],
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            ans_val = normalize_answer_for_candidates(
                candidate_text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
        candidates.append(ans_val)
        candidate_texts.append(candidate_text)
        if ans_val is not None:
            counts[ans_val] += 1

        t = len(candidates)
        top_ans, top_cnt = (None, 0)
        if counts:
            top_ans, top_cnt = counts.most_common(1)[0]

        if t >= min_samples:
            if stop_count is not None and top_cnt >= stop_count:
                stop_reason = "stop_count"
                break
            if stop_ratio is not None and top_cnt / t >= stop_ratio:
                stop_reason = "stop_ratio"
                break

    valid_candidates = [c for c in candidates if c is not None]
    unique_candidates = set(valid_candidates)
    unique_frac = len(unique_candidates) / len(candidates) if candidates else 0.0

    if not valid_candidates:
        top_ans = None
    else:
        counter = collections.Counter(valid_candidates)
        top_ans, _ = counter.most_common(1)[0]

    is_correct = False
    pred_text = None
    if top_ans is not None:
        for idx, val in enumerate(candidates):
            if val == top_ans:
                pred_text = candidate_texts[idx]
                break
        if pred_text is None and candidate_texts:
            pred_text = candidate_texts[0]
        if profiler:
            with profiler.track("scoring"):
                is_correct = evaluate_prediction(pred_text or "", top_ans, example)
        else:
            is_correct = evaluate_prediction(pred_text or "", top_ans, example)

    return {
        "example_id": example['id'],
        "method": "self_consistency_early_stop",
        "policy": policy.name,
        "n": max_n,
        "pred": top_ans,
        "is_correct": is_correct,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "time_s": total_time,
        "extra": {
            "candidates": candidates,
            "unique_candidate_frac": unique_frac,
            "num_candidates": len(candidates),
            "n_used": len(candidates),
            "stop_reason": stop_reason,
            "stop_ratio": stop_ratio,
            "stop_count": stop_count,
            "min_samples": min_samples,
            "latency": profiler.summary() if profiler else None,
        }
    }

def run_best_of_n_early_stop(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    max_n: int,
    score_threshold: float = 0.7,
    min_samples: int = 1,
    seed: int = 42,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    prompt = make_prompt(policy, example['question'])
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")

    best_score = -1.0
    best_ans_val = None
    best_text = None
    candidates = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    stop_reason = "max_n"

    for i in range(max_n):
        if profiler:
            with profiler.track("sampling", use_cuda=True):
                res = model.generate(
                    prompt,
                    temperature=getattr(policy, 'temperature', 0.7),
                    top_p=getattr(policy, 'top_p', 1.0),
                    top_k=getattr(policy, 'top_k', 50),
                    do_sample=True,
                    seed=seed + i
                )
        else:
            res = model.generate(
                prompt,
                temperature=getattr(policy, 'temperature', 0.7),
                top_p=getattr(policy, 'top_p', 1.0),
                top_k=getattr(policy, 'top_k', 50),
                do_sample=True,
                seed=seed + i
            )
        total_prompt_tokens += res['prompt_tokens']
        total_completion_tokens += res['completion_tokens']
        total_time += res['time_s']

        if profiler:
            with profiler.track("scoring"):
                candidate_text = extract_candidate_answer(
                    res["text"],
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                sc = score_candidate(
                    res["text"],
                    answer_type=answer_type,
                    example=example,
                    choices=choices,
                    choice_labels=choice_labels,
                )
                ans_val = normalize_answer_for_candidates(
                    candidate_text,
                    answer_type=answer_type,
                    choices=choices,
                    choice_labels=choice_labels,
                )
        else:
            candidate_text = extract_candidate_answer(
                res["text"],
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
            sc = score_candidate(
                res["text"],
                answer_type=answer_type,
                example=example,
                choices=choices,
                choice_labels=choice_labels,
            )
            ans_val = normalize_answer_for_candidates(
                candidate_text,
                answer_type=answer_type,
                choices=choices,
                choice_labels=choice_labels,
            )
        candidates.append(ans_val)

        if sc > best_score:
            best_score = sc
            best_ans_val = ans_val
            best_text = res["text"]

        if (i + 1) >= min_samples and sc >= score_threshold:
            stop_reason = "score_threshold"
            break

    unique_candidates = set([c for c in candidates if c is not None])
    unique_frac = len(unique_candidates) / len(candidates) if candidates else 0.0

    is_correct = False
    if best_ans_val is not None:
        pred_text = best_text or ""
        if profiler:
            with profiler.track("scoring"):
                is_correct = evaluate_prediction(pred_text, best_ans_val, example)
        else:
            is_correct = evaluate_prediction(pred_text, best_ans_val, example)

    return {
        "example_id": example['id'],
        "method": "best_of_n_early_stop",
        "policy": policy.name,
        "n": max_n,
        "pred": best_ans_val,
        "is_correct": is_correct,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "time_s": total_time,
        "extra": {
            "best_score": best_score,
            "unique_candidate_frac": unique_frac,
            "num_candidates": len(candidates),
            "n_used": len(candidates),
            "stop_reason": stop_reason,
            "score_threshold": score_threshold,
            "min_samples": min_samples,
            "latency": profiler.summary() if profiler else None,
        }
    }
