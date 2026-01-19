import collections
from .models import ModelRunner
from .policies import Policy, make_prompt
from .scoring import (
    get_answer_type,
    extract_candidate_answer,
    normalize_answer_for_candidates,
    score_candidate,
    build_verifier_prompt,
    evaluate_prediction,
)

def run_greedy(model: ModelRunner, policy: Policy, example: dict, seed: int = 42) -> dict:
    """Run single greedy sample (or near greedy)."""
    if policy is None:
        prompt = example['question']
        policy_name = "raw"
    else:
        prompt = make_prompt(policy, example['question'])
        policy_name = policy.name
    
    res = model.generate(
        prompt,
        do_sample=False,
        seed=seed
    )
    
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    pred_text = res["text"]
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
        "extra": {"full_text": pred_text}
    }

def run_self_consistency(
    model: ModelRunner,
    policy: Policy,
    example: dict,
    n: int,
    seed: int = 42,
    batched: bool = False,
    batched_seeded: bool = False,
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
            "num_candidates": len(candidates)
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
            "num_candidates": len(candidates)
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
        candidates.append(ans_val)
        verifier_prompt = build_verifier_prompt(example["question"], text, candidate_text)
        score = verifier.score_candidate(verifier_prompt)
        verifier_scores.append(score)

    best_idx = int(max(range(len(verifier_scores)), key=lambda i: verifier_scores[i]))
    best_ans_val = candidates[best_idx]

    unique_candidates = set([c for c in candidates if c is not None])
    unique_frac = len(unique_candidates) / len(candidates) if candidates else 0.0

    is_correct = False
    if best_ans_val is not None:
        pred_text = candidate_texts[best_idx] if candidate_texts else ""
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
        }
    }
