import collections
import math
from typing import List, Dict, Any, Optional
import numpy as np

from .models import ModelRunner
from .policies import Policy, make_prompt
from .scoring import extract_final_answer, normalize_answer_for_candidates, compare_answer_values, score_candidate

def _gini(values: List[int]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total <= 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(sorted_vals, start=1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n

def _entropy_from_counts(counts: collections.Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent

def _majority_vote(candidates: List[Any]) -> Optional[Any]:
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    counter = collections.Counter(valid)
    top_ans, _ = counter.most_common(1)[0]
    return top_ans

def run_global_anytime_sc(
    model: ModelRunner,
    policy: Optional[Policy],
    examples: List[Dict[str, Any]],
    global_budget_tokens: int,
    init_k: int = 1,
    allocation_policy: str = "uniform",
    per_example_budget_tokens: Optional[int] = None,
    ucb_c: float = 1.0,
    max_samples_per_item: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 50,
    finalize: str = "majority",
    store_allocation_steps: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    allocation_policy = (allocation_policy or "uniform").lower()
    finalize = (finalize or "majority").lower()
    max_samples = None if max_samples_per_item is None else int(max_samples_per_item)
    if max_samples is not None and max_samples <= 0:
        max_samples = None
    if per_example_budget_tokens is not None:
        per_example_budget_tokens = int(per_example_budget_tokens)

    rng = np.random.RandomState(seed)

    prompts = []
    for ex in examples:
        if policy is None:
            prompts.append(ex["question"])
        else:
            prompts.append(make_prompt(policy, ex["question"]))

    states = []
    for _ in examples:
        states.append({
            "candidates": [],
            "candidate_scores": [],
            "counts": collections.Counter(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "time_s": 0.0,
            "n_samples": 0,
            "reward_sum": 0.0,
            "reward_n": 0,
        })

    total_tokens_global = 0
    allocation_idx = 0
    finish_idx = 0
    stop = False
    steps = []

    def sample_once(ex_idx: int, sample_idx: int) -> None:
        nonlocal total_tokens_global
        state = states[ex_idx]
        before_counts = state["counts"].copy()
        before_total = sum(before_counts.values())
        before_top = max(before_counts.values()) if before_counts else 0
        before_conf = (before_top / before_total) if before_total > 0 else 0.0
        before_entropy = _entropy_from_counts(before_counts) if before_counts else 0.0

        res = model.generate(
            prompts[ex_idx],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            seed=seed + ex_idx * 1000 + sample_idx,
        )
        total_tokens_global += res["total_tokens"]
        state["prompt_tokens"] += res["prompt_tokens"]
        state["completion_tokens"] += res["completion_tokens"]
        state["total_tokens"] += res["total_tokens"]
        state["time_s"] += res["time_s"]
        state["n_samples"] += 1

        ans_str = extract_final_answer(res["text"])
        ans_val = normalize_answer_for_candidates(ans_str)
        state["candidates"].append(ans_val)
        if finalize == "best_of":
            state["candidate_scores"].append(score_candidate(res["text"]))
        else:
            state["candidate_scores"].append(None)
        if ans_val is not None:
            state["counts"][ans_val] += 1

        after_counts = state["counts"]
        after_total = sum(after_counts.values())
        after_top = max(after_counts.values()) if after_counts else 0
        after_conf = (after_top / after_total) if after_total > 0 else 0.0
        after_entropy = _entropy_from_counts(after_counts) if after_counts else 0.0
        reward = max(0.0, after_conf - before_conf) + max(0.0, before_entropy - after_entropy)
        state["reward_sum"] += reward
        state["reward_n"] += 1

        steps.append({
            "t": len(steps) + 1,
            "example_id": examples[ex_idx].get("id"),
            "allocation": allocation_policy,
            "tokens_total": total_tokens_global,
            "n_samples": state["n_samples"],
            "reward": reward,
            "confidence": after_conf,
            "entropy": after_entropy,
        })

    def eligible_indices() -> List[int]:
        if max_samples is None:
            base = list(range(len(examples)))
        else:
            base = [i for i, s in enumerate(states) if s["n_samples"] < max_samples]
        if allocation_policy == "per_example_budget":
            if per_example_budget_tokens is None:
                target = max(1, int(global_budget_tokens / max(1, len(examples))))
            else:
                target = per_example_budget_tokens
            return [i for i in base if states[i]["total_tokens"] < target]
        return base

    def pick_uniform(idx_list: List[int]) -> int:
        nonlocal allocation_idx
        if not idx_list:
            return -1
        start = allocation_idx % len(examples)
        for offset in range(len(examples)):
            i = (start + offset) % len(examples)
            if i in idx_list:
                allocation_idx = i + 1
                return i
        return -1

    def pick_random(idx_list: List[int]) -> int:
        if not idx_list:
            return -1
        return rng.choice(idx_list)

    def pick_finish_one(idx_list: List[int]) -> int:
        nonlocal finish_idx
        if not idx_list:
            return -1
        start = finish_idx % len(examples)
        for offset in range(len(examples)):
            i = (start + offset) % len(examples)
            if i in idx_list:
                finish_idx = i
                return i
        return -1

    def pick_uncertainty(idx_list: List[int]) -> int:
        best = []
        best_margin = None
        best_entropy = None
        for i in idx_list:
            counts = states[i]["counts"]
            total = sum(counts.values())
            if total == 0:
                margin = 0.0
                ent = 1.0
            else:
                sorted_counts = sorted(counts.values(), reverse=True)
                top1 = sorted_counts[0]
                top2 = sorted_counts[1] if len(sorted_counts) > 1 else 0
                margin = (top1 - top2) / total
                ent = _entropy_from_counts(counts)
            if best_margin is None or margin < best_margin or (margin == best_margin and ent > best_entropy):
                best_margin = margin
                best_entropy = ent
                best = [i]
            elif margin == best_margin and ent == best_entropy:
                best.append(i)
        return rng.choice(best) if best else -1

    def pick_voi(idx_list: List[int]) -> int:
        best = []
        best_score = None
        best_entropy = None
        for i in idx_list:
            counts = states[i]["counts"]
            total = sum(counts.values())
            if total == 0:
                score = 1.0
                ent = 1.0
            else:
                sorted_counts = sorted(counts.values(), reverse=True)
                top1 = sorted_counts[0]
                top2 = sorted_counts[1] if len(sorted_counts) > 1 else 0
                confidence = top1 / total
                margin = top1 - top2
                prob_flip = (1.0 - confidence) if margin <= 1 else 0.0
                score = prob_flip * (1.0 - confidence)
                ent = _entropy_from_counts(counts)
            if best_score is None or score > best_score or (score == best_score and ent > best_entropy):
                best_score = score
                best_entropy = ent
                best = [i]
            elif score == best_score and ent == best_entropy:
                best.append(i)
        return rng.choice(best) if best else -1

    def pick_ucb(idx_list: List[int], t: int) -> int:
        best = []
        best_score = None
        for i in idx_list:
            state = states[i]
            n = state["reward_n"]
            if n == 0:
                score = float("inf")
            else:
                mean = state["reward_sum"] / n
                score = mean + ucb_c * math.sqrt(math.log(max(2, t)) / n)
            if best_score is None or score > best_score:
                best_score = score
                best = [i]
            elif score == best_score:
                best.append(i)
        return rng.choice(best) if best else -1

    # Initial samples
    for ex_idx in range(len(examples)):
        for k in range(init_k):
            if max_samples is not None and states[ex_idx]["n_samples"] >= max_samples:
                break
            sample_once(ex_idx, states[ex_idx]["n_samples"])
            if total_tokens_global >= global_budget_tokens:
                stop = True
                break
        if stop:
            break

    # Global allocation loop
    while total_tokens_global < global_budget_tokens:
        eligible = eligible_indices()
        if not eligible:
            break
        if allocation_policy == "uniform":
            pick = pick_uniform(eligible)
        elif allocation_policy == "random":
            pick = pick_random(eligible)
        elif allocation_policy == "finish_one":
            pick = pick_finish_one(eligible)
        elif allocation_policy in {"uncertainty", "entropy", "margin"}:
            pick = pick_uncertainty(eligible)
        elif allocation_policy in {"voi", "voi_lite", "value_of_information"}:
            pick = pick_voi(eligible)
        elif allocation_policy == "ucb":
            pick = pick_ucb(eligible, len(steps) + 1)
        else:
            pick = pick_uniform(eligible)
        if pick == -1:
            break
        sample_once(pick, states[pick]["n_samples"])
        if total_tokens_global >= global_budget_tokens:
            break

    results = []
    samples_per_item = [s["n_samples"] for s in states]
    alloc_summary = {
        "mean_samples": float(np.mean(samples_per_item)) if samples_per_item else 0.0,
        "std_samples": float(np.std(samples_per_item, ddof=1)) if len(samples_per_item) > 1 else 0.0,
        "min_samples": min(samples_per_item) if samples_per_item else 0,
        "max_samples": max(samples_per_item) if samples_per_item else 0,
        "gini": _gini(samples_per_item),
        "total_tokens_global": total_tokens_global,
    }
    if allocation_policy == "per_example_budget":
        alloc_summary["per_example_budget_tokens"] = per_example_budget_tokens or int(global_budget_tokens / max(1, len(examples)))

    for ex_idx, ex in enumerate(examples):
        state = states[ex_idx]
        candidates = state["candidates"]
        valid = [c for c in candidates if c is not None]
        unique_frac = len(set(valid)) / len(candidates) if candidates else 0.0
        if finalize == "best_of" and state["candidate_scores"]:
            scored = [
                (score, val)
                for score, val in zip(state["candidate_scores"], candidates)
                if score is not None
            ]
            if scored:
                best_score, best_val = max(scored, key=lambda x: x[0])
                pred = best_val
            else:
                pred = _majority_vote(candidates)
        else:
            pred = _majority_vote(candidates)

        is_correct = False
        if pred is not None:
            is_correct = compare_answer_values(pred, ex.get("target"))

        counts = state["counts"]
        total_valid = sum(counts.values())
        if total_valid > 0:
            sorted_counts = sorted(counts.values(), reverse=True)
            top1 = sorted_counts[0]
            top2 = sorted_counts[1] if len(sorted_counts) > 1 else 0
            margin = (top1 - top2) / total_valid
            confidence = top1 / total_valid
            ent = _entropy_from_counts(counts)
        else:
            margin = 0.0
            confidence = 0.0
            ent = 0.0

        results.append({
            "example_id": ex.get("id"),
            "method": "global_anytime_sc",
            "policy": policy.name if policy is not None else "raw",
            "pred": pred,
            "is_correct": is_correct,
            "prompt_tokens": state["prompt_tokens"],
            "completion_tokens": state["completion_tokens"],
            "total_tokens": state["total_tokens"],
            "time_s": state["time_s"],
            "global_budget_tokens": global_budget_tokens,
            "allocation": allocation_policy,
            "init_k": init_k,
            "max_samples_per_item": max_samples,
            "global_tokens_used": total_tokens_global,
            "extra": {
                "candidates": candidates,
                "candidate_scores": state["candidate_scores"],
                "unique_candidate_frac": unique_frac,
                "num_candidates": len(candidates),
                "n_used": len(candidates),
                "vote_margin": margin,
                "confidence": confidence,
                "entropy": ent,
                "finalize": finalize,
                "allocation_summary": alloc_summary,
            }
        })

    if store_allocation_steps and results:
        results[0]["extra"]["allocation_steps"] = steps

    return results
