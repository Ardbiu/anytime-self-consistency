import collections
import math
from typing import List, Dict, Any, Optional
import numpy as np

from .models import ModelRunner
from .policies import Policy, make_prompt, BwKShadowPricePolicy, ContextConfig, bucketize_context
from .scoring import (
    get_answer_type,
    extract_candidate_answer,
    normalize_answer_for_candidates,
    evaluate_prediction,
    score_candidate,
)

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

def _resolve_context_config(context_config: Optional[object]) -> Optional[ContextConfig]:
    if context_config is None:
        return None
    if isinstance(context_config, ContextConfig):
        return context_config
    if isinstance(context_config, dict):
        try:
            return ContextConfig(**context_config)
        except TypeError:
            return None
    return None

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
    context_config: Optional[object] = None,
    context_policy: Optional[Policy] = None,
    bwk_lambda_init: float = 0.01,
    bwk_eta: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Global allocation with a single knapsack budget (BwK setting).
    If allocation_policy == "bwk", uses a primal-dual shadow price
    (Agrawal & Devanur, 2014) to trade off utility vs cost.
    """
    allocation_policy = (allocation_policy or "uniform").lower()
    finalize = (finalize or "majority").lower()
    max_samples = None if max_samples_per_item is None else int(max_samples_per_item)
    if max_samples is not None and max_samples <= 0:
        max_samples = None
    if per_example_budget_tokens is not None:
        per_example_budget_tokens = int(per_example_budget_tokens)

    context_cfg = _resolve_context_config(context_config)
    context_mode = None if context_cfg is None else (context_cfg.mode or "length").lower()

    rng = np.random.RandomState(seed)

    prompts = []
    for ex in examples:
        if policy is None:
            prompts.append(ex["question"])
        else:
            prompts.append(make_prompt(policy, ex["question"]))

    context_keys = ["default"] * len(examples)
    context_features = [None] * len(examples)
    if context_cfg is not None:
        for idx, ex in enumerate(examples):
            if context_policy is None:
                context_prompt = prompts[idx]
            else:
                context_prompt = make_prompt(context_policy, ex.get("question", ""))
            use_hidden_state = context_mode in {"hidden_state", "embedding"}
            context_info = model.get_prompt_context(context_prompt, use_hidden_state=use_hidden_state)
            context_keys[idx] = bucketize_context(
                int(context_info.get("prompt_tokens") or 0),
                context_info.get("embedding_norm"),
                context_cfg,
            )
            context_features[idx] = {
                "prompt_tokens": context_info.get("prompt_tokens"),
                "embedding_norm": context_info.get("embedding_norm"),
                "context_mode": context_mode,
            }

    states = []
    for _ in examples:
        states.append({
            "candidates": [],
            "candidate_texts": [],
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

    total_tokens_consumed = 0
    allocation_idx = 0
    finish_idx = 0
    stop = False
    steps = []
    use_contextual_bwk = allocation_policy in {"contextual_bwk", "bwk_contextual"}
    bwk_policy = (
        BwKShadowPricePolicy(lambda_init=bwk_lambda_init, eta=bwk_eta)
        if allocation_policy == "bwk"
        else None
    )
    bwk_policies = {} if use_contextual_bwk else None
    target_tokens = float(global_budget_tokens) / max(1, len(examples))

    def _get_bwk_policy(ex_idx: int) -> Optional[BwKShadowPricePolicy]:
        if bwk_policy is not None:
            return bwk_policy
        if bwk_policies is None:
            return None
        key = context_keys[ex_idx]
        policy_obj = bwk_policies.get(key)
        if policy_obj is None:
            policy_obj = BwKShadowPricePolicy(lambda_init=bwk_lambda_init, eta=bwk_eta)
            bwk_policies[key] = policy_obj
        return policy_obj

    def sample_once(ex_idx: int, sample_idx: int) -> None:
        nonlocal total_tokens_consumed
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
        total_tokens_consumed += res["total_tokens"]
        state["prompt_tokens"] += res["prompt_tokens"]
        state["completion_tokens"] += res["completion_tokens"]
        state["total_tokens"] += res["total_tokens"]
        state["time_s"] += res["time_s"]
        state["n_samples"] += 1

        ex = examples[ex_idx]
        answer_type = get_answer_type(ex)
        choices = ex.get("choices")
        choice_labels = ex.get("choice_labels")
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
        state["candidates"].append(ans_val)
        state["candidate_texts"].append(res["text"])
        if finalize == "best_of":
            state["candidate_scores"].append(
                score_candidate(
                    res["text"],
                    answer_type=answer_type,
                    example=ex,
                    choices=choices,
                    choice_labels=choice_labels,
                )
            )
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

        bwk_for_ex = _get_bwk_policy(ex_idx)
        if bwk_for_ex is not None:
            bwk_for_ex.update_price(res["total_tokens"], target_tokens)

        steps.append({
            "t": len(steps) + 1,
            "example_id": examples[ex_idx].get("id"),
            "allocation": allocation_policy,
            "tokens_total": total_tokens_consumed,
            "n_samples": state["n_samples"],
            "reward": reward,
            "confidence": after_conf,
            "entropy": after_entropy,
            "shadow_price": bwk_for_ex.lambda_price if bwk_for_ex is not None else None,
            "context_key": context_keys[ex_idx],
        })

    def is_eligible(i: int) -> bool:
        if max_samples is not None and states[i]["n_samples"] >= max_samples:
            return False
        if allocation_policy == "per_example_budget":
            if per_example_budget_tokens is None:
                target_per_ex = max(1, int(global_budget_tokens / max(1, len(examples))))
            else:
                target_per_ex = per_example_budget_tokens
            if states[i]["total_tokens"] >= target_per_ex:
                return False
        return True

    def pick_uniform(idx_list: List[int]) -> int:
        nonlocal allocation_idx
        if not idx_list:
            return -1
        # idx_list is sorted. We want the first i in idx_list such that i >= allocation_idx % len(examples)
        # If none, we wrap around to idx_list[0].
        # Since idx_list is sorted, we can just iterate or binary search.
        # Linear scan is O(M), sufficient.
        
        threshold = allocation_idx % len(examples)
        
        # Fast path: prediction check
        # verification: find first >= threshold
        chosen = -1
        for i in idx_list:
            if i >= threshold:
                chosen = i
                break
        
        if chosen == -1:
            chosen = idx_list[0]
            
        allocation_idx = chosen + 1
        return chosen

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

    def pick_voc(idx_list: List[int]) -> int:
        """
        Value of computation heuristic:
        prioritize items where an extra sample is most likely to change the vote.
        """
        best = []
        best_score = None
        for i in idx_list:
            counts = states[i]["counts"]
            total = sum(counts.values())
            if total == 0:
                score = 1.0
            else:
                sorted_counts = sorted(counts.values(), reverse=True)
                top1 = sorted_counts[0]
                top2 = sorted_counts[1] if len(sorted_counts) > 1 else 0
                p1 = top1 / total
                p2 = top2 / total
                margin = max(0.0, p1 - p2)
                variance = p1 * (1.0 - p1)
                score = variance * (1.0 - margin)
            if best_score is None or score > best_score:
                best_score = score
                best = [i]
            elif score == best_score:
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

    def pick_bwk(idx_list: List[int]) -> int:
        if not idx_list or (bwk_policy is None and bwk_policies is None):
            return -1
        total_samples = sum(s["n_samples"] for s in states)
        avg_cost = (total_tokens_consumed / total_samples) if total_samples > 0 else target_tokens
        best = []
        best_score = None
        for i in idx_list:
            bwk_for_ex = _get_bwk_policy(i)
            if bwk_for_ex is None:
                continue
            state = states[i]
            counts = state["counts"]
            total = sum(counts.values())
            if total == 0:
                p_correct = 0.5
            else:
                top1 = max(counts.values())
                p_correct = top1 / total
            cost = (state["total_tokens"] / state["n_samples"]) if state["n_samples"] > 0 else avg_cost
            normalized_cost = cost / max(1.0, target_tokens)
            score = bwk_for_ex.score(p_correct, normalized_cost)
            if best_score is None or score > best_score:
                best_score = score
                best = [i]
            elif score == best_score:
                best.append(i)
        return rng.choice(best) if best else -1

    # Initial samples
    # We must maintain active_indices
    active_indices = set()
    for i in range(len(examples)):
        if is_eligible(i):
            active_indices.add(i)

    for ex_idx in range(len(examples)):
        for k in range(init_k):
            # Check eligibility before sampling (in case init_k > max_samples)
            if not is_eligible(ex_idx):
                if ex_idx in active_indices:
                    active_indices.remove(ex_idx)
                break
                
            sample_once(ex_idx, states[ex_idx]["n_samples"])
            
            # Post-sample check
            if not is_eligible(ex_idx):
                if ex_idx in active_indices:
                    active_indices.remove(ex_idx)
                # Break inner loop if this example is done
                break
                
            if total_tokens_consumed >= global_budget_tokens:
                stop = True
                break
        if stop:
            break

    # Global allocation loop
    while total_tokens_consumed < global_budget_tokens:
        if not active_indices:
            break
            
        # Convert to sorted list for pickers
        eligible = sorted(list(active_indices))
        
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
        elif allocation_policy in {"voc", "voc_anytime", "value_of_computation"}:
            pick = pick_voc(eligible)
        elif allocation_policy == "ucb":
            pick = pick_ucb(eligible, len(steps) + 1)
        elif allocation_policy in {"bwk", "contextual_bwk", "bwk_contextual"}:
            pick = pick_bwk(eligible)
        else:
            pick = pick_uniform(eligible)
            
        if pick == -1:
            break
            
        sample_once(pick, states[pick]["n_samples"])
        
        # Update eligibility
        if not is_eligible(pick):
            active_indices.remove(pick)
            
        if total_tokens_consumed >= global_budget_tokens:
            break

    results = []
    samples_per_item = [s["n_samples"] for s in states]
    alloc_summary = {
        "mean_samples": float(np.mean(samples_per_item)) if samples_per_item else 0.0,
        "std_samples": float(np.std(samples_per_item, ddof=1)) if len(samples_per_item) > 1 else 0.0,
        "min_samples": min(samples_per_item) if samples_per_item else 0,
        "max_samples": max(samples_per_item) if samples_per_item else 0,
        "gini": _gini(samples_per_item),
        "total_tokens_global": total_tokens_consumed,
    }
    if allocation_policy == "per_example_budget":
        alloc_summary["per_example_budget_tokens"] = per_example_budget_tokens or int(global_budget_tokens / max(1, len(examples)))
    if bwk_policy is not None:
        alloc_summary["shadow_price_final"] = bwk_policy.lambda_price
    if bwk_policies:
        alloc_summary["shadow_price_by_context"] = {
            key: policy.lambda_price for key, policy in bwk_policies.items()
        }

    for ex_idx, ex in enumerate(examples):
        state = states[ex_idx]
        candidates = state["candidates"]
        valid = [c for c in candidates if c is not None]
        unique_frac = len(set(valid)) / len(candidates) if candidates else 0.0
        pred = None
        pred_text = None
        if finalize == "best_of" and state["candidate_scores"]:
            scored = [
                (score, val, idx)
                for idx, (score, val) in enumerate(zip(state["candidate_scores"], candidates))
                if score is not None
            ]
            if scored:
                _, best_val, best_idx = max(scored, key=lambda x: x[0])
                pred = best_val
                if best_idx < len(state["candidate_texts"]):
                    pred_text = state["candidate_texts"][best_idx]
            else:
                pred = _majority_vote(candidates)
        else:
            pred = _majority_vote(candidates)

        if pred_text is None and pred is not None:
            for idx, val in enumerate(candidates):
                if val == pred:
                    if idx < len(state["candidate_texts"]):
                        pred_text = state["candidate_texts"][idx]
                    break
            if pred_text is None and state["candidate_texts"]:
                pred_text = state["candidate_texts"][-1]

        is_correct = False
        if pred is not None:
            is_correct = evaluate_prediction(pred_text or "", pred, ex)

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
            "global_tokens_used": total_tokens_consumed,
            "context_key": context_keys[ex_idx],
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
                "context_features": context_features[ex_idx],
            }
        })

    if store_allocation_steps and results:
        results[0]["extra"]["allocation_steps"] = steps

    return results
