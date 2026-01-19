import math
import collections
import numpy as np
from .models import ModelRunner
from .policies import Policy, make_prompt
from .scoring import (
    get_answer_type,
    extract_candidate_answer,
    normalize_answer_for_candidates,
    evaluate_prediction,
)

def _select_policy(
    policies: list[Policy],
    t: int,
    allocation: str,
    answer_counts: dict,
    steps: list[dict],
    policy_stats: dict,
    rng: np.random.RandomState,
    ucb_window: int = None,
    ucb_discount: float = None,
) -> Policy:
    allocation = (allocation or "ucb").lower()
    use_window = allocation in {"ucb_window", "ucb_sliding"}
    use_discount = allocation in {"ucb_discount", "ucb_discounted"}

    if ucb_window is not None and int(ucb_window) <= 0:
        ucb_window = None
    if ucb_discount is not None:
        if not (0.0 < float(ucb_discount) < 1.0):
            ucb_discount = None

    if t < len(policies):
        return policies[t % len(policies)]
    if allocation == "uniform":
        return policies[t % len(policies)]
    if allocation in {"ucb", "ucb_window", "ucb_sliding", "ucb_discount", "ucb_discounted", "thompson"}:
        if not answer_counts:
            best_pol_idx = rng.randint(len(policies))
            return policies[best_pol_idx]

        current_leader = max(answer_counts, key=answer_counts.get)
        best_ucb = -float("inf")
        best_p = None
        log_t = math.log(max(t, 1))

        for p in policies:
            if use_window or use_discount:
                nk = 0.0
                mk = 0.0
                for step in steps:
                    if step["policy"] != p.name:
                        continue
                    if use_window and ucb_window is not None:
                        if step["t"] < (t - int(ucb_window) + 1):
                            continue
                    weight = 1.0
                    if use_discount and ucb_discount is not None:
                        weight = float(ucb_discount) ** max(0, t - step["t"])
                    nk += weight
                    if str(step["raw_val"]) == current_leader:
                        mk += weight
            else:
                stats = policy_stats[p.name]
                nk = stats["nk"]
                mk = sum(
                    1 for s in steps
                    if s["policy"] == p.name and str(s["raw_val"]) == current_leader
                )
            if nk == 0:
                ucb = float("inf")
            else:
                mean = mk / nk
                exploration = math.sqrt(2 * log_t / nk)
                ucb = mean + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_p = p
        return best_p
    return policies[t % len(policies)]

def run_anytime_sc(
    model: ModelRunner, 
    policies: list[Policy], 
    example: dict, 
    budget_tokens: int,
    delta: float,
    allocation: str = "ucb",
    min_samples: int = 3,
    seed: int = 42,
    batch_size: int = 1,
    allow_unseeded_batch: bool = False,
    ucb_window: int = None,
    ucb_discount: float = None,
    prompt_cost: float = 1.0,
    completion_cost: float = 1.0,
) -> dict:
    """
    Runs Anytime Self-Consistency with bandits and stopping.
    If batch_size > 1, sampling happens in batches before re-evaluating the stop rule.
    """
    
    # State tracking
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    total_time = 0.0
    
    # Bandit stats per policy k:
    # nk = times sampled
    # mk = times it produced the current leading answer (dynamic reward)
    # We will compute rewards dynamically based on the GLOBAL top answer.
    policy_stats = {p.name: {"nk": 0, "mk": 0, "rewards": []} for p in policies}
    
    # Answer counts
    # We need to map float answers to counts
    # But floats are tricky keys. We verify equality carefully or strict mapping.
    # We'll use a string repr for keys, but value for checking.
    answer_counts = collections.defaultdict(int)
    answer_to_val = {}
    answer_to_text = {}
    
    steps = []
    
    # Initialize seeds
    rng = np.random.RandomState(seed)
    
    t = 0
    batch_id = 0
    stop_triggered = False
    final_pred = None
    final_pred_text = None
    
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")

    budget_cost = float(budget_tokens)
    while total_cost < budget_cost:
        batch_id += 1

        # 1. Select Policy
        policy = _select_policy(
            policies,
            t,
            allocation,
            answer_counts,
            steps,
            policy_stats,
            rng,
            ucb_window=ucb_window,
            ucb_discount=ucb_discount,
        )

        # 2. Sample (batch)
        prompt = make_prompt(policy, example['question'])
        effective_batch_size = max(1, int(batch_size))
        if t < len(policies):
            effective_batch_size = 1

        prompts = [prompt] * effective_batch_size
        seeds = [seed + (t + i + 1) * 100 for i in range(effective_batch_size)]

        if effective_batch_size > 1 and hasattr(model, "generate_batch"):
            batch_seeds = None if allow_unseeded_batch else seeds
            results = model.generate_batch(
                prompts,
                temperature=getattr(policy, 'temperature', 0.7),
                top_p=getattr(policy, 'top_p', 1.0),
                top_k=getattr(policy, 'top_k', 50),
                do_sample=True,
                seeds=batch_seeds,
            )
        else:
            results = []
            for i in range(effective_batch_size):
                res = model.generate(
                    prompt,
                    temperature=getattr(policy, 'temperature', 0.7),
                    top_p=getattr(policy, 'top_p', 1.0),
                    top_k=getattr(policy, 'top_k', 50),
                    do_sample=True,
                    seed=seeds[i],
                )
                results.append(res)

        for idx, res in enumerate(results):
            t += 1
            total_tokens += res["total_tokens"]
            total_prompt_tokens += res.get("prompt_tokens", 0)
            total_completion_tokens += res.get("completion_tokens", 0)
            total_cost += (
                float(prompt_cost) * res.get("prompt_tokens", 0)
                + float(completion_cost) * res.get("completion_tokens", 0)
            )
            total_time += res["time_s"]

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

            ans_key = str(ans_val)
            if ans_val is not None:
                answer_counts[ans_key] += 1
                answer_to_val[ans_key] = ans_val
                if ans_key not in answer_to_text:
                    answer_to_text[ans_key] = candidate_text

            policy_stats[policy.name]["nk"] += 1

            if not answer_counts:
                c1, c2 = 0, 0
                leading_ans = None
            else:
                sorted_counts = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
                c1 = sorted_counts[0][1]
                leading_ans = sorted_counts[0][0]
                if len(sorted_counts) > 1:
                    c2 = sorted_counts[1][1]
                else:
                    c2 = 0

            threshold = math.sqrt(2 * t * math.log(1/delta)) if t > 0 else 0
            margin = c1 - c2

            steps.append({
                "t": t,
                "batch_id": batch_id,
                "batch_pos": idx,
                "batch_size": effective_batch_size,
                "policy": policy.name,
                "answer": ans_key,
                "raw_val": ans_val,
                "tokens": res["total_tokens"],
                "cost": (
                    float(prompt_cost) * res.get("prompt_tokens", 0)
                    + float(completion_cost) * res.get("completion_tokens", 0)
                ),
                "total_cost": total_cost,
                "c1": c1,
                "c2": c2,
                "stop": False,
                "margin": margin,
                "threshold": threshold
            })

        if steps:
            last = steps[-1]
            margin = last["margin"]
            threshold = last["threshold"]
            should_stop = (margin >= threshold) and (t >= min_samples)
            last["stop"] = should_stop

            if should_stop:
                stop_triggered = True
                final_pred = answer_to_val.get(leading_ans)
                final_pred_text = answer_to_text.get(leading_ans)
                break
            
    # If exhausted budget without stopping
    if final_pred is None and answer_counts:
        best_key = max(answer_counts, key=answer_counts.get)
        final_pred = answer_to_val.get(best_key)
        final_pred_text = answer_to_text.get(best_key)
        
    is_correct = False
    if final_pred is not None:
        pred_text = final_pred_text or ""
        is_correct = evaluate_prediction(pred_text, final_pred, example)
         
    # Calculate diversity
    # t is total samples taken
    # answer_counts keys are the unique answers found
    unique_frac = len(answer_counts) / t if t > 0 else 0.0

    return {
        "example_id": example['id'],
        "method": "anytime_sc",
        "budget_tokens": budget_tokens,
        "budget_cost": budget_cost,
        "delta": delta,
        "allocation": allocation,
        "pred": final_pred,
        "is_correct": is_correct,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "time_s": total_time,
        "steps": steps, # Detailed log
        "extra": {
            "num_candidates": t,
            "unique_candidate_frac": unique_frac
        }
    }

def run_oracle_stopping(
    model: ModelRunner,
    policies: list[Policy],
    example: dict,
    budget_tokens: int,
    allocation: str = "ucb",
    seed: int = 42,
    batch_size: int = 1,
    allow_unseeded_batch: bool = False,
    ucb_window: int = None,
    ucb_discount: float = None,
    prompt_cost: float = 1.0,
    completion_cost: float = 1.0,
) -> dict:
    """
    Oracle stopping baseline: stop as soon as a correct sample appears.
    """
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    total_time = 0.0

    policy_stats = {p.name: {"nk": 0, "mk": 0, "rewards": []} for p in policies}
    answer_counts = collections.defaultdict(int)
    answer_to_val = {}
    answer_to_text = {}
    steps = []

    rng = np.random.RandomState(seed)
    t = 0
    batch_id = 0
    stop_triggered = False
    final_pred = None
    final_pred_text = None
    stop_reason = "budget"

    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")
    budget_cost = float(budget_tokens)

    while total_cost < budget_cost:
        batch_id += 1
        policy = _select_policy(
            policies,
            t,
            allocation,
            answer_counts,
            steps,
            policy_stats,
            rng,
            ucb_window=ucb_window,
            ucb_discount=ucb_discount,
        )

        prompt = make_prompt(policy, example["question"])
        effective_batch_size = max(1, int(batch_size))
        if t < len(policies):
            effective_batch_size = 1

        prompts = [prompt] * effective_batch_size
        seeds = [seed + (t + i + 1) * 100 for i in range(effective_batch_size)]

        if effective_batch_size > 1 and hasattr(model, "generate_batch"):
            batch_seeds = None if allow_unseeded_batch else seeds
            results = model.generate_batch(
                prompts,
                temperature=getattr(policy, "temperature", 0.7),
                top_p=getattr(policy, "top_p", 1.0),
                top_k=getattr(policy, "top_k", 50),
                do_sample=True,
                seeds=batch_seeds,
            )
        else:
            results = []
            for i in range(effective_batch_size):
                res = model.generate(
                    prompt,
                    temperature=getattr(policy, "temperature", 0.7),
                    top_p=getattr(policy, "top_p", 1.0),
                    top_k=getattr(policy, "top_k", 50),
                    do_sample=True,
                    seed=seeds[i],
                )
                results.append(res)

        for idx, res in enumerate(results):
            t += 1
            total_tokens += res["total_tokens"]
            total_prompt_tokens += res.get("prompt_tokens", 0)
            total_completion_tokens += res.get("completion_tokens", 0)
            total_cost += (
                float(prompt_cost) * res.get("prompt_tokens", 0)
                + float(completion_cost) * res.get("completion_tokens", 0)
            )
            total_time += res["time_s"]

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

            ans_key = str(ans_val)
            if ans_val is not None:
                answer_counts[ans_key] += 1
                answer_to_val[ans_key] = ans_val
                if ans_key not in answer_to_text:
                    answer_to_text[ans_key] = candidate_text

            policy_stats[policy.name]["nk"] += 1

            is_correct_sample = evaluate_prediction(res["text"], ans_val, example)
            step = {
                "t": t,
                "batch_id": batch_id,
                "batch_pos": idx,
                "batch_size": effective_batch_size,
                "policy": policy.name,
                "answer": ans_key,
                "raw_val": ans_val,
                "tokens": res["total_tokens"],
                "cost": (
                    float(prompt_cost) * res.get("prompt_tokens", 0)
                    + float(completion_cost) * res.get("completion_tokens", 0)
                ),
                "total_cost": total_cost,
                "stop": False,
                "oracle_hit": bool(is_correct_sample),
            }

            if is_correct_sample and final_pred is None:
                final_pred = ans_val
                final_pred_text = candidate_text
                step["stop"] = True
                stop_triggered = True
                stop_reason = "oracle_hit"

            steps.append(step)

            if stop_triggered:
                break

        if stop_triggered:
            break

    if final_pred is None and answer_counts:
        best_key = max(answer_counts, key=answer_counts.get)
        final_pred = answer_to_val.get(best_key)
        final_pred_text = answer_to_text.get(best_key)

    is_correct = False
    if final_pred is not None:
        pred_text = final_pred_text or ""
        is_correct = evaluate_prediction(pred_text, final_pred, example)

    unique_frac = len(answer_counts) / t if t > 0 else 0.0

    return {
        "example_id": example["id"],
        "method": "oracle_stopping",
        "budget_tokens": budget_tokens,
        "budget_cost": budget_cost,
        "allocation": allocation,
        "pred": final_pred,
        "is_correct": is_correct,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "time_s": total_time,
        "steps": steps,
        "extra": {
            "num_candidates": t,
            "unique_candidate_frac": unique_frac,
            "stop_reason": stop_reason,
            "oracle_hit": stop_triggered,
        },
    }
