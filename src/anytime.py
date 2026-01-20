import math
import collections
from typing import Optional, Dict, Any
import numpy as np
from .models import ModelRunner
from .policies import Policy, make_prompt, ContextConfig, bucketize_context, BwKShadowPricePolicy
from .profiler import LatencyProfiler
from .scoring import (
    get_answer_type,
    extract_candidate_answer,
    normalize_answer_for_candidates,
    evaluate_prediction,
)

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

def _empirical_bernstein_eps(p_hat: float, n: float, delta: float) -> float:
    if n <= 1:
        return float("inf")
    delta = max(float(delta), 1e-12)
    v_hat = p_hat * (1.0 - p_hat)
    log_term = math.log(3.0 / delta)
    return math.sqrt(2.0 * v_hat * log_term / n) + (3.0 * log_term) / max(1.0, n - 1.0)

def _compute_weighted_counts(
    steps: list,
    t: int,
    window: Optional[int] = None,
    discount: Optional[float] = None,
    current_step: Optional[dict] = None,
) -> tuple[dict, float]:
    counts: Dict[str, float] = collections.defaultdict(float)
    total_weight = 0.0
    for step in steps:
        ans_val = step.get("raw_val")
        if ans_val is None:
            continue
        step_t = int(step.get("t", 0))
        if window is not None and step_t < (t - int(window) + 1):
            continue
        weight = 1.0
        if discount is not None:
            weight = float(discount) ** max(0, t - step_t)
        counts[str(ans_val)] += weight
        total_weight += weight
    if current_step:
        ans_val = current_step.get("raw_val")
        if ans_val is not None:
            step_t = int(current_step.get("t", t))
            if window is None or step_t >= (t - int(window) + 1):
                weight = 1.0
                if discount is not None:
                    weight = float(discount) ** max(0, t - step_t)
                counts[str(ans_val)] += weight
                total_weight += weight
    return counts, total_weight

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
    bwk_policy: Optional[BwKShadowPricePolicy] = None,
    bwk_target_cost: Optional[float] = None,
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
    if allocation in {"bwk", "contextual_bwk"} and bwk_policy is not None:
        if not answer_counts:
            return policies[t % len(policies)]

        current_leader = max(answer_counts, key=answer_counts.get)
        best_score = -float("inf")
        best_p = None
        target_cost = float(bwk_target_cost) if bwk_target_cost is not None else 1.0

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
                p_correct = 0.5
            else:
                p_correct = mk / nk

            stats = policy_stats[p.name]
            if stats.get("nk", 0) > 0 and stats.get("cost_sum", 0.0) > 0:
                avg_cost = stats["cost_sum"] / stats["nk"]
            else:
                avg_cost = target_cost
            normalized_cost = avg_cost / max(1.0, target_cost)
            score = bwk_policy.score(p_correct, normalized_cost)

            if score > best_score:
                best_score = score
                best_p = p
        return best_p

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
    bound_method: str = "hoeffding",
    bound_window: int = None,
    bound_discount: float = None,
    seed: int = 42,
    batch_size: int = 1,
    allow_unseeded_batch: bool = False,
    ucb_window: int = None,
    ucb_discount: float = None,
    prompt_cost: float = 1.0,
    completion_cost: float = 1.0,
    context_config: Optional[object] = None,
    context_policy: Optional[Policy] = None,
    context_state: Optional[Dict[str, Any]] = None,
    bwk_lambda_init: float = 0.01,
    bwk_eta: float = 0.01,
    bwk_target_cost: Optional[float] = None,
    safety_valve: bool = False,
    safety_n: Optional[int] = None,
    safety_allocation: str = "uniform",
    safety_max_cost: Optional[float] = None,
    profiler: Optional[LatencyProfiler] = None,
) -> dict:
    """
    Runs Anytime Self-Consistency with bandits and stopping.
    If batch_size > 1, sampling happens in batches before re-evaluating the stop rule.
    """

    allocation = (allocation or "ucb").lower()
    bound_method = (bound_method or "hoeffding").lower()
    if bound_window is not None and int(bound_window) <= 0:
        bound_window = None
    if bound_discount is not None:
        if not (0.0 < float(bound_discount) < 1.0):
            bound_discount = None

    context_cfg = _resolve_context_config(context_config)
    context_key = "default"
    context_features: Dict[str, Any] = {}
    if context_cfg is not None:
        context_prompt = example.get("question", "")
        if context_policy is not None:
            context_prompt = make_prompt(context_policy, example.get("question", ""))
        mode = (context_cfg.mode or "length").lower()
        use_hidden_state = mode in {"hidden_state", "embedding"}
        context_info = model.get_prompt_context(context_prompt, use_hidden_state=use_hidden_state)
        context_key = bucketize_context(
            int(context_info.get("prompt_tokens") or 0),
            context_info.get("embedding_norm"),
            context_cfg,
        )
        context_features = {
            "prompt_tokens": context_info.get("prompt_tokens"),
            "embedding_norm": context_info.get("embedding_norm"),
            "context_mode": mode,
        }
    
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
    policy_stats = {
        p.name: {"nk": 0, "mk": 0, "rewards": [], "cost_sum": 0.0}
        for p in policies
    }
    
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
    
    bwk_policy = None
    if allocation in {"bwk", "contextual_bwk"}:
        if allocation == "contextual_bwk":
            if context_state is None:
                context_state = {}
            bwk_map = context_state.setdefault("bwk_policies", {})
            bwk_policy = bwk_map.get(context_key)
            if bwk_policy is None:
                bwk_policy = BwKShadowPricePolicy(lambda_init=bwk_lambda_init, eta=bwk_eta)
                bwk_map[context_key] = bwk_policy
        else:
            bwk_policy = BwKShadowPricePolicy(lambda_init=bwk_lambda_init, eta=bwk_eta)

    if bwk_policy is not None and bwk_target_cost is None:
        expected_costs = []
        for p in policies:
            est_prompt = make_prompt(p, example.get("question", ""))
            prompt_info = model.get_prompt_context(est_prompt, use_hidden_state=False)
            est_cost = (
                float(prompt_cost) * float(prompt_info.get("prompt_tokens") or 0)
                + float(completion_cost) * float(getattr(p, "max_new_tokens", model.max_new_tokens))
            )
            expected_costs.append(est_cost)
        if expected_costs:
            bwk_target_cost = float(np.mean(expected_costs))

    t = 0
    batch_id = 0
    stop_triggered = False
    final_pred = None
    final_pred_text = None
    safety_triggered = False
    safety_samples = 0
    safety_reason = None
    
    answer_type = get_answer_type(example)
    choices = example.get("choices")
    choice_labels = example.get("choice_labels")

    budget_cost = float(budget_tokens)
    while total_cost < budget_cost:
        batch_id += 1

        # 1. Select Policy
        if profiler:
            with profiler.track("bandit"):
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
                    bwk_policy=bwk_policy,
                    bwk_target_cost=bwk_target_cost,
                )
        else:
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
                bwk_policy=bwk_policy,
                bwk_target_cost=bwk_target_cost,
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
            if profiler:
                with profiler.track("sampling", use_cuda=True):
                    results = model.generate_batch(
                        prompts,
                        temperature=getattr(policy, 'temperature', 0.7),
                        top_p=getattr(policy, 'top_p', 1.0),
                        top_k=getattr(policy, 'top_k', 50),
                        do_sample=True,
                        seeds=batch_seeds,
                    )
            else:
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
                if profiler:
                    with profiler.track("sampling", use_cuda=True):
                        res = model.generate(
                            prompt,
                            temperature=getattr(policy, 'temperature', 0.7),
                            top_p=getattr(policy, 'top_p', 1.0),
                            top_k=getattr(policy, 'top_k', 50),
                            do_sample=True,
                            seed=seeds[i],
                        )
                else:
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
            step_cost = (
                float(prompt_cost) * res.get("prompt_tokens", 0)
                + float(completion_cost) * res.get("completion_tokens", 0)
            )
            total_cost += step_cost
            total_time += res["time_s"]

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

            ans_key = str(ans_val)
            if ans_val is not None:
                answer_counts[ans_key] += 1
                answer_to_val[ans_key] = ans_val
                if ans_key not in answer_to_text:
                    answer_to_text[ans_key] = candidate_text

            policy_stats[policy.name]["nk"] += 1
            policy_stats[policy.name]["cost_sum"] += step_cost
            if bwk_policy is not None and bwk_target_cost is not None:
                bwk_policy.update_price(step_cost, bwk_target_cost)

            if profiler:
                with profiler.track("bandit"):
                    counts_for_bound = answer_counts
                    effective_t = float(t)
                    if bound_window is not None or bound_discount is not None:
                        counts_for_bound, effective_t = _compute_weighted_counts(
                            steps,
                            t,
                            window=bound_window,
                            discount=bound_discount,
                            current_step={"t": t, "raw_val": ans_val},
                        )

                    if not counts_for_bound:
                        c1, c2 = 0.0, 0.0
                        leading_ans = None
                    else:
                        sorted_counts = sorted(counts_for_bound.items(), key=lambda x: x[1], reverse=True)
                        c1 = float(sorted_counts[0][1])
                        leading_ans = sorted_counts[0][0]
                        if len(sorted_counts) > 1:
                            c2 = float(sorted_counts[1][1])
                        else:
                            c2 = 0.0

                    if bound_method in {"empirical_bernstein", "bernstein", "eb"}:
                        if effective_t > 0:
                            p1 = c1 / effective_t
                            p2 = c2 / effective_t
                            eps1 = _empirical_bernstein_eps(p1, effective_t, delta / 2.0)
                            eps2 = _empirical_bernstein_eps(p2, effective_t, delta / 2.0)
                            threshold = effective_t * (eps1 + eps2)
                        else:
                            threshold = float("inf")
                    else:
                        threshold = math.sqrt(2 * effective_t * math.log(1 / delta)) if effective_t > 0 else 0.0
                    margin = c1 - c2
            else:
                counts_for_bound = answer_counts
                effective_t = float(t)
                if bound_window is not None or bound_discount is not None:
                    counts_for_bound, effective_t = _compute_weighted_counts(
                        steps,
                        t,
                        window=bound_window,
                        discount=bound_discount,
                        current_step={"t": t, "raw_val": ans_val},
                    )

                if not counts_for_bound:
                    c1, c2 = 0.0, 0.0
                    leading_ans = None
                else:
                    sorted_counts = sorted(counts_for_bound.items(), key=lambda x: x[1], reverse=True)
                    c1 = float(sorted_counts[0][1])
                    leading_ans = sorted_counts[0][0]
                    if len(sorted_counts) > 1:
                        c2 = float(sorted_counts[1][1])
                    else:
                        c2 = 0.0

                if bound_method in {"empirical_bernstein", "bernstein", "eb"}:
                    if effective_t > 0:
                        p1 = c1 / effective_t
                        p2 = c2 / effective_t
                        eps1 = _empirical_bernstein_eps(p1, effective_t, delta / 2.0)
                        eps2 = _empirical_bernstein_eps(p2, effective_t, delta / 2.0)
                        threshold = effective_t * (eps1 + eps2)
                    else:
                        threshold = float("inf")
                else:
                    threshold = math.sqrt(2 * effective_t * math.log(1 / delta)) if effective_t > 0 else 0.0
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
                "cost": step_cost,
                "total_cost": total_cost,
                "c1": c1,
                "c2": c2,
                "stop": False,
                "margin": margin,
                "threshold": threshold,
                "effective_t": effective_t,
                "bound_method": bound_method,
                "context_key": context_key,
                "shadow_price": bwk_policy.lambda_price if bwk_policy is not None else None,
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

    if not stop_triggered and safety_valve:
        safety_triggered = True
        target_safety_n = safety_n if safety_n is not None else max(min_samples, len(policies) * 2)
        max_cost = float(safety_max_cost) if safety_max_cost is not None else float("inf")
        safety_reason = "budget_exhausted" if total_cost >= budget_cost else "no_stop"

        while t < target_safety_n and total_cost < max_cost:
            if profiler:
                with profiler.track("bandit"):
                    safety_policy = _select_policy(
                        policies,
                        t,
                        safety_allocation,
                        answer_counts,
                        steps,
                        policy_stats,
                        rng,
                        ucb_window=ucb_window,
                        ucb_discount=ucb_discount,
                        bwk_policy=bwk_policy,
                        bwk_target_cost=bwk_target_cost,
                    )
            else:
                safety_policy = _select_policy(
                    policies,
                    t,
                    safety_allocation,
                    answer_counts,
                    steps,
                    policy_stats,
                    rng,
                    ucb_window=ucb_window,
                    ucb_discount=ucb_discount,
                    bwk_policy=bwk_policy,
                    bwk_target_cost=bwk_target_cost,
                )
            prompt = make_prompt(safety_policy, example.get("question", ""))
            if profiler:
                with profiler.track("sampling", use_cuda=True):
                    res = model.generate(
                        prompt,
                        temperature=getattr(safety_policy, "temperature", 0.7),
                        top_p=getattr(safety_policy, "top_p", 1.0),
                        top_k=getattr(safety_policy, "top_k", 50),
                        do_sample=True,
                        seed=seed + (t + 1) * 100,
                    )
            else:
                res = model.generate(
                    prompt,
                    temperature=getattr(safety_policy, "temperature", 0.7),
                    top_p=getattr(safety_policy, "top_p", 1.0),
                    top_k=getattr(safety_policy, "top_k", 50),
                    do_sample=True,
                    seed=seed + (t + 1) * 100,
                )

            t += 1
            safety_samples += 1
            total_tokens += res["total_tokens"]
            total_prompt_tokens += res.get("prompt_tokens", 0)
            total_completion_tokens += res.get("completion_tokens", 0)
            step_cost = (
                float(prompt_cost) * res.get("prompt_tokens", 0)
                + float(completion_cost) * res.get("completion_tokens", 0)
            )
            total_cost += step_cost
            total_time += res["time_s"]

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

            ans_key = str(ans_val)
            if ans_val is not None:
                answer_counts[ans_key] += 1
                answer_to_val[ans_key] = ans_val
                if ans_key not in answer_to_text:
                    answer_to_text[ans_key] = candidate_text

            policy_stats[safety_policy.name]["nk"] += 1
            policy_stats[safety_policy.name]["cost_sum"] += step_cost
            if bwk_policy is not None and bwk_target_cost is not None:
                bwk_policy.update_price(step_cost, bwk_target_cost)

            if profiler:
                with profiler.track("bandit"):
                    counts_for_bound = answer_counts
                    effective_t = float(t)
                    if bound_window is not None or bound_discount is not None:
                        counts_for_bound, effective_t = _compute_weighted_counts(
                            steps,
                            t,
                            window=bound_window,
                            discount=bound_discount,
                            current_step={"t": t, "raw_val": ans_val},
                        )

                    if not counts_for_bound:
                        c1, c2 = 0.0, 0.0
                        leading_ans = None
                    else:
                        sorted_counts = sorted(counts_for_bound.items(), key=lambda x: x[1], reverse=True)
                        c1 = float(sorted_counts[0][1])
                        leading_ans = sorted_counts[0][0]
                        if len(sorted_counts) > 1:
                            c2 = float(sorted_counts[1][1])
                        else:
                            c2 = 0.0

                    if bound_method in {"empirical_bernstein", "bernstein", "eb"}:
                        if effective_t > 0:
                            p1 = c1 / effective_t
                            p2 = c2 / effective_t
                            eps1 = _empirical_bernstein_eps(p1, effective_t, delta / 2.0)
                            eps2 = _empirical_bernstein_eps(p2, effective_t, delta / 2.0)
                            threshold = effective_t * (eps1 + eps2)
                        else:
                            threshold = float("inf")
                    else:
                        threshold = math.sqrt(2 * effective_t * math.log(1 / delta)) if effective_t > 0 else 0.0
                    margin = c1 - c2
            else:
                counts_for_bound = answer_counts
                effective_t = float(t)
                if bound_window is not None or bound_discount is not None:
                    counts_for_bound, effective_t = _compute_weighted_counts(
                        steps,
                        t,
                        window=bound_window,
                        discount=bound_discount,
                        current_step={"t": t, "raw_val": ans_val},
                    )

                if not counts_for_bound:
                    c1, c2 = 0.0, 0.0
                    leading_ans = None
                else:
                    sorted_counts = sorted(counts_for_bound.items(), key=lambda x: x[1], reverse=True)
                    c1 = float(sorted_counts[0][1])
                    leading_ans = sorted_counts[0][0]
                    if len(sorted_counts) > 1:
                        c2 = float(sorted_counts[1][1])
                    else:
                        c2 = 0.0

                if bound_method in {"empirical_bernstein", "bernstein", "eb"}:
                    if effective_t > 0:
                        p1 = c1 / effective_t
                        p2 = c2 / effective_t
                        eps1 = _empirical_bernstein_eps(p1, effective_t, delta / 2.0)
                        eps2 = _empirical_bernstein_eps(p2, effective_t, delta / 2.0)
                        threshold = effective_t * (eps1 + eps2)
                    else:
                        threshold = float("inf")
                else:
                    threshold = math.sqrt(2 * effective_t * math.log(1 / delta)) if effective_t > 0 else 0.0
                margin = c1 - c2

            steps.append({
                "t": t,
                "batch_id": batch_id,
                "batch_pos": 0,
                "batch_size": 1,
                "policy": safety_policy.name,
                "answer": ans_key,
                "raw_val": ans_val,
                "tokens": res["total_tokens"],
                "cost": step_cost,
                "total_cost": total_cost,
                "c1": c1,
                "c2": c2,
                "stop": False,
                "margin": margin,
                "threshold": threshold,
                "effective_t": effective_t,
                "bound_method": bound_method,
                "context_key": context_key,
                "shadow_price": bwk_policy.lambda_price if bwk_policy is not None else None,
                "safety_valve": True,
            })

            should_stop = (margin >= threshold) and (t >= min_samples)
            steps[-1]["stop"] = should_stop
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
        "bound_method": bound_method,
        "bound_window": bound_window,
        "bound_discount": bound_discount,
        "pred": final_pred,
        "is_correct": is_correct,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "time_s": total_time,
        "context_key": context_key,
        "bwk_target_cost": bwk_target_cost,
        "bwk_lambda_final": bwk_policy.lambda_price if bwk_policy is not None else None,
        "safety_valve": safety_valve,
        "safety_triggered": safety_triggered,
        "safety_samples": safety_samples,
        "safety_reason": safety_reason,
        "steps": steps, # Detailed log
        "extra": {
            "num_candidates": t,
            "unique_candidate_frac": unique_frac,
            "context_features": context_features,
            "latency": profiler.summary() if profiler else None,
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
    profiler: Optional[LatencyProfiler] = None,
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
        if profiler:
            with profiler.track("bandit"):
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
        else:
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
            if profiler:
                with profiler.track("sampling", use_cuda=True):
                    results = model.generate_batch(
                        prompts,
                        temperature=getattr(policy, "temperature", 0.7),
                        top_p=getattr(policy, "top_p", 1.0),
                        top_k=getattr(policy, "top_k", 50),
                        do_sample=True,
                        seeds=batch_seeds,
                    )
            else:
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
                if profiler:
                    with profiler.track("sampling", use_cuda=True):
                        res = model.generate(
                            prompt,
                            temperature=getattr(policy, "temperature", 0.7),
                            top_p=getattr(policy, "top_p", 1.0),
                            top_k=getattr(policy, "top_k", 50),
                            do_sample=True,
                            seed=seeds[i],
                        )
                else:
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

            ans_key = str(ans_val)
            if ans_val is not None:
                answer_counts[ans_key] += 1
                answer_to_val[ans_key] = ans_val
                if ans_key not in answer_to_text:
                    answer_to_text[ans_key] = candidate_text

            policy_stats[policy.name]["nk"] += 1

            if profiler:
                with profiler.track("scoring"):
                    is_correct_sample = evaluate_prediction(res["text"], ans_val, example)
            else:
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
            "latency": profiler.summary() if profiler else None,
        },
    }
