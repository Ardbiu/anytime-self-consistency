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
    allow_unseeded_batch: bool = False
) -> dict:
    """
    Runs Anytime Self-Consistency with bandits and stopping.
    If batch_size > 1, sampling happens in batches before re-evaluating the stop rule.
    """
    
    # State tracking
    total_tokens = 0
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

    while total_tokens < budget_tokens:
        batch_id += 1

        # 1. Select Policy
        if t < len(policies):
            policy = policies[t % len(policies)]
        else:
            if allocation == "uniform":
                policy = policies[t % len(policies)]
            elif allocation in ["ucb", "thompson"]:
                if not answer_counts:
                    best_pol_idx = rng.randint(len(policies))
                    policy = policies[best_pol_idx]
                else:
                    current_leader = max(answer_counts, key=answer_counts.get)

                    best_ucb = -float('inf')
                    best_p = None
                    log_t = math.log(max(t, 1))

                    for p in policies:
                        stats = policy_stats[p.name]
                        nk = stats["nk"]
                        if nk == 0:
                            ucb = float('inf')
                        else:
                            mk = sum(1 for s in steps if s["policy"] == p.name and str(s["raw_val"]) == current_leader)
                            mean = mk / nk
                            exploration = math.sqrt(2 * log_t / nk)
                            ucb = mean + exploration

                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_p = p
                    policy = best_p
            else:
                policy = policies[t % len(policies)]

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
            total_tokens += res['total_tokens']
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
                "tokens": res['total_tokens'],
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
        "delta": delta,
        "allocation": allocation,
        "pred": final_pred,
        "is_correct": is_correct,
        "total_tokens": total_tokens,
        "time_s": total_time,
        "steps": steps, # Detailed log
        "extra": {
            "num_candidates": t,
            "unique_candidate_frac": unique_frac
        }
    }
