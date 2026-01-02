import math
import collections
import numpy as np
from .models import ModelRunner
from .policies import Policy, make_prompt
from .scoring import extract_final_answer, normalize_numeric_answer

def run_anytime_sc(
    model: ModelRunner, 
    policies: list[Policy], 
    example: dict, 
    budget_tokens: int,
    delta: float,
    allocation: str = "ucb",
    min_samples: int = 3,
    seed: int = 42
) -> dict:
    """
    Runs Anytime Self-Consistency with bandits and stopping.
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
    
    steps = []
    
    # Initialize seeds
    rng = np.random.RandomState(seed)
    
    t = 0
    stop_triggered = False
    final_pred = None
    
    while total_tokens < budget_tokens:
        t += 1
        
        # 1. Select Policy
        if t <= len(policies) * 1: # Force traverse all policies at least once
            policy = policies[(t - 1) % len(policies)]
        else:
            # Bandit Selection
            if allocation == "uniform":
                 policy = policies[t % len(policies)]
            elif allocation == "ucb" or allocation == "thompson":
                 # Calculate UCB for each
                 # Metric: Mean reward + C * sqrt(log(t) / nk)
                 # Reward definition: 1 if matches current Plurality, 0 otherwise? 
                 # Or 1 if matches current Majority?
                 # Problem: The "correct" answer changes as we sample.
                 # Solution: Re-evaluate historic rewards based on CURRENT top answer (a bit expensive but accurate)
                 # fast approx: just keep running average of "agreed with consensus at the time".
                 # Let's use the robust version: "Reward = 1 if answer == current_top_answer".
                 # But current_top_answer changes.
                 # Let's stick to the prompt spec: "mk (# times it produced the current leading answer)"
                 
                 # First identify current leader to compute stats
                 if not answer_counts:
                     best_pol_idx = rng.randint(len(policies))
                     policy = policies[best_pol_idx]
                 else:
                     # Find leader
                     current_leader = max(answer_counts, key=answer_counts.get)
                     
                     best_ucb = -float('inf')
                     best_p = None
                     
                     for p in policies:
                         stats = policy_stats[p.name]
                         nk = stats["nk"]
                         if nk == 0: 
                             ucb = float('inf')
                         else:
                             # Recompute mk based on current leader
                             # We need to store history of answers per policy to do this perfectly.
                             # Let's simplify: Estimate mk from stored history if we have it.
                             # If we don't store full history, we can't perfectly recompute mk.
                             # Let's store full simple history: list of (policy, answer_key)
                             
                             # Count how many times this policy produced current_leader
                             # This requires iterating our step history.
                             mk = sum(1 for s in steps if s['policy'] == p.name and str(s['raw_val']) == current_leader)
                             
                             mean = mk / nk
                             exploration = math.sqrt(2 * math.log(t) / nk)
                             ucb = mean + exploration
                         
                         if ucb > best_ucb:
                             best_ucb = ucb
                             best_p = p
                     policy = best_p

        # 2. Sample
        prompt = make_prompt(policy, example['question'])
        # Use simple seed increment
        step_seed = seed + t * 100
        
        # Check budget before generating? 
        # We don't know exact cost, but we can guess. If budget is very tight, we might overspend. 
        # That's acceptable for research code usually (soft limit).
        
        res = model.generate(
            prompt,
            temperature=policy.temperature,
            top_p=policy.top_p,
            do_sample=policy.do_sample,
            seed=step_seed
        )
        
        total_tokens += res['total_tokens']
        total_time += res['time_s']
        
        ans_str = extract_final_answer(res['text'])
        ans_val = normalize_numeric_answer(ans_str)
        
        # Update Stats
        ans_key = str(ans_val) # utilize string as key
        if ans_val is not None:
             answer_counts[ans_key] += 1
             answer_to_val[ans_key] = ans_val
        
        policy_stats[policy.name]["nk"] += 1
        
        # 3. Check Stopping Condition
        # Need top 2 counts
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
                
        # Margin check
        # bound = sqrt(2 * t * log(1/delta))? 
        # Actually usually it's (1/epsilon^2)*log(...) for Chernoff.
        # User requested: (c1 - c2) >= sqrt(2 * t * log(1/delta))
        # wait, if t is large, sqrt(t) grows. c1-c2 also grows.
        # Standard Hoeffding based stopping usually compares counts.
        # Let's implement the requested formula exactly.
        
        threshold = math.sqrt(2 * t * math.log(1/delta)) if t > 0 else 0
        margin = c1 - c2
        
        should_stop = (margin >= threshold) and (t >= min_samples)
        
        steps.append({
            "t": t,
            "policy": policy.name,
            "answer": ans_key,
            "raw_val": ans_val,
            "tokens": res['total_tokens'],
            "c1": c1,
            "c2": c2,
            "stop": should_stop,
            "margin": margin,
            "threshold": threshold
        })
        
        if should_stop:
            stop_triggered = True
            final_pred = answer_to_val.get(leading_ans)
            break
            
    # If exhausted budget without stopping
    if final_pred is None and answer_counts:
        final_pred = answer_to_val.get(max(answer_counts, key=answer_counts.get))
        
    is_correct = False
    if final_pred is not None and example['gold'] is not None:
         is_correct = abs(final_pred - example['gold']) < 1e-6
         
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
        "extra": {}
    }
