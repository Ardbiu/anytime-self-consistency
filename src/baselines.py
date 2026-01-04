import collections
from .models import ModelRunner
from .policies import Policy, make_prompt
from .scoring import extract_final_answer, normalize_answer_for_candidates, correctness_from_target, compare_answer_values, score_candidate

def run_greedy(model: ModelRunner, policy: Policy, example: dict) -> dict:
    """Run single greedy sample (or near greedy)."""
    # Force greedy params
    prompt = make_prompt(policy, example['question'])
    
    res = model.generate(
        prompt,
        do_sample=False,
        seed=42
    )
    
    pred_text = res['text']
    final_ans = extract_final_answer(pred_text)
    pred_val = normalize_answer_for_candidates(final_ans)
    is_correct = correctness_from_target(pred_text, example.get("target"))
    
    return {
        "example_id": example['id'],
        "method": "greedy",
        "policy": policy.name,
        "n": 1,
        "pred": final_ans, # Raw string extracted
        "pred_val": pred_val, # Numeric
        "is_correct": is_correct,
        "prompt_tokens": res['prompt_tokens'],
        "completion_tokens": res['completion_tokens'],
        "total_tokens": res['total_tokens'],
        "time_s": res['time_s'],
        "extra": {"full_text": pred_text}
    }

def run_self_consistency(model: ModelRunner, policy: Policy, example: dict, n: int, seed: int = 42) -> dict:
    """Run Self-Consistency with Majority Vote."""
    prompt = make_prompt(policy, example['question'])
    
    candidates = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    
    # Generate N times
    # Note: In production, batching is better. Here we iterate for simplicity/memory safety on smaller devices.
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
        
        ans_str = extract_final_answer(res['text'])
        ans_val = normalize_answer_for_candidates(ans_str)
        candidates.append(ans_val)
        
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
    if top_ans is not None:
         is_correct = compare_answer_values(top_ans, example.get("target"))

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

def run_best_of_n(model: ModelRunner, policy: Policy, example: dict, n: int, seed: int = 42) -> dict:
    """Run Best-of-N using a heuristic scorer."""
    prompt = make_prompt(policy, example['question'])
    
    best_score = -1.0
    best_res = None
    best_ans_val = None
    candidates = []
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    
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
        
        # Score
        sc = score_candidate(res['text'])
        ans_val = normalize_answer_for_candidates(extract_final_answer(res['text']))
        candidates.append(ans_val)

        if sc > best_score:
            best_score = sc
            best_res = res
            best_ans_val = ans_val
    
    unique_candidates = set([c for c in candidates if c is not None])
    unique_frac = len(unique_candidates) / len(candidates) if candidates else 0.0
    
    # Check correctness of chosen one
    is_correct = False
    if best_ans_val is not None:
         is_correct = compare_answer_values(best_ans_val, example.get("target"))

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
