import argparse
import yaml
import json
import os
import time
from tqdm import tqdm
from .utils import setup_logging, set_seed, ensure_dir
from .data import load_gsm8k
from .models import ModelRunner
from .policies import Policy, load_policies_from_config
from .baselines import run_greedy, run_self_consistency, run_best_of_n
from .anytime import run_anytime_sc

logger = setup_logging("run_eval")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup
    set_seed(config.get("seed", 42))
    ensure_dir(config["output_dir"])
    
    # Load Data
    data = load_gsm8k(
        split=config.get("split", "test"),
        limit=config.get("limit", None),
        seed=config.get("seed", 42)
    )
    if not data:
        return
        
    # Load Model
    model = ModelRunner(
        model_name=config["model_name"],
        dtype="float16" if "gpu" in str(config).lower() else "auto" # simple heuristic
    )
    
    # Base output filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    import uuid
    run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    logger.info(f"Global Run ID: {run_id}")
    
    # Iterate Methods
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # repo root
    
    for method_cfg in config["methods"]:
        m_name = method_cfg["name"]
        logger.info(f"Running Method: {m_name}")
        
        # Expand configs (e.g. multiple budgets, multiple n)
        # We simplify by flattening the tasks list
        tasks = []
        
        # Common Policy Loading
        # If method has specific policy list, use it. Else use global single policy name mapping.
        if "policies" in method_cfg:
            policies = load_policies_from_config(method_cfg, root_dir)
        elif "policy" in method_cfg:
            # Create single policy list
            dummy_cfg = {"policies": [method_cfg["policy"]]}
            policies = load_policies_from_config(dummy_cfg, root_dir)
            single_policy = policies[0]
        
        if m_name == "greedy":
            tasks.append(lambda ex: run_greedy(model, single_policy, ex))
            run_lbl = f"greedy_{single_policy.name}"
            
        elif m_name == "self_consistency":
            for n in method_cfg["n_values"]:
                tasks.append(lambda ex, n=n: run_self_consistency(model, single_policy, ex, n))
            run_lbl = f"sc_{single_policy.name}"

        elif m_name == "best_of_n":
            for n in method_cfg["n_values"]:
                tasks.append(lambda ex, n=n: run_best_of_n(model, single_policy, ex, n))
            run_lbl = f"bon_{single_policy.name}"
                
        elif m_name == "anytime_sc":
            # Cartesian product of budgets, deltas, allocations
            for b in method_cfg["budgets"]:
                for d in method_cfg["deltas"]:
                    alloc = method_cfg.get("allocation", "ucb")
                    tasks.append(lambda ex, b=b, d=d, a=alloc: run_anytime_sc(model, policies, ex, b, d, a))
            run_lbl = "anytime"
        
        # Execute
        # We run all tasks for a method? No, usually we want one file per parameter setting or one big file.
        # The prompt says: "Ensure each output file has a unique name...".
        # It's better to structure this loop: Loop over Parameters -> Loop over Examples -> Write File.
        
        # Refactor execution loop
        if m_name == "greedy":
            configs = [{"policy": single_policy}]
        elif m_name == "self_consistency":
            configs = [{"n": n, "policy": single_policy} for n in method_cfg["n_values"]]
        elif m_name == "best_of_n":
             configs = [{"n": n, "policy": single_policy} for n in method_cfg["n_values"]]
        elif m_name == "anytime_sc":
            configs = []
            for b in method_cfg["budgets"]:
                for d in method_cfg["deltas"]:
                     configs.append({"budget": b, "delta": d, "allocation": method_cfg.get("allocation", "ucb"), "policies": policies})
        
        for run_cfg in configs:
            # Construct filename
            if m_name == "greedy":
                fname = f"{config['dataset']}_{m_name}_{run_cfg['policy'].name}_{run_id}.jsonl"
            elif m_name in ["self_consistency", "best_of_n"]:
                fname = f"{config['dataset']}_{m_name}_n{run_cfg['n']}_{run_cfg['policy'].name}_{run_id}.jsonl"
            elif m_name == "anytime_sc":
                fname = f"{config['dataset']}_{m_name}_b{run_cfg['budget']}_d{run_cfg['delta']}_{run_cfg['allocation']}_{run_id}.jsonl"
            
            out_path = os.path.join(config["output_dir"], fname)
            logger.info(f"Starting run for {fname}...")
            

            with open(out_path, 'w') as f_out:
                for example in tqdm(data):
                    try:
                        # Dispatch
                        if m_name == "greedy":
                            res = run_greedy(model, run_cfg["policy"], example)
                        elif m_name == "self_consistency":
                            res = run_self_consistency(model, run_cfg["policy"], example, run_cfg["n"])
                        elif m_name == "best_of_n":
                            res = run_best_of_n(model, run_cfg["policy"], example, run_cfg["n"])
                        elif m_name == "anytime_sc":
                            res = run_anytime_sc(model, run_cfg["policies"], example, run_cfg["budget"], run_cfg["delta"], run_cfg["allocation"])
                        
                        # Inject Metadata
                        res["dataset"] = config.get("dataset", "unknown")
                        res["split"] = config.get("split", "unknown")
                        res["model_name"] = config.get("model_name", "unknown")
                        res["run_id"] = run_id
                        res["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                        
                        # Ensure fields exist
                        if "is_correct" not in res: res["is_correct"] = None
                        if "gold" not in res: res["gold"] = example.get("gold")

                        f_out.write(json.dumps(res) + "\n")
                        f_out.flush() # Ensure safety
                    except Exception as e:
                        logger.error(f"Error processing example {example.get('id')}: {e}")
                        continue

if __name__ == "__main__":
    main()
