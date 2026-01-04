import argparse
import yaml
import json
import os
import time
from tqdm import tqdm
from .utils import setup_logging, set_seed, ensure_dir
from .data import load_dataset_records
from .models import ModelRunner
from .policies import load_policies_from_config
from .baselines import run_greedy, run_self_consistency, run_best_of_n
from .anytime import run_anytime_sc

logger = setup_logging("run_eval")

def run_eval(
    config: dict,
    dataset_override: str = None,
    limit_override: int = None,
    seed_override: int = None,
    run_group: str = None,
) -> None:
    if "methods" not in config or not isinstance(config["methods"], list):
        raise ValueError("Config must include a list of methods.")
    if "output_dir" not in config:
        raise ValueError("Config missing required output_dir.")

    dataset_name = dataset_override or config.get("dataset")
    if not dataset_name:
        logger.error("No dataset specified.")
        return
    split = config.get("split", "test")
    seed = seed_override if seed_override is not None else config.get("seed", 42)
    run_group = run_group or config.get("run_group")

    set_seed(seed)
    ensure_dir(config["output_dir"])

    data = load_dataset_records(
        dataset_name,
        split=split,
        limit=limit_override if limit_override is not None else config.get("limit", None),
        seed=seed,
    )
    if not data:
        return

    model = ModelRunner(
        model_name=config["model_name"],
        dtype="float16" if "gpu" in str(config).lower() else "auto",
        max_new_tokens=config.get("max_new_tokens", 512),
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    import uuid
    run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    logger.info(f"Global Run ID: {run_id}")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for method_cfg in config["methods"]:
        m_name = method_cfg["name"]
        logger.info(f"Running Method: {m_name}")

        def resolve_single_policy_from_name(policy_name):
            if not policy_name:
                return None, None
            if policy_name in {"raw", "none", "question"}:
                return "raw", None
            policies = load_policies_from_config({"policies": [policy_name]}, root_dir)
            if not policies:
                raise ValueError(f"Unknown policy '{policy_name}' for method '{m_name}'.")
            return policy_name, policies[0]

        def resolve_single_policy():
            policy_name = method_cfg.get("policy") or method_cfg.get("prompt")
            return resolve_single_policy_from_name(policy_name)

        def resolve_policy_list():
            policy_names = method_cfg.get("policies") or config.get("policies") or []
            if not policy_names:
                fallback_name = method_cfg.get("policy") or method_cfg.get("prompt")
                if fallback_name:
                    policy_names = [fallback_name]
            if not policy_names:
                return []
            policies = load_policies_from_config({"policies": policy_names}, root_dir)
            if len(policies) != len(policy_names):
                raise ValueError(f"One or more policies are unknown for method '{m_name}': {policy_names}")
            return policies

        # Refactor execution loop
        if m_name == "greedy":
            policy_name, single_policy = resolve_single_policy()
            if policy_name is None:
                try:
                    policy_name, single_policy = resolve_single_policy_from_name("direct")
                except ValueError:
                    policy_name = "raw"
                    single_policy = None
            configs = [{"policy": single_policy, "policy_name": policy_name}]
        elif m_name == "self_consistency":
            if "n_values" not in method_cfg:
                raise ValueError("self_consistency requires n_values.")
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("self_consistency requires a valid policy/prompt.")
            configs = [{"n": n, "policy": single_policy, "policy_name": policy_name} for n in method_cfg["n_values"]]
        elif m_name == "best_of_n":
             if "n_values" not in method_cfg:
                 raise ValueError("best_of_n requires n_values.")
             policy_name, single_policy = resolve_single_policy()
             if not single_policy:
                 raise ValueError("best_of_n requires a valid policy/prompt.")
             configs = [{"n": n, "policy": single_policy, "policy_name": policy_name} for n in method_cfg["n_values"]]
        elif m_name == "anytime_sc":
            if "budgets" not in method_cfg or "deltas" not in method_cfg:
                raise ValueError("anytime_sc requires budgets and deltas.")
            policies = resolve_policy_list()
            if not policies:
                raise ValueError("anytime_sc requires policies.")
            configs = []
            for b in method_cfg["budgets"]:
                for d in method_cfg["deltas"]:
                     configs.append({"budget": b, "delta": d, "allocation": method_cfg.get("allocation", "ucb"), "policies": policies})
        else:
            raise ValueError(f"Unknown method name: {m_name}")
        
        for run_cfg in configs:
            # Construct filename
            suffix_parts = []
            if run_group:
                suffix_parts.append(run_group)
            if run_group or seed_override is not None:
                suffix_parts.append(f"seed{seed}")
            suffix = "_".join(suffix_parts)

            if m_name == "greedy":
                fname = f"{dataset_name}_{m_name}_{run_cfg['policy_name']}"
            elif m_name in ["self_consistency", "best_of_n"]:
                fname = f"{dataset_name}_{m_name}_n{run_cfg['n']}_{run_cfg['policy'].name}"
            elif m_name == "anytime_sc":
                fname = f"{dataset_name}_{m_name}_b{run_cfg['budget']}_d{run_cfg['delta']}_{run_cfg['allocation']}"

            if suffix:
                fname = f"{fname}_{suffix}_{run_id}.jsonl"
            else:
                fname = f"{fname}_{run_id}.jsonl"
            
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
                        res["dataset"] = dataset_name or "unknown"
                        res["split"] = split
                        res["model_name"] = config.get("model_name", "unknown")
                        res["run_id"] = run_id
                        res["run_group"] = run_group
                        res["seed"] = seed
                        res["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                        
                        # Ensure fields exist
                        if "is_correct" not in res: res["is_correct"] = None
                        if "target" not in res: res["target"] = example.get("target")
                        if "parse_error" not in res: res["parse_error"] = example.get("parse_error", False)

                        f_out.write(json.dumps(res) + "\n")
                        f_out.flush() # Ensure safety
                    except Exception as e:
                        logger.error(f"Error processing example {example.get('id')}: {e}")
                        continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--seed", type=int, help="Override seed for sampling/shuffle")
    parser.add_argument("--run_group", type=str, help="Run group identifier for multi-seed suites")
    parser.add_argument("--dataset", type=str, help="Override dataset name")
    parser.add_argument("--limit", type=int, help="Override dataset limit")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_eval(
        config=config,
        dataset_override=args.dataset,
        limit_override=args.limit,
        seed_override=args.seed,
        run_group=args.run_group,
    )

if __name__ == "__main__":
    main()
