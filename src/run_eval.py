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
from .baselines import (
    run_greedy,
    run_self_consistency,
    run_best_of_n,
    run_best_of_n_verifier,
    run_self_consistency_early_stop,
    run_best_of_n_early_stop,
)
from .anytime import run_anytime_sc
from .global_anytime import run_global_anytime_sc

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
        dtype=config.get("dtype", "auto"),
        max_new_tokens=config.get("max_new_tokens", 512),
        use_flash_attention=config.get("use_flash_attention", False),
        use_compile=config.get("use_compile", False),
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    import uuid
    run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    logger.info(f"Global Run ID: {run_id}")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for method_cfg in config["methods"]:
        m_name = method_cfg["name"]
        logger.info(f"Running Method: {m_name}")
        verifier_model = None

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

        def build_fixed_n_configs(method_cfg, policy_name, single_policy, method_label):
            configs = []
            n_values = method_cfg.get("n_values", [])
            if n_values:
                configs.extend([
                    {"n": n, "policy": single_policy, "policy_name": policy_name}
                    for n in n_values
                ])

            match_budgets = method_cfg.get("match_budgets")
            if match_budgets:
                tokens_per_sample = method_cfg.get("tokens_per_sample")
                if tokens_per_sample is None or float(tokens_per_sample) <= 0:
                    raise ValueError(f"{method_label} match_budgets requires tokens_per_sample > 0.")
                for b in match_budgets:
                    n = max(1, int(round(float(b) / float(tokens_per_sample))))
                    configs.append({
                        "n": n,
                        "policy": single_policy,
                        "policy_name": policy_name,
                        "budget_tokens": int(b),
                        "tokens_per_sample": float(tokens_per_sample),
                    })

            if not configs:
                raise ValueError(f"{method_label} requires n_values or match_budgets.")
            return configs

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
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("self_consistency requires a valid policy/prompt.")
            configs = build_fixed_n_configs(method_cfg, policy_name, single_policy, "self_consistency")
            # Add batched flag to all configs
            batched = method_cfg.get("batched", False)
            for cfg in configs:
                cfg["batched"] = batched
        elif m_name == "best_of_n":
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("best_of_n requires a valid policy/prompt.")
            configs = build_fixed_n_configs(method_cfg, policy_name, single_policy, "best_of_n")
            # Add batched flag to all configs
            batched = method_cfg.get("batched", False)
            for cfg in configs:
                cfg["batched"] = batched
        elif m_name == "best_of_n_verifier":
            if "n_values" not in method_cfg:
                raise ValueError("best_of_n_verifier requires n_values.")
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("best_of_n_verifier requires a valid policy/prompt.")
            verifier_model_name = method_cfg.get("verifier_model_name")
            if not verifier_model_name:
                raise ValueError("best_of_n_verifier requires verifier_model_name.")
            verifier_max_new_tokens = int(method_cfg.get("verifier_max_new_tokens", 8))
            verifier_model = ModelRunner(
                model_name=verifier_model_name,
                dtype="float16" if "gpu" in str(config).lower() else "auto",
                max_new_tokens=verifier_max_new_tokens,
            )
            configs = [
                {
                    "n": n,
                    "policy": single_policy,
                    "policy_name": policy_name,
                    "verifier_model_name": verifier_model_name,
                }
                for n in method_cfg["n_values"]
            ]
        elif m_name == "self_consistency_early_stop":
            if "n_values" not in method_cfg:
                raise ValueError("self_consistency_early_stop requires n_values.")
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("self_consistency_early_stop requires a valid policy/prompt.")
            stop_ratio = method_cfg.get("stop_ratio", 0.6)
            stop_count = method_cfg.get("stop_count")
            min_samples = method_cfg.get("min_samples", 2)
            configs = [
                {
                    "n": n,
                    "policy": single_policy,
                    "policy_name": policy_name,
                    "stop_ratio": stop_ratio,
                    "stop_count": stop_count,
                    "min_samples": min_samples,
                }
                for n in method_cfg["n_values"]
            ]
        elif m_name == "best_of_n_early_stop":
            if "n_values" not in method_cfg:
                raise ValueError("best_of_n_early_stop requires n_values.")
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("best_of_n_early_stop requires a valid policy/prompt.")
            score_threshold = method_cfg.get("score_threshold", 0.7)
            min_samples = method_cfg.get("min_samples", 1)
            configs = [
                {
                    "n": n,
                    "policy": single_policy,
                    "policy_name": policy_name,
                    "score_threshold": score_threshold,
                    "min_samples": min_samples,
                }
                for n in method_cfg["n_values"]
            ]
        elif m_name == "anytime_sc":
            if "budgets" not in method_cfg or "deltas" not in method_cfg:
                raise ValueError("anytime_sc requires budgets and deltas.")
            policies = resolve_policy_list()
            if not policies:
                raise ValueError("anytime_sc requires policies.")
            batch_size = int(method_cfg.get("batch_size", 1))
            allow_unseeded_batch = bool(method_cfg.get("allow_unseeded_batch", False))
            configs = []
            for b in method_cfg["budgets"]:
                for d in method_cfg["deltas"]:
                    configs.append({
                        "budget": b,
                        "delta": d,
                        "allocation": method_cfg.get("allocation", "ucb"),
                        "policies": policies,
                        "batch_size": batch_size,
                        "allow_unseeded_batch": allow_unseeded_batch,
                    })
        elif m_name == "global_anytime_sc":
            policy_name, single_policy = resolve_single_policy()
            if policy_name is None:
                try:
                    policy_name, single_policy = resolve_single_policy_from_name("direct")
                except ValueError:
                    policy_name = "raw"
                    single_policy = None
            if not single_policy and policy_name != "raw":
                raise ValueError("global_anytime_sc requires a valid policy/prompt (or policy=raw).")
            budgets = method_cfg.get("global_budget_tokens")
            if budgets is None:
                raise ValueError("global_anytime_sc requires global_budget_tokens.")
            if isinstance(budgets, (int, float)):
                budgets = [int(budgets)]
            allocation_policies = method_cfg.get("allocation_policy", "uniform")
            if isinstance(allocation_policies, str):
                allocation_policies = [allocation_policies]
            init_k = int(method_cfg.get("init_k", 1))
            max_samples_per_item = method_cfg.get("max_samples_per_item")
            per_example_budget_tokens = method_cfg.get("per_example_budget_tokens")
            ucb_c = float(method_cfg.get("ucb_c", 1.0))
            store_allocation_steps = bool(method_cfg.get("store_allocation_steps", False))
            temperature = method_cfg.get("temperature", getattr(single_policy, "temperature", 0.7) if single_policy else 0.7)
            top_p = method_cfg.get("top_p", getattr(single_policy, "top_p", 1.0) if single_policy else 1.0)
            top_k = method_cfg.get("top_k", getattr(single_policy, "top_k", 50) if single_policy else 50)
            finalize = method_cfg.get("finalize", "majority")

            configs = []
            for b in budgets:
                for alloc in allocation_policies:
                    configs.append({
                        "global_budget_tokens": int(b),
                        "allocation_policy": alloc,
                        "init_k": init_k,
                        "max_samples_per_item": max_samples_per_item,
                        "per_example_budget_tokens": per_example_budget_tokens,
                        "ucb_c": ucb_c,
                        "store_allocation_steps": store_allocation_steps,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "top_k": int(top_k),
                        "policy": single_policy,
                        "policy_name": policy_name,
                        "finalize": finalize,
                    })
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
            elif m_name in ["self_consistency", "best_of_n", "best_of_n_verifier", "self_consistency_early_stop", "best_of_n_early_stop"]:
                fname = f"{dataset_name}_{m_name}_n{run_cfg['n']}"
                if run_cfg.get("budget_tokens") is not None:
                    fname = f"{fname}_b{int(run_cfg['budget_tokens'])}"
                fname = f"{fname}_{run_cfg['policy'].name}"
                if m_name == "best_of_n_verifier":
                    verifier_label = run_cfg["verifier_model_name"].split("/")[-1]
                    fname = f"{fname}_verifier{verifier_label}"
            elif m_name == "anytime_sc":
                fname = f"{dataset_name}_{m_name}_b{run_cfg['budget']}_d{run_cfg['delta']}_{run_cfg['allocation']}"
            elif m_name == "global_anytime_sc":
                fname = f"{dataset_name}_{m_name}_T{run_cfg['global_budget_tokens']}_init{run_cfg['init_k']}_{run_cfg['allocation_policy']}_{run_cfg['policy_name']}"

            if suffix:
                fname = f"{fname}_{suffix}_{run_id}.jsonl"
            else:
                fname = f"{fname}_{run_id}.jsonl"
            
            out_path = os.path.join(config["output_dir"], fname)
            logger.info(f"Starting run for {fname}...")
            

            with open(out_path, 'w') as f_out:
                if m_name == "global_anytime_sc":
                    example_by_id = {ex.get("id"): ex for ex in data}
                    try:
                        results = run_global_anytime_sc(
                            model,
                            run_cfg["policy"],
                            data,
                            run_cfg["global_budget_tokens"],
                            init_k=run_cfg["init_k"],
                            allocation_policy=run_cfg["allocation_policy"],
                            per_example_budget_tokens=run_cfg["per_example_budget_tokens"],
                            ucb_c=run_cfg["ucb_c"],
                            max_samples_per_item=run_cfg["max_samples_per_item"],
                            temperature=run_cfg["temperature"],
                            top_p=run_cfg["top_p"],
                            top_k=run_cfg["top_k"],
                            finalize=run_cfg["finalize"],
                            store_allocation_steps=run_cfg["store_allocation_steps"],
                            seed=seed,
                        )
                    except Exception as e:
                        logger.error(f"Error running global_anytime_sc: {e}")
                        continue

                    for res in results:
                        res["dataset"] = dataset_name or "unknown"
                        res["split"] = split
                        res["model_name"] = config.get("model_name", "unknown")
                        res["run_id"] = run_id
                        res["run_group"] = run_group
                        res["seed"] = seed
                        res["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                        res["budget_tokens"] = run_cfg["global_budget_tokens"]
                        res["allocation"] = run_cfg["allocation_policy"]
                        res["init_k"] = run_cfg["init_k"]
                        res["max_samples_per_item"] = run_cfg["max_samples_per_item"]
                        res["per_example_budget_tokens"] = run_cfg["per_example_budget_tokens"]
                        res["ucb_c"] = run_cfg["ucb_c"]
                        res["store_allocation_steps"] = run_cfg["store_allocation_steps"]
                        res["temperature"] = run_cfg["temperature"]
                        res["top_p"] = run_cfg["top_p"]
                        res["top_k"] = run_cfg["top_k"]
                        res["finalize"] = run_cfg["finalize"]
                        if "target" not in res:
                            res["target"] = example_by_id.get(res.get("example_id"), {}).get("target")
                        if "parse_error" not in res or "subject" not in res:
                            ex = example_by_id.get(res.get("example_id"), {})
                            if "parse_error" not in res:
                                res["parse_error"] = ex.get("parse_error", False)
                            if "subject" not in res and "subject" in ex:
                                res["subject"] = ex.get("subject")

                        f_out.write(json.dumps(res) + "\n")
                    f_out.flush()
                else:
                    for example in tqdm(data):
                        try:
                            # Dispatch
                            if m_name == "greedy":
                                res = run_greedy(model, run_cfg["policy"], example)
                            elif m_name == \"self_consistency\":
                                res = run_self_consistency(model, run_cfg[\"policy\"], example, run_cfg[\"n\"], batched=run_cfg.get(\"batched\", False))
                            elif m_name == \"best_of_n\":
                                res = run_best_of_n(model, run_cfg[\"policy\"], example, run_cfg[\"n\"], batched=run_cfg.get(\"batched\", False))
                            elif m_name == "best_of_n_verifier":
                                res = run_best_of_n_verifier(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    run_cfg["n"],
                                    verifier_model,
                                )
                            elif m_name == "self_consistency_early_stop":
                                res = run_self_consistency_early_stop(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    run_cfg["n"],
                                    stop_ratio=run_cfg["stop_ratio"],
                                    stop_count=run_cfg["stop_count"],
                                    min_samples=run_cfg["min_samples"],
                                )
                            elif m_name == "best_of_n_early_stop":
                                res = run_best_of_n_early_stop(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    run_cfg["n"],
                                    score_threshold=run_cfg["score_threshold"],
                                    min_samples=run_cfg["min_samples"],
                                )
                            elif m_name == "anytime_sc":
                                res = run_anytime_sc(
                                    model,
                                    run_cfg["policies"],
                                    example,
                                    run_cfg["budget"],
                                    run_cfg["delta"],
                                    run_cfg["allocation"],
                                    batch_size=run_cfg["batch_size"],
                                    allow_unseeded_batch=run_cfg["allow_unseeded_batch"],
                                )
                            else:
                                raise ValueError(f"Unknown method name: {m_name}")

                            # Inject Metadata
                            res["dataset"] = dataset_name or "unknown"
                            res["split"] = split
                            res["model_name"] = config.get("model_name", "unknown")
                            res["run_id"] = run_id
                            res["run_group"] = run_group
                            res["seed"] = seed
                            res["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                            if run_cfg.get("budget_tokens") is not None:
                                res["budget_tokens"] = run_cfg["budget_tokens"]
                            if run_cfg.get("tokens_per_sample") is not None:
                                res["tokens_per_sample"] = run_cfg["tokens_per_sample"]
                            if run_cfg.get("verifier_model_name") is not None:
                                res["verifier_model_name"] = run_cfg["verifier_model_name"]
                            if run_cfg.get("batch_size") is not None:
                                res["batch_size"] = run_cfg["batch_size"]
                            if run_cfg.get("allow_unseeded_batch") is not None:
                                res["allow_unseeded_batch"] = run_cfg["allow_unseeded_batch"]

                            # Ensure fields exist
                            if "is_correct" not in res:
                                res["is_correct"] = None
                            if "target" not in res:
                                res["target"] = example.get("target")
                            if "parse_error" not in res:
                                res["parse_error"] = example.get("parse_error", False)
                            if "subject" not in res and "subject" in example:
                                res["subject"] = example.get("subject")

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
