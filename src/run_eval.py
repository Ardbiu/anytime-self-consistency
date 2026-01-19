import argparse
import yaml
import json
import os
import time
import hashlib
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
    shard_id: int = 0,
    num_shards: int = 1,
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
    cache_enabled = bool(config.get("cache_enabled", True))
    cache_path = config.get("cache_path") or "outputs/cache/cache.jsonl"

    set_seed(seed)
    ensure_dir(config["output_dir"])
    if cache_enabled:
        ensure_dir(os.path.dirname(cache_path))

    data = load_dataset_records(
        dataset_name,
        split=split,
        limit=limit_override if limit_override is not None else config.get("limit", None),
        seed=seed,
    )
    if not data:
        return

    # Sharding
    if num_shards > 1:
        total_examples = len(data)
        data = data[shard_id::num_shards]
        logger.info(f"Shard {shard_id}/{num_shards}: Process {len(data)}/{total_examples} examples.")

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

    cache_records = {}
    cache_global_done = set()
    if cache_enabled and os.path.exists(cache_path):
        with open(cache_path, "r") as cache_file:
            for line in cache_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("cache_key")
                scope = rec.get("cache_scope")
                if scope == "global" and rec.get("completed"):
                    if key:
                        cache_global_done.add(key)
                elif key:
                    cache_records[key] = rec

    def _policy_name(policy_obj):
        if policy_obj is None:
            return "raw"
        return getattr(policy_obj, "name", "unknown")

    def _params_hash(method_name, run_cfg):
        payload = {
            "dataset": dataset_name,
            "split": split,
            "seed": seed,
            "model_name": config.get("model_name"),
            "max_new_tokens": config.get("max_new_tokens", 512),
            "use_flash_attention": config.get("use_flash_attention", False),
            "use_compile": config.get("use_compile", False),
            "method": method_name,
            "policy": _policy_name(run_cfg.get("policy")),
            "policy_name": run_cfg.get("policy_name"),
            "policies": [p.name for p in run_cfg.get("policies", [])] if run_cfg.get("policies") else None,
            "n": run_cfg.get("n"),
            "budget": run_cfg.get("budget"),
            "budget_tokens": run_cfg.get("budget_tokens"),
            "delta": run_cfg.get("delta"),
            "allocation": run_cfg.get("allocation"),
            "global_budget_tokens": run_cfg.get("global_budget_tokens"),
            "allocation_policy": run_cfg.get("allocation_policy"),
            "init_k": run_cfg.get("init_k"),
            "max_samples_per_item": run_cfg.get("max_samples_per_item"),
            "per_example_budget_tokens": run_cfg.get("per_example_budget_tokens"),
            "ucb_c": run_cfg.get("ucb_c"),
            "temperature": run_cfg.get("temperature"),
            "top_p": run_cfg.get("top_p"),
            "top_k": run_cfg.get("top_k"),
            "finalize": run_cfg.get("finalize"),
            "batch_size": run_cfg.get("batch_size"),
            "allow_unseeded_batch": run_cfg.get("allow_unseeded_batch"),
            "batched": run_cfg.get("batched"),
            "batched_seeded": run_cfg.get("batched_seeded"),
            "verifier_model_name": run_cfg.get("verifier_model_name"),
            "verifier_max_new_tokens": run_cfg.get("verifier_max_new_tokens"),
            "verifier_task": run_cfg.get("verifier_task"),
            "verifier_use_flash_attention": run_cfg.get("verifier_use_flash_attention"),
            "verifier_use_compile": run_cfg.get("verifier_use_compile"),
        }
        data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:12]

    def _cache_key(qid, method_name, params_hash):
        return f"{dataset_name}||{split}||{qid}||{method_name}||{params_hash}"

    def _global_cache_key(method_name, params_hash):
        return f"{dataset_name}||{split}||{method_name}||{params_hash}||global"

    for method_cfg in config["methods"]:
        m_name = method_cfg["name"]
        logger.info(f"Running Method: {m_name}")
        verifier_model = None

        def resolve_single_policy_from_name(policy_name):
            if not policy_name:
                return None, None
            if policy_name in {"raw", "none", "question"}:
                policies = load_policies_from_config({"policies": ["raw"]}, root_dir)
                raw_policy = policies[0] if policies else None
                return "raw", raw_policy
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
            batched_seeded = method_cfg.get("batched_seeded", False)
            for cfg in configs:
                cfg["batched"] = batched
                cfg["batched_seeded"] = batched_seeded
        elif m_name == "best_of_n":
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("best_of_n requires a valid policy/prompt.")
            configs = build_fixed_n_configs(method_cfg, policy_name, single_policy, "best_of_n")
            # Add batched flag to all configs
            batched = method_cfg.get("batched", False)
            batched_seeded = method_cfg.get("batched_seeded", False)
            for cfg in configs:
                cfg["batched"] = batched
                cfg["batched_seeded"] = batched_seeded
        elif m_name == "best_of_n_verifier":
            if "n_values" not in method_cfg:
                raise ValueError("best_of_n_verifier requires n_values.")
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("best_of_n_verifier requires a valid policy/prompt.")
            verifier_model_name = method_cfg.get("verifier_model_name")
            if not verifier_model_name:
                raise ValueError("best_of_n_verifier requires verifier_model_name.")
            verifier_task = method_cfg.get("verifier_task", "yes_no")
            verifier_max_new_tokens = int(method_cfg.get("verifier_max_new_tokens", 8))
            verifier_use_flash_attention = bool(method_cfg.get("verifier_use_flash_attention", config.get("use_flash_attention", False)))
            verifier_use_compile = bool(method_cfg.get("verifier_use_compile", config.get("use_compile", False)))
            verifier_model = ModelRunner(
                model_name=verifier_model_name,
                dtype="float16" if "gpu" in str(config).lower() else "auto",
                max_new_tokens=verifier_max_new_tokens,
                use_flash_attention=verifier_use_flash_attention,
                use_compile=verifier_use_compile,
                task=verifier_task,
            )
            batched = method_cfg.get("batched", False)
            batched_seeded = method_cfg.get("batched_seeded", False)
            configs = [
                {
                    "n": n,
                    "policy": single_policy,
                    "policy_name": policy_name,
                    "verifier_model_name": verifier_model_name,
                    "verifier_max_new_tokens": verifier_max_new_tokens,
                    "verifier_task": verifier_task,
                    "verifier_use_flash_attention": verifier_use_flash_attention,
                    "verifier_use_compile": verifier_use_compile,
                    "batched": batched,
                    "batched_seeded": batched_seeded,
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
            params_hash = _params_hash(m_name, run_cfg)
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

            if m_name == "global_anytime_sc":
                global_cache_key = _global_cache_key(m_name, params_hash)
                if cache_enabled and global_cache_key in cache_global_done:
                    logger.info(f"Skipping global run (cached): {global_cache_key}")
                    continue

            with open(out_path, 'w') as f_out:
                cache_file = None
                if cache_enabled:
                    cache_file = open(cache_path, "a")
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
                        res["params_hash"] = params_hash
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
                        ex = example_by_id.get(res.get("example_id"), {})
                        if "parse_error" not in res:
                            res["parse_error"] = ex.get("parse_error", False)
                        if "subject" not in res and "subject" in ex:
                            res["subject"] = ex.get("subject")
                        if "answer_type" not in res and "answer_type" in ex:
                            res["answer_type"] = ex.get("answer_type")
                        if "code_task" not in res and "code_task" in ex:
                            res["code_task"] = ex.get("code_task")

                        f_out.write(json.dumps(res) + "\n")
                    f_out.flush()
                    if cache_file:
                        cache_file.write(json.dumps({
                            "cache_scope": "global",
                            "cache_key": global_cache_key,
                            "completed": True,
                            "dataset": dataset_name,
                            "split": split,
                            "method": m_name,
                            "params_hash": params_hash,
                            "global_budget_tokens": run_cfg["global_budget_tokens"],
                            "allocation": run_cfg["allocation_policy"],
                            "run_id": run_id,
                            "run_group": run_group,
                            "seed": seed,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }) + "\n")
                        cache_file.flush()
                else:
                    for example in tqdm(data):
                        try:
                            cache_key = _cache_key(example.get("id"), m_name, params_hash)
                            if cache_enabled and cache_key in cache_records:
                                cached = dict(cache_records[cache_key])
                                cached["cached"] = True
                                cached["run_id"] = run_id
                                cached["run_group"] = run_group
                                cached["seed"] = seed
                                cached["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                                if "dataset" not in cached:
                                    cached["dataset"] = dataset_name or "unknown"
                                if "split" not in cached:
                                    cached["split"] = split
                                if "method" not in cached:
                                    cached["method"] = m_name
                                if "model_name" not in cached:
                                    cached["model_name"] = config.get("model_name", "unknown")
                                if "total_tokens" not in cached and "tokens_used" in cached:
                                    cached["total_tokens"] = cached.get("tokens_used")
                                f_out.write(json.dumps(cached) + "\n")
                                continue
                            # Dispatch
                            if m_name == "greedy":
                                res = run_greedy(model, run_cfg["policy"], example, seed=seed)
                            elif m_name == "self_consistency":
                                res = run_self_consistency(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    run_cfg["n"],
                                    seed=seed,
                                    batched=run_cfg.get("batched", False),
                                    batched_seeded=run_cfg.get("batched_seeded", False),
                                )
                            elif m_name == "best_of_n":
                                res = run_best_of_n(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    run_cfg["n"],
                                    seed=seed,
                                    batched=run_cfg.get("batched", False),
                                    batched_seeded=run_cfg.get("batched_seeded", False),
                                )
                            elif m_name == "best_of_n_verifier":
                                res = run_best_of_n_verifier(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    run_cfg["n"],
                                    verifier_model,
                                    seed=seed,
                                    batched=run_cfg.get("batched", False),
                                    batched_seeded=run_cfg.get("batched_seeded", False),
                                )
                            elif m_name == "self_consistency_early_stop":
                                res = run_self_consistency_early_stop(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    run_cfg["n"],
                                    seed=seed,
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
                                    seed=seed,
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
                                    seed=seed,
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
                            res["params_hash"] = params_hash
                            if run_cfg.get("budget_tokens") is not None:
                                res["budget_tokens"] = run_cfg["budget_tokens"]
                            if run_cfg.get("tokens_per_sample") is not None:
                                res["tokens_per_sample"] = run_cfg["tokens_per_sample"]
                            if run_cfg.get("verifier_model_name") is not None:
                                res["verifier_model_name"] = run_cfg["verifier_model_name"]
                            if run_cfg.get("verifier_max_new_tokens") is not None:
                                res["verifier_max_new_tokens"] = run_cfg["verifier_max_new_tokens"]
                            if run_cfg.get("verifier_task") is not None:
                                res["verifier_task"] = run_cfg["verifier_task"]
                            if run_cfg.get("verifier_use_flash_attention") is not None:
                                res["verifier_use_flash_attention"] = run_cfg["verifier_use_flash_attention"]
                            if run_cfg.get("verifier_use_compile") is not None:
                                res["verifier_use_compile"] = run_cfg["verifier_use_compile"]
                            if run_cfg.get("batch_size") is not None:
                                res["batch_size"] = run_cfg["batch_size"]
                            if run_cfg.get("allow_unseeded_batch") is not None:
                                res["allow_unseeded_batch"] = run_cfg["allow_unseeded_batch"]
                            if run_cfg.get("batched") is not None:
                                res["batched"] = run_cfg["batched"]
                            if run_cfg.get("batched_seeded") is not None:
                                res["batched_seeded"] = run_cfg["batched_seeded"]

                            # Ensure fields exist
                            if "is_correct" not in res:
                                res["is_correct"] = None
                            if "target" not in res:
                                res["target"] = example.get("target")
                            if "parse_error" not in res:
                                res["parse_error"] = example.get("parse_error", False)
                            if "subject" not in res and "subject" in example:
                                res["subject"] = example.get("subject")
                            if "answer_type" not in res and "answer_type" in example:
                                res["answer_type"] = example.get("answer_type")
                            if "code_task" not in res and "code_task" in example:
                                res["code_task"] = example.get("code_task")

                            f_out.write(json.dumps(res) + "\n")
                            f_out.flush() # Ensure safety
                            if cache_file:
                                cache_record = dict(res)
                                cache_record.update({
                                    "cache_scope": "example",
                                    "cache_key": cache_key,
                                    "dataset": dataset_name or "unknown",
                                    "split": split,
                                    "qid": example.get("id"),
                                    "method": m_name,
                                    "params_hash": params_hash,
                                    "final_answer": res.get("pred"),
                                    "tokens_used": res.get("total_tokens"),
                                    "time_s": res.get("time_s"),
                                    "cached": False,
                                })
                                extra = res.get("extra", {})
                                if isinstance(extra, dict) and "candidates" in extra:
                                    cache_record["all_answers"] = extra.get("candidates")
                                cache_file.write(json.dumps(cache_record) + "\n")
                                cache_file.flush()
                        except Exception as e:
                            logger.error(f"Error processing example {example.get('id')}: {e}")
                            continue
                if cache_file:
                    cache_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--seed", type=int, help="Override seed for sampling/shuffle")
    parser.add_argument("--run_group", type=str, help="Run group identifier for multi-seed suites")
    parser.add_argument("--dataset", type=str, help="Override dataset name")
    parser.add_argument("--limit", type=int, help="Override dataset limit")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_eval(
        config=config,
        dataset_override=args.dataset,
        limit_override=args.limit,
        seed_override=args.seed,
        run_group=args.run_group,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

if __name__ == "__main__":
    main()
