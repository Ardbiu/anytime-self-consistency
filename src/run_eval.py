import argparse
import yaml
import json
import os
import time
import hashlib
import signal
from tqdm import tqdm
from .utils import setup_logging, set_seed, ensure_dir
from .data import load_dataset_records
from .models import ModelRunner
from .policies import load_policies_from_config, BwKShadowPricePolicy
from .profiler import LatencyProfiler
from .baselines import (
    run_greedy,
    run_self_correction,
    run_self_consistency,
    run_best_of_n,
    run_speculative_decoding,
    run_medusa,
    run_best_of_n_verifier,
    run_self_consistency_early_stop,
    run_best_of_n_early_stop,
)
from .anytime import run_anytime_sc, run_oracle_stopping
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
    resume: bool = False,
    save_interval: int = 0,
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
    profile_latency_default = bool(config.get("profile_latency", False))
    resume = bool(resume or config.get("resume", False))
    save_interval = int(save_interval or config.get("save_interval", 0) or 0)
    checkpoint_cfg = config.get("checkpoint") or {}
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", False))
    checkpoint_examples = int(checkpoint_cfg.get("examples", 0) or 0)
    checkpoint_max_degradation = float(checkpoint_cfg.get("max_degradation", 0.2))
    checkpoint_policy_name = checkpoint_cfg.get("policy")

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

    stop_requested = {"flag": False}

    def _handle_sigterm(signum, frame):
        stop_requested["flag"] = True
        logger.warning("SIGTERM received; will stop after current example and save state.")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

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

    def resolve_policy_from_name(policy_name):
        if not policy_name:
            return None, None
        if policy_name in {"raw", "none", "question"}:
            policies = load_policies_from_config({"policies": ["raw"]}, root_dir)
            raw_policy = policies[0] if policies else None
            return "raw", raw_policy
        policies = load_policies_from_config({"policies": [policy_name]}, root_dir)
        if not policies:
            raise ValueError(f"Unknown policy '{policy_name}'.")
        return policy_name, policies[0]

    def _params_hash(method_name, run_cfg):
        payload = {
            "dataset": dataset_name,
            "split": split,
            "seed": seed,
            "model_name": config.get("model_name"),
            "max_new_tokens": config.get("max_new_tokens", 512),
            "method_max_new_tokens": run_cfg.get("max_new_tokens"),
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
            "bound_method": run_cfg.get("bound_method"),
            "bound_window": run_cfg.get("bound_window"),
            "bound_discount": run_cfg.get("bound_discount"),
            "prompt_cost": run_cfg.get("prompt_cost"),
            "completion_cost": run_cfg.get("completion_cost"),
            "context_config": run_cfg.get("context_config"),
            "context_policy_name": run_cfg.get("context_policy_name"),
            "bwk_lambda_init": run_cfg.get("bwk_lambda_init"),
            "bwk_eta": run_cfg.get("bwk_eta"),
            "bwk_target_cost": run_cfg.get("bwk_target_cost"),
            "safety_valve": run_cfg.get("safety_valve"),
            "safety_n": run_cfg.get("safety_n"),
            "safety_allocation": run_cfg.get("safety_allocation"),
            "safety_max_cost": run_cfg.get("safety_max_cost"),
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
            "ucb_window": run_cfg.get("ucb_window"),
            "ucb_discount": run_cfg.get("ucb_discount"),
            "batched": run_cfg.get("batched"),
            "batched_seeded": run_cfg.get("batched_seeded"),
            "verifier_model_name": run_cfg.get("verifier_model_name"),
            "verifier_max_new_tokens": run_cfg.get("verifier_max_new_tokens"),
            "verifier_task": run_cfg.get("verifier_task"),
            "verifier_use_flash_attention": run_cfg.get("verifier_use_flash_attention"),
            "verifier_use_compile": run_cfg.get("verifier_use_compile"),
            "draft_model_name": run_cfg.get("draft_model_name"),
            "medusa_model_name": run_cfg.get("medusa_model_name"),
            "medusa_heads": run_cfg.get("medusa_heads"),
            "correction_prompt": run_cfg.get("correction_prompt"),
            "profile_latency": profile_latency,
        }
        data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:12]

    def _cache_key(qid, method_name, params_hash):
        return f"{dataset_name}||{split}||{qid}||{method_name}||{params_hash}"

    def _global_cache_key(method_name, params_hash):
        return f"{dataset_name}||{split}||{method_name}||{params_hash}||global"

    def _find_existing_run_file(prefix: str) -> str:
        if not resume:
            return ""
        if not os.path.exists(config["output_dir"]):
            return ""
        candidates = []
        for fname in os.listdir(config["output_dir"]):
            if not fname.startswith(prefix):
                continue
            if fname.endswith(".jsonl") or fname.endswith(".jsonl.tmp"):
                candidates.append(fname)
        if not candidates:
            return ""
        candidates.sort(key=lambda f: os.path.getmtime(os.path.join(config["output_dir"], f)), reverse=True)
        return os.path.join(config["output_dir"], candidates[0])

    def _parse_run_id_from_path(path: str, prefix: str) -> str:
        base = os.path.basename(path)
        if base.endswith(".jsonl.tmp"):
            base = base[:-len(".jsonl.tmp")]
        elif base.endswith(".jsonl"):
            base = base[:-len(".jsonl")]
        if base.startswith(prefix):
            return base[len(prefix):]
        return run_id

    def _load_processed_ids(path: str) -> set:
        processed = set()
        if not path or not os.path.exists(path):
            return processed
        with open(path, "r") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ex_id = rec.get("example_id")
                if ex_id:
                    processed.add(ex_id)
        return processed

    def _load_existing_records(path: str) -> dict:
        records = {}
        if not path or not os.path.exists(path):
            return records
        with open(path, "r") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ex_id = rec.get("example_id")
                if ex_id:
                    records[ex_id] = rec
        return records

    def _save_state(path: str, state: dict) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f_state:
            json.dump(state, f_state)
        os.replace(tmp_path, path)

    checkpoint_baseline_accuracy = None
    if checkpoint_enabled and checkpoint_examples > 0:
        baseline_policy = None
        baseline_policy_name = None
        if checkpoint_policy_name:
            baseline_policy_name, baseline_policy = resolve_policy_from_name(checkpoint_policy_name)
        if baseline_policy is None:
            for method_cfg in config["methods"]:
                if method_cfg.get("name") == "greedy":
                    baseline_policy_name, baseline_policy = resolve_policy_from_name(
                        method_cfg.get("policy") or method_cfg.get("prompt")
                    )
                    break
        if baseline_policy is None:
            baseline_policy_name, baseline_policy = resolve_policy_from_name("direct")

        baseline_count = min(checkpoint_examples, len(data))
        if baseline_count > 0:
            logger.info(
                f"Checkpoint baseline: greedy/{baseline_policy_name} on first {baseline_count} examples"
            )
            baseline_correct = 0
            for ex in data[:baseline_count]:
                res = run_greedy(model, baseline_policy, ex, seed=seed)
                if res.get("is_correct"):
                    baseline_correct += 1
            checkpoint_baseline_accuracy = baseline_correct / baseline_count
            logger.info(f"Checkpoint baseline accuracy: {checkpoint_baseline_accuracy:.4f}")

    baseline_latency = {}

    for method_cfg in config["methods"]:
        m_name = method_cfg["name"]
        logger.info(f"Running Method: {m_name}")
        profile_latency = bool(method_cfg.get("profile_latency", profile_latency_default))
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
        if m_name == "greedy" or m_name.startswith("greedy_"):
            policy_name, single_policy = resolve_single_policy()
            if policy_name is None:
                try:
                    policy_name, single_policy = resolve_single_policy_from_name("direct")
                except ValueError:
                    policy_name = "raw"
                    single_policy = None
            configs = [{"policy": single_policy, "policy_name": policy_name}]
        elif m_name == "self_correction":
            policy_name, single_policy = resolve_single_policy()
            if policy_name is None:
                try:
                    policy_name, single_policy = resolve_single_policy_from_name("direct")
                except ValueError:
                    policy_name = "raw"
                    single_policy = None
            correction_prompt = method_cfg.get("correction_prompt")
            configs = [{
                "policy": single_policy,
                "policy_name": policy_name,
                "correction_prompt": correction_prompt,
            }]
        elif m_name == "speculative_decoding":
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("speculative_decoding requires a valid policy/prompt.")
            draft_model_name = method_cfg.get("draft_model_name")
            if not draft_model_name:
                raise ValueError("speculative_decoding requires draft_model_name.")
            draft_max_new_tokens = int(method_cfg.get("draft_max_new_tokens", config.get("max_new_tokens", 512)))
            draft_use_flash_attention = bool(method_cfg.get("draft_use_flash_attention", config.get("use_flash_attention", False)))
            draft_use_compile = bool(method_cfg.get("draft_use_compile", config.get("use_compile", False)))
            draft_model = ModelRunner(
                model_name=draft_model_name,
                dtype=config.get("dtype", "auto"),
                max_new_tokens=draft_max_new_tokens,
                use_flash_attention=draft_use_flash_attention,
                use_compile=draft_use_compile,
            )
            configs = [{
                "policy": single_policy,
                "policy_name": policy_name,
                "draft_model": draft_model,
                "draft_model_name": draft_model_name,
                "draft_max_new_tokens": draft_max_new_tokens,
            }]
        elif m_name == "medusa":
            policy_name, single_policy = resolve_single_policy()
            if not single_policy:
                raise ValueError("medusa requires a valid policy/prompt.")
            medusa_heads = int(method_cfg.get("medusa_heads", 4))
            medusa_model_name = method_cfg.get("medusa_model_name")
            medusa_model = None
            if medusa_model_name:
                medusa_model = ModelRunner(
                    model_name=medusa_model_name,
                    dtype=config.get("dtype", "auto"),
                    max_new_tokens=config.get("max_new_tokens", 512),
                    use_flash_attention=config.get("use_flash_attention", False),
                    use_compile=config.get("use_compile", False),
                )
            configs = [{
                "policy": single_policy,
                "policy_name": policy_name,
                "medusa_heads": medusa_heads,
                "medusa_model_name": medusa_model_name,
                "medusa_model": medusa_model,
            }]
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
            prompt_cost = method_cfg.get("prompt_cost", 1.0)
            completion_cost = method_cfg.get("completion_cost", 1.0)
            bound_method = method_cfg.get("bound_method") or method_cfg.get("bound") or "hoeffding"
            bound_window = method_cfg.get("bound_window")
            bound_discount = method_cfg.get("bound_discount")
            context_config = method_cfg.get("context") or method_cfg.get("context_config")
            context_policy_name = method_cfg.get("context_policy")
            context_policy = None
            if context_policy_name:
                _, context_policy = resolve_single_policy_from_name(context_policy_name)
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
                        "ucb_window": method_cfg.get("ucb_window"),
                        "ucb_discount": method_cfg.get("ucb_discount"),
                        "prompt_cost": prompt_cost,
                        "completion_cost": completion_cost,
                        "bound_method": bound_method,
                        "bound_window": bound_window,
                        "bound_discount": bound_discount,
                        "context_config": context_config,
                        "context_policy": context_policy,
                        "context_policy_name": context_policy_name,
                        "bwk_lambda_init": method_cfg.get("bwk_lambda_init", 0.01),
                        "bwk_eta": method_cfg.get("bwk_eta", 0.01),
                        "bwk_target_cost": method_cfg.get("bwk_target_cost"),
                        "safety_valve": bool(method_cfg.get("safety_valve", False)),
                        "safety_n": method_cfg.get("safety_n"),
                        "safety_allocation": method_cfg.get("safety_allocation", "uniform"),
                        "safety_max_cost": method_cfg.get("safety_max_cost"),
                    })
        elif m_name == "oracle_stopping":
            if "budgets" not in method_cfg:
                raise ValueError("oracle_stopping requires budgets.")
            policies = resolve_policy_list()
            if not policies:
                raise ValueError("oracle_stopping requires policies.")
            batch_size = int(method_cfg.get("batch_size", 1))
            allow_unseeded_batch = bool(method_cfg.get("allow_unseeded_batch", False))
            prompt_cost = method_cfg.get("prompt_cost", 1.0)
            completion_cost = method_cfg.get("completion_cost", 1.0)
            configs = []
            for b in method_cfg["budgets"]:
                configs.append({
                    "budget": b,
                    "allocation": method_cfg.get("allocation", "ucb"),
                    "policies": policies,
                    "batch_size": batch_size,
                    "allow_unseeded_batch": allow_unseeded_batch,
                    "ucb_window": method_cfg.get("ucb_window"),
                    "ucb_discount": method_cfg.get("ucb_discount"),
                    "prompt_cost": prompt_cost,
                    "completion_cost": completion_cost,
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
            context_config = method_cfg.get("context") or method_cfg.get("context_config")
            context_policy_name = method_cfg.get("context_policy")
            context_policy = None
            if context_policy_name:
                _, context_policy = resolve_single_policy_from_name(context_policy_name)
            bwk_lambda_init = method_cfg.get("bwk_lambda_init", 0.01)
            bwk_eta = method_cfg.get("bwk_eta", 0.01)

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
                        "context_config": context_config,
                        "context_policy": context_policy,
                        "context_policy_name": context_policy_name,
                        "bwk_lambda_init": bwk_lambda_init,
                        "bwk_eta": bwk_eta,
                    })
        else:
            raise ValueError(f"Unknown method name: {m_name}")

        method_max_new_tokens = method_cfg.get("max_new_tokens")
        if method_max_new_tokens is not None:
            for cfg in configs:
                cfg["max_new_tokens"] = int(method_max_new_tokens)
        
        for run_cfg in configs:
            params_hash = _params_hash(m_name, run_cfg)
            # Construct filename
            suffix_parts = []
            if run_group:
                suffix_parts.append(run_group)
            if num_shards > 1:
                suffix_parts.append(f"shard{shard_id}of{num_shards}")
            if run_group or seed_override is not None:
                suffix_parts.append(f"seed{seed}")
            suffix = "_".join(suffix_parts)

            if m_name == "greedy" or m_name.startswith("greedy_") or m_name in ["self_correction", "speculative_decoding", "medusa"]:
                fname = f"{dataset_name}_{m_name}_{run_cfg['policy_name']}"
                if m_name == "speculative_decoding":
                    draft_label = run_cfg["draft_model_name"].split("/")[-1]
                    fname = f"{fname}_draft{draft_label}"
                if m_name == "medusa":
                    medusa_label = run_cfg.get("medusa_model_name")
                    if medusa_label:
                        medusa_label = medusa_label.split("/")[-1]
                        fname = f"{fname}_medusa{medusa_label}"
                    fname = f"{fname}_h{run_cfg.get('medusa_heads', 4)}"
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
            elif m_name == "oracle_stopping":
                fname = f"{dataset_name}_{m_name}_b{run_cfg['budget']}_{run_cfg['allocation']}"
            elif m_name == "global_anytime_sc":
                fname = f"{dataset_name}_{m_name}_T{run_cfg['global_budget_tokens']}_init{run_cfg['init_k']}_{run_cfg['allocation_policy']}_{run_cfg['policy_name']}"

            if suffix:
                prefix = f"{fname}_{suffix}_"
            else:
                prefix = f"{fname}_"

            existing_path = _find_existing_run_file(prefix)
            run_id_for_run = run_id
            if existing_path:
                run_id_for_run = _parse_run_id_from_path(existing_path, prefix)
            final_path = os.path.join(config["output_dir"], f"{prefix}{run_id_for_run}.jsonl")
            active_path = f"{final_path}.tmp"
            if existing_path and m_name == "global_anytime_sc" and resume:
                logger.info(f"Skipping global run (resume detected): {os.path.basename(existing_path)}")
                continue
            if existing_path:
                if existing_path.endswith(".jsonl.tmp"):
                    active_path = existing_path
                else:
                    os.replace(existing_path, active_path)
            if not resume and os.path.exists(active_path):
                os.remove(active_path)

            logger.info(f"Starting run for {os.path.basename(final_path)}...")

            state_path = f"{final_path}.state.json"
            processed_records = {}
            processed_ids = set()
            saved_state = None
            if resume:
                if os.path.exists(state_path):
                    try:
                        with open(state_path, "r") as f_state:
                            saved_state = json.load(f_state)
                        processed_ids = set(saved_state.get("processed_ids", []))
                    except Exception:
                        saved_state = None
                processed_records = _load_existing_records(active_path)
                if not processed_ids:
                    processed_ids = set(processed_records.keys())

            context_state = None
            if m_name == "anytime_sc" and run_cfg.get("allocation") in {"contextual_bwk", "bwk_contextual"}:
                context_state = {"bwk_policies": {}}
                if saved_state and isinstance(saved_state.get("bwk_state"), dict):
                    eta_val = float(saved_state.get("bwk_eta", run_cfg.get("bwk_eta", 0.01)))
                    for key, lambda_val in saved_state.get("bwk_state", {}).items():
                        context_state["bwk_policies"][key] = BwKShadowPricePolicy(
                            lambda_init=float(lambda_val),
                            eta=eta_val,
                        )

            def _build_state() -> dict:
                state = {
                    "dataset": dataset_name,
                    "split": split,
                    "method": m_name,
                    "params_hash": params_hash,
                    "run_id": run_id_for_run,
                    "run_group": run_group,
                    "seed": seed,
                    "processed_ids": sorted(processed_ids),
                    "budget_tokens": run_cfg.get("budget") or run_cfg.get("global_budget_tokens"),
                }
                if context_state and "bwk_policies" in context_state:
                    state["bwk_state"] = {
                        key: policy.lambda_price
                        for key, policy in context_state["bwk_policies"].items()
                    }
                    state["bwk_eta"] = run_cfg.get("bwk_eta", 0.01)
                return state

            default_max_new_tokens = model.max_new_tokens
            if run_cfg.get("max_new_tokens") is not None:
                model.max_new_tokens = int(run_cfg["max_new_tokens"])
            else:
                model.max_new_tokens = default_max_new_tokens

            if m_name == "global_anytime_sc":
                global_cache_key = _global_cache_key(m_name, params_hash)
                if cache_enabled and global_cache_key in cache_global_done:
                    logger.info(f"Skipping global run (cached): {global_cache_key}")
                    continue

            run_completed = True
            with open(active_path, 'a') as f_out:
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
                            context_config=run_cfg.get("context_config"),
                            context_policy=run_cfg.get("context_policy"),
                            bwk_lambda_init=run_cfg.get("bwk_lambda_init", 0.01),
                            bwk_eta=run_cfg.get("bwk_eta", 0.01),
                        )
                    except Exception as e:
                        logger.error(f"Error running global_anytime_sc: {e}")
                        continue

                    for res in results:
                        res["dataset"] = dataset_name or "unknown"
                        res["split"] = split
                        res["model_name"] = config.get("model_name", "unknown")
                        res["run_id"] = run_id_for_run
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
                        if run_cfg.get("context_config") is not None:
                            res["context_config"] = run_cfg["context_config"]
                        if run_cfg.get("context_policy_name") is not None:
                            res["context_policy_name"] = run_cfg["context_policy_name"]
                        if run_cfg.get("bwk_lambda_init") is not None:
                            res["bwk_lambda_init"] = run_cfg["bwk_lambda_init"]
                        if run_cfg.get("bwk_eta") is not None:
                            res["bwk_eta"] = run_cfg["bwk_eta"]
                        if run_cfg.get("max_new_tokens") is not None:
                            res["method_max_new_tokens"] = run_cfg["max_new_tokens"]
                        res["latency_benefit"] = None
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
                            "run_id": run_id_for_run,
                            "run_group": run_group,
                            "seed": seed,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }) + "\n")
                        cache_file.flush()
                else:
                    checkpoint_active = (
                        checkpoint_baseline_accuracy is not None
                        and checkpoint_examples > 0
                        and m_name != "greedy"
                    )
                    seen_examples = 0
                    correct_examples = 0
                    examples_since_save = 0
                    run_completed = True

                    for example in tqdm(data):
                        try:
                            ex_id = example.get("id")
                            if resume and ex_id in processed_ids:
                                if checkpoint_active and ex_id in processed_records:
                                    seen_examples += 1
                                    if processed_records[ex_id].get("is_correct"):
                                        correct_examples += 1
                                continue
                            cache_key = _cache_key(example.get("id"), m_name, params_hash)
                            if cache_enabled and cache_key in cache_records:
                                cached = dict(cache_records[cache_key])
                                cached["cached"] = True
                                cached["run_id"] = run_id_for_run
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
                                if "answer_type" not in cached and "answer_type" in example:
                                    cached["answer_type"] = example.get("answer_type")
                                if "code_task" not in cached and "code_task" in example:
                                    cached["code_task"] = example.get("code_task")
                                if "parse_error" not in cached:
                                    cached["parse_error"] = example.get("parse_error", False)
                                if "subject" not in cached and "subject" in example:
                                    cached["subject"] = example.get("subject")
                                if ex_id and m_name == "greedy" and cached.get("time_s") is not None:
                                    baseline_latency[ex_id] = cached.get("time_s")
                                baseline_time = baseline_latency.get(ex_id)
                                if baseline_time is not None and cached.get("time_s") is not None:
                                    cached["latency_benefit"] = baseline_time - cached.get("time_s")
                                else:
                                    cached["latency_benefit"] = None
                                if checkpoint_active:
                                    seen_examples += 1
                                    if cached.get("is_correct"):
                                        correct_examples += 1
                                    if seen_examples >= checkpoint_examples:
                                        current_acc = correct_examples / max(seen_examples, 1)
                                        threshold = checkpoint_baseline_accuracy * (1.0 - checkpoint_max_degradation)
                                        if current_acc < threshold:
                                            cached["early_stop_triggered"] = True
                                            f_out.write(json.dumps(cached) + "\n")
                                            f_out.flush()
                                            logger.warning(
                                                f"Checkpoint stop for {m_name}: acc={current_acc:.3f} < {threshold:.3f}"
                                            )
                                            break
                                    cached["early_stop_triggered"] = False
                                f_out.write(json.dumps(cached) + "\n")
                                f_out.flush()
                                if ex_id:
                                    processed_ids.add(ex_id)
                                examples_since_save += 1
                                if save_interval and examples_since_save >= save_interval:
                                    _save_state(state_path, _build_state())
                                    examples_since_save = 0
                                if stop_requested["flag"]:
                                    _save_state(state_path, _build_state())
                                    run_completed = False
                                    break
                                continue
                            profiler = LatencyProfiler(enabled=profile_latency)
                            # Dispatch
                            if m_name == "greedy" or m_name.startswith("greedy_"):
                                res = run_greedy(model, run_cfg["policy"], example, seed=seed, profiler=profiler)
                            elif m_name == "self_correction":
                                res = run_self_correction(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    seed=seed,
                                    correction_prompt=run_cfg.get("correction_prompt"),
                                    profiler=profiler,
                                )
                            elif m_name == "speculative_decoding":
                                res = run_speculative_decoding(
                                    model,
                                    run_cfg["draft_model"],
                                    run_cfg["policy"],
                                    example,
                                    seed=seed,
                                    profiler=profiler,
                                )
                            elif m_name == "medusa":
                                medusa_model = run_cfg.get("medusa_model") or model
                                res = run_medusa(
                                    medusa_model,
                                    run_cfg["policy"],
                                    example,
                                    seed=seed,
                                    medusa_heads=run_cfg.get("medusa_heads", 4),
                                    profiler=profiler,
                                )
                            elif m_name == "self_consistency":
                                res = run_self_consistency(
                                    model,
                                    run_cfg["policy"],
                                    example,
                                    run_cfg["n"],
                                    seed=seed,
                                    batched=run_cfg.get("batched", False),
                                    batched_seeded=run_cfg.get("batched_seeded", False),
                                    profiler=profiler,
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
                                    profiler=profiler,
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
                                    profiler=profiler,
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
                                    profiler=profiler,
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
                                    profiler=profiler,
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
                                    ucb_window=run_cfg.get("ucb_window"),
                                    ucb_discount=run_cfg.get("ucb_discount"),
                                    prompt_cost=run_cfg.get("prompt_cost", 1.0),
                                    completion_cost=run_cfg.get("completion_cost", 1.0),
                                    bound_method=run_cfg.get("bound_method", "hoeffding"),
                                    bound_window=run_cfg.get("bound_window"),
                                    bound_discount=run_cfg.get("bound_discount"),
                                    context_config=run_cfg.get("context_config"),
                                    context_policy=run_cfg.get("context_policy"),
                                    context_state=context_state,
                                    bwk_lambda_init=run_cfg.get("bwk_lambda_init", 0.01),
                                    bwk_eta=run_cfg.get("bwk_eta", 0.01),
                                    bwk_target_cost=run_cfg.get("bwk_target_cost"),
                                    safety_valve=run_cfg.get("safety_valve", False),
                                    safety_n=run_cfg.get("safety_n"),
                                    safety_allocation=run_cfg.get("safety_allocation", "uniform"),
                                    safety_max_cost=run_cfg.get("safety_max_cost"),
                                    profiler=profiler,
                                )
                            elif m_name == "oracle_stopping":
                                res = run_oracle_stopping(
                                    model,
                                    run_cfg["policies"],
                                    example,
                                    run_cfg["budget"],
                                    run_cfg["allocation"],
                                    seed=seed,
                                    batch_size=run_cfg["batch_size"],
                                    allow_unseeded_batch=run_cfg["allow_unseeded_batch"],
                                    ucb_window=run_cfg.get("ucb_window"),
                                    ucb_discount=run_cfg.get("ucb_discount"),
                                    prompt_cost=run_cfg.get("prompt_cost", 1.0),
                                    completion_cost=run_cfg.get("completion_cost", 1.0),
                                    profiler=profiler,
                                )
                            else:
                                raise ValueError(f"Unknown method name: {m_name}")

                            # Inject Metadata
                            res["dataset"] = dataset_name or "unknown"
                            res["split"] = split
                            res["model_name"] = config.get("model_name", "unknown")
                            res["run_id"] = run_id_for_run
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
                            if run_cfg.get("draft_model_name") is not None:
                                res["draft_model_name"] = run_cfg["draft_model_name"]
                            if run_cfg.get("medusa_model_name") is not None:
                                res["medusa_model_name"] = run_cfg["medusa_model_name"]
                            if run_cfg.get("medusa_heads") is not None:
                                res["medusa_heads"] = run_cfg["medusa_heads"]
                            if run_cfg.get("correction_prompt") is not None:
                                res["correction_prompt"] = run_cfg["correction_prompt"]
                            if run_cfg.get("bound_method") is not None:
                                res["bound_method"] = run_cfg["bound_method"]
                            if run_cfg.get("bound_window") is not None:
                                res["bound_window"] = run_cfg["bound_window"]
                            if run_cfg.get("bound_discount") is not None:
                                res["bound_discount"] = run_cfg["bound_discount"]
                            if run_cfg.get("context_config") is not None:
                                res["context_config"] = run_cfg["context_config"]
                            if run_cfg.get("context_policy_name") is not None:
                                res["context_policy_name"] = run_cfg["context_policy_name"]
                            if run_cfg.get("bwk_lambda_init") is not None:
                                res["bwk_lambda_init"] = run_cfg["bwk_lambda_init"]
                            if run_cfg.get("bwk_eta") is not None:
                                res["bwk_eta"] = run_cfg["bwk_eta"]
                            if run_cfg.get("bwk_target_cost") is not None:
                                res["bwk_target_cost"] = run_cfg["bwk_target_cost"]
                            if run_cfg.get("safety_valve") is not None:
                                res["safety_valve"] = run_cfg["safety_valve"]
                            if run_cfg.get("safety_n") is not None:
                                res["safety_n"] = run_cfg["safety_n"]
                            if run_cfg.get("safety_allocation") is not None:
                                res["safety_allocation"] = run_cfg["safety_allocation"]
                            if run_cfg.get("safety_max_cost") is not None:
                                res["safety_max_cost"] = run_cfg["safety_max_cost"]
                            if run_cfg.get("batch_size") is not None:
                                res["batch_size"] = run_cfg["batch_size"]
                            if run_cfg.get("allow_unseeded_batch") is not None:
                                res["allow_unseeded_batch"] = run_cfg["allow_unseeded_batch"]
                            if run_cfg.get("ucb_window") is not None:
                                res["ucb_window"] = run_cfg["ucb_window"]
                            if run_cfg.get("ucb_discount") is not None:
                                res["ucb_discount"] = run_cfg["ucb_discount"]
                            if run_cfg.get("prompt_cost") is not None:
                                res["prompt_cost"] = run_cfg["prompt_cost"]
                            if run_cfg.get("completion_cost") is not None:
                                res["completion_cost"] = run_cfg["completion_cost"]
                            if run_cfg.get("batched") is not None:
                                res["batched"] = run_cfg["batched"]
                            if run_cfg.get("batched_seeded") is not None:
                                res["batched_seeded"] = run_cfg["batched_seeded"]
                            if run_cfg.get("max_new_tokens") is not None:
                                res["method_max_new_tokens"] = run_cfg["max_new_tokens"]
                            if ex_id and m_name == "greedy" and res.get("time_s") is not None:
                                baseline_latency[ex_id] = res.get("time_s")
                            baseline_time = baseline_latency.get(ex_id)
                            if baseline_time is not None and res.get("time_s") is not None:
                                res["latency_benefit"] = baseline_time - res.get("time_s")
                            else:
                                res["latency_benefit"] = None

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
                            if checkpoint_active:
                                seen_examples += 1
                                if res.get("is_correct"):
                                    correct_examples += 1
                                if seen_examples >= checkpoint_examples:
                                    current_acc = correct_examples / max(seen_examples, 1)
                                    threshold = checkpoint_baseline_accuracy * (1.0 - checkpoint_max_degradation)
                                    if current_acc < threshold:
                                        res["early_stop_triggered"] = True
                                        f_out.write(json.dumps(res) + "\n")
                                        f_out.flush()
                                        logger.warning(
                                            f"Checkpoint stop for {m_name}: acc={current_acc:.3f} < {threshold:.3f}"
                                        )
                                        break
                                res["early_stop_triggered"] = False

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
                            if ex_id:
                                processed_ids.add(ex_id)
                            examples_since_save += 1
                            if save_interval and examples_since_save >= save_interval:
                                _save_state(state_path, _build_state())
                                examples_since_save = 0
                            if stop_requested["flag"]:
                                _save_state(state_path, _build_state())
                                run_completed = False
                                break
                        except Exception as e:
                            logger.error(f"Error processing example {example.get('id')}: {e}")
                            continue
                if cache_file:
                    cache_file.close()
            if run_completed and os.path.exists(active_path):
                os.replace(active_path, final_path)
            if stop_requested["flag"]:
                return
            model.max_new_tokens = default_max_new_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--seed", type=int, help="Override seed for sampling/shuffle")
    parser.add_argument("--run_group", type=str, help="Run group identifier for multi-seed suites")
    parser.add_argument("--dataset", type=str, help="Override dataset name")
    parser.add_argument("--limit", type=int, help="Override dataset limit")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs")
    parser.add_argument("--save_interval", type=int, default=0, help="Save state every N examples")
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
        resume=args.resume,
        save_interval=args.save_interval,
    )

if __name__ == "__main__":
    main()
