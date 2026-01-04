#!/usr/bin/env python
"""
Latency Benchmark: Serial vs Batched Inference

This script compares wall-clock latency between:
1. Serial generation (one prompt at a time)
2. Batched generation (multiple prompts in parallel)

This addresses the ICML reviewer concern that "token savings ≠ latency savings"
by providing real-world timing comparisons.
"""

import argparse
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import ModelRunner


def run_serial(model: ModelRunner, prompts: list, temperature: float = 0.7) -> dict:
    """Run generation serially (one at a time)."""
    results = []
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        res = model.generate(
            prompt,
            temperature=temperature,
            do_sample=True,
            seed=42 + i,
        )
        results.append(res)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    total_tokens = sum(r["completion_tokens"] for r in results)
    
    return {
        "mode": "serial",
        "n_prompts": len(prompts),
        "total_time_s": total_time,
        "time_per_prompt_s": total_time / len(prompts),
        "total_tokens_generated": total_tokens,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
    }


def run_batched(model: ModelRunner, prompts: list, temperature: float = 0.7) -> dict:
    """Run generation in a batch."""
    start_time = time.time()
    
    # No seeds for true batched mode (faster)
    results = model.generate_batch(
        prompts,
        temperature=temperature,
        do_sample=True,
        seeds=None,  # Don't use seeds to enable true batching
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    total_tokens = sum(r["completion_tokens"] for r in results)
    
    return {
        "mode": "batched",
        "n_prompts": len(prompts),
        "total_time_s": total_time,
        "time_per_prompt_s": total_time / len(prompts),
        "total_tokens_generated": total_tokens,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
    }


def format_results(serial: dict, batched: dict) -> str:
    """Format comparison results as a table."""
    speedup = serial["total_time_s"] / batched["total_time_s"] if batched["total_time_s"] > 0 else 0
    
    lines = [
        "",
        "=" * 70,
        "LATENCY BENCHMARK: Serial vs Batched Inference",
        "=" * 70,
        "",
        f"{'Metric':<30} {'Serial':>15} {'Batched':>15}",
        "-" * 70,
        f"{'Number of Prompts':<30} {serial['n_prompts']:>15} {batched['n_prompts']:>15}",
        f"{'Total Time (s)':<30} {serial['total_time_s']:>15.2f} {batched['total_time_s']:>15.2f}",
        f"{'Time per Prompt (s)':<30} {serial['time_per_prompt_s']:>15.3f} {batched['time_per_prompt_s']:>15.3f}",
        f"{'Total Tokens Generated':<30} {serial['total_tokens_generated']:>15} {batched['total_tokens_generated']:>15}",
        f"{'Tokens per Second':<30} {serial['tokens_per_second']:>15.1f} {batched['tokens_per_second']:>15.1f}",
        "-" * 70,
        f"{'SPEEDUP (Serial / Batched)':<30} {speedup:>15.2f}x",
        "",
        "=" * 70,
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark serial vs batched inference latency"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model to benchmark"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of prompts to generate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens per generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup generations before timing"
    )
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = ModelRunner(args.model, max_new_tokens=args.max_new_tokens)
    
    # Create test prompts (GSM8K-style)
    prompts = [
        f"Question: What is {i+2} + {i+3}? Think step by step. Final:"
        for i in range(args.n)
    ]
    
    # Warmup
    print(f"Running {args.warmup} warmup generations...")
    for i in range(args.warmup):
        model.generate(prompts[0], do_sample=True, seed=i)
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\nBenchmarking with {args.n} prompts...")
    
    # Run serial benchmark
    print("  Running serial mode...")
    serial_results = run_serial(model, prompts, args.temperature)
    
    # Clear cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Run batched benchmark
    print("  Running batched mode...")
    batched_results = run_batched(model, prompts, args.temperature)
    
    # Print results
    print(format_results(serial_results, batched_results))
    
    # Interpretation
    speedup = serial_results["total_time_s"] / batched_results["total_time_s"]
    if speedup > 1.5:
        print("✓ Batching provides significant speedup. Wall-clock savings are real.")
    elif speedup > 1.1:
        print("~ Batching provides modest speedup. Consider larger batch sizes.")
    else:
        print("✗ No speedup from batching. This may be due to memory constraints or small batch size.")


if __name__ == "__main__":
    main()
