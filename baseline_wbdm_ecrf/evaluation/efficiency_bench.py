"""
efficiency_bench.py — Params/FLOPs/memory/inference-time benchmark,
matching Table 8 (aIoU/Time comparison against MedSegDiff/BGDiffSeg) and
Table 9 (Time vs. sampling-step count T=3..1000).

Usage:
    python evaluation/efficiency_bench.py --model_config configs/model_adaptive.yaml --steps 3
    python evaluation/efficiency_bench.py --model_config configs/model_adaptive.yaml --sweep_steps 3 10 50 100 200 500 1000
"""
import argparse
import time

import torch
import yaml

from models import build_model_from_config

try:
    from thop import profile as thop_profile
    _HAS_THOP = True
except ImportError:
    _HAS_THOP = False


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_inference(model, image_size: int, channels: int, device: str, num_steps: int, warmup: int = 3, iters: int = 20) -> float:
    model.eval()
    dummy = torch.randn(1, channels, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model.predict(dummy, num_sample_steps=num_steps)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model.predict(dummy, num_sample_steps=num_steps)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) / iters
    return elapsed


def peak_memory_gb(device: str) -> float:
    if device != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--steps", type=int, default=None, help="single sampling-step count to benchmark")
    parser.add_argument("--sweep_steps", type=int, nargs="+", default=None, help="benchmark across multiple step counts (Table 9 style)")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    cfg = dict(base_cfg)
    cfg["model"] = model_cfg

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    image_size = cfg["data"]["image_size"]
    channels = cfg["data"]["channels"]

    model = build_model_from_config(cfg).to(device)
    params = count_params(model)
    print(f"Params: {params/1e6:.2f}M")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_counts = args.sweep_steps or [args.steps or cfg["diffusion"].get("T_sample", 3)]
    for steps in step_counts:
        t = benchmark_inference(model, image_size, channels, device, num_steps=steps)
        mem = peak_memory_gb(device)
        print(f"T_sample={steps:>5d}  time/image={t:.4f}s  peak_mem={mem:.2f}GB")


if __name__ == "__main__":
    main()
