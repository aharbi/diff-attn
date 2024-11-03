import os
import random
import numpy as np
import torch
import triton
import triton.testing

from pathlib import Path
from typing import Callable
from layers import *


def set_deterministic(seed=0):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096],
        y_log=True,
        ylabel="Runtime (ms)",
        line_arg="implementation",
        line_vals=["A", "FA", "DA", "DFA"],
        line_names=["A", "FA", "DA", "DFA"],
        plot_name="runtime",
        args={"embed_dim": 512, "num_heads": 16, "batch_size": 1},
    )
)
def benchmark_runtime(
    seq_len: int, embed_dim: int, num_heads: int, batch_size: int, implementation: int
):
    q = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")

    quantiles = [0.5, 0.1, 0.9]

    if implementation == "A":
        fn = MultiheadAttn
    elif implementation == "FA":
        fn = MultiheadFlashAttn
    elif implementation == "DA":
        fn = MultiheadDiffAttn
    elif implementation == "DFA":
        fn = MultiheadFlashDiffAttn
    else:
        raise ValueError("Invalid implementation")

    def run():
        fn(q, k, v, embed_dim, num_heads)

    ms, min_ms, max_ms = triton.testing.do_bench(lambda: run(), quantiles=quantiles)

    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096],
        y_log=True,
        ylabel="Peak Memory Usage (MB)",
        line_arg="implementation",
        line_vals=["A", "FA", "DA", "DFA"],
        line_names=["A", "FA", "DA", "DFA"],
        plot_name="memory",
        args={"embed_dim": 512, "num_heads": 16, "batch_size": 1},
    )
)
def benchmark_memory_usage(
    seq_len: int, embed_dim: int, num_heads: int, batch_size: int, implementation: str
):
    q = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")

    if implementation == "A":
        fn = MultiheadAttn
    elif implementation == "FA":
        fn = MultiheadFlashAttn
    elif implementation == "DA":
        fn = MultiheadDiffAttn
    elif implementation == "DFA":
        fn = MultiheadFlashDiffAttn
    else:
        raise ValueError("Invalid implementation")

    torch.cuda.reset_peak_memory_stats()
    fn(q, k, v, embed_dim, num_heads)
    max_memory = torch.cuda.max_memory_allocated() / (1024**2)

    return max_memory


def benchmark_difference(
    fn1: Callable = MultiheadDiffAttn,
    fn2: Callable = MultiheadFlashDiffAttn,
    seq_len: int = 16,
    embed_dim: int = 512,
    num_heads: int = 16,
    batch_size: int = 1,
    N: int = 100,
    save_path: Path = "./results/difference/",
    benchmark_name: str = "difference",
):

    set_deterministic(0)

    diff = np.zeros(N)

    for i in range(N):
        q = torch.rand(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
        k = torch.rand(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
        v = torch.rand(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")

        y1 = fn1(q, k, v, embed_dim, num_heads)
        y2 = fn2(q, k, v, embed_dim, num_heads)

        diff[i] = torch.abs(y1 - y2).mean().item()

    np.savetxt(os.path.join(save_path, f"{benchmark_name}.csv"), diff, delimiter=",")


if __name__ == "__main__":
    benchmark_memory_usage.run(print_data=True, save_path="./results/memory/")
    benchmark_runtime.run(print_data=True, save_path="./results/runtime/")
    benchmark_difference()
