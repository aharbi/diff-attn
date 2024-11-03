import os
import random
import numpy as np
import torch
import triton
import triton.testing

from pathlib import Path
from typing import Callable
from layers import (
    MultiheadAttn,
    MultiheadFlashAttn,
    MultiheadDiffAttn,
    MultiheadFlashDiffAttn,
)


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


def random_tensor_generator(*shape, dtype: torch.dtype, device: str):
    return torch.empty(*shape, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)


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
        args={"num_heads": 16, "head_dim": 64, "batch_size": 1},
    )
)
def benchmark_runtime(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    implementation: str,
    dtype=torch.float16,
    device="cuda",
):

    quantiles = [0.5, 0.1, 0.9]

    random_tensor = lambda *shape: random_tensor_generator(
        *shape, dtype=dtype, device=device
    )

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

    if fn in [MultiheadDiffAttn, MultiheadFlashDiffAttn]:
        q1 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
        q2 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
        k1 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
        k2 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
        v = random_tensor(batch_size, num_heads, seq_len, head_dim)
    else:
        q = random_tensor(batch_size, num_heads, seq_len, head_dim)
        k = random_tensor(batch_size, num_heads, seq_len, head_dim)
        v = random_tensor(batch_size, num_heads, seq_len, head_dim)

    def run():
        if fn in [MultiheadDiffAttn, MultiheadFlashDiffAttn]:
            fn(q1, q2, k1, k2, v)
        else:
            fn(q, k, v)

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
        args={"num_heads": 16, "head_dim": 64, "batch_size": 1},
    )
)
def benchmark_memory_usage(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    implementation: str,
    dtype=torch.float16,
    device="cuda",
):

    random_tensor = lambda *shape: random_tensor_generator(
        *shape, dtype=dtype, device=device
    )

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

    if fn in [MultiheadDiffAttn, MultiheadFlashDiffAttn]:
        q1 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
        q2 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
        k1 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
        k2 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
        v = random_tensor(batch_size, num_heads, seq_len, head_dim)
    else:
        q = random_tensor(batch_size, num_heads, seq_len, head_dim)
        k = random_tensor(batch_size, num_heads, seq_len, head_dim)
        v = random_tensor(batch_size, num_heads, seq_len, head_dim)

    torch.cuda.reset_peak_memory_stats()

    if fn in [MultiheadDiffAttn, MultiheadFlashDiffAttn]:
        fn(q1, q2, k1, k2, v)
    else:
        fn(q, k, v)

    max_memory = torch.cuda.max_memory_allocated() / (1024**2)

    return max_memory


def benchmark_difference(
    fn1: Callable = MultiheadDiffAttn,
    fn2: Callable = MultiheadFlashDiffAttn,
    seq_len: int = 32,
    head_dim: int = 128,
    num_heads: int = 16,
    batch_size: int = 1,
    N: int = 100,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    save_path: Path = "./results/difference/",
    benchmark_name: str = "difference",
):

    random_tensor = lambda *shape: random_tensor_generator(
        *shape, dtype=dtype, device=device
    )

    diff = np.zeros(N)

    for i in range(N):
        if (fn1 and fn2) in [MultiheadDiffAttn, MultiheadFlashDiffAttn]:
            q1 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
            q2 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
            k1 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
            k2 = random_tensor(batch_size, num_heads, seq_len, head_dim // 2)
            v = random_tensor(batch_size, num_heads, seq_len, head_dim)

            y1 = fn1(q1, q2, k1, k2, v)
            y2 = fn2(q1, q2, k1, k2, v)
        else:
            q = random_tensor(batch_size, num_heads, seq_len, head_dim)
            k = random_tensor(batch_size, num_heads, seq_len, head_dim)
            v = random_tensor(batch_size, num_heads, seq_len, head_dim)

            y1 = fn1(q, k, v)
            y2 = fn2(q, k, v)

        assert torch.allclose(y1, y2, atol=1e-2, rtol=0)

        diff[i] = torch.abs(y1 - y2).mean().item()

    np.savetxt(os.path.join(save_path, f"{benchmark_name}.csv"), diff, delimiter=",")


if __name__ == "__main__":
    benchmark_memory_usage.run(print_data=True, save_path="./results/memory/")
    benchmark_runtime.run(print_data=True, save_path="./results/runtime/")
    benchmark_difference()
