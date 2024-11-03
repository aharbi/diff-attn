import torch
import triton
import triton.testing

from src.layers import *


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="implementation",
        line_vals=["standard", "differential"],
        line_names=["MultiheadFlashAttn", "MultiheadFlashDiffAttn"],
        plot_name="Attention Mechanism Performance",
        args={"embed_dim": 512, "num_heads": 16, "batch_size": 64},
    )
)
def benchmark_attention(seq_len, embed_dim, num_heads, batch_size, implementation):
    x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")

    if implementation == "standard":
        fn = MultiheadFlashAttn
    elif implementation == "differential":
        fn = MultiheadFlashDiffAttn

    def run():
        fn(x, x, x, embed_dim, num_heads)

    ms = triton.testing.do_bench(run)

    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096],
        line_arg="implementation",
        line_vals=["standard", "differential"],
        line_names=["MultiheadFlashAttn", "MultiheadFlashDiffAttn"],
        plot_name="Attention Mechanism Performance",
        args={"embed_dim": 512, "num_heads": 16, "batch_size": 1},
    )
)
def benchmark_memory_usage(seq_len, embed_dim, num_heads, batch_size, implementation):
    x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")

    if implementation == "standard":
        fn = MultiheadAttn
    elif implementation == "differential":
        fn = MultiheadDiffAttn

    torch.cuda.reset_peak_memory_stats()
    fn(x, x, x, embed_dim, num_heads)
    max_memory = torch.cuda.max_memory_allocated() / (1024**2)

    return max_memory


if __name__ == "__main__":
    benchmark_memory_usage.run(print_data=True, show_plots=True)
    benchmark_attention.run(print_data=True, show_plots=True)
