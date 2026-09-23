import argparse
import statistics
import time

import torch

import sgl_kernel  # noqa: F401
from sglang.kernels.ops.moe.router import fused_moe_router_shim


def reference_router(hidden_states, router_weight, topk, softcap):
    logits = hidden_states.float() @ router_weight.t()
    logits = torch.tanh(logits / softcap) * softcap
    scores = torch.softmax(logits, dim=-1)
    return torch.topk(scores, topk, dim=-1)


def fused_router(hidden_states, router_weight, topk, softcap):
    return fused_moe_router_shim(
        softcap,
        hidden_states,
        router_weight,
        topk,
        False,
    )


def benchmark(function, hidden_states, router_weight, topk, softcap, warmup, repeats):
    for _ in range(warmup):
        function(hidden_states, router_weight, topk, softcap)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function(hidden_states, router_weight, topk, softcap)
        samples.append((time.perf_counter_ns() - start) / 1e3)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Xeon fused Grok MoE router")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    compiled_router = torch.compile(reference_router, fullgraph=True)

    print(f"threads={args.threads} hidden=6144 experts=8 topk=2 softcap=30")
    print("dtype tokens fused_us eager_us compile_us speedup_eager speedup_compile")
    for dtype in (torch.bfloat16, torch.float16):
        router_weight = torch.randn(8, 6144, dtype=torch.float32)
        for tokens in args.tokens:
            hidden_states = torch.randn(tokens, 6144, dtype=dtype) / 6144**0.5
            fused_us = benchmark(
                fused_router,
                hidden_states,
                router_weight,
                2,
                30.0,
                args.warmup,
                args.repeats,
            )
            eager_us = benchmark(
                reference_router,
                hidden_states,
                router_weight,
                2,
                30.0,
                args.warmup,
                args.repeats,
            )
            compile_us = benchmark(
                compiled_router,
                hidden_states,
                router_weight,
                2,
                30.0,
                args.warmup,
                args.repeats,
            )
            print(
                f"{str(dtype).removeprefix('torch.'):>8} {tokens:6d} "
                f"{fused_us:8.2f} {eager_us:8.2f} {compile_us:10.2f} "
                f"{eager_us / fused_us:13.2f}x {compile_us / fused_us:15.2f}x"
            )


if __name__ == "__main__":
    main()
