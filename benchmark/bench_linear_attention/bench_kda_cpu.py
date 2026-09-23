"""Benchmark Xeon chunk_kda against eager PyTorch and TorchInductor.

Run on the target Xeon host; this script prints measured median latency and
speedup and intentionally contains no checked-in synthetic results.
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.kda_cpu import chunk_kda


SHAPES = (
    (128, 8, 128, 128),
    (512, 8, 128, 128),
    (2048, 8, 128, 128),
    (4096, 4, 192, 128),
)


def make_inputs(tokens, heads, key_dim, value_dim, dtype):
    torch.manual_seed(42)
    q = torch.randn(1, tokens, heads, key_dim).to(dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, heads, value_dim).to(dtype)
    gate = (torch.randn_like(q) * 0.05 - 0.1).contiguous()
    beta = torch.sigmoid(torch.randn(1, tokens, heads)).to(dtype)
    state = torch.randn(1, heads, value_dim, key_dim) * 0.01
    indices = torch.zeros(1, dtype=torch.int32)
    offsets = torch.tensor([0, tokens], dtype=torch.int32)
    return q, k, v, gate, beta, state, indices, offsets


def eager_reference(q, k, v, gate, beta, state):
    scale = q.shape[-1] ** -0.5
    outputs = []
    current = state[0]
    for token in range(q.shape[1]):
        qt = F.normalize(q[0, token].float(), dim=-1, eps=1e-6)
        kt = F.normalize(k[0, token].float(), dim=-1, eps=1e-6)
        current = current * gate[0, token].float().exp().unsqueeze(-2)
        residual = v[0, token].float() - torch.einsum("hvk,hk->hv", current, kt)
        current = current + torch.einsum(
            "hv,hk->hvk", residual * beta[0, token].float()[:, None], kt
        )
        outputs.append(torch.einsum("hvk,hk->hv", current, qt) * scale)
    return torch.stack(outputs).unsqueeze(0).to(v.dtype), current.unsqueeze(0)


def measure(fn, state, warmup, iterations):
    for _ in range(warmup):
        fn(state.clone())
    samples = []
    for _ in range(iterations):
        working_state = state.clone()
        start = time.perf_counter()
        fn(working_state)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def run_case(shape, dtype, warmup, iterations):
    q, k, v, gate, beta, state, indices, offsets = make_inputs(*shape, dtype)

    def kernel(working_state):
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            initial_state=working_state,
            initial_state_indices=indices,
            cu_seqlens=offsets,
            use_qk_l2norm_in_kernel=True,
        )

    def eager(working_state):
        return eager_reference(q, k, v, gate, beta, working_state)

    kernel_ms = measure(kernel, state, warmup, iterations)
    eager_ms = measure(eager, state, warmup, iterations)
    results = [("sgl-kernel", kernel_ms, eager_ms / kernel_ms)]
    try:
        compiled = torch.compile(eager_reference, fullgraph=True)

        def inductor(working_state):
            return compiled(q, k, v, gate, beta, working_state)

        inductor_ms = measure(inductor, state, warmup, iterations)
        results.append(("torch.compile", inductor_ms, eager_ms / inductor_ms))
    except (RuntimeError, torch._dynamo.exc.Unsupported) as error:
        print(f"  torch.compile unavailable: {error}")
    print(f"T={shape[0]:4d} H={shape[1]} K={shape[2]} V={shape[3]} {dtype}")
    print(f"  {'eager':13s} {eager_ms:9.3f} ms  1.00x")
    for name, latency, speedup in results:
        print(f"  {name:13s} {latency:9.3f} ms  {speedup:5.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "both"), default="both")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dtypes = (torch.bfloat16, torch.float16) if args.dtype == "both" else (
        torch.bfloat16 if args.dtype == "bf16" else torch.float16,
    )
    print(f"threads={args.threads}; latency is median of {args.iterations} runs")
    for dtype in dtypes:
        for shape in SHAPES:
            run_case(shape, dtype, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
