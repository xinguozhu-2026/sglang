import pytest
import torch

import sgl_kernel  # noqa: F401
from sglang.kernels.ops.moe.router import fused_moe_router_shim
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu-intel")


def _reference(hidden_states, router_weight, topk, softcap, correction_bias=None):
    logits = hidden_states.float() @ router_weight.float().t()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    if correction_bias is not None:
        logits = logits + correction_bias
    scores = torch.softmax(logits, dim=-1)
    return torch.topk(scores, topk, dim=-1)


def _run_case(tokens, hidden_size, experts, topk, dtype, with_bias=False):
    torch.manual_seed(41)
    hidden_states = torch.randn(tokens, hidden_size, dtype=dtype) / hidden_size**0.5
    router_weight = torch.randn(experts, hidden_size, dtype=torch.float32)
    correction_bias = torch.randn(experts, dtype=torch.float32) if with_bias else None

    actual_weights, actual_ids = fused_moe_router_shim(
        30.0,
        hidden_states,
        router_weight,
        topk,
        False,
        correction_bias,
    )
    expected_weights, expected_ids = _reference(
        hidden_states,
        router_weight,
        topk,
        30.0,
        correction_bias,
    )

    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(actual_weights, expected_weights, rtol=4e-3, atol=2e-5)
    assert actual_weights.dtype == torch.float32
    assert actual_ids.dtype == torch.int32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 17, 128])
def test_fused_moe_router_grok_shape(dtype, tokens):
    _run_case(tokens, 6144, 8, 2, dtype)


@pytest.mark.parametrize(
    "hidden_size,experts,topk",
    [
        (1, 3, 1),
        (31, 7, 2),
        (33, 17, 4),
        (65, 9, 8),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_router_vector_tails(hidden_size, experts, topk, dtype):
    _run_case(5, hidden_size, experts, topk, dtype, with_bias=True)


def test_fused_moe_router_ties_choose_lowest_expert_ids():
    hidden_states = torch.ones(2, 33, dtype=torch.bfloat16)
    router_weight = torch.zeros(8, 33, dtype=torch.float32)

    weights, ids = fused_moe_router_shim(
        30.0,
        hidden_states,
        router_weight,
        2,
        False,
    )

    torch.testing.assert_close(ids, torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))
    torch.testing.assert_close(weights, torch.full((2, 2), 0.125))


def test_fused_moe_router_empty_batch():
    hidden_states = torch.empty(0, 65, dtype=torch.float16)
    router_weight = torch.randn(8, 65, dtype=torch.float32)

    weights, ids = fused_moe_router_shim(
        30.0,
        hidden_states,
        router_weight,
        2,
        False,
    )

    assert weights.shape == (0, 2)
    assert ids.shape == (0, 2)
