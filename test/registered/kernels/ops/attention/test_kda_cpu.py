import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.kda_cpu import chunk_kda


def _reference(q, k, v, gate, beta, state_pool, indices, offsets, scale, normalize):
    output = torch.empty_like(v)
    state_pool = state_pool.clone()
    for sequence, state_index in enumerate(indices.tolist()):
        state = state_pool[state_index]
        for token in range(offsets[sequence], offsets[sequence + 1]):
            qt = q[0, token].float()
            kt = k[0, token].float()
            if normalize:
                qt = F.normalize(qt, dim=-1, eps=1e-6)
                kt = F.normalize(kt, dim=-1, eps=1e-6)
            state = state * gate[0, token].float().exp().unsqueeze(-2)
            residual = v[0, token].float() - torch.einsum("hvk,hk->hv", state, kt)
            state = state + torch.einsum(
                "hv,hk->hvk", residual * beta[0, token].float()[:, None], kt
            )
            output[0, token] = (torch.einsum("hvk,hk->hv", state, qt) * scale).to(v.dtype)
        state_pool[state_index] = state
    return output, state_pool


@unittest.skipUnless(torch.backends.cpu.get_cpu_capability() == "AVX512", "requires AVX512 CPU")
class TestChunkKDACPU(unittest.TestCase):
    """Guard packed CPU KDA tails and indexed state-pool mutation semantics."""

    @torch.inference_mode()
    def test_reduced_dtypes_ragged_state_indices_and_tails(self):
        for dtype in (torch.bfloat16, torch.float16):
            for lengths, key_dim, value_dim in (([1, 63, 64, 65], 64, 64), ([3, 17], 47, 33)):
                with self.subTest(dtype=dtype, lengths=lengths, key_dim=key_dim, value_dim=value_dim):
                    torch.manual_seed(7)
                    tokens, heads = sum(lengths), 2
                    q = torch.randn(1, tokens, heads, key_dim, dtype=dtype)
                    k = torch.randn_like(q)
                    v = torch.randn(1, tokens, heads, value_dim, dtype=dtype) * 0.1
                    gate = (torch.randn_like(q) * 0.05 - 0.1).contiguous()
                    beta = torch.sigmoid(torch.randn(1, tokens, heads)).to(dtype)
                    state = torch.randn(6, heads, value_dim, key_dim) * 0.01
                    original = state.clone()
                    indices = torch.tensor([4, 1, 5, 2], dtype=torch.int32)[: len(lengths)]
                    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
                    scale = key_dim**-0.5
                    expected, expected_state = _reference(
                        q, k, v, gate, beta, state, indices, offsets.tolist(), scale, True
                    )
                    actual = chunk_kda(
                        q=q,
                        k=k,
                        v=v,
                        g=gate,
                        beta=beta,
                        scale=scale,
                        initial_state=state,
                        initial_state_indices=indices,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=offsets,
                    )
                    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-2
                    torch.testing.assert_close(actual.float(), expected.float(), atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(state, expected_state, atol=tolerance, rtol=tolerance)
                    untouched = sorted(set(range(6)) - set(indices.tolist()))
                    torch.testing.assert_close(state[untouched], original[untouched], atol=0, rtol=0)

    @torch.inference_mode()
    def test_raw_gate_and_beta_activation(self):
        torch.manual_seed(11)
        dtype = torch.bfloat16
        tokens, heads, key_dim, value_dim = 9, 2, 32, 24
        q = torch.randn(1, tokens, heads, key_dim, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn(1, tokens, heads, value_dim, dtype=dtype)
        raw_gate = torch.randn_like(q) * 0.2 - 1
        raw_beta = torch.randn(1, tokens, heads, dtype=dtype)
        a_log = torch.randn(heads) * 0.1
        dt_bias = torch.randn(heads, key_dim) * 0.1
        gate = -a_log.exp().view(1, 1, heads, 1) * F.softplus(
            raw_gate.float() + dt_bias.view(1, 1, heads, key_dim)
        )
        beta = raw_beta.float().sigmoid()
        state = torch.randn(3, heads, value_dim, key_dim) * 0.01
        indices = torch.tensor([2], dtype=torch.int32)
        scale = key_dim**-0.5
        expected, expected_state = _reference(
            q, k, v, gate, beta, state, indices, [0, tokens], scale, False
        )
        actual = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=raw_gate,
            beta=raw_beta,
            initial_state=state,
            initial_state_indices=indices,
            A_log=a_log,
            dt_bias=dt_bias,
            beta_is_raw=True,
        )
        torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(state, expected_state, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
