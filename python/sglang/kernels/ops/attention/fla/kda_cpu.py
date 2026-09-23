from typing import Optional

import torch


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    output_intermediate_states: bool = False,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
    beta_is_raw: bool = False,
    **kwargs,
):
    """Run packed KDA prefill on CPU and update selected state rows in place."""
    if output_intermediate_states or track_state is not None or track_chunk_idx is not None:
        raise NotImplementedError("CPU chunk_kda does not support intermediate state snapshots")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if initial_state is None or initial_state_indices is None:
        raise ValueError("CPU chunk_kda requires initial_state and initial_state_indices")
    return torch.ops.sgl_kernel.chunk_kda_cpu(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state,
        initial_state_indices,
        cu_seqlens,
        A_log,
        dt_bias,
        scale,
        use_qk_l2norm_in_kernel,
        beta_is_raw,
        lower_bound,
    )
