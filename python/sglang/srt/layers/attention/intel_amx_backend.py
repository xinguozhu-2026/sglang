from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, window_write_locs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel, get_spec

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class IntelAMXAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        import sgl_kernel  # noqa: F401

        super().__init__()
        self.forward_metadata = None
        self.extend_metadata = None
        self.draft_decode_metadata = None
        self.device = model_runner.device
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.runner_ids = model_runner.runner_ids
        self.is_dsv4 = False
        if self._init_dsv4_if_needed(model_runner):
            return
        self.use_mla = model_runner.use_mla_backend
        self.max_context_len = model_runner.model_config.context_len

        # full->SWA translated out_cache_loc, computed once per forward (the only
        # set_kv_buffer is in eager forward_extend; decode writes KV in-kernel).
        self.use_sliding_window_kv_pool = self.token_to_kv_pool.layer_router.routes_to(
            IdSpaceKind.SLIDING_WINDOW
        )
        self.swa_out_cache_loc = None
        self._swa_req_to_token = None

        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )

        # The first layer that holds KV: not every layer does (a hybrid
        # linear-attention model's recurrent layers keep state elsewhere).
        layer_id = model_runner.token_to_kv_pool.layer_router.layer_ids()[0]
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
            layer_id
        ).shape[-1]
        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu
        self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu

        # Number of KV splits used by decode_attention_cpu; attn_logits is
        # sized [bs, num_head, num_kv_splits, v_head_dim + 1] to match.
        self.num_kv_splits = 8

        self._attn_logits_buffers: dict[tuple[int, int], torch.Tensor] = {}

        # speculative decoding params
        self.num_draft_tokens = get_spec().speculative_num_draft_tokens

    def _build_extend_metadata(self, forward_batch: ForwardBatch):
        """Resolve (seq_lens, extend_seq_lens, extend_start_loc, tree_mask) for
        forward_extend, once per forward pass.

        In TARGET_VERIFY mode the batch carries no extend_* fields, so they are
        derived from spec_info (mirrors the CUDA unified path in
        triton_backend.py); each request extends by exactly num_draft_tokens
        tokens. Outside spec decoding the fields are passed through.
        """
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        tree_mask = None

        if forward_batch.forward_mode.is_target_verify():
            spec_info = forward_batch.spec_info
            if spec_info is None:
                raise RuntimeError(
                    "spec_info is unset in TARGET_VERIFY mode; the extend_* "
                    "metadata can only be derived from spec_info for "
                    "speculative verify batches."
                )
            num_draft_tokens = spec_info.draft_token_num
            extend_seq_lens = torch.full(
                (bs,), num_draft_tokens, dtype=torch.int32, device=self.device
            )
            # Uniform extend lengths: start locations form a plain range.
            extend_start_loc = torch.arange(
                0,
                bs * num_draft_tokens,
                num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            seq_lens = forward_batch.seq_lens + num_draft_tokens
            # Speculative verify with a token tree: each draft token may only
            # attend to its ancestors among the draft tokens (the committed
            # prefix stays fully visible).
            #
            # NOTE: unlike triton_backend.py, which forwards spec_info.custom_mask
            # unconditionally, the mask is gated on tree_topk here. tree_topk == 1
            # means the draft tokens form a simple chain whose visibility is
            # exactly the kernel's built-in causal masking, and skipping the explicit
            # mask lets extend_attention_cpu take its faster mask-free path. EAGLE
            # has tree_topk == topk (> 1 for real trees); NGRAM has tree_topk == -1
            # (irregular tree); both need the mask.
            if spec_info.tree_topk != 1:
                custom_mask = spec_info.custom_mask
                if custom_mask is not None and custom_mask.numel() > 0:
                    tree_mask = custom_mask
        else:
            extend_seq_lens = forward_batch.extend_seq_lens
            extend_start_loc = forward_batch.extend_start_loc

        return seq_lens, extend_seq_lens, extend_start_loc, tree_mask

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        if self.is_dsv4:
            return self._init_dsv4_forward_metadata(forward_batch)
        bs = forward_batch.batch_size
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = None
            self.extend_metadata = None
        elif forward_batch.forward_mode.is_target_verify():
            max_extend_len = self.num_draft_tokens
            self.extend_metadata = self._build_extend_metadata(forward_batch)
        else:
            # Pass -1 to let C++ kernel compute max_len_extend dynamically on C++ side.
            # Avoids calling Python .item() which triggers torch.compile recompile guards.
            max_extend_len = -1
            self.extend_metadata = self._build_extend_metadata(forward_batch)
        self.forward_metadata = (attn_logits, max_extend_len)

        self._init_swa_forward_metadata(forward_batch.out_cache_loc)

    def _init_swa_forward_metadata(self, out_cache_loc: torch.Tensor | None) -> None:
        if self.use_sliding_window_kv_pool and out_cache_loc is not None:
            self.swa_out_cache_loc = self.runner_ids.ids_in(
                IdSpaceKind.SLIDING_WINDOW, out_cache_loc
            )
        else:
            self.swa_out_cache_loc = None
        self._swa_req_to_token = None

    def _is_swa_layer(self, layer_id: int) -> bool:
        return (
            self.use_sliding_window_kv_pool
            and self.token_to_kv_pool.layer_router.plays(
                layer_id, IdSpaceKind.SLIDING_WINDOW
            )
        )

    def _swa_kv_indices(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """This batch's rows of ``req_to_token``, in the SWA pool's index space.

        ``req_to_token`` holds full-pool indices, but a sliding-window layer's
        K/V buffer only has ``swa_size`` rows. Feeding it full indices reads out
        of bounds as soon as the allocator hands one out past that -- and since
        the allocator walks its free list in order, that first happens long
        after startup, once cumulative allocations exceed ``swa_size``.

        Returns the compacted table plus the ``req_pool_indices`` that address
        it; the kernels take their row stride from ``req_to_token.size(1)``.
        """
        cached = self._swa_req_to_token
        if cached is not None and cached[0] is req_to_token:
            return cached[1], cached[2]

        rows = req_to_token[req_pool_indices, : int(seq_lens.max())]
        translated = self.runner_ids.ids_in(IdSpaceKind.SLIDING_WINDOW, rows).to(
            req_to_token.dtype
        )
        indices = torch.arange(
            translated.shape[0], dtype=torch.int64, device=translated.device
        )
        self._swa_req_to_token = (req_to_token, translated, indices)
        return translated, indices

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        if self.is_dsv4:
            return
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        # Pass -1 to let C++ kernel compute max_len_extend dynamically on C++ side.
        max_extend_len = -1
        self.forward_metadata = (attn_logits, max_extend_len)
        self.extend_metadata = None
        self._init_swa_forward_metadata(out_cache_loc)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        *,
        compress_ratio: int = 0,
        attn_sink: torch.Tensor | None = None,
        **kwargs,
    ):
        if self.is_dsv4:
            return self._forward_dsv4_attention(
                q,
                k,
                layer,
                forward_batch,
                compress_ratio=compress_ratio,
                attn_sink=attn_sink if attn_sink is not None else sinks,
                save_kv_cache=save_kv_cache,
            )
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        key_buffer = self.kv_layers.read(layer.layer_id).key()
        value_buffer = self.kv_layers.read(layer.layer_id).value()
        if save_kv_cache and k is not None and v is not None:
            # Cross-attention never writes to the SWA pool, so only thread the
            # full->SWA location for non-cross-attention layers.
            swa_loc = None if layer.is_cross_attention else self.swa_out_cache_loc
            write_loc = KVWriteLoc(
                cache_loc, window_locs=window_write_locs(self.runner_ids, swa_loc)
            )
            if not self.use_mla and key_buffer.dtype == torch.float8_e4m3fn:
                self.kv_layers.layer(layer.layer_id).store(
                    layer,
                    write_loc,
                    k,
                    v,
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                )
            else:
                self.kv_layers.layer(layer.layer_id).store(layer, write_loc, k, v)

        # Precomputed once per forward pass in init_forward_metadata (spec
        # verify batches carry no extend_* fields; see _build_extend_metadata).
        seq_lens, extend_seq_lens, extend_start_loc, tree_mask = self.extend_metadata

        _, max_extend_len = self.forward_metadata
        if seq_lens.dtype != torch.int64:
            seq_lens = seq_lens.to(torch.int64)

        key_scale = layer.k_scale_float or 1.0
        value_scale = layer.v_scale_float or 1.0
        is_causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            is_causal = False

        # Gemma4's KV-shared layers pass k=v=None - the layer they share with
        # already wrote their extend K/V to the cache
        req_to_token = self.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        if self._is_swa_layer(layer.layer_id) and not layer.is_cross_attention:
            req_to_token, req_pool_indices = self._swa_kv_indices(
                req_to_token, req_pool_indices, seq_lens
            )
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            key_buffer,
            value_buffer,
            key_scale,
            value_scale,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
            layer.is_cross_attention,
            layer.sliding_window_size + 1,
            forward_batch.encoder_lens,
            sinks,
            tree_mask,
            is_causal,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        *,
        compress_ratio: int = 0,
        attn_sink: torch.Tensor | None = None,
        **kwargs,
    ):
        if self.is_dsv4:
            return self._forward_dsv4_attention(
                q,
                k,
                layer,
                forward_batch,
                compress_ratio=compress_ratio,
                attn_sink=attn_sink if attn_sink is not None else sinks,
                save_kv_cache=save_kv_cache,
            )
        if self.draft_decode_metadata is not None:
            req_to_token, seq_lens, req_pool_indices = self.draft_decode_metadata
        else:
            req_to_token = self.req_to_token_pool.req_to_token
            req_pool_indices = forward_batch.req_pool_indices
            seq_lens = forward_batch.seq_lens

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if seq_lens.dtype != torch.int64:
            seq_lens = seq_lens.to(torch.int64)

        if layer.v_head_dim == self.v_head_dim and layer.tp_q_head_num == self.num_head:
            attn_logits, _ = self.forward_metadata
        else:
            # This layer's shape differs from the model-wide metadata buffer -
            # size from the same seq_lens the kernel derives num_seqs from
            attn_logits = self._get_attn_logits_buffer(
                seq_lens.shape[0], layer.tp_q_head_num, layer.v_head_dim
            )

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        key_buffer = self.kv_layers.read(layer.layer_id).key()
        value_buffer = self.kv_layers.read(layer.layer_id).value()
        key_scale = layer.k_scale_float or 1.0
        value_scale = layer.v_scale_float or 1.0
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if (
            save_kv_cache
            and k is not None
            and v is not None
            and key_buffer.dtype == torch.float8_e4m3fn
        ):
            swa_loc = None if layer.is_cross_attention else self.swa_out_cache_loc
            self.kv_layers.layer(layer.layer_id).store(
                layer,
                KVWriteLoc(
                    cache_loc, window_locs=window_write_locs(self.runner_ids, swa_loc)
                ),
                k,
                v,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            k = None
            v = None
        if self._is_swa_layer(layer.layer_id) and not layer.is_cross_attention:
            # When k/v survive the block above the kernel writes them itself, so
            # the write location needs the same translation set_kv_buffer got.
            cache_loc = self.swa_out_cache_loc
            req_to_token, req_pool_indices = self._swa_kv_indices(
                req_to_token, req_pool_indices, seq_lens
            )
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            key_buffer,
            value_buffer,
            key_scale,
            value_scale,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k,
            v,
            cache_loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            layer.scaling,
            layer.logit_cap,
            layer.is_cross_attention,
            layer.sliding_window_size + 1,
            forward_batch.encoder_lens,
            sinks,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _get_attn_logits_buffer(
        self, num_seqs: int, num_heads: int, v_head_dim: int
    ) -> torch.Tensor:
        key = (num_heads, v_head_dim)
        buffer = self._attn_logits_buffers.get(key)
        if buffer is None or buffer.shape[0] < num_seqs:
            # decode_attention_cpu writes every element it later reads, so the
            # buffer needs no initialization and can be reused; it only grows.
            buffer = torch.empty(
                (num_seqs, num_heads, self.num_kv_splits, v_head_dim + 1),
                dtype=torch.float32,
                device=self.device,
            )
            self._attn_logits_buffers[key] = buffer
        return buffer[:num_seqs]

    def support_triton(self):
        return False

    def _init_dsv4_if_needed(self, model_runner: ModelRunner) -> bool:
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        if not isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            return False
        assert model_runner.spec_algorithm.is_none(), (
            "IntelAMXAttnBackend DeepSeek-V4.1 path does not yet support speculative decoding"
        )
        text_cfg = model_runner.model_config.hf_text_config
        self.is_dsv4 = True
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.dsv4_window = text_cfg.sliding_window
        self.dsv4_topk = getattr(text_cfg, "index_topk", 0)
        self.v_head_dim = text_cfg.qk_nope_head_dim + text_cfg.qk_rope_head_dim
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.swa_out_cache_loc = None
        self._dsv4_meta: dict | None = None
        self.dsv4_sparse_attn_fwd = torch.ops.sgl_kernel.dsv41_sparse_attention_cpu
        self.dsv4_dequant_index_k_fwd = (
            torch.ops.sgl_kernel.dsv4_gather_dequant_fp4_index_k_cpu
        )
        return True

    def _init_dsv4_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices

        if forward_batch.forward_mode.is_idle():
            self._dsv4_meta = None
            self.swa_out_cache_loc = None
            return
        num_tokens = forward_batch.positions.numel()
        if forward_batch.forward_mode.is_decode():
            req_indices = forward_batch.req_pool_indices.long()
        else:
            req_indices = token_req_indices(forward_batch, num_tokens=num_tokens).long()

        self.swa_out_cache_loc = self.token_to_kv_pool.translate_loc_from_full_to_swa(
            forward_batch.out_cache_loc
        )
        win_positions = (
            forward_batch.positions[:, None]
            - self.dsv4_window
            + 1
            + torch.arange(self.dsv4_window, device=forward_batch.positions.device)
        )
        full_slots = self.req_to_token[
            req_indices[:, None], win_positions.clamp_min(0)
        ].long()
        swa_slots = self.token_to_kv_pool.translate_loc_from_full_to_swa(
            full_slots
        ).long()
        swa_slots = swa_slots.masked_fill((win_positions < 0) | (swa_slots <= 0), -1)

        self._dsv4_meta = {
            "req_indices": req_indices,
            "swa_slots": swa_slots,
            "empty_extra": torch.empty((num_tokens, 0), dtype=torch.long),
            "selected": {},
            "candidates": None,
        }

    def get_swa_out_cache_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
        if self.swa_out_cache_loc is not None:
            return self.swa_out_cache_loc
        return self.token_to_kv_pool.translate_loc_from_full_to_swa(
            forward_batch.out_cache_loc
        )

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
            layer_id, self.get_swa_out_cache_loc(forward_batch), swa_k
        )

    def forward_low_ratio_sources(
        self, *, layer, x, q_lora, positions, forward_batch: ForwardBatch
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices

        meta = self._dsv4_meta
        req = (
            meta["req_indices"]
            if meta is not None and meta["req_indices"].numel() == x.size(0)
            else token_req_indices(forward_batch, num_tokens=x.size(0)).long()
        )
        if layer.compressor is not None:
            self._dsv4_compress(layer, x, req, positions, forward_batch)
        if layer.indexer is not None:
            self._dsv4_index(layer, x, q_lora, req, positions)

    def _dsv4_compress(self, layer, x, req, pos, batch: ForwardBatch) -> None:
        from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore

        pool = self.token_to_kv_pool
        ratio = layer.compress_ratio
        kv, score = layer.compressor.project(x)
        if ratio == 2:
            state = pool.get_attention_compress_states(layer.layer_id)
            carried = state.get_state_by_state_loc(
                state.translate_from_req_position_to_state_loc(req, pos - 1)
            )
            adjacent = torch.zeros(pos.numel(), dtype=torch.bool, device=pos.device)
            adjacent[slice(1, None)] = (req[slice(1, None)] == req[slice(None, -1)]) & (
                pos[slice(1, None)] == pos[slice(None, -1)] + 1
            )
            partner_kv = torch.where(adjacent[:, None], kv.roll(1, 0), carried.kv)
            partner_score = torch.where(
                adjacent[:, None], score.roll(1, 0), carried.score
            )
            pooled = layer.compressor.pool_pairs(
                torch.stack((partner_kv, kv), dim=1),
                torch.stack((partner_score, score), dim=1),
            )
            keep = torch.ones(pos.numel(), dtype=torch.bool, device=pos.device)
            ring = state.ring_size
            if pos.numel() > ring:
                keep[:-ring] = req[:-ring] != req[ring:]
            state.set_state_by_state_loc(
                state.translate_from_req_position_to_state_loc(req[keep], pos[keep]),
                KVAndScore.from_kv_score(kv=kv[keep], score=score[keep]),
            )
            valid = pos % 2 == 1
            pooled = pooled[valid]
        else:
            valid = torch.ones(pos.numel(), dtype=torch.bool, device=pos.device)
            pooled = kv

        group_pos = pos[valid] - (ratio - 1)
        slots = batch.out_cache_loc[valid] // ratio
        latent = layer.compressor.finish(pooled)
        freqs = layer.freqs_cis[group_pos]
        if layer.indexer is not None and layer.indexer.owns_k:
            pool.set_index_k_fp4(
                layer.layer_id, slots, layer.indexer.index_keys(latent, freqs)
            )
        pool.set_extra_key_buffer_fused(layer.layer_id, slots, latent, freqs_cis=freqs)

    def _dsv4_compressed_slots(self, req: int, length: int, ratio: int) -> torch.Tensor:
        return (
            self.req_to_token[req, ratio - 1 : length * ratio : ratio].long() // ratio
        )

    def _dsv4_index(self, layer, x, q_lora, req, pos) -> None:
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_mask,
            select_candidate_block_ids,
        )

        meta = self._dsv4_meta
        assert meta is not None, "init_forward_metadata must run before _dsv4_index"
        indexer = layer.indexer
        ratio = layer.compress_ratio
        query = indexer.queries(q_lora, layer.freqs_cis, positions=pos)
        weights = indexer.head_weights(x)
        selected = torch.full((pos.numel(), self.dsv4_topk), -1, dtype=torch.long)
        published: list[torch.Tensor] = [
            torch.empty(0, dtype=torch.long) for _ in range(pos.numel())
        ]

        req_list = req.tolist()
        pos_list = pos.tolist()
        max_len_per_req: dict[int, int] = {}
        for request, position in zip(req_list, pos_list):
            length = (position + 1) // ratio
            if length > max_len_per_req.get(request, 0):
                max_len_per_req[request] = length

        pool = self.token_to_kv_pool
        idx_buf = pool.get_index_k_with_scale_buffer(layer.layer_id)
        idx_page_size = pool.get_index_k_page_size(ratio)
        req_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for request, max_len in max_len_per_req.items():
            if max_len > 0:
                all_slots = self._dsv4_compressed_slots(request, max_len, ratio)
                all_keys = self.dsv4_dequant_index_k_fwd(
                    idx_buf, all_slots, idx_page_size
                )
                req_cache[request] = (all_slots, all_keys)

        n_rows = len(req_list)
        r0 = 0
        tile_limit = 64
        while r0 < n_rows:
            request = req_list[r0]
            len0 = (pos_list[r0] + 1) // ratio
            if len0 == 0:
                r0 += 1
                continue
            r1 = r0 + 1
            tile_max_len = len0
            while (
                r1 < n_rows
                and (r1 - r0) < tile_limit
                and req_list[r1] == request
                and ((pos_list[r1] + 1) // ratio) > 0
            ):
                tile_max_len = max(tile_max_len, (pos_list[r1] + 1) // ratio)
                r1 += 1

            all_slots, all_keys = req_cache[request]
            tile_scores = indexer.scores(
                query[r0:r1], all_keys[:tile_max_len], weights[r0:r1]
            )
            for local_i, row in enumerate(range(r0, r1)):
                length = (pos_list[row] + 1) // ratio
                slots = all_slots[:length]
                scores = tile_scores.select(0, local_i)[:length].clone()
                if indexer.is_candidate_source:
                    published[row] = select_candidate_block_ids(
                        scores.unsqueeze(0),
                        length,
                        topk_blocks=indexer.candidate_topk_blocks,
                        block_size=indexer.candidate_block_size,
                    ).select(0, 0)
                elif indexer.uses_candidates:
                    candidates = meta["candidates"]
                    assert candidates is not None, (
                        "CSA2 candidate source has not run yet"
                    )
                    allowed = candidate_block_mask(
                        candidates[row],
                        width=length,
                        block_size=indexer.candidate_block_size,
                    )
                    scores = scores.masked_fill(~allowed, -torch.inf)
                count = min(self.dsv4_topk, length)
                indices = scores.topk(count, sorted=False).indices
                chosen = slots[indices].masked_fill(~scores[indices].isfinite(), -1)
                selected[row, :count] = chosen
            r0 = r1

        meta["selected"][ratio] = selected
        if indexer.is_candidate_source:
            meta["candidates"] = published

    def _forward_dsv4_attention(
        self,
        q,
        k,
        layer,
        forward_batch: ForwardBatch,
        *,
        compress_ratio,
        attn_sink,
        save_kv_cache,
    ):
        if save_kv_cache and k is not None:
            self.store_cache(layer.layer_id, k, forward_batch)
        meta = self._dsv4_meta
        assert meta is not None, "init_forward_metadata must run before attention"
        pool = self.token_to_kv_pool
        swa_cache = pool.get_swa_key_buffer_radix(layer.layer_id)
        extra_cache = (
            pool.get_extra_key_buffer(layer.layer_id) if compress_ratio else swa_cache
        )
        extra_slots = (
            meta["selected"][compress_ratio] if compress_ratio else meta["empty_extra"]
        )
        return self.dsv4_sparse_attn_fwd(
            q.contiguous(),
            swa_cache,
            extra_cache,
            meta["swa_slots"],
            extra_slots,
            attn_sink.float().contiguous(),
            float(layer.scaling),
        )

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        if self.is_dsv4:
            if forward_batch.forward_mode.is_idle():
                return torch.empty_like(q)
            if forward_batch.forward_mode.is_decode():
                return self.forward_decode(
                    q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
                )
            return self.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        return super().forward(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )


class IntelAMXMultiStepDraftBackend:
    """
    Wrap multiple intel amx attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sgl_kernel import build_draft_decode_metadata_cpu

        self.build_draft_decode_metadata = build_draft_decode_metadata_cpu
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: list[IntelAMXAttnBackend] = []
        for _ in range(self.speculative_num_steps - 1):
            self.attn_backends.append(IntelAMXAttnBackend(model_runner))
        self.device = model_runner.device
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        num_seqs = forward_batch.batch_size
        topk = self.topk
        bs = num_seqs * topk
        num_steps = self.speculative_num_steps
        req_to_token = self.attn_backends[0].req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        pool_len = self.pool_len
        num_head = self.attn_backends[0].num_head
        v_head_dim = self.attn_backends[0].v_head_dim
        device = self.device

        # Build expanded req_to_token via C++ kernel
        req_to_token_draft = self.build_draft_decode_metadata(
            req_to_token,
            forward_batch.req_pool_indices,
            seq_lens,
            topk,
            num_steps,
            pool_len,
        )

        req_pool_indices_expanded = torch.arange(bs, dtype=torch.int64, device=device)

        num_kv_splits = self.attn_backends[0].num_kv_splits
        for step in range(num_steps - 1):
            # Each candidate sees prefix + (step + 1) draft tokens.
            seq_lens_expanded = seq_lens.repeat_interleave(topk) + step + 1
            attn_logits = torch.zeros(
                (bs, num_head, num_kv_splits, v_head_dim + 1),
                dtype=torch.float32,
                device=device,
            )
            self.attn_backends[step].forward_metadata = (attn_logits, None)
            self.attn_backends[step].draft_decode_metadata = (
                req_to_token_draft,
                seq_lens_expanded,
                req_pool_indices_expanded,
            )
