#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "common.h"
#include "vec.h"

#if defined(__x86_64__) && defined(__AVX512BF16__) && defined(__AVX512BW__)
#include <immintrin.h>
#define SGLANG_DSV41_HAS_AVX512_BF16 1
#else
#define SGLANG_DSV41_HAS_AVX512_BF16 0
#endif

namespace {

inline float ue8m0_to_float(uint8_t exp_u8) {
  const uint32_t bits = (exp_u8 == 0) ? (1u << 22) : (static_cast<uint32_t>(exp_u8) << 23);
  float val;
  std::memcpy(&val, &bits, sizeof(float));
  return val;
}

inline uint8_t encode_e2m1_nibble(float v) {
  const uint8_t sign = (v < 0.0f) ? 8u : 0u;
  const float ax = std::abs(v);
  uint8_t mag = 0u;
  if (ax > 5.0f) {
    mag = 7u;
  } else if (ax >= 3.5f) {
    mag = 6u;
  } else if (ax > 2.5f) {
    mag = 5u;
  } else if (ax >= 1.75f) {
    mag = 4u;
  } else if (ax > 1.25f) {
    mag = 3u;
  } else if (ax >= 0.75f) {
    mag = 2u;
  } else if (ax > 0.25f) {
    mag = 1u;
  }
  return sign | mag;
}

inline float decode_e2m1_nibble(uint8_t code) {
  static constexpr float kLut[] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return *(kLut + (code & 0x0Fu));
}

#if SGLANG_DSV41_HAS_AVX512_BF16
void csa2_sparse_attention_avx512_bf16(
    at::BFloat16* __restrict__ out_ptr,
    const at::BFloat16* __restrict__ q_ptr,
    const at::BFloat16* __restrict__ swa_ptr,
    const at::BFloat16* __restrict__ extra_ptr,
    const int64_t* __restrict__ swa_slot_ptr,
    const int64_t* __restrict__ extra_slot_ptr,
    const float* __restrict__ sink_ptr,
    int64_t T,
    int64_t H,
    int64_t D,
    int64_t W,
    int64_t K_sp,
    int64_t swa_rows,
    int64_t extra_rows,
    float sm_scale) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kHeadTile = 16;
  const int64_t num_h_tiles = (H + kHeadTile - 1) / kHeadTile;

  at::parallel_for(0, T * num_h_tiles, 1, [&](int64_t begin, int64_t end) {
    std::vector<float> acc_tile(kHeadTile * D);
    std::vector<float> kv_f32(D);
    float m_arr[kHeadTile];
    float l_arr[kHeadTile];

    for (int64_t task = begin; task < end; ++task) {
      const int64_t t = task / num_h_tiles;
      const int64_t h0 = (task % num_h_tiles) * kHeadTile;
      const int64_t bh = std::min(kHeadTile, H - h0);

      std::fill(acc_tile.begin(), acc_tile.begin() + bh * D, 0.0f);
      for (int64_t hh = 0; hh < bh; ++hh) {
        const float s_init = *(sink_ptr + h0 + hh);
        if (std::isfinite(s_init)) {
          *(m_arr + hh) = s_init;
          *(l_arr + hh) = 1.0f;
        } else {
          *(m_arr + hh) = -std::numeric_limits<float>::infinity();
          *(l_arr + hh) = 0.0f;
        }
      }

      auto run_branch =
          [&](const at::BFloat16* cache_base, int64_t max_rows, const int64_t* slots_row, int64_t num_slots) {
            for (int64_t j = 0; j < num_slots; ++j) {
              const int64_t slot = *(slots_row + j);
              if (slot < 0 || slot >= max_rows) {
                continue;
              }
              if (j + 2 < num_slots) {
                const int64_t pf_slot = *(slots_row + j + 2);
                if (pf_slot >= 0 && pf_slot < max_rows) {
                  const char* pf_base = reinterpret_cast<const char*>(cache_base + pf_slot * D);
                  for (int64_t off = 0; off < D * 2; off += 64) {
                    _mm_prefetch(pf_base + off, _MM_HINT_T0);
                  }
                }
              }

              const at::BFloat16* kv_src = cache_base + slot * D;
              float* kv_f = kv_f32.data();
              int64_t u = 0;
              for (; u + 32 <= D; u += 32) {
                __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(kv_src + u));
                __m256i lo16 = _mm512_castsi512_si256(raw);
                __m256i hi16 = _mm512_extracti64x4_epi64(raw, 1);
                __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo16), 16));
                __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi16), 16));
                _mm512_storeu_ps(kv_f + u, f0);
                _mm512_storeu_ps(kv_f + u + 16, f1);
              }
              for (; u < D; ++u) {
                *(kv_f + u) = static_cast<float>(*(kv_src + u));
              }

              for (int64_t hh = 0; hh < bh; ++hh) {
                const at::BFloat16* q_src = q_ptr + (t * H + h0 + hh) * D;
                __m512 dot_acc = _mm512_setzero_ps();
                int64_t d = 0;
                for (; d + 32 <= D; d += 32) {
                  __m512bh q_bh = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(q_src + d));
                  __m512bh k_bh = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(kv_src + d));
                  dot_acc = _mm512_dpbf16_ps(dot_acc, q_bh, k_bh);
                }
                float dot = _mm512_reduce_add_ps(dot_acc);
                for (; d < D; ++d) {
                  dot += static_cast<float>(*(q_src + d)) * (*(kv_f + d));
                }

                const float score = dot * sm_scale;
                const float m_prev = *(m_arr + hh);
                float* acc_f = acc_tile.data() + hh * D;
                if (score > m_prev) {
                  const float alpha = std::isinf(m_prev) ? 0.0f : std::exp(m_prev - score);
                  *(l_arr + hh) = (*(l_arr + hh)) * alpha + 1.0f;
                  *(m_arr + hh) = score;
                  const __m512 alpha_v = _mm512_set1_ps(alpha);
                  int64_t k = 0;
                  for (; k + 16 <= D; k += 16) {
                    __m512 av = _mm512_fmadd_ps(_mm512_loadu_ps(acc_f + k), alpha_v, _mm512_loadu_ps(kv_f + k));
                    _mm512_storeu_ps(acc_f + k, av);
                  }
                  for (; k < D; ++k) {
                    *(acc_f + k) = (*(acc_f + k)) * alpha + (*(kv_f + k));
                  }
                } else {
                  const float beta = std::exp(score - m_prev);
                  *(l_arr + hh) += beta;
                  const __m512 beta_v = _mm512_set1_ps(beta);
                  int64_t k = 0;
                  for (; k + 16 <= D; k += 16) {
                    __m512 av = _mm512_fmadd_ps(_mm512_loadu_ps(kv_f + k), beta_v, _mm512_loadu_ps(acc_f + k));
                    _mm512_storeu_ps(acc_f + k, av);
                  }
                  for (; k < D; ++k) {
                    *(acc_f + k) += beta * (*(kv_f + k));
                  }
                }
              }
            }
          };

      run_branch(swa_ptr, swa_rows, swa_slot_ptr + t * W, W);
      if (K_sp > 0) {
        run_branch(extra_ptr, extra_rows, extra_slot_ptr + t * K_sp, K_sp);
      }

      for (int64_t hh = 0; hh < bh; ++hh) {
        const float denom = *(l_arr + hh);
        const float inv_l = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        const fVec inv_v(inv_l);
        const float* acc_f = acc_tile.data() + hh * D;
        at::BFloat16* o_dst = out_ptr + (t * H + h0 + hh) * D;
        int64_t d = 0;
        for (; d + 32 <= D; d += 32) {
          fVec o0 = fVec::loadu(acc_f + d) * inv_v;
          fVec o1 = fVec::loadu(acc_f + d + 16) * inv_v;
          store_float_vec2(o_dst + d, o0, o1);
        }
        for (; d < D; ++d) {
          *(o_dst + d) = static_cast<at::BFloat16>((*(acc_f + d)) * inv_l);
        }
      }
    }
  });
}
#endif

}  // namespace

void dsv4_norm_rope_cpu(
    at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    std::optional<double> eps,
    bool inverse) {
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "input must be 2D or 3D");
  TORCH_CHECK(input.stride(-1) == 1, "input last dim must be contiguous");

  const int64_t B = input.size(0);
  const int64_t H = (input.dim() == 3) ? input.size(1) : 1;
  const int64_t D = input.size(-1);
  const int64_t stride_b = input.stride(0);
  const int64_t stride_h = (input.dim() == 3) ? input.stride(1) : D;

  at::Tensor freqs_real = freqs_cis.is_complex() ? at::view_as_real(freqs_cis).flatten(-2).to(at::kFloat).contiguous()
                                                 : freqs_cis.to(at::kFloat).contiguous();
  TORCH_CHECK(freqs_real.dim() == 2, "freqs_cis must flatten to 2D");
  const int64_t max_pos = freqs_real.size(0);
  const int64_t rope_dim = freqs_real.size(1);
  TORCH_CHECK(rope_dim % 2 == 0 && rope_dim <= D, "invalid rope_dim");

  CHECK_DIM(1, positions);
  CHECK_EQ(positions.size(0), B);
  at::Tensor pos_i64 = positions.to(at::kLong).contiguous();
  const int64_t* pos_ptr = pos_i64.const_data_ptr<int64_t>();
  for (int64_t b = 0; b < B; ++b) {
    TORCH_CHECK((*(pos_ptr + b)) >= 0 && (*(pos_ptr + b)) < max_pos, "position out of range");
  }
  if (B * H == 0) {
    return;
  }

  const bool has_norm = weight.has_value();
  int64_t norm_dim = 0;
  at::Tensor weight_f;
  const float* w_ptr = nullptr;
  if (has_norm) {
    norm_dim = weight->numel();
    TORCH_CHECK(norm_dim == D || norm_dim == D - rope_dim, "invalid norm weight size");
    weight_f = weight->to(at::kFloat).contiguous();
    w_ptr = weight_f.const_data_ptr<float>();
  }
  const float eps_f = static_cast<float>(eps.value_or(1e-6));
  const float* freqs_ptr = freqs_real.const_data_ptr<float>();
  const int64_t rope_offset = D - rope_dim;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "dsv4_norm_rope_cpu", [&] {
        scalar_t* data_ptr = input.data_ptr<scalar_t>();
        at::parallel_for(0, B * H, 0, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t b = idx / H;
            const int64_t h = idx % H;
            scalar_t* row = data_ptr + b * stride_b + h * stride_h;

            if (has_norm && norm_dim > 0) {
              double ss = 0.0;
              for (int64_t d = 0; d < norm_dim; ++d) {
                float v = static_cast<float>(*(row + d));
                ss += static_cast<double>(v) * static_cast<double>(v);
              }
              const float inv_rms = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(norm_dim)) + eps_f);
              for (int64_t d = 0; d < norm_dim; ++d) {
                float v = static_cast<float>(*(row + d)) * inv_rms * (*(w_ptr + d));
                *(row + d) = static_cast<scalar_t>(v);
              }
            }

            const float* f_row = freqs_ptr + (*(pos_ptr + b)) * rope_dim;
            scalar_t* r_ptr = row + rope_offset;
            for (int64_t d = 0; d < rope_dim; d += 2) {
              const float x = static_cast<float>(*(r_ptr + d));
              const float y = static_cast<float>(*(r_ptr + d + 1));
              const float c = *(f_row + d);
              const float s = inverse ? -(*(f_row + d + 1)) : (*(f_row + d + 1));
              *(r_ptr + d) = static_cast<scalar_t>(x * c - y * s);
              *(r_ptr + d + 1) = static_cast<scalar_t>(y * c + x * s);
            }
          }
        });
      });
}

std::tuple<at::Tensor, at::Tensor> dsv41_sqrtsoftplus_topk_cpu(
    const at::Tensor& logits, const std::optional<at::Tensor>& bias, int64_t topk, bool renormalize) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(logits);
  CHECK_DIM(2, logits);
  const int64_t M = logits.size(0);
  const int64_t E = logits.size(1);
  TORCH_CHECK(topk > 0 && topk <= E, "Invalid topk: ", topk);

  at::Tensor out_weights = at::empty({M, topk}, logits.options().dtype(at::kFloat));
  at::Tensor out_ids = at::empty({M, topk}, logits.options().dtype(at::kInt));
  if (M == 0) {
    return std::make_tuple(out_weights, out_ids);
  }

  at::Tensor bias_f;
  const float* bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->numel() == E, "bias size mismatch");
    bias_f = bias->to(at::kFloat).contiguous();
    bias_ptr = bias_f.const_data_ptr<float>();
  }

  float* w_out = out_weights.data_ptr<float>();
  int32_t* id_out = out_ids.data_ptr<int32_t>();
  const int64_t row_stride = logits.stride(0);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, logits.scalar_type(), "dsv41_sqrtsoftplus_topk_cpu", [&] {
        const scalar_t* logits_ptr = logits.const_data_ptr<scalar_t>();
        at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
          std::vector<float> weights_row(E);
          std::vector<std::pair<float, int32_t>> scored(E);
          for (int64_t m = begin; m < end; ++m) {
            const scalar_t* l_row = logits_ptr + m * row_stride;
            for (int64_t e = 0; e < E; ++e) {
              const float x = static_cast<float>(*(l_row + e));
              const float sp = std::max(x, 0.0f) + std::log1p(std::exp(-std::abs(x)));
              const float w = std::sqrt(sp);
              weights_row.at(e) = w;
              const float s = bias_ptr ? (w + (*(bias_ptr + e))) : w;
              scored.at(e) = {s, static_cast<int32_t>(e)};
            }
            std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(), [](const auto& a, const auto& b) {
              return a.first != b.first ? (a.first > b.first) : (a.second < b.second);
            });
            float w_sum = 0.0f;
            for (int64_t k = 0; k < topk; ++k) {
              const int32_t idx = scored.at(k).second;
              const float wk = weights_row.at(idx);
              *(id_out + m * topk + k) = idx;
              *(w_out + m * topk + k) = wk;
              w_sum += wk;
            }
            if (renormalize && w_sum > 0.0f) {
              const float inv = 1.0f / w_sum;
              for (int64_t k = 0; k < topk; ++k) {
                *(w_out + m * topk + k) *= inv;
              }
            }
          }
        });
      });

  return std::make_tuple(out_weights, out_ids);
}

void dsv4_pack_fp4_index_k_cpu(const at::Tensor& input, at::Tensor& cache, const at::Tensor& loc, int64_t page_size) {
  CHECK_DIM(2, input);
  CHECK_DIM(2, cache);
  CHECK_EQ(input.size(1), 128);
  TORCH_CHECK(cache.scalar_type() == at::kByte, "cache must be uint8");
  const int64_t N = input.size(0);
  if (N == 0) {
    return;
  }

  at::Tensor in_c = input.contiguous();
  at::Tensor loc_i64 = loc.to(at::kLong).contiguous();
  const int64_t* loc_ptr = loc_i64.const_data_ptr<int64_t>();
  uint8_t* cache_ptr = cache.data_ptr<uint8_t>();
  const int64_t row_bytes = cache.stride(0);
  const int64_t max_pages = cache.size(0);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, in_c.scalar_type(), "dsv4_pack_fp4_index_k_cpu", [&] {
        const scalar_t* in_ptr = in_c.const_data_ptr<scalar_t>();
        at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            const int64_t slot = *(loc_ptr + i);
            if (slot <= 0) {
              continue;
            }
            const int64_t page = slot / page_size;
            const int64_t off = slot % page_size;
            if (page >= max_pages) {
              continue;
            }
            const scalar_t* src = in_ptr + i * 128;
            uint8_t* dst_payload = cache_ptr + page * row_bytes + off * 64;
            uint8_t* dst_scale = cache_ptr + page * row_bytes + page_size * 64 + off * 4;

            for (int64_t blk = 0; blk < 4; ++blk) {
              const scalar_t* blk_src = src + blk * 32;
              float amax = std::ldexp(1.0f, -126);
              for (int64_t k = 0; k < 32; ++k) {
                amax = std::max(amax, std::abs(static_cast<float>(*(blk_src + k))));
              }
              int32_t exp_i = static_cast<int32_t>(std::ceil(std::log2(amax / 6.0f))) + 127;
              exp_i = std::clamp(exp_i, 0, 254);
              const uint8_t exp_u8 = static_cast<uint8_t>(exp_i);
              *(dst_scale + blk) = exp_u8;
              const float inv_scale = 1.0f / ue8m0_to_float(exp_u8);

              uint8_t* blk_dst = dst_payload + blk * 16;
              for (int64_t b = 0; b < 16; ++b) {
                const float v0 = static_cast<float>(*(blk_src + 2 * b)) * inv_scale;
                const float v1 = static_cast<float>(*(blk_src + 2 * b + 1)) * inv_scale;
                *(blk_dst + b) = encode_e2m1_nibble(v0) | (encode_e2m1_nibble(v1) << 4);
              }
            }
          }
        });
      });
}

at::Tensor dsv4_gather_dequant_fp4_index_k_cpu(const at::Tensor& cache, const at::Tensor& slots, int64_t page_size) {
  CHECK_DIM(2, cache);
  TORCH_CHECK(cache.scalar_type() == at::kByte, "cache must be uint8");
  at::Tensor slots_i64 = slots.to(at::kLong).contiguous();
  const int64_t N = slots_i64.numel();
  at::Tensor out = at::empty({N, 128}, cache.options().dtype(at::kBFloat16));
  if (N == 0) {
    return out;
  }

  const uint8_t* cache_ptr = cache.const_data_ptr<uint8_t>();
  const int64_t row_bytes = cache.stride(0);
  const int64_t max_pages = cache.size(0);
  const int64_t* slot_ptr = slots_i64.const_data_ptr<int64_t>();
  at::BFloat16* out_ptr = out.data_ptr<at::BFloat16>();

  at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t slot = *(slot_ptr + i);
      at::BFloat16* dst = out_ptr + i * 128;
      if (slot < 0 || (slot / page_size) >= max_pages) {
        std::fill(dst, dst + 128, static_cast<at::BFloat16>(0.0f));
        continue;
      }
      const int64_t page = slot / page_size;
      const int64_t off = slot % page_size;
      const uint8_t* src_payload = cache_ptr + page * row_bytes + off * 64;
      const uint8_t* src_scale = cache_ptr + page * row_bytes + page_size * 64 + off * 4;
      for (int64_t blk = 0; blk < 4; ++blk) {
        const float scale = ue8m0_to_float(*(src_scale + blk));
        const uint8_t* blk_src = src_payload + blk * 16;
        at::BFloat16* blk_dst = dst + blk * 32;
        for (int64_t b = 0; b < 16; ++b) {
          const uint8_t byte_v = *(blk_src + b);
          *(blk_dst + 2 * b) = static_cast<at::BFloat16>(decode_e2m1_nibble(byte_v) * scale);
          *(blk_dst + 2 * b + 1) = static_cast<at::BFloat16>(decode_e2m1_nibble(byte_v >> 4) * scale);
        }
      }
    }
  });
  return out;
}

at::Tensor dsv41_sparse_attention_cpu(
    const at::Tensor& q,
    const at::Tensor& swa_cache,
    const at::Tensor& extra_cache,
    const at::Tensor& swa_slots,
    const at::Tensor& extra_slots,
    const at::Tensor& attn_sink,
    double scaling) {
  CHECK_INPUT(q);
  CHECK_INPUT(swa_cache);
  CHECK_INPUT(extra_cache);
  TORCH_CHECK(q.dim() == 2 || q.dim() == 3, "q must be 2D or 3D");

  const int64_t H = attn_sink.numel();
  TORCH_CHECK(H > 0, "attn_sink must be non-empty");
  const int64_t T = q.size(0);
  const int64_t D = (q.dim() == 3) ? q.size(2) : (q.size(1) / H);
  if (q.dim() == 3) {
    CHECK_EQ(q.size(1), H);
  } else {
    CHECK_EQ(q.size(1), H * D);
  }
  CHECK_EQ(swa_cache.size(-1), D);
  CHECK_EQ(extra_cache.size(-1), D);
  TORCH_CHECK(swa_cache.scalar_type() == extra_cache.scalar_type(), "cache dtype mismatch");

  at::Tensor out = at::empty_like(q);
  if (T == 0) {
    return out;
  }

  at::Tensor swa_idx = swa_slots.to(at::kLong).contiguous();
  at::Tensor extra_idx = extra_slots.to(at::kLong).contiguous();
  at::Tensor sink_f = attn_sink.to(at::kFloat).contiguous();
  at::Tensor q_matched =
      (q.scalar_type() == swa_cache.scalar_type()) ? q.contiguous() : q.to(swa_cache.scalar_type()).contiguous();
  at::Tensor out_matched = (out.scalar_type() == swa_cache.scalar_type()) ? out : at::empty_like(q_matched);

  const int64_t W = swa_idx.size(1);
  const int64_t K_sp = extra_idx.size(1);
  const int64_t swa_rows = swa_cache.size(0);
  const int64_t extra_rows = extra_cache.size(0);
  const int64_t* swa_slot_ptr = swa_idx.const_data_ptr<int64_t>();
  const int64_t* extra_slot_ptr = extra_idx.const_data_ptr<int64_t>();
  const float* sink_ptr = sink_f.const_data_ptr<float>();
  const float sm_scale = static_cast<float>(scaling);

#if SGLANG_DSV41_HAS_AVX512_BF16
  if (swa_cache.scalar_type() == at::kBFloat16) {
    csa2_sparse_attention_avx512_bf16(
        out_matched.data_ptr<at::BFloat16>(),
        q_matched.const_data_ptr<at::BFloat16>(),
        swa_cache.const_data_ptr<at::BFloat16>(),
        extra_cache.const_data_ptr<at::BFloat16>(),
        swa_slot_ptr,
        extra_slot_ptr,
        sink_ptr,
        T,
        H,
        D,
        W,
        K_sp,
        swa_rows,
        extra_rows,
        sm_scale);
    return (out.scalar_type() == out_matched.scalar_type()) ? out_matched : out_matched.to(out.scalar_type());
  }
#endif

  constexpr int64_t kHeadTile = 8;
  const int64_t num_h_tiles = (H + kHeadTile - 1) / kHeadTile;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, swa_cache.scalar_type(), "dsv41_sparse_attention_cpu", [&] {
        using fVec = at::vec::Vectorized<float>;
        constexpr int64_t kVec = fVec::size();
        const scalar_t* q_ptr = q_matched.const_data_ptr<scalar_t>();
        const scalar_t* swa_ptr = swa_cache.const_data_ptr<scalar_t>();
        const scalar_t* extra_ptr = extra_cache.const_data_ptr<scalar_t>();
        scalar_t* out_ptr = out_matched.data_ptr<scalar_t>();

        at::parallel_for(0, T * num_h_tiles, 1, [&](int64_t begin, int64_t end) {
          std::vector<float> q_tile(kHeadTile * D);
          std::vector<float> acc_tile(kHeadTile * D);
          std::vector<float> kv_row(D);
          float m_arr[kHeadTile];
          float l_arr[kHeadTile];

          for (int64_t task = begin; task < end; ++task) {
            const int64_t t = task / num_h_tiles;
            const int64_t h0 = (task % num_h_tiles) * kHeadTile;
            const int64_t bh = std::min(kHeadTile, H - h0);

            std::fill(acc_tile.begin(), acc_tile.begin() + bh * D, 0.0f);
            for (int64_t hh = 0; hh < bh; ++hh) {
              const scalar_t* q_src = q_ptr + (t * H + h0 + hh) * D;
              float* q_dst = q_tile.data() + hh * D;
              for (int64_t d = 0; d < D; ++d) {
                *(q_dst + d) = static_cast<float>(*(q_src + d));
              }
              const float s_init = *(sink_ptr + h0 + hh);
              if (std::isfinite(s_init)) {
                *(m_arr + hh) = s_init;
                *(l_arr + hh) = 1.0f;
              } else {
                *(m_arr + hh) = -std::numeric_limits<float>::infinity();
                *(l_arr + hh) = 0.0f;
              }
            }

            auto accumulate_branch =
                [&](const scalar_t* cache_base, int64_t max_rows, const int64_t* slots_row, int64_t num_slots) {
                  for (int64_t j = 0; j < num_slots; ++j) {
                    const int64_t slot = *(slots_row + j);
                    if (slot < 0 || slot >= max_rows) {
                      continue;
                    }
                    const scalar_t* kv_src = cache_base + slot * D;
                    float* kv_f = kv_row.data();
                    for (int64_t d = 0; d < D; ++d) {
                      *(kv_f + d) = static_cast<float>(*(kv_src + d));
                    }
                    for (int64_t hh = 0; hh < bh; ++hh) {
                      const float* q_f = q_tile.data() + hh * D;
                      float* acc_f = acc_tile.data() + hh * D;
                      fVec dot_v(0.0f);
                      int64_t d = 0;
                      for (; d + kVec <= D; d += kVec) {
                        dot_v = at::vec::fmadd(fVec::loadu(q_f + d), fVec::loadu(kv_f + d), dot_v);
                      }
                      float dot = at::vec::vec_reduce_all<float>([](fVec& x, fVec& y) { return x + y; }, dot_v);
                      for (; d < D; ++d) {
                        dot += (*(q_f + d)) * (*(kv_f + d));
                      }
                      const float score = dot * sm_scale;
                      const float m_prev = *(m_arr + hh);
                      if (score > m_prev) {
                        const float alpha = std::isinf(m_prev) ? 0.0f : std::exp(m_prev - score);
                        *(l_arr + hh) = (*(l_arr + hh)) * alpha + 1.0f;
                        *(m_arr + hh) = score;
                        const fVec alpha_v(alpha);
                        int64_t k = 0;
                        for (; k + kVec <= D; k += kVec) {
                          fVec a_v = fVec::loadu(acc_f + k) * alpha_v + fVec::loadu(kv_f + k);
                          a_v.store(acc_f + k);
                        }
                        for (; k < D; ++k) {
                          *(acc_f + k) = (*(acc_f + k)) * alpha + (*(kv_f + k));
                        }
                      } else {
                        const float beta = std::exp(score - m_prev);
                        *(l_arr + hh) += beta;
                        const fVec beta_v(beta);
                        int64_t k = 0;
                        for (; k + kVec <= D; k += kVec) {
                          fVec a_v = at::vec::fmadd(fVec::loadu(kv_f + k), beta_v, fVec::loadu(acc_f + k));
                          a_v.store(acc_f + k);
                        }
                        for (; k < D; ++k) {
                          *(acc_f + k) += beta * (*(kv_f + k));
                        }
                      }
                    }
                  }
                };

            accumulate_branch(swa_ptr, swa_rows, swa_slot_ptr + t * W, W);
            if (K_sp > 0) {
              accumulate_branch(extra_ptr, extra_rows, extra_slot_ptr + t * K_sp, K_sp);
            }

            for (int64_t hh = 0; hh < bh; ++hh) {
              const float denom = *(l_arr + hh);
              const float inv_l = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
              const float* acc_f = acc_tile.data() + hh * D;
              scalar_t* o_dst = out_ptr + (t * H + h0 + hh) * D;
              for (int64_t d = 0; d < D; ++d) {
                *(o_dst + d) = static_cast<scalar_t>((*(acc_f + d)) * inv_l);
              }
            }
          }
        });
      });

  return (out.scalar_type() == out_matched.scalar_type()) ? out_matched : out_matched.to(out.scalar_type());
}
