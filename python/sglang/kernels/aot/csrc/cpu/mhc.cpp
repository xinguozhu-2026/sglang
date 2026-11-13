#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
void hc_combine_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x_flat,
    const float* __restrict__ pre,
    int64_t M,
    int64_t K,
    int64_t H) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = 2 * fVec::size();

  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      const scalar_t* x_m = x_flat + m * K * H;
      const float* pre_m = pre + m * K;
      scalar_t* out_m = out + m * H;

      int64_t d = 0;
      for (; d + kVecSize <= H; d += kVecSize) {
        fVec acc0(0.0f), acc1(0.0f);
        for (int64_t k = 0; k < K; ++k) {
          const fVec w(pre_m[k]);
          auto [v0, v1] = load_float_vec2(x_m + k * H + d);
          acc0 = at::vec::fmadd(v0, w, acc0);
          acc1 = at::vec::fmadd(v1, w, acc1);
        }
        store_float_vec2(out_m + d, acc0, acc1);
      }
      for (; d < H; ++d) {
        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          acc += pre_m[k] * static_cast<float>(x_m[k * H + d]);
        }
        out_m[d] = static_cast<scalar_t>(acc);
      }
    }
  });
}

// out[m, j, d] = sum_i comb[m, i, j] * residual[m, i, d] + post_mix[m, j] * x[m, d]
template <typename scalar_t>
void hc_post_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb,
    int64_t M,
    int64_t K,
    int64_t H) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = 2 * fVec::size();

  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      const scalar_t* x_m = x + m * H;
      const scalar_t* res_m = residual + m * K * H;
      const float* post_m = post_mix + m * K;
      const float* comb_m = comb + m * K * K;
      scalar_t* out_m = out + m * K * H;

      for (int64_t j = 0; j < K; ++j) {
        const fVec p_vec(post_m[j]);
        scalar_t* out_mj = out_m + j * H;

        int64_t d = 0;
        for (; d + kVecSize <= H; d += kVecSize) {
          auto [x0, x1] = load_float_vec2(x_m + d);
          fVec acc0 = x0 * p_vec;
          fVec acc1 = x1 * p_vec;
          for (int64_t i = 0; i < K; ++i) {
            // Column-wise consumption: comb[m, i, j]
            const fVec c_vec(comb_m[i * K + j]);
            auto [r0, r1] = load_float_vec2(res_m + i * H + d);
            acc0 = at::vec::fmadd(r0, c_vec, acc0);
            acc1 = at::vec::fmadd(r1, c_vec, acc1);
          }
          store_float_vec2(out_mj + d, acc0, acc1);
        }
        for (; d < H; ++d) {
          float acc = post_m[j] * static_cast<float>(x_m[d]);
          for (int64_t i = 0; i < K; ++i) {
            acc += comb_m[i * K + j] * static_cast<float>(res_m[i * H + d]);
          }
          out_mj[d] = static_cast<scalar_t>(acc);
        }
      }
    }
  });
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_split_sinkhorn_cpu(
    const at::Tensor& mixes,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double hc_mult,
    int64_t sinkhorn_iters,
    double eps) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mixes);
  TORCH_CHECK(mixes.dim() >= 1, "mixes must be at least 1D");
  const int64_t D = mixes.size(-1);
  const int64_t K = static_cast<int64_t>(std::round(std::sqrt(static_cast<double>(D + 1)))) - 1;
  TORCH_CHECK(K > 0 && 2 * K + K * K == D, "Invalid mixes last dim: ", D);
  TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
  TORCH_CHECK(hc_base.numel() == D, "hc_base numel mismatch");

  auto lead_sizes = mixes.sizes().slice(0, mixes.dim() - 1).vec();
  auto pre_sizes = lead_sizes;
  pre_sizes.push_back(K);
  auto comb_sizes = lead_sizes;
  comb_sizes.push_back(K);
  comb_sizes.push_back(K);

  auto opts = mixes.options().dtype(at::kFloat);
  at::Tensor pre = at::empty(pre_sizes, opts);
  at::Tensor post = at::empty(pre_sizes, opts);
  at::Tensor comb = at::empty(comb_sizes, opts);

  const int64_t M = mixes.numel() / D;
  if (M == 0) {
    return std::make_tuple(pre, post, comb);
  }

  at::Tensor scale_f = hc_scale.to(at::kFloat).contiguous();
  at::Tensor base_f = hc_base.to(at::kFloat).contiguous();
  const float* scale_ptr = scale_f.const_data_ptr<float>();
  const float* base_ptr = base_f.const_data_ptr<float>();
  const float s_pre = scale_ptr[0];
  const float s_post = scale_ptr[1];
  const float s_comb = scale_ptr[2];
  const float mult_f = static_cast<float>(hc_mult);
  const float eps_f = static_cast<float>(eps);

  float* pre_ptr = pre.data_ptr<float>();
  float* post_ptr = post.data_ptr<float>();
  float* comb_ptr = comb.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      mixes.scalar_type(),
      "hc_split_sinkhorn_cpu",
      [&] {
        const scalar_t* mix_ptr = mixes.const_data_ptr<scalar_t>();
        at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
          for (int64_t m = begin; m < end; ++m) {
            const scalar_t* m_row = mix_ptr + m * D;
            float* p_pre = pre_ptr + m * K;
            float* p_post = post_ptr + m * K;
            float* c = comb_ptr + m * K * K;

            for (int64_t k = 0; k < K; ++k) {
              float zp = static_cast<float>(m_row[k]) * s_pre + base_ptr[k];
              p_pre[k] = 1.0f / (1.0f + std::exp(-zp)) + eps_f;

              float zq = static_cast<float>(m_row[K + k]) * s_post + base_ptr[K + k];
              p_post[k] = (1.0f / (1.0f + std::exp(-zq))) * mult_f;
            }

            // Row-wise softmax(-1) + eps
            for (int64_t i = 0; i < K; ++i) {
              float max_v = -std::numeric_limits<float>::infinity();
              for (int64_t j = 0; j < K; ++j) {
                int64_t idx = 2 * K + i * K + j;
                float z = static_cast<float>(m_row[idx]) * s_comb + base_ptr[idx];
                c[i * K + j] = z;
                max_v = std::max(max_v, z);
              }
              float row_sum = 0.0f;
              for (int64_t j = 0; j < K; ++j) {
                float e = std::exp(c[i * K + j] - max_v);
                c[i * K + j] = e;
                row_sum += e;
              }
              for (int64_t j = 0; j < K; ++j) {
                c[i * K + j] = c[i * K + j] / row_sum + eps_f;
              }
            }

            // Initial column normalization
            for (int64_t j = 0; j < K; ++j) {
              float col_sum = 0.0f;
              for (int64_t i = 0; i < K; ++i) {
                col_sum += c[i * K + j];
              }
              float inv = 1.0f / (col_sum + eps_f);
              for (int64_t i = 0; i < K; ++i) {
                c[i * K + j] *= inv;
              }
            }

            // Remaining Sinkhorn-Knopp iterations
            for (int64_t it = 1; it < sinkhorn_iters; ++it) {
              for (int64_t i = 0; i < K; ++i) {
                float row_sum = 0.0f;
                for (int64_t j = 0; j < K; ++j) {
                  row_sum += c[i * K + j];
                }
                float inv = 1.0f / (row_sum + eps_f);
                for (int64_t j = 0; j < K; ++j) {
                  c[i * K + j] *= inv;
                }
              }
              for (int64_t j = 0; j < K; ++j) {
                float col_sum = 0.0f;
                for (int64_t i = 0; i < K; ++i) {
                  col_sum += c[i * K + j];
                }
                float inv = 1.0f / (col_sum + eps_f);
                for (int64_t i = 0; i < K; ++i) {
                  c[i * K + j] *= inv;
                }
              }
            }
          }
        });
      });

  return std::make_tuple(pre, post, comb);
}

at::Tensor hc_combine_cpu(const at::Tensor& x_flat, const at::Tensor& pre, int64_t hc) {
  CHECK_INPUT(x_flat);
  CHECK_DIM(2, x_flat);
  TORCH_CHECK(hc > 0 && x_flat.size(1) % hc == 0, "x_flat dim 1 must be divisible by hc");

  const int64_t M = x_flat.size(0);
  const int64_t H = x_flat.size(1) / hc;
  TORCH_CHECK(pre.numel() == M * hc, "pre numel mismatch");

  at::Tensor out = at::empty({M, H}, x_flat.options());
  if (M == 0) {
    return out;
  }

  at::Tensor pre_f = pre.to(at::kFloat).contiguous();
  const float* pre_ptr = pre_f.const_data_ptr<float>();

  if (x_flat.scalar_type() == at::kFloat) {
    at::Tensor x_bf16_out = (x_flat.view({M, hc, H}) * pre_f.view({M, hc, 1})).sum(1);
    return x_bf16_out;
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(x_flat.scalar_type(), "hc_combine_cpu", [&] {
    hc_combine_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        x_flat.const_data_ptr<scalar_t>(),
        pre_ptr,
        M,
        hc,
        H);
  });
  return out;
}

at::Tensor mhc_post_cpu(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_mix,
    const at::Tensor& comb) {
  CHECK_INPUT(x);
  CHECK_INPUT(residual);
  TORCH_CHECK(x.scalar_type() == residual.scalar_type(), "x and residual dtype mismatch");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, H]");

  const int64_t M = x.size(0);
  const int64_t H = x.size(1);
  at::Tensor out = at::empty_like(residual);
  if (M == 0) {
    return out;
  }

  TORCH_CHECK(residual.numel() % (M * H) == 0, "residual shape mismatch with x");
  const int64_t K = residual.numel() / (M * H);
  TORCH_CHECK(post_mix.numel() == M * K, "post_mix numel mismatch");
  TORCH_CHECK(comb.numel() == M * K * K, "comb numel mismatch");

  at::Tensor post_f = post_mix.to(at::kFloat).contiguous();
  at::Tensor comb_f = comb.to(at::kFloat).contiguous();

  if (x.scalar_type() == at::kFloat) {
    auto res_3d = residual.view({M, K, H});
    auto comb_3d = comb_f.view({M, K, K});
    auto post_3d = post_f.view({M, K, 1});
    return (at::bmm(comb_3d.transpose(1, 2), res_3d) + post_3d * x.unsqueeze(1))
        .reshape(residual.sizes());
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "hc_post_cpu", [&] {
    hc_post_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        x.const_data_ptr<scalar_t>(),
        residual.const_data_ptr<scalar_t>(),
        post_f.const_data_ptr<float>(),
        comb_f.const_data_ptr<float>(),
        M,
        K,
        H);
  });
  return out;
}
