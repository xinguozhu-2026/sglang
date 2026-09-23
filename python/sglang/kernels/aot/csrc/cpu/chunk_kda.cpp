#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/CPUBlas.h>

#include <cmath>
#include <immintrin.h>

#include "common.h"

namespace {

inline float softplus(float x) {
  if (x > 20.0f) {
    return x;
  }
  return std::log1p(std::exp(x));
}

template <typename scalar_t>
void chunk_kda_recurrent(
    scalar_t* output,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* g,
    const scalar_t* beta,
    float* state,
    const float* a_log,
    const float* dt_bias,
    int64_t begin,
    int64_t end,
    int64_t heads,
    int64_t key_dim,
    int64_t value_dim,
    float scale,
    bool normalize_qk,
    bool activate_gate,
    bool activate_beta,
    const std::optional<double>& lower_bound) {
  std::vector<float> qf(key_dim);
  std::vector<float> kf(key_dim);
  std::vector<float> gate(key_dim);
  std::vector<float> residual(value_dim);

  for (int64_t token = begin; token < end; ++token) {
    for (int64_t head = 0; head < heads; ++head) {
      const int64_t qkv_offset = (token * heads + head) * key_dim;
      const int64_t v_offset = (token * heads + head) * value_dim;
      float q_norm = 0.0f;
      float k_norm = 0.0f;
      for (int64_t d = 0; d < key_dim; ++d) {
        qf[d] = static_cast<float>(q[qkv_offset + d]);
        kf[d] = static_cast<float>(k[qkv_offset + d]);
        q_norm += qf[d] * qf[d];
        k_norm += kf[d] * kf[d];
        float gate_value = static_cast<float>(g[qkv_offset + d]);
        if (activate_gate) {
          const float x = gate_value + dt_bias[head * key_dim + d];
          gate_value = lower_bound.has_value()
              ? static_cast<float>(*lower_bound) / (1.0f + std::exp(-std::exp(a_log[head]) * x))
              : -std::exp(a_log[head]) * softplus(x);
        }
        gate[d] = std::exp(gate_value);
      }
      if (normalize_qk) {
        q_norm = 1.0f / std::sqrt(q_norm + 1e-6f);
        k_norm = 1.0f / std::sqrt(k_norm + 1e-6f);
        for (int64_t d = 0; d < key_dim; ++d) {
          qf[d] *= q_norm;
          kf[d] *= k_norm;
        }
      }
      float beta_value = static_cast<float>(beta[token * heads + head]);
      if (activate_beta) {
        beta_value = 1.0f / (1.0f + std::exp(-beta_value));
      }
      float* head_state = state + head * value_dim * key_dim;
      for (int64_t row = 0; row < value_dim; ++row) {
        float projection = 0.0f;
        float* state_row = head_state + row * key_dim;
        for (int64_t d = 0; d < key_dim; ++d) {
          state_row[d] *= gate[d];
          projection += state_row[d] * kf[d];
        }
        residual[row] = (static_cast<float>(v[v_offset + row]) - projection) * beta_value;
        for (int64_t d = 0; d < key_dim; ++d) {
          state_row[d] += residual[row] * kf[d];
        }
        float out = 0.0f;
        for (int64_t d = 0; d < key_dim; ++d) {
          out += state_row[d] * qf[d];
        }
        output[v_offset + row] = static_cast<scalar_t>(out * scale);
      }
    }
  }
}

}  // namespace

at::Tensor chunk_kda_cpu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& initial_state,
    const at::Tensor& initial_state_indices,
    const std::optional<at::Tensor>& cu_seqlens,
    const std::optional<at::Tensor>& a_log,
    const std::optional<at::Tensor>& dt_bias,
    double scale,
    bool use_qk_l2norm,
    bool beta_is_raw,
    const std::optional<double>& lower_bound) {
  CHECK_CPU(q);
  CHECK_CPU(k);
  CHECK_CPU(v);
  CHECK_CPU(g);
  CHECK_CPU(beta);
  CHECK_CPU(initial_state);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(g);
  CHECK_CONTIGUOUS(beta);
  CHECK_CONTIGUOUS(initial_state);
  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(4, g);
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  q.scalar_type() == g.scalar_type() && q.scalar_type() == beta.scalar_type(),
              "chunk_kda_cpu expects q, k, v, g, and beta to share a dtype");
  TORCH_CHECK(initial_state.scalar_type() == at::kFloat, "chunk_kda_cpu state must be float32");
  TORCH_CHECK(q.size(0) == 1, "chunk_kda_cpu expects packed batch dimension 1");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == g.sizes(), "q, k, and g shapes must match");
  const int64_t tokens = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  TORCH_CHECK(v.size(0) == 1 && v.size(1) == tokens && v.size(2) == heads, "v shape mismatch");
  TORCH_CHECK(beta.sizes() == at::IntArrayRef({1, tokens, heads}), "beta shape mismatch");
  TORCH_CHECK(initial_state.size(1) == heads && initial_state.size(2) == value_dim &&
                  initial_state.size(3) == key_dim,
              "initial_state shape mismatch");
  TORCH_CHECK(a_log.has_value() == dt_bias.has_value(), "A_log and dt_bias must be provided together");
  if (a_log.has_value()) {
    TORCH_CHECK(a_log->scalar_type() == at::kFloat && dt_bias->scalar_type() == at::kFloat,
                "A_log and dt_bias must be float32");
    TORCH_CHECK(a_log->numel() == heads && dt_bias->numel() == heads * key_dim, "gate parameter shape mismatch");
  }

  auto output = at::empty_like(v);
  auto indices = initial_state_indices.contiguous().to(at::kLong);
  at::Tensor offsets;
  if (cu_seqlens.has_value()) {
    offsets = cu_seqlens->contiguous().to(at::kLong);
  } else {
    offsets = at::tensor({0, tokens}, at::TensorOptions().dtype(at::kLong));
  }
  const int64_t sequences = offsets.numel() - 1;
  TORCH_CHECK(indices.numel() == sequences, "initial_state_indices length mismatch");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "chunk_kda_cpu", [&] {
    const auto* offsets_ptr = offsets.const_data_ptr<int64_t>();
    const auto* indices_ptr = indices.const_data_ptr<int64_t>();
    const float* a_ptr = a_log.has_value() ? a_log->contiguous().const_data_ptr<float>() : nullptr;
    const float* bias_ptr = dt_bias.has_value() ? dt_bias->contiguous().const_data_ptr<float>() : nullptr;
    at::parallel_for(0, sequences, 1, [&](int64_t seq_begin, int64_t seq_end) {
      for (int64_t seq = seq_begin; seq < seq_end; ++seq) {
        const int64_t state_index = indices_ptr[seq];
        TORCH_CHECK(state_index >= 0 && state_index < initial_state.size(0), "state index out of range");
        chunk_kda_recurrent<scalar_t>(
            output.data_ptr<scalar_t>(), q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
            v.const_data_ptr<scalar_t>(), g.const_data_ptr<scalar_t>(), beta.const_data_ptr<scalar_t>(),
            initial_state.data_ptr<float>() + state_index * heads * value_dim * key_dim, a_ptr, bias_ptr,
            offsets_ptr[seq], offsets_ptr[seq + 1], heads, key_dim, value_dim, static_cast<float>(scale),
            use_qk_l2norm, a_log.has_value(), beta_is_raw, lower_bound);
      }
    });
  });
  return output;
}
