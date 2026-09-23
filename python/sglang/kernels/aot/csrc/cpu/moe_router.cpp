#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "common.h"
#include "vec.h"

namespace {

template <typename input_t, typename weight_t>
float router_dot(
    const input_t* __restrict__ input,
    const weight_t* __restrict__ weight,
    int64_t hidden_size) {
  using input_vec = at::vec::Vectorized<input_t>;
  using weight_vec = at::vec::Vectorized<weight_t>;
  using float_vec = at::vec::Vectorized<float>;

  constexpr int64_t input_width = input_vec::size();
  constexpr int64_t weight_width = weight_vec::size();
  static_assert(input_width == weight_width);

  float_vec sum0(0.0f);
  float_vec sum1(0.0f);
  int64_t d = 0;
  for (; d <= hidden_size - input_width; d += input_width) {
    input_vec x = input_vec::loadu(input + d);
    weight_vec w = weight_vec::loadu(weight + d);
    float_vec x0, x1, w0, w1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    std::tie(w0, w1) = at::vec::convert_to_float(w);
    sum0 = sum0 + x0 * w0;
    sum1 = sum1 + x1 * w1;
  }

  float sum = vec_reduce_sum(sum0 + sum1);
  for (; d < hidden_size; ++d) {
    sum += static_cast<float>(input[d]) * static_cast<float>(weight[d]);
  }
  return sum;
}

template <typename input_t>
float router_dot_float_weight(
    const input_t* __restrict__ input,
    const float* __restrict__ weight,
    int64_t hidden_size) {
  using input_vec = at::vec::Vectorized<input_t>;
  using float_vec = at::vec::Vectorized<float>;

  constexpr int64_t input_width = input_vec::size();
  constexpr int64_t float_width = float_vec::size();
  float_vec sum0(0.0f);
  float_vec sum1(0.0f);
  int64_t d = 0;
  for (; d <= hidden_size - input_width; d += input_width) {
    input_vec x = input_vec::loadu(input + d);
    float_vec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    sum0 = sum0 + x0 * float_vec::loadu(weight + d);
    sum1 = sum1 + x1 * float_vec::loadu(weight + d + float_width);
  }

  float sum = vec_reduce_sum(sum0 + sum1);
  for (; d < hidden_size; ++d) {
    sum += static_cast<float>(input[d]) * weight[d];
  }
  return sum;
}

template <typename input_t, typename weight_t>
void fused_moe_router_kernel(
    const input_t* __restrict__ input,
    const weight_t* __restrict__ weight,
    const float* __restrict__ correction_bias,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t hidden_size,
    int64_t topk,
    float softcap) {
  at::parallel_for(0, num_tokens, 1, [&](int64_t begin, int64_t end) {
    std::vector<float> logits(num_experts);
    std::vector<int32_t> order(num_experts);
    for (int64_t token = begin; token < end; ++token) {
      const input_t* token_input = input + token * hidden_size;
      float max_logit = -std::numeric_limits<float>::infinity();
      for (int64_t expert = 0; expert < num_experts; ++expert) {
        const weight_t* expert_weight = weight + expert * hidden_size;
        float logit;
        if constexpr (std::is_same_v<weight_t, float>) {
          logit = router_dot_float_weight(token_input, expert_weight, hidden_size);
        } else {
          logit = router_dot(token_input, expert_weight, hidden_size);
        }
        if (softcap != 0.0f) {
          logit = softcap * std::tanh(logit / softcap);
        }
        if (correction_bias != nullptr) {
          logit += correction_bias[expert];
        }
        logits[expert] = logit;
        max_logit = std::max(max_logit, logit);
        order[expert] = expert;
      }

      float sum_exp = 0.0f;
      for (int64_t expert = 0; expert < num_experts; ++expert) {
        logits[expert] = std::exp(logits[expert] - max_logit);
        sum_exp += logits[expert];
      }
      std::partial_sort(order.begin(), order.begin() + topk, order.end(), [&](int32_t lhs, int32_t rhs) {
        if (logits[lhs] == logits[rhs]) {
          return lhs < rhs;
        }
        return logits[lhs] > logits[rhs];
      });
      const float inverse_sum = 1.0f / sum_exp;
      for (int64_t rank = 0; rank < topk; ++rank) {
        const int32_t expert = order[rank];
        topk_ids[token * topk + rank] = expert;
        topk_weights[token * topk + rank] = logits[expert] * inverse_sum;
      }
    }
  });
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> fused_moe_router_cpu(
    const at::Tensor& hidden_states,
    const at::Tensor& router_weight,
    int64_t topk,
    double moe_softcapping,
    const std::optional<at::Tensor>& correction_bias) {
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(router_weight);
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be a 2D tensor");
  TORCH_CHECK(router_weight.dim() == 2, "router_weight must be a 2D tensor");
  TORCH_CHECK(hidden_states.size(1) == router_weight.size(1), "Router hidden size mismatch");
  TORCH_CHECK(
      hidden_states.scalar_type() == at::kBFloat16 || hidden_states.scalar_type() == at::kHalf,
      "hidden_states must be bfloat16 or float16");
  TORCH_CHECK(
      router_weight.scalar_type() == at::kFloat || router_weight.scalar_type() == hidden_states.scalar_type(),
      "router_weight must be float32 or match hidden_states dtype");

  const int64_t num_tokens = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  const int64_t num_experts = router_weight.size(0);
  TORCH_CHECK(topk > 0 && topk <= num_experts, "topk must satisfy 0 < topk <= num_experts");
  TORCH_CHECK(moe_softcapping >= 0.0, "moe_softcapping must be non-negative");

  const float* correction_bias_ptr = nullptr;
  if (correction_bias.has_value()) {
    const at::Tensor& bias = correction_bias.value();
    CHECK_INPUT_SHAPE_DTYPE<false>(bias, {num_experts}, at::kFloat);
    correction_bias_ptr = bias.data_ptr<float>();
  }

  at::Tensor topk_weights = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kFloat));
  at::Tensor topk_ids = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kInt));
  const float softcap = static_cast<float>(moe_softcapping);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(hidden_states.scalar_type(), "fused_moe_router_cpu_input", [&] {
    if (router_weight.scalar_type() == at::kFloat) {
      fused_moe_router_kernel<scalar_t, float>(
          hidden_states.data_ptr<scalar_t>(),
          router_weight.data_ptr<float>(),
          correction_bias_ptr,
          topk_weights.data_ptr<float>(),
          topk_ids.data_ptr<int32_t>(),
          num_tokens,
          num_experts,
          hidden_size,
          topk,
          softcap);
    } else {
      fused_moe_router_kernel<scalar_t, scalar_t>(
          hidden_states.data_ptr<scalar_t>(),
          router_weight.data_ptr<scalar_t>(),
          correction_bias_ptr,
          topk_weights.data_ptr<float>(),
          topk_ids.data_ptr<int32_t>(),
          num_tokens,
          num_experts,
          hidden_size,
          topk,
          softcap);
    }
  });
  return std::make_tuple(topk_weights, topk_ids);
}
