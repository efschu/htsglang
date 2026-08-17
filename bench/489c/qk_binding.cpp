// SPDX-License-Identifier: Apache-2.0
//
// #489 (c) / #726 -- torch-facing binding for the three QK arms.
//
// Plain C++ on purpose. The kernel launches live in qk_arms.cu behind
// `extern "C"` host functions, because `<<<>>>` is CUDA syntax the host
// compiler cannot parse -- and keeping this file torch-only is what lets the
// .cu stay checkable with `nvcc -cubin` for sm_86 and sm_120a without a
// device, which is the toolchain pin the whole ticket rests on.

#include <torch/extension.h>

extern "C" {
void c_launch_int8_imma(const void*, const void*, const void*, float*, int, int,
                        int, int);
void c_launch_fp8_deployed(const void*, const void*, const void*, float*, int,
                           int, int, int);
void c_launch_bf16_reference(const float*, const float*, float*, int, int, int);
int c_arm_b_native(void);
}

#define CHECK(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

void launch_int8_imma(at::Tensor q, at::Tensor k, at::Tensor k_scale,
                      at::Tensor out, int64_t tokens, int64_t heads,
                      int64_t groups) {
  CHECK(q); CHECK(k); CHECK(k_scale); CHECK(out);
  c_launch_int8_imma(q.data_ptr(), k.data_ptr(), k_scale.data_ptr(),
                     out.data_ptr<float>(), (int)tokens, (int)heads,
                     (int)k.size(1), (int)groups);
}

void launch_fp8_deployed(at::Tensor q, at::Tensor k, at::Tensor k_scale,
                         at::Tensor out, int64_t tokens, int64_t heads,
                         int64_t d, int64_t groups) {
  CHECK(q); CHECK(k); CHECK(k_scale); CHECK(out);
  c_launch_fp8_deployed(q.data_ptr(), k.data_ptr(), k_scale.data_ptr(),
                        out.data_ptr<float>(), (int)tokens, (int)heads, (int)d,
                        (int)groups);
}

void launch_bf16_reference(at::Tensor q, at::Tensor k, at::Tensor out,
                           int64_t tokens, int64_t heads, int64_t d) {
  CHECK(q); CHECK(k); CHECK(out);
  c_launch_bf16_reference(q.data_ptr<float>(), k.data_ptr<float>(),
                          out.data_ptr<float>(), (int)tokens, (int)heads,
                          (int)d);
}

// Read back FROM THE DEVICE which shape arm B compiled to. #489 (c) requires
// the log to say what each arm actually selected, not what the runner inferred
// from an arch string.
int64_t arm_b_native() { return (int64_t)c_arm_b_native(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_int8_imma", &launch_int8_imma);
  m.def("launch_fp8_deployed", &launch_fp8_deployed);
  m.def("launch_bf16_reference", &launch_bf16_reference);
  m.def("arm_b_native", &arm_b_native);
}
