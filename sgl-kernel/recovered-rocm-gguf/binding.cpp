// Minimal pybind binding for the GGUF kernels, so a ROCm/gfx1103 build of
// sgl-kernel's csrc/quantization/gguf can be compiled and exercised in
// isolation -- without building the whole sgl-kernel extension.
#include <torch/extension.h>

torch::Tensor ggml_dequantize(
    torch::Tensor W, int64_t type, int64_t m, int64_t n,
    std::optional<at::ScalarType> const& dtype, std::optional<torch::Tensor> out);
torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X, int64_t type, int64_t row);
torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type, int64_t row);
torch::Tensor ggml_moe_a8(
    torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t type, int64_t row, int64_t top_k, int64_t tokens);
torch::Tensor ggml_moe_a8_vec(
    torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
    int64_t top_k, int64_t type, int64_t row, int64_t tokens);
int64_t ggml_moe_get_block_size(int64_t type);
int64_t ggml_mmvq_kq_tuned();
int64_t ggml_mxfp4_native();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize", &ggml_dequantize, "",
        pybind11::arg("W"), pybind11::arg("type"), pybind11::arg("m"), pybind11::arg("n"),
        pybind11::arg("dtype") = std::nullopt, pybind11::arg("out") = std::nullopt);
  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8);
  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8);
  m.def("ggml_moe_a8", &ggml_moe_a8);
  m.def("ggml_moe_a8_vec", &ggml_moe_a8_vec);
  m.def("ggml_moe_get_block_size", &ggml_moe_get_block_size);
  m.def("ggml_mmvq_kq_tuned", &ggml_mmvq_kq_tuned);
  m.def("ggml_mxfp4_native", &ggml_mxfp4_native);
}
