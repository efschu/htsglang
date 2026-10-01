// efeu-TP14 2026-10-01: standalone ROCm/gfx1103 build of sgl-kernel's EAGLE /
// NEXTN tree kernels (build_tree_kernel_efficient, verify_tree_greedy).
// csrc/speculative/eagle_utils.cu is a VERBATIM copy of the upstream source
// (sgl-kernel/csrc/speculative/eagle_utils.cu @ 13dc5f2dc7), which upstream
// setup_rocm.py already compiles for HIP (gfx942/950 only). The kernels use no
// warp intrinsics, so the port to gfx11 (wave32) is build wiring, not code.
// Ops are registered as torch.ops.sgl_spec_rocm.* (not sgl_kernel, so a real
// sgl_kernel can never be shadowed); sgl_spec_rocm.py mirrors the upstream
// python wrappers' signatures.
#include <torch/extension.h>
#include <torch/library.h>

void build_tree_kernel_efficient(
    at::Tensor parent_list, at::Tensor selected_index, at::Tensor verified_seq_len,
    at::Tensor tree_mask, at::Tensor positions, at::Tensor retrive_index,
    at::Tensor retrive_next_token, at::Tensor retrive_next_sibling,
    int64_t topk, int64_t depth, int64_t draft_token_num, int64_t tree_mask_mode);

void verify_tree_greedy(
    at::Tensor predicts, at::Tensor accept_index, at::Tensor accept_token_num,
    at::Tensor candidates, at::Tensor retrive_index, at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling, at::Tensor target_predict);

TORCH_LIBRARY(sgl_spec_rocm, m) {
  m.def(
      "verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor target_predict) -> ()");
  m.def(
      "build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
      "Tensor! tree_mask, Tensor! positions, Tensor! retrive_index, Tensor! retrive_next_token, "
      "Tensor! retrive_next_sibling, int topk, int depth, int draft_token_num, int tree_mask_mode) -> "
      "()");
}

TORCH_LIBRARY_IMPL(sgl_spec_rocm, CUDA, m) {
  m.impl("verify_tree_greedy", &verify_tree_greedy);
  m.impl("build_tree_kernel_efficient", &build_tree_kernel_efficient);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
