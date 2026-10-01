"""efeu-TP14: python wrappers for the gfx1103 build of sgl-kernel's EAGLE/NEXTN
tree kernels. Signatures mirror sgl_kernel.speculative (upstream) so the
serving tree can import these as a drop-in fallback when sgl_kernel is absent.
"""

import torch

import _sgl_spec_rocm  # noqa: F401  (registers torch.ops.sgl_spec_rocm.*)


def verify_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    torch.ops.sgl_spec_rocm.verify_tree_greedy.default(
        predicts, accept_index, accept_token_num, candidates, retrive_index,
        retrive_next_token, retrive_next_sibling, target_predict,
    )


def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    torch.ops.sgl_spec_rocm.build_tree_kernel_efficient.default(
        parent_list, selected_index, verified_seq_len, tree_mask, positions,
        retrive_index, retrive_next_token, retrive_next_sibling,
        topk, depth, draft_token_num, tree_mask_mode,
    )
