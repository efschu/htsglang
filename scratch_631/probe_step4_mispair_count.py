# SPDX-License-Identifier: Apache-2.0
"""Count mispaired buffers produced by get_mha_kv_ptrs_with_pp (#646).

Runs the REAL split against labelled synthetic registrations and prints the
mispairing count per PP stage, for the draft arm and for the no-draft control.
Run this at the unmodified base AND after the fix; the numbers are the
evidence.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python python <this file>
"""

import sys
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager

TARGET_ITEM = 4096
DRAFT_ITEM = 4096
ALL_LAYERS = list(range(48))
STAGES = [(0, list(range(0, 16))), (16, list(range(16, 32))), (32, list(range(32, 48)))]


class Registration:
    def __init__(self):
        self.ptrs = []
        self.item_lens = []
        self.labels = {}

    def _add(self, label, item_len):
        ptr = 0x1000 + len(self.ptrs) * 0x100
        self.ptrs.append(ptr)
        self.item_lens.append(item_len)
        self.labels[ptr] = label
        return ptr

    @classmethod
    def build(cls, layer_ids, target_item_len, draft_item_len=None):
        r = cls()
        for lid in layer_ids:
            r._add(f"K{lid}", target_item_len)
        for lid in layer_ids:
            r._add(f"V{lid}", target_item_len)
        if draft_item_len is not None:
            r._add("Kd", draft_item_len)
            r._add("Vd", draft_item_len)
        return r


def _manager_for(start_layer, end_layer):
    # Real class, no __init__: the split reads kv_args and nothing else.
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(
        prefill_start_layer=start_layer, prefill_end_layer=end_layer
    )
    return mgr


def pairs_for(src, dst, start_layer, end_layer, dst_num_draft_buffers=None):
    mgr = _manager_for(start_layer, end_layer)
    try:
        out = mgr.get_mha_kv_ptrs_with_pp(src.ptrs, dst.ptrs, dst_num_draft_buffers)
    except TypeError:
        # Base signature has no dst_num_draft_buffers parameter.
        out = mgr.get_mha_kv_ptrs_with_pp(src.ptrs, dst.ptrs)
    src_k, src_v, dst_k, dst_v, stage = out[0], out[1], out[2], out[3], out[4]
    draft_pairs = out[5] if len(out) > 5 else []
    res = []
    for i in range(stage):
        res.append(
            (src.labels.get(src_k[i], "<oob>"), dst.labels.get(dst_k[i], "<oob>"))
        )
    for i in range(stage):
        res.append(
            (src.labels.get(src_v[i], "<oob>"), dst.labels.get(dst_v[i], "<oob>"))
        )
    for s_ptr, d_ptr, _idx in draft_pairs:
        res.append((src.labels.get(s_ptr, "<oob>"), dst.labels.get(d_ptr, "<oob>")))
    return res


def report(title, with_draft):
    print(f"--- {title} ---")
    dst = Registration.build(
        ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM if with_draft else None
    )
    total_bad = 0
    for start, layers in STAGES:
        src = Registration.build(
            layers, TARGET_ITEM, DRAFT_ITEM if with_draft else None
        )
        # 2 buffers (Kd + Vd) per draft layer.
        prs = pairs_for(src, dst, start, start + len(layers), 2 if with_draft else 0)
        bad = [(s, d) for s, d in prs if s != d]
        total_bad += len(bad)
        example = f"  example src {bad[0][0]} -> dst {bad[0][1]}" if bad else ""
        print(
            f"  stage start={start:2d}: {len(bad):2d} of {len(prs)} mispaired{example}"
        )
    print(f"  TOTAL mispaired across stages: {total_bad}")
    return total_bad


if __name__ == "__main__":
    bad_draft = report("PP prefill + draft pool on both arms", True)
    bad_ctrl = report("CONTROL: same PP geometry, no draft pool", False)
    print()
    print(f"draft arm total mispaired = {bad_draft}")
    print(f"control arm total mispaired = {bad_ctrl}")
    sys.exit(0)
