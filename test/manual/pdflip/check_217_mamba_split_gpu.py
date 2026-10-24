# SPDX-License-Identifier: Apache-2.0
"""#217 review M1: the vectorised mamba state split ON THE GPU, byte for byte against the per-layer loop and the arena.

The desk proves the split on CPU tensors only; ``index_copy_`` with a strided source is the same TensorIterator on
CUDA, but a wrong split would not crash -- it would silently corrupt the mamba state after every handback. This script
runs the three split cases of the desk with the DEVICE POOL on ``cuda:0`` (MiB-sized, seconds):

  1. one block, the pinned-stage ("cpu") path      -- test_27b_admit_peak_1024::test_the_vectorised_split_lands_the_loops_bytes
  2. blocks 3 + 3 + 1 (partial last block)          -- review pin test_the_vectorised_split_with_a_partial_last_block_...
  3. "dma" mode, one row per block, five slots      -- test_27b_admit_peak_1024::test_dma_mode_reuses_one_row_and_lands_...

each with FLLIPER_PDFLIP_MAMBA_STATE_SPLIT_VEC=1 and =0; both must land exactly the arena's bytes in exactly the asked
rows and leave every other row zero. Prints one PASS/FAIL line per case and exits non-zero on any failure.

RUN ONLY INSIDE A BOOKED GPU WINDOW (gpuq + the old arbitration), on the card the window names, e.g.::

    CUDA_VISIBLE_DEVICES=<card> /opt/venv/bin/python test/manual/pdflip/check_217_mamba_split_gpu.py

Footprint: the CUDA context (~300-500 MiB) plus < 1 MiB of tensors. No model, no server, no NCCL.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest.mock as mock

import torch

from flliper.srt.mem_cache.pool_host import arena_mamba_pool as AMP

L, A, T_SHAPE, WIDTH, CONV_SHAPE, E = 3, 9, (2, 4), 3, (6, 3), 2
T_ROW = 8 * E
SPLIT = (2, 3, 1)
ROWS = 16


def _pool(page_bytes=None, seed=217):
    """Per layer [foreign][temporal][foreign][conv q][conv k][foreign][conv v]: other ranks' bytes in the slot."""
    t_ext, c_ext, off = [], [], 0
    for _l in range(L):
        off += 5
        t_ext.append((off, T_ROW)); off += T_ROW
        off += 3
        segs = []
        for j, n_j in enumerate(SPLIT):
            if j == 2:
                off += 2
            segs.append((off, n_j * WIDTH * E)); off += n_j * WIDTH * E
        c_ext.append(segs)
    blob = torch.randint(0, 255, (A, off), dtype=torch.uint8, generator=torch.Generator().manual_seed(seed))
    pool = AMP.ArenaMambaPoolHost.__new__(AMP.ArenaMambaPoolHost)
    pool._slot_view = blob
    pool._page_bytes = off if page_bytes is None else page_bytes
    pool._state_stage = None
    pool.temporal_dtype = torch.bfloat16
    pool.conv_dtype = torch.bfloat16
    pool._layout = {"L": L, "t_shape": T_SHAPE, "conv_shape": CONV_SHAPE, "width": WIDTH, "t_ext": t_ext,
                    "c_ext": c_ext}
    return pool, blob, t_ext, c_ext


def _dev_pool(dev):
    return types.SimpleNamespace(mamba_cache=types.SimpleNamespace(
        temporal=torch.zeros((L, ROWS) + T_SHAPE, dtype=torch.bfloat16, device=dev),
        conv=[torch.zeros((L, ROWS) + CONV_SHAPE, dtype=torch.bfloat16, device=dev)]))


def _u8(t):
    return t.contiguous().view(torch.uint8).cpu()


def _check(dp, blob, t_ext, c_ext, slots, didx) -> list:
    bad = []
    t = _u8(dp.mamba_cache.temporal)
    c = _u8(dp.mamba_cache.conv[0])
    for l in range(L):
        for s, d in zip(slots.tolist(), didx.tolist()):
            off, ln = t_ext[l]
            if not torch.equal(t[l, d].reshape(-1), blob[s, off:off + ln]):
                bad.append(f"temporal l={l} slot={s} row={d}")
            got = c[l, d].reshape(-1)
            want = torch.cat([blob[s, o:o + n] for o, n in c_ext[l]])
            if not torch.equal(got, want):
                bad.append(f"conv l={l} slot={s} row={d}")
    rest = [r for r in range(ROWS) if r not in didx.tolist()]
    if t[:, rest].any() or c[:, rest].any():
        bad.append("an unasked row was written")
    return bad


def _run(case, vec, dev, slots, didx, page_bytes=None, rows_per_block=None):
    env = {AMP.ENV_STATE_SPLIT_VEC: vec, AMP.ENV_STATE_LOAD_BLOCK_BYTES: "",
           "FLLIPER_PDFLIP_ARENA_PAGE_LOAD_MODE": "", "FLLIPER_PDFLIP_ADMIT_PEAK_STAGES": "1"}
    pool, blob, t_ext, c_ext = _pool(page_bytes)
    dp = _dev_pool(dev)
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.dict(os.environ, env))
        if rows_per_block is not None:
            st.enter_context(mock.patch.object(AMP, "_state_load_block_rows",
                                               lambda rb, dma, env=None: rows_per_block))
        pool._load_states_all_layers(dp, slots, didx.to(dev))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    took = pool._state_split_vec(dp, pool._compact[0], pool._compact[1]) is not None
    bad = _check(dp, blob, t_ext, c_ext, slots, didx)
    if vec == "1" and not took:
        bad.append("the vectorised split was NOT taken (shape guard declined) -- the case proves nothing")
    print(f"{'PASS' if not bad else 'FAIL'} {case} vec={vec} stage_rows={tuple(pool._state_dev_stage.shape)[0]} "
          f"split={'vec' if (vec == '1' and took) else 'layers'}" + ("" if not bad else f" :: {bad[:4]}"))
    return dp, not bad


def main(device: str = "cuda:0") -> int:
    """``device`` "cpu" is the desk smoke of this script itself (test_27b_admit_peak_1024); M1 is "cuda:0"."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            print("FAIL no CUDA device visible -- run inside the booked window with CUDA_VISIBLE_DEVICES set")
            return 2
        print(f"device {torch.cuda.get_device_name(dev)} torch {torch.__version__}")
    ok = True
    cases = [
        ("1-one-block", torch.tensor([5, 1, 7, 2]), torch.tensor([3, 9, 0, 12]), None, None),
        ("2-partial-last-block", torch.tensor([8, 1, 6, 2, 0, 5, 3]), torch.tensor([4, 15, 0, 9, 11, 2, 7]), None, 3),
        ("3-dma-one-row", torch.tensor([6, 0, 3, 4, 2]), torch.tensor([1, 14, 7, 2, 11]), 1 << 20, None),
    ]
    for case, slots, didx, pb, rpb in cases:
        a, oka = _run(case, "1", dev, slots, didx, pb, rpb)
        b, okb = _run(case, "0", dev, slots, didx, pb, rpb)
        same = (torch.equal(_u8(a.mamba_cache.temporal), _u8(b.mamba_cache.temporal))
                and torch.equal(_u8(a.mamba_cache.conv[0]), _u8(b.mamba_cache.conv[0])))
        print(f"{'PASS' if same else 'FAIL'} {case} vec == layers (whole device pools, byte-equal)")
        ok = ok and oka and okb and same
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0"))
