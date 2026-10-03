# SPDX-License-Identifier: Apache-2.0
"""Item 290 part 1: SEQ_SYNC_BATCH_MIB/_UNITS 256/128 is the CODE default.

Cost bookkeeping (host + staging VRAM) for the NF lanes, pinned as tests so the
change cannot grow into a reserve unnoticed:

* host: the ring files are ``cards x depth x (4 KiB + RING_SLOTS x batch)``;
  NF (3 cards, depth 2, 4 slots) 1.5 GiB (64 MiB) -> 6.0 GiB (256 MiB), the
  +4.5 GiB is below the measured reap_headroom of the NF boots (16.4-16.6 GiB)
  and is priced by the measured flip_ratchet, not by a reserve term;
* VRAM: the on-card IPC staging is sized by the TAG (``total_bytes``), never by
  the sync batch -- per-lane staging VRAM is independent of the batch default.

27B line: the default itself landed with 110-4 (eb40f738ad); this file adds the cost
bookkeeping (shared with the NF branch desk/nf-q-290-seqsync-1003, where it is red on
b2ec1a6d0f). Hermetic: no GPU, no boot.
"""

import inspect
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import host_ledger as hl  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402

GIB = 1 << 30
#: measured NF reap_headroom (hard bound 104.44 - predicted run peak):
#: boot 10020634 16.61 GiB, h6-abl boot 10030641 16.44 GiB.
NF_REAP_HEADROOM_GIB = 16.44


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(bx.SEQ_SYNC_BATCH_MIB_ENV, raising=False)
    monkeypatch.delenv(bx.SEQ_SYNC_BATCH_UNITS_ENV, raising=False)
    return monkeypatch


def test_code_default_is_256_mib_128_units(clean):
    assert bx.seq_sync_batch() == (256 << 20, 128)


def test_env_still_wins_and_garbage_falls_back_to_the_new_default(clean):
    clean.setenv(bx.SEQ_SYNC_BATCH_MIB_ENV, "64")
    clean.setenv(bx.SEQ_SYNC_BATCH_UNITS_ENV, "32")
    assert bx.seq_sync_batch() == (64 << 20, 32)
    clean.setenv(bx.SEQ_SYNC_BATCH_MIB_ENV, "x")
    clean.setenv(bx.SEQ_SYNC_BATCH_UNITS_ENV, "y")
    assert bx.seq_sync_batch() == (256 << 20, 128)
    clean.setenv(bx.SEQ_SYNC_BATCH_UNITS_ENV, "1")  # UNITS=1 = per-unit form
    assert bx.seq_sync_batch()[1] == 1


def test_sync_groups_follow_the_new_bounds(clean):
    class P:
        def __init__(self, n):
            self.nbytes = n

    mb, un = bx.seq_sync_batch()
    pieces = [P(2 << 20) for _ in range(300)]  # 600 MiB, 300 units
    groups = bx._sync_groups(pieces, mb, un)
    assert sum(len(g) for g in groups) == 300
    assert len(groups) == 3  # 128 + 128 + 44 units (bytes bound 256 MiB = 128 units)
    assert max(len(g) for g in groups) == 128


def test_nf_ring_host_cost_fits_the_headroom_without_a_reserve(clean):
    """NF: 3 cards, depth 2, 4 ring slots (nf-int4*.env: SEQ_BUFFER_DEPTH=2)."""
    cards, depth, slots = xr.N_CARDS, 2, bx.seq_lane_ring_slots()
    assert (cards, slots) == (3, 4)
    old = hl.seq_lanes_priced_bytes(ring_on=True, cards=cards, depth=depth,
                                    slots=slots, slot_bytes=64 << 20)
    new = hl.seq_lanes_priced_bytes(ring_on=True, cards=cards, depth=depth,
                                    slots=slots, slot_bytes=bx.seq_sync_batch()[0])
    assert old == 3 * 2 * (4096 + 4 * (64 << 20))        # 1.500 GiB (metal: priced=1.500)
    assert new == 3 * 2 * (4096 + 4 * (256 << 20))       # 6.000 GiB
    assert round((new - old) / GIB, 3) == 4.5
    # fits the measured headroom; not a reserve (the term stays out of charge_terms
    # and is carried by the measured flip_ratchet)
    assert (new - old) / GIB < NF_REAP_HEADROOM_GIB
    # the priced Lanes figure and the preregistered files agree (self-pricing)
    files = bx.ring_preregister_files("n", ["c0", "c1", "c2"], depth=2)
    assert sum(n for _, _, n in files) == new


def test_27b_depth1_ring_is_the_already_proven_3_gib(clean):
    new = hl.seq_lanes_priced_bytes(ring_on=True, cards=3, depth=1, slots=4,
                                    slot_bytes=bx.seq_sync_batch()[0])
    assert round(new / GIB, 3) == 3.0


def test_staging_vram_is_sized_by_the_tag_not_by_the_batch():
    src = inspect.getsource(bx.run_sequential_units)
    m = re.search(r"_stage_alloc\((.*?)charge=stage_charge\)", src, re.S)
    assert m, "stage allocation call not found"
    args = m.group(1)
    assert "total_bytes" in args
    assert "seq_sync_batch" not in args and "_bat_bytes" not in args
    # the batch bound is only ever used to group SYNCS in the copy loop
    assert "_bat_bytes" not in src.split("_stage_alloc(")[0]
