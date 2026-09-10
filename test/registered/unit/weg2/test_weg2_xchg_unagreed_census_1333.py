# SPDX-License-Identifier: Apache-2.0
"""#1333 B1a -- THE PER-CLASS CENSUS OF WHAT THE PAIR DID **NOT** AGREE.

THE MEASUREMENT THIS ADDS, and why a boot needs it.  Boot weg2xsn8
(`f03e405e7d`, record `BOOT_weg2xsn8_0910.md`) was the first boot on which the
card manifest ever agreed on metal.  It printed, per card and per leg:

    agreed=370 manifest_mine=862 manifest_theirs=1200   bytes=2803456
    agreed=235 manifest_mine=505 manifest_theirs=1200   bytes=1195520
    agreed=230 manifest_mine=495 manifest_theirs=1200   bytes=1143296

-- three counts and the agreed bytes, and NOTHING about the remainder.  The
remainder is the quantity that sizes the S6 host bounce: the user's decision of
2026-09-11 is that non-identical layers are assembled in a small host buffer
and sliced per card, so "how many pieces, of which class, worth how many bytes,
does this card NOT hold identically with its peer" is the number the buffer is
built against.  It had to be approximated from the checkpoint (an upper bound
per class, `weg2/tools/class_byte_census.py`) because no log line carried it.

So the plan line now carries, for BOTH sides of the intersection:

    mine_unagreed=<n> mine_unagreed_bytes=<b> mine_unagreed_by_class=cls:n:b|...
    theirs_unagreed=<n> theirs_unagreed_bytes=<b> theirs_unagreed_by_class=...

THE DOUBLES ARE THE SMOKE'S, imported rather than rebuilt: a second pair of
fake models would be a second set of assumptions about the same asymmetry, and
these two are the ones whose agreement is already pinned
(`test_weg2_shadow_execution_smoke_1329.py`).  Their arithmetic is hand-checkable,
which is what makes this a measurement test rather than a snapshot:

    P (the PP stage, layers 8..15):  8 qkv_proj @ 64x32x2 = 4096 B
                                     8 down_proj @ 32x32x2 = 2048 B
                                     1 embed_tokens @ 128x32x2 = 8192 B  -> 17
    D (the TP shards, layers 0..15): 16 qkv_proj @ 21x32x2 = 1344 B
                                     16 down_proj @ 32x32x2 = 2048 B
                                     1 embed_tokens @ 128x32x2 = 8192 B  -> 33
    AGREED = the 8 replicated down_proj of the overlapping layers + embed = 9
    P's remainder  =  8 pieces, ONE class:  qkv_proj  8 x 4096  = 32768 B
    D's remainder  = 24 pieces, TWO classes: qkv_proj 16 x 1344 = 21504 B
                                             down_proj 8 x 2048 = 16384 B
                                             (layers 0..7, outside P's stage)

RED ON `f03e405e7d`: the fields do not exist, so every test below fails on the
attribute or on the missing token in the line.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402

from .test_weg2_shadow_execution_smoke_1329 import (  # noqa: E402
    DOWN_ROWS,
    P_STAGE,
    QKV_ROWS,
    _d_shard_model,
    _p_stage_model,
)
from .test_weg2_xchg_transport_1273 import _fresh_boot  # noqa: E402

COLS = 32
ITEMSIZE = 2
QKV_BYTES_P = QKV_ROWS * COLS * ITEMSIZE            # 4096, the whole tensor
QKV_BYTES_D = (QKV_ROWS // 3) * COLS * ITEMSIZE     # 1344, one TP shard
DOWN_BYTES = DOWN_ROWS * COLS * ITEMSIZE            # 2048, replicated
N_STAGE = len(P_STAGE)                              # 8 layers in P's stage


@pytest.fixture()
def chunked(monkeypatch):
    from sglang.srt.managers import weg2_memory_saver as ms

    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "8")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "4")
    return ms


@pytest.fixture()
def region(tmp_path):
    boot = _fresh_boot()
    r = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    r.begin_flip(f"{boot}.1")
    yield r
    r.close()


def _manifests():
    p, reason_p = sh.derive_card_manifest(rank=0, model=_p_stage_model())
    d, reason_d = sh.derive_card_manifest(rank=0, model=_d_shard_model())
    assert p is not None, reason_p
    assert d is not None, reason_d
    return p, d


def _agree(region):
    """The two-flip handshake, then the intersection, as the product does it."""
    p, d = _manifests()
    p_row, d_row = xr.rank_row("P", 0), xr.rank_row("D", 0)
    for _ in range(2):
        p_agreed, p_state = sh.reconcile_card_manifest(
            region, row=p_row, peer_row=d_row, entries=p)
        d_agreed, d_state = sh.reconcile_card_manifest(
            region, row=d_row, peer_row=p_row, entries=d)
    assert p_state == d_state == sh.MANIFEST_AGREED, (p_state, d_state)
    return p_agreed, d_agreed


def _as_dict(census):
    """``((cls, n, bytes), ...)`` -> ``{cls: (n, bytes)}``, refusing duplicates."""
    out = {}
    for cls, n, nbytes in census:
        assert cls not in out, f"class {cls} appears twice in one census"
        out[cls] = (int(n), int(nbytes))
    return out


# ===========================================================================
# THE MEASUREMENT.
# ===========================================================================


def test_the_unagreed_census_names_every_class_with_count_and_bytes(
        chunked, region):
    """Both remainders, per class, against the hand-computed arithmetic."""
    p_agreed, d_agreed = _agree(region)

    # P's view: its own remainder is the sharded qkv_proj it holds whole.
    mine = _as_dict(p_agreed.mine_unagreed)
    assert mine == {"qkv_proj": (N_STAGE, N_STAGE * QKV_BYTES_P)}, mine
    # ... and the peer's remainder, from the SAME two rows, is D's.
    theirs = _as_dict(p_agreed.theirs_unagreed)
    assert theirs == {
        "qkv_proj": (2 * N_STAGE, 2 * N_STAGE * QKV_BYTES_D),
        "down_proj": (N_STAGE, N_STAGE * DOWN_BYTES),
    }, theirs

    # D's view is the MIRROR of P's, computed independently on the other end.
    # Both ends must produce the same two censuses with the sides swapped, or
    # the two boot logs would size two different buffers.
    assert _as_dict(d_agreed.mine_unagreed) == theirs
    assert _as_dict(d_agreed.theirs_unagreed) == mine


def test_the_agreed_pieces_never_appear_in_the_unagreed_census(chunked, region):
    """MUTANT DIRECTION 1: the census computed over the WHOLE side.

    If the difference is not taken (or is taken against the wrong set), the
    replicated ``down_proj`` of the overlapping layers -- the pieces that DID
    agree -- show up in ``mine_unagreed`` and the bounce is sized for bytes
    that never need to travel.  The count is the whole assertion: P's remainder
    is 17 - 9 = 8 pieces, and 9 of its 17 are agreed.
    """
    p_agreed, _ = _agree(region)
    mine = _as_dict(p_agreed.mine_unagreed)
    assert sum(n for n, _b in mine.values()) == p_agreed.mine - p_agreed.count
    assert p_agreed.count == N_STAGE + 1, p_agreed.count
    assert sum(n for n, _b in mine.values()) == N_STAGE
    # The agreed class must not carry a remainder on P at all: every down_proj
    # P holds is one D holds identically.
    assert "down_proj" not in mine, mine
    theirs = _as_dict(p_agreed.theirs_unagreed)
    assert sum(n for n, _b in theirs.values()) == p_agreed.theirs - p_agreed.count


def test_every_class_in_the_census_carries_bytes(chunked, region):
    """MUTANT DIRECTION 2: a class row with a count and no bytes.

    A census whose bytes column is zero (or missing) reads as "nothing to
    carry" and is the exact shape that would size a host buffer at zero. Every
    row must carry bytes > 0, and the per-row bytes must equal the sum of its
    pieces' own extents -- checked here against the totals rather than against
    a repetition of the implementation's own multiplication.
    """
    p_agreed, d_agreed = _agree(region)
    for side in (p_agreed.mine_unagreed, p_agreed.theirs_unagreed,
                 d_agreed.mine_unagreed, d_agreed.theirs_unagreed):
        assert side, "an empty census on an asymmetric pair is the defect"
        for cls, n, nbytes in side:
            assert n > 0, (cls, n)
            assert nbytes > 0, (cls, nbytes)
            assert nbytes % n == 0, (cls, n, nbytes)
    # The two totals are the two sides' own remainders and must differ here:
    # P holds 8 whole qkv tensors (32768 B), D holds 16 shards plus 8 extra
    # down_proj (37888 B).  Equal totals would mean one side was counted twice.
    p_total = sum(b for _c, _n, b in p_agreed.mine_unagreed)
    d_total = sum(b for _c, _n, b in p_agreed.theirs_unagreed)
    assert p_total == N_STAGE * QKV_BYTES_P == 32768, p_total
    assert d_total == 2 * N_STAGE * QKV_BYTES_D + N_STAGE * DOWN_BYTES, d_total
    assert p_total != d_total


def test_the_plan_line_carries_both_censuses_with_their_totals(chunked, region):
    """The line is the deliverable: a boot record must be able to grep it."""
    p_agreed, _ = _agree(region)
    plan, reason = sh.derive_leg_plan(
        hook=sh.HOOK_SOURCE, group="P", peer="D", rank=0,
        model=_p_stage_model(), agreed=p_agreed, require_agreement=True)
    assert plan is not None, reason
    line = plan.line()
    for token in (f"mine_unagreed={N_STAGE} ",
                  f"mine_unagreed_bytes={N_STAGE * QKV_BYTES_P} ",
                  f"mine_unagreed_by_class=qkv_proj:{N_STAGE}:"
                  f"{N_STAGE * QKV_BYTES_P}",
                  f"theirs_unagreed={2 * N_STAGE + N_STAGE} ",
                  "theirs_unagreed_by_class="):
        assert token in line, (token, line)
    # BYTES-DESCENDING, so the first class on the line is the one that sizes
    # the buffer.  On D's remainder qkv_proj (21504 B) outweighs down_proj
    # (16384 B), and a reader must be able to trust that order.
    tail = line.split("theirs_unagreed_by_class=")[1].split()[0]
    assert tail.startswith(f"qkv_proj:{2 * N_STAGE}:{2 * N_STAGE * QKV_BYTES_D}"), tail
    assert "down_proj" in tail, tail
