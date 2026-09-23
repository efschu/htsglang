# SPDX-License-Identifier: Apache-2.0
"""fnFL2x18: the declared pad is zeroed locally and never becomes a lane.

THE SPECIMEN (boot fnFL2x18, 2026-09-23 02:36:10Z, the first P->D wake of the
Next-Flash Platztausch). Every Form-A D rank carries a zero pad expert row per
expert tensor (#74), declared as ZEROFILL (``src_rank=-1``). D TP2 collected
tag weights_9 -- layers 27-29, which the co-located P stage (layers 40-47)
does not hold -- and died:

    W68 Weg2XchgPlanDisagree: the join's plan carries NO desc for lane
    src=2 dst=2 tag='weights_9' (direction=pp_to_tp), which THIS rank owns

``group_descs_by_pair`` files a ZEROFILL desc under its destination's
diagonal; the diagonal lane then asked the join for (2, 2) pieces, and a pad
row has none. The pad was not zeroed on the sequential path either -- only the
ring transport called ``apply_zerofill``.

Hermetic: real POSIX semaphores, a fake device over files, the real mixin
method.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_undrained_lane_refusal_1391 import _bare_manager  # noqa: E402
from test_weg2_xchg_transport_1273 import (  # noqa: E402
    FakeDeviceOps,
    dev_ptr,
    read,
    write,
)

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_transport as tp
from sglang.srt.weg2 import xchg_bounce as xb

PAD_BYTES = 16


def _pad_desc(*, dst_ptr):
    return wx.XchgDesc(
        tag="weights_9", src_rank=-1, dst_rank=2,
        param_name="model.layers.27.mlp.experts.w13_weight_shape",
        dst_ptr=dst_ptr, kind=wx.ZEROFILL, nbytes=PAD_BYTES, rows=1,
        run_bytes=PAD_BYTES, spitch=0, dpitch=PAD_BYTES, src_off=0, dst_off=0,
    )


def _terms():
    return xb.BounceTerms(
        bytes_per_direction=64, n_layers=1, widest_layer_bytes=64, depth=1,
        pairs=6, slot_bytes=4096, mean_layer_bytes=64, buffer_bytes=4096,
        staging_bytes=4096, n_lanes=1, max_tag_bytes=0, lanes_concurrent=0,
    )


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    nonce = f"x18-zerofill-{os.getpid()}"
    xr.unlink_semaphores(nonce)
    xr.create_semaphores(nonce)

    def _no_pad_lane(self, *, hook, group, rank, pair, card, tag, log=None):
        raise AssertionError(
            f"a lane was derived for pair={pair} card={card} tag={tag} -- the "
            f"only descs of this leg are pad")

    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager,
                        "_weg2_seq_lane_descs", _no_pad_lane, raising=True)
    ops = FakeDeviceOps(str(tmp_path), 2)
    ops.raw_malloc(0, 4096)
    write(ops, dev_ptr(2, 0), b"\xab" * PAD_BYTES)
    try:
        yield nonce, ops, str(tmp_path)
    finally:
        xr.unlink_semaphores(nonce)


def _leg(rig, *, hook, dst_ptr):
    nonce, ops, root = rig
    _bare_manager()._weg2_xchg_bounce_leg(
        descs=[_pad_desc(dst_ptr=dst_ptr)], ops=ops, boot_nonce=nonce,
        terms=_terms(), mode=wx.INJECT_AUTHORITATIVE, shm_root=root, device=0,
        hook=hook, sems=tp.SemSet(nonce), tag="weights_9", rank=2,
    )
    return ops


def test_x18_a_pad_only_diagonal_derives_no_lane_and_zeroes_the_pad(rig):
    ops = _leg(rig, hook="authoritative", dst_ptr=dev_ptr(2, 0))
    assert read(ops, dev_ptr(2, 0), PAD_BYTES) == b"\0" * PAD_BYTES


def test_the_depositor_never_zeroes_a_destination_pad(rig):
    """The pad belongs to the collecting rank. A depositor that zeroed it would
    write through an address of the OTHER group's pages."""
    ops = _leg(rig, hook="source", dst_ptr=dev_ptr(2, 0))
    assert read(ops, dev_ptr(2, 0), PAD_BYTES) == b"\xab" * PAD_BYTES


def test_a_pad_without_an_address_is_refused_by_name(rig):
    with pytest.raises(wx.Weg2XchgPlanDisagree) as e:
        _leg(rig, hook="authoritative", dst_ptr=None)
    assert "ZEROFILL" in str(e.value) and "weights_9" in str(e.value)
