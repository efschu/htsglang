"""Review of b386b53661 (D-FORK-ANCHOR x MIXED-D, nf-rc31-forkanchor-review-1022).

(R1) THE CONSUMED DESCRIPTOR RIDES INTO THE DECODE. After a MIXED-D forward the mixed batch object becomes the running
     batch (scheduler.py ``running_batch = last_batch`` -- mix_with_running emptied the running batch), filter_batch
     returns early when nothing finished, and prepare_for_decode keeps ``pdflip_turn_tracks``. Before b386b53661 a mixed
     batch carried None; now it carries the consumed plan with ``gdn`` done and ``dst_phys`` set. The GDN side refuses
     it (``forward_rows``: extend, not verify); the PLE side (``qwen4_exp._ple_turn_rows``) had no mode gate, so a decode
     forward on the general PLE path (FLLIPER_ENABLE_QWEN4_PLE_FUSION=0, no spec) would write the PLE side states at
     the stale rows/offsets into a slot that is by then a tree anchor or freed. Gate: the same as forward_rows.
(R2) Two prefill rows, a plan on EACH, spec tails behind: both fork rows equal the upstream default-track rows at t.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from flliper.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


FA = _load("_rc31_forkanchor_mixed_for_review", _HERE / "test_nf_rc31_forkanchor_mixed_1021.py")
d_args = FA.d_args


def _mixed_after_forward(n_tails=3):
    from flliper.srt.pdflip import turn_anchor as ta

    rig = FA.MX.Rig()
    pre, end, t, main = 125696, 126119, 125952, 126080
    turn = ta.TurnTracks(1, need=FA.METAL_NEED)
    turn.add(0, torch.tensor([16]), t, pre, end - pre)
    new = FA._prefill(rig, [pre], [end - pre], [main + 1], [12], turn)
    ScheduleBatch.mix_with_running(new, FA._running(rig, n_tails))
    FA._forward(FA._fb_of(new))
    assert ta.insert_verdict((t, turn.slots[0], turn, pre, end)) == (True, "ok")
    return new, turn


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.IDLE])
def test_r1_a_consumed_mixed_plan_writes_nothing_in_a_later_non_extend_forward(d_args, mode):
    from flliper.srt.models import qwen4_exp as q
    from flliper.srt.pdflip import turn_anchor as ta

    new, turn = _mixed_after_forward()
    for r in new.reqs:
        r.finished = lambda: False
    ScheduleBatch.filter_batch(new)                # nothing finished: the early return keeps every field
    assert new.pdflip_turn_tracks is turn, "the premise: the running batch (= the mixed batch) still carries the plan"

    fb = FA._fb_of(new)
    fb.forward_mode = mode                          # the next round on the SAME batch object
    assert ta.forward_rows(fb) is None             # GDN side: no rows (unchanged)
    for kind in FA.METAL_NEED:
        assert q._ple_turn_rows(fb, kind) is None, (mode, kind)


def test_r1_the_mixed_extend_itself_still_writes_both_ple_kinds(d_args):
    from flliper.srt.models import qwen4_exp as q

    new, turn = _mixed_after_forward()
    fb = FA._fb_of(new)
    assert fb.forward_mode == ForwardMode.MIXED
    for kind in FA.METAL_NEED:
        assert q._ple_turn_rows(fb, kind) is turn


def test_r2_two_prefill_rows_each_with_a_plan_and_spec_tails(d_args):
    from flliper.srt.pdflip import turn_anchor as ta

    prefixes, exts = [42048, 126208], [1088, 515]
    t0, t1 = 42048 + 512, 126656
    turn = ta.TurnTracks(2, need=FA.METAL_NEED)
    turn.add(0, torch.tensor([8]), t0, prefixes[0], exts[0])
    turn.add(1, torch.tensor([9]), t1, prefixes[1], exts[1])
    new = FA._prefill(FA.MX.Rig(), prefixes, exts, [43072 + 1, 126720 + 1], [40, 41], turn)
    ScheduleBatch.mix_with_running(new, FA._running(FA.MX.Rig(), 3))
    assert new.extend_lens == [1088, 515, 1, 1, 1] and turn.bs == 5
    be = FA._forward(FA._fb_of(new))
    md = be.forward_metadata
    assert turn.complete()
    assert md.track_ssm_h_dst[-2:].tolist() == [8, 9]

    for row, t, slots in ((0, t0, [8, 41]), (1, t1, [40, 9])):
        seqlens = [43072 + 1, 126720 + 1]
        seqlens[row] = t + 1                         # a default track at t on that row (upstream math)
        ref_new = FA._prefill(FA.MX.Rig(), prefixes, exts, seqlens, slots, None)
        ScheduleBatch.mix_with_running(ref_new, FA._running(FA.MX.Rig(), 3))
        ref = FA.TA0._metadata(FA.TA0._backend(), FA._fb_of(ref_new))
        k = -2 + row
        assert md.track_ssm_h_src[k].item() == ref.track_ssm_h_src[row].item(), row
        assert md.track_conv_indices[k].tolist() == ref.track_conv_indices[row].tolist(), row
