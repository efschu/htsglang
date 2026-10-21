"""D-FORK-ANCHOR x MIXED-D (NF rc31b 82784e1138, boot ..._1021_202608, D TP0, 21.10.).

THE MEASURED GAP: D TP0 wrote 62 x ``D-FORK-ANCHOR SET ... state=split_only why=unmarked:gdn,ple_conv,ple_ngram`` against
64 x ``taken`` (logged lines; cumulative at 21:37:34: d_set:split_only 193, d_set:taken 91). Every one of the 62 unmarked
plans ran in a MIXED-D forward (``MIXED-D mixed n_prefill_tok=<the step>`` between the PLAN line and the forward), every
one of the 64 taken plans in an unmixed one. Chain pdflip-2-9:
  PLAN n=1 rid=pdflip-2-9 kind=turn at=125952 step=[125696,126119) role=host state=slot
  MIXED-D mixed n=6 n_prefill_tok=423 n_decode_rows=3
  PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=426
  SET n=1 rid=pdflip-2-9 kind=turn at=125952 state=split_only why=unmarked:gdn,ple_conv,ple_ngram
THE ROOT: ``ScheduleBatch.mix_with_running`` merges the running decodes in with ``merge_batch``, which nulls
``pdflip_turn_tracks`` (and the mamba track triple). MIXED-D (S32, 18.10.) restores the track triple for the prefill rows,
D-FORK-ANCHOR (21.10.) came later and nothing restored its descriptor: the forward got ``pdflip_turn_tracks=None``, the GDN
backend never reached ``_pdflip_turn_rows`` (no ``TURN-ANCHOR SKIP reason=geometry`` either), the PLE layers saw no turn
(``gdn`` not done) -- all three kinds unmarked, the plan inserted split-only, its slot freed.

Pinned (hermetic, CPU, the REAL ``ScheduleBatch.mix_with_running`` / ``turn_anchor.forward_rows`` / GDN
``_pdflip_turn_rows`` / qwen4_exp ``_ple_turn_rows`` / ``turn_anchor.insert_verdict``):
  (1) metal repro 2-9: the mixed batch keeps the plan, the forward marks gdn + both PLE kinds, the verdict is ``ok``
      (red on 82784e1138: ``unmarked:gdn,ple_conv,ple_ngram``, the metal string);
  (2) the fork row in a mixed batch is exactly the h row / conv window the upstream track math picks for a default track
      at t on that row (two prefill rows, the plan on the second, three tails behind);
  (3) the tails are counted, never planned; a mixed batch whose planned row changed still gets no rows; no descriptor
      stays no descriptor; no track triple, no plan (the fork rows ride the default track gathers).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

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


TA0 = _load("_turn_anchor_0929_for_rc31_mixed", _HERE / "test_pdflip_turn_anchor_0929.py")
MX = _load("_mixed_d_s32_for_rc31_mixed", _HERE / "test_pdflip_mixed_d_short_s32_1018.py")

PAGE = CHUNK = 64
METAL_NEED = ("ple_conv", "ple_ngram")          # NF D: the pool carries both Qwen4-Exp PLE side states


@pytest.fixture
def d_args():
    sa = TA0.ServerArgs(model_path="dummy", page_size=PAGE)
    sa._mamba_cache_chunk_size = CHUNK
    TA0.set_global_server_args_for_scheduler(sa)
    return sa


def _running(rig, n):
    """n decoding requests (rows 0..n-1 of the toy pool), the running batch MIXED-D folds in."""
    reqs = []
    for i in range(n):
        rig.alloc(i, 0, 16)
        reqs.append(MX.FakeReq(f"tail-{i}", [3 + i, 5, 8], [13 + i]))
    idx = torch.arange(n)
    return MX._ns(reqs=reqs, req_pool_indices=idx, req_pool_indices_cpu=idx.clone(),
                  seq_lens=torch.tensor([r.seqlen for r in reqs]), orig_seq_lens=torch.zeros(n, dtype=torch.int64),
                  seq_lens_cpu=None, spec_info=object())


def _prefill(rig, prefixes, exts, track_seqlens, track_slots, turn):
    """The short prefill batch as prepare_for_extend leaves it (prefill rows only, its plan in ``pdflip_turn_tracks``)."""
    n = len(prefixes)
    rows = list(range(8 - n, 8))                   # pool rows behind the tails
    reqs = [MX.FakeReq(f"pdflip-pre-{i}", [7] * (p + e), []) for i, (p, e) in enumerate(zip(prefixes, exts))]
    return MX._ns(
        reqs=reqs, req_pool_indices=torch.tensor(rows), req_pool_indices_cpu=torch.tensor(rows),
        seq_lens=torch.tensor([p + e for p, e in zip(prefixes, exts)]),
        orig_seq_lens=torch.tensor([p + e for p, e in zip(prefixes, exts)]),
        seq_lens_cpu=torch.tensor([p + e for p, e in zip(prefixes, exts)]),
        out_cache_loc=torch.arange(sum(exts)), extend_num_tokens=sum(exts), prefix_lens=list(prefixes),
        extend_lens=list(exts), extend_logprob_start_lens=[0] * n,
        req_to_token_pool=types.SimpleNamespace(req_to_token=rig.req_to_token),
        mamba_track_indices=torch.tensor(track_slots, dtype=torch.int64),
        mamba_track_mask=torch.ones(n, dtype=torch.bool),
        mamba_track_seqlens=torch.tensor(track_seqlens, dtype=torch.int64),
        pdflip_turn_tracks=turn,
    )


def _fb_of(batch):
    """The ForwardBatch fields the GDN backend / turn_anchor read, taken from the (mixed) schedule batch itself."""
    fb = types.SimpleNamespace(
        batch_size=batch.batch_size(), forward_mode=batch.forward_mode,
        extend_prefix_lens_cpu=list(batch.prefix_lens), extend_seq_lens_cpu=list(batch.extend_lens),
        extend_prefix_lens=torch.tensor(batch.prefix_lens, dtype=torch.int64),
        extend_seq_lens=torch.tensor(batch.extend_lens, dtype=torch.int64),
        mamba_track_mask=batch.mamba_track_mask, mamba_track_indices=batch.mamba_track_indices,
        mamba_track_seqlens=batch.mamba_track_seqlens,
        pdflip_turn_tracks=batch.pdflip_turn_tracks,
    )
    return fb


def _forward(fb):
    """One mixed forward's marks, the production call sites in order: ``GDNAttnBackend.init_forward_metadata`` (rows +
    ``gdn``) then every PLE layer (``_ple_turn_rows`` registers the kind, the layer writes and marks it)."""
    from flliper.srt.models import qwen4_exp as q

    be = TA0._backend()
    be.forward_metadata = TA0._metadata(be, fb)
    be.forward_metadata.conv_states_mask_indices = fb.mamba_track_indices[fb.mamba_track_mask]
    if be.forward_metadata.has_mamba_track_mask and getattr(fb, "pdflip_turn_tracks", None) is not None:
        be._pdflip_turn_rows(fb)                   # gdn_backend.py init_forward_metadata, the same guard
    for kind in METAL_NEED:
        turn = q._ple_turn_rows(fb, kind)
        if turn is not None:
            turn.done.add(kind)
    return be


# -- (1) metal repro ---------------------------------------------------------------------------------------------------
def test_metal_2_9_a_mixed_short_d_prefill_writes_its_fork_state(d_args):
    """pdflip-2-9: step [125696,126119) (423 tokens < halbX 3200), turn fork at 125952, main track 126080, three decodes
    folded in (MIXED-D n=6). Red on 82784e1138: the plan reaches the insert ``unmarked:gdn,ple_conv,ple_ngram``."""
    from flliper.srt.pdflip import turn_anchor as ta

    rig = MX.Rig()
    pre, end, t, main = 125696, 126119, 125952, 126080
    turn = ta.TurnTracks(1, need=METAL_NEED)
    turn.add(0, torch.tensor([16]), t, pre, end - pre)
    plan = (t, turn.slots[0], turn, pre, end)       # what the request carries (PENDING_ATTR)
    new = _prefill(rig, [pre], [end - pre], [main + 1], [12], turn)
    ScheduleBatch.mix_with_running(new, _running(rig, 3))
    assert new.forward_mode == ForwardMode.MIXED and new.batch_size() == 4
    assert new.extend_lens == [423, 1, 1, 1]

    fb = _fb_of(new)
    _forward(fb)
    assert ta.insert_verdict(plan) == (True, "ok"), ta.insert_verdict(plan)
    assert ta.forward_rows(fb) is turn, "the mixed forward must see the plan the scheduler drew the slot for"
    assert turn.dst_phys.tolist() == [16] and turn.rows_dev.tolist() == [0] and turn.offsets_dev.tolist() == [t - pre]


# -- (2) the row is the upstream row ------------------------------------------------------------------------------------
def test_the_fork_row_in_a_mixed_batch_is_the_upstream_track_row(d_args):
    """Two prefill rows (a 1088-token step and the 2-10-like gen-fork step [126208,126723) at 126656), the plan on row 1,
    three tails behind: the appended h row / conv window equal what the REAL ``_init_track_ssm_indices`` /
    ``_init_track_conv_indices`` pick for a default track at t on row 1 of the same mixed geometry."""
    from flliper.srt.pdflip import turn_anchor as ta

    rig = MX.Rig()
    prefixes, exts = [42048, 126208], [1088, 515]
    t = 126656
    turn = ta.TurnTracks(2, need=METAL_NEED)
    turn.add(1, torch.tensor([9]), t, 126208, 515)
    new = _prefill(rig, prefixes, exts, [43072 + 1, 126720 + 1], [40, 41], turn)
    ScheduleBatch.mix_with_running(new, _running(rig, 3))
    fb = _fb_of(new)
    base = TA0._metadata(TA0._backend(), _fb_of(new))
    be = _forward(fb)
    md = be.forward_metadata
    assert turn.complete()
    assert md.track_ssm_h_src[:-1].tolist() == base.track_ssm_h_src.tolist(), "the default rows first, unchanged"
    assert md.track_ssm_h_dst[-1].item() == 9

    ref_new = _prefill(MX.Rig(), prefixes, exts, [43072 + 1, t + 1], [40, 9], None)
    ScheduleBatch.mix_with_running(ref_new, _running(MX.Rig(), 3))
    ref = TA0._metadata(TA0._backend(), _fb_of(ref_new))
    assert md.track_ssm_h_src[-1].item() == ref.track_ssm_h_src[-1].item()
    assert md.track_conv_indices[-1].tolist() == ref.track_conv_indices[-1].tolist()
    # the tails own one h state each, AFTER the prefill rows: they never shift a planned row
    assert ta.h_rows(fb.extend_seq_lens_cpu, [1], [t], [126208], CHUNK) == [(1088 - 1) // 64 + 1 + (t - 126208) // 64]


# -- (3) bounds ---------------------------------------------------------------------------------------------------------
def test_the_tails_are_counted_never_planned(d_args):
    from flliper.srt.pdflip import turn_anchor as ta

    rig = MX.Rig()
    turn = ta.TurnTracks(1, need=METAL_NEED)
    turn.add(0, torch.tensor([16]), 125952, 125696, 423)
    new = _prefill(rig, [125696], [423], [126081], [12], turn)
    ScheduleBatch.mix_with_running(new, _running(rig, 2))
    assert (turn.bs, turn.rows, turn.prefixes, turn.ext_lens) == (3, [0], [125696], [423])
    assert new.mamba_track_mask.tolist() == [True, False, False]
    with pytest.raises(ValueError):
        turn.append_tail_rows(-1)


def test_a_mixed_batch_whose_planned_row_changed_still_gets_no_rows(d_args):
    from flliper.srt.pdflip import turn_anchor as ta

    rig = MX.Rig()
    turn = ta.TurnTracks(1, need=METAL_NEED)
    turn.add(0, torch.tensor([16]), 125952, 125696, 423)
    new = _prefill(rig, [125696], [423], [126081], [12], turn)
    ScheduleBatch.mix_with_running(new, _running(rig, 3))
    fb = _fb_of(new)
    fb.extend_prefix_lens_cpu = [125632] + fb.extend_prefix_lens_cpu[1:]
    assert ta.forward_rows(fb) is None
    _forward(fb)
    assert not turn.complete() and ta.insert_verdict((125952, turn.slots[0], turn, 125696, 126119))[1].startswith(
        "unmarked:gdn")


def test_no_descriptor_stays_none_and_no_track_triple_carries_no_plan(d_args):
    from flliper.srt.pdflip import turn_anchor as ta

    rig = MX.Rig()
    new = _prefill(rig, [125696], [423], [126081], [12], None)
    ScheduleBatch.mix_with_running(new, _running(rig, 3))
    assert new.pdflip_turn_tracks is None

    turn = ta.TurnTracks(1, need=METAL_NEED)
    turn.add(0, torch.tensor([16]), 125952, 125696, 423)
    new2 = _prefill(MX.Rig(), [125696], [423], [126081], [12], turn)
    new2.mamba_track_seqlens = None                # no default track gather: the fork rows have nothing to ride
    ScheduleBatch.mix_with_running(new2, _running(MX.Rig(), 3))
    assert new2.pdflip_turn_tracks is None and turn.bs == 1
