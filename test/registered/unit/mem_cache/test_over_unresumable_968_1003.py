"""OVER-UNRESUMABLE (27B 673cc89f6a, boot dkr27browauthoritybar1fs10031707
1003_170754, P group stop 17:23:06Z on PP1, rid weg2-24-204).

MEASURED (P log + debug-hold dump rank1_pid710):

    17:22:16 PP0/PP2  WEG2-READ-STAGES pages=57041   (early read, store anchor 57041)
    17:22:16 PP1      WEG2-READ-STAGES pages=59392   (D's flush tail 57041..59392 seen)
    17:22:21 PP1      #1400 FOLLOWER-EARLY-SETTLE told=57041 own=59392 -> over at=ack
    17:22:21 PP0      #988 LOADBACK prefix moved to 57041 ... mamba_restored=58
    17:22:21 PP1      [#904 match-census] refused=57041 MambaComponent:absent=57041
    17:22:21 PP1      #631 BULLETIN UNDER-COVERAGE told=57041, local=0, host_hit=0
    17:23:06 PP1      #968 PREFIX MATERIALISATION SHORTFALL ... holds 0 after 45.02s

"over" marked the rid SATISFIED at told and ``own_prefix`` acked told without
asking whether the rank can RESUME there: its tree held one node 0..59392 with
the recurrent state at 59392 only. Now the over-settle asks the told-fidelity
probe at told; unresumable -> ``over_unresumable``, not satisfied, the ack
names the own read (!= told) and PP0's PF answers told=0 for every rank.

Real ``check_prefetch_progress`` (#1157 harness) on a real FULL+MAMBA tree:
the read completes 16 tokens with its anchor at 16; PP0's told is 12."""

import importlib.util
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_store_told as st  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.managers import weg2_told_fidelity as tf  # noqa: E402
from sglang.srt.mem_cache.hicache_phase_binding import binding_state  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_n1_told_pin_968", os.path.join(os.path.dirname(__file__), "test_told_anchor_pin_ack_n1_1001.py"))
n1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n1)

REQ = n1.REQ
OWN = n1.TOLD          # the follower's early read: 16 tokens, anchor at 16
TOLD = OWN - 4         # PP0's (absolute) told: shallower than this rank's read


def _follower(cache):
    return SimpleNamespace(tree_cache=cache, _weg2_store_told={REQ: TOLD},
                           ps=SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1))


def test_the_tree_cannot_resume_at_told(monkeypatch):
    """The premise, green before and after: KV 0..16 with the state at 16 only
    -- a match capped at told=12 resumes nothing (the metal #904 refusal)."""
    try:
        cache, _ = n1._completed(monkeypatch, with_anchor=True)
        assert tf.pp0_admissible(_follower(cache), n1._req(), TOLD) == 0
        assert tf.pp0_admissible(_follower(cache), n1._req(), OWN) == OWN
    finally:
        binding_state().reset()


def test_an_over_read_without_a_state_at_told_is_not_satisfied(monkeypatch):
    """RED on 673cc89f6a: "over", SATISFIED, and the PF ack said told (12) --
    PP0 admitted a prefix this rank could not materialise (#968 on the metal)."""
    monkeypatch.delenv("SGLANG_WEG2_DUAL_LAYOUT", raising=False)
    try:
        cache, _ = n1._completed(monkeypatch, with_anchor=True)
        sched, req = _follower(cache), n1._req()
        how = st.follower_early_settle(sched, req, TOLD, OWN, absolute=True)
        ack = fb.own_prefix(sched, req, REQ, TOLD)
        assert ack != TOLD, (how, ack)         # PP0's pp0_decide -> Admit(0) for every rank
        assert ack == OWN                      # the ack names what this rank CAN resume
        assert how == "over_unresumable", how
        assert REQ not in (getattr(sched, "_weg2_store_told_satisfied", None) or {})
        assert sched._weg2_over_unresumable_n == 1
    finally:
        binding_state().reset()


def test_dual_layout_keeps_the_old_over(monkeypatch):
    """Dual layout: byte-identical -- no probe, "over" as before."""
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    try:
        cache, _ = n1._completed(monkeypatch, with_anchor=True)
        sched, req = _follower(cache), n1._req()
        assert st.follower_early_settle(sched, req, TOLD, OWN, absolute=True) == "over"
        assert sched._weg2_store_told_satisfied[REQ] == TOLD
    finally:
        binding_state().reset()
