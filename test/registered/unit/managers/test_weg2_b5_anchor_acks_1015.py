"""B5 ANCHOR-ACKS (NF h16p 226aad7105, boot ..._1015_115748, P.log): PF TOLD-FALLBACK ~14/h, 497k tokens
re-prefilled in 17 min. Three cases:

* 12:04:24 weg2-3-27 ``told=91712 -> 0 reason=mismatch acks={1: 91712, 2: 87296}``: the acks differ, H64's
  floor needs EQUAL acks -> told=0. A follower's ack is its resumable REACH; a stage that reached 91712 proved
  nothing about 87296 unless it has a state anchor there. B5: the ack carries the anchors below the reach; PP0
  floors at the deepest depth EVERY stage proved.
* 12:16:46 weg2-22-70 ``told=117184 -> 0 reason=told_fidelity acks={2: 107584, 1: 107584}`` and ``#TF
  TOLD-FIDELITY ... told=107584 pp0_admissible=67584``: the followers agreed on 107584, PP0's own tree resumes
  only at 67584. B5: PP0 takes the next deeper common depth it resumes from itself.
* 12:05:49 weg2-3-28 ``reason=frist acks={}``: no ack, nothing proved -- stays told=0.

PP0 decides alone (its acks, its own tree); the followers only report what they can resume from.
"""

from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (fb.ENV_ANCHOR_ACKS, fb.ENV_FLOOR, fb.ENV_MIN_PASSES):
        monkeypatch.delenv(k, raising=False)


def _sched():
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3, pp_rank=0))


def _open(s, rid, told, acks, anchors=None):
    fb.pp0_open(s, rid, told, published_at=0.0, own_read_s=0.0, window_s=1.0, published_pass=0)
    o = fb._pp0_open_map(s)[rid]
    o.acks.update(acks)
    o.anchors.update(anchors or {})
    return o


# ---------------------------------------------------------------------------
# the three h16p cases
# ---------------------------------------------------------------------------


def test_mismatch_120424_red_without_anchors(monkeypatch):
    monkeypatch.setenv(fb.ENV_ANCHOR_ACKS, "0")
    s = _sched()
    _open(s, "weg2-3-27", 91712, {1: 91712, 2: 87296}, {1: (87296, 40960)})
    assert fb.pp0_decide(s, "weg2-3-27", now=100.0, pass_n=10) == (0, fb.REASON_MISMATCH)


def test_mismatch_120424_green_floors_at_the_common_anchor():
    s = _sched()
    _open(s, "weg2-3-27", 91712, {1: 91712, 2: 87296}, {1: (87296, 40960), 2: (40960,)})
    assert fb.pp0_decide(s, "weg2-3-27", now=100.0, pass_n=10) == (87296, fb.REASON_FLOOR)


def test_mismatch_without_a_common_depth_stays_zero():
    """PP1 has no anchor at 87296: nobody proved a depth every stage resumes from."""
    s = _sched()
    _open(s, "weg2-3-27", 91712, {1: 91712, 2: 87296}, {1: (64000,), 2: ()})
    assert fb.pp0_decide(s, "weg2-3-27", now=100.0, pass_n=10) == (0, fb.REASON_MISMATCH)


def _tf_admits_up_to(monkeypatch, limit):
    from sglang.srt.managers import weg2_told_fidelity as tf

    monkeypatch.setattr(tf, "pp0_admissible", lambda sched, req, depth: min(int(depth), int(limit)))


def test_told_fidelity_121646_red_without_anchors(monkeypatch):
    monkeypatch.setenv(fb.ENV_ANCHOR_ACKS, "0")
    _tf_admits_up_to(monkeypatch, 67584)
    s = _sched()
    _open(s, "weg2-22-70", 117184, {2: 107584, 1: 107584}, {1: (67584,), 2: (67584,)})
    assert fb.pp0_decide(s, "weg2-22-70", now=100.0, pass_n=10) == (107584, fb.REASON_FLOOR)
    assert fb.pp0_floor_below(s, "weg2-22-70", object(), 107584, True) == 0  # -> told_fidelity 0, the h16p log


def test_told_fidelity_121646_green_pp0_takes_the_next_common_depth(monkeypatch):
    _tf_admits_up_to(monkeypatch, 67584)
    s = _sched()
    _open(s, "weg2-22-70", 117184, {2: 107584, 1: 107584}, {1: (67584, 32768), 2: (67584,)})
    assert fb.pp0_decide(s, "weg2-22-70", now=100.0, pass_n=10) == (107584, fb.REASON_FLOOR)
    assert fb.pp0_floor_below(s, "weg2-22-70", object(), 107584, True) == 67584


def test_told_fidelity_pp0_without_any_common_anchor_stays_zero(monkeypatch):
    _tf_admits_up_to(monkeypatch, 50000)
    s = _sched()
    _open(s, "weg2-22-70", 117184, {2: 107584, 1: 107584}, {1: (67584,), 2: (67584,)})
    assert fb.pp0_floor_below(s, "weg2-22-70", object(), 107584, True) == 0


def test_frist_120549_without_acks_stays_zero():
    s = _sched()
    _open(s, "weg2-3-28", 119360, {})
    assert fb.pp0_decide(s, "weg2-3-28", now=1e9, pass_n=10_000) == (0, fb.REASON_FRIST)


def test_equal_acks_at_told_are_unchanged():
    s = _sched()
    _open(s, "r", 50000, {1: 50000, 2: 50000}, {1: (40000,), 2: (40000,)})
    assert fb.pp0_decide(s, "r", now=100.0, pass_n=10) == (50000, fb.REASON_ACKS)


def test_an_incomplete_ack_set_still_waits():
    s = _sched()
    _open(s, "r", 50000, {1: 40000}, {1: (30000,)})
    assert fb.pp0_decide(s, "r", now=0.0, pass_n=0) is None


def test_a_follower_acking_zero_can_still_prove_a_lower_anchor():
    """NO-ROOM at its reach (ack 0), room at a lower anchor: that anchor is proved (room-checked on the follower)."""
    s = _sched()
    _open(s, "r", 90000, {1: 0, 2: 90000}, {1: (40000,), 2: (40000,)})
    assert fb.pp0_decide(s, "r", now=100.0, pass_n=10) == (40000, fb.REASON_FLOOR)


# ---------------------------------------------------------------------------
# the wire: PP0 reads both entry forms
# ---------------------------------------------------------------------------


def test_harvest_reads_the_long_and_the_short_entry(monkeypatch):
    s = _sched()
    _open(s, "a", 91712, {})
    _open(s, "b", 5000, {})
    acks = [fb.Weg2ToldReadAck(rank=1, seq=0, reads=[("a", 91712, (87296, 40960)), ("b", 5000)]),
            fb.Weg2ToldReadAck(rank=2, seq=0, reads=[("a", 87296, ())])]
    monkeypatch.setattr(fb, "_channel", lambda sched: types.SimpleNamespace(harvest=lambda: acks))
    assert fb.pp0_harvest(s) == 3
    o = fb._pp0_open_map(s)["a"]
    assert o.acks == {1: 91712, 2: 87296} and o.anchors == {1: (87296, 40960), 2: ()}
    assert fb._pp0_open_map(s)["b"].anchors == {1: ()}


# ---------------------------------------------------------------------------
# the follower's anchors: the tree's own validators on the matched path
# ---------------------------------------------------------------------------


class _Node:
    def __init__(self, n, parent, state):
        self.key = [0] * n
        self.parent = parent
        self.state = state


class _Comp:
    def create_match_validator(self, match_device_only=False):
        return lambda node, depth: bool(node.state)


def test_path_anchor_depths_are_the_validated_boundaries_deepest_first():
    root = _Node(0, None, True)
    a = _Node(32768, root, True)      # 32768
    b = _Node(34816, a, True)         # 67584
    c = _Node(40000, b, False)        # 107584: no state
    d = _Node(9600, c, True)          # 117184
    tree = types.SimpleNamespace(_components_tuple=(_Comp(),), root_node=root, cache_controller=object())
    assert fb._path_anchor_depths(tree, d) == [117184, 67584, 32768]


def test_ack_anchors_respect_told_own_and_the_room(monkeypatch):
    root = _Node(0, None, True)
    a = _Node(32768, root, True)
    b = _Node(34816, a, True)
    c = _Node(40000, b, True)
    tree = types.SimpleNamespace(_components_tuple=(_Comp(),), root_node=root, cache_controller=object(),
                                 match_prefix=lambda params: types.SimpleNamespace(best_match_node=c))
    sched = types.SimpleNamespace(tree_cache=tree, ps=types.SimpleNamespace(pp_rank=1))
    req = types.SimpleNamespace(rid="r", full_untruncated_fill_ids=list(range(120000)), extra_key=None)
    from sglang.srt.managers import weg2_store_told as st

    monkeypatch.setattr(st, "_probe_key", lambda *a, **k: (None, False))
    monkeypatch.setattr(fb, "_room_own", lambda sched, req, rid, d: d)
    assert fb.ack_anchors(sched, req, "r", 107584, 117184) == [67584, 32768]
    monkeypatch.setattr(fb, "_room_own", lambda sched, req, rid, d: d if d <= 40000 else 0)
    assert fb.ack_anchors(sched, req, "r", 107584, 117184) == [32768]


def test_follower_pump_sends_the_long_entry(monkeypatch):
    sent = []
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1),
                                  tree_cache=types.SimpleNamespace(check_prefetch_progress=lambda rid: True))
    st = fb._fstate(sched, create=True)
    st.expect["r"] = 117184
    st.registered["r"] = types.SimpleNamespace()
    monkeypatch.setattr(fb, "own_prefix", lambda *a: 107584)
    monkeypatch.setattr(fb, "ack_anchors", lambda *a: [67584])
    monkeypatch.setattr(fb, "_channel", lambda s: types.SimpleNamespace(
        send_nowait=lambda ack: sent.append(ack) or True, pump=lambda: None))
    fb.follower_pump(sched)
    assert sent[0].reads == [("r", 107584, (67584,))]


# ---------------------------------------------------------------------------
# mutants and wiring
# ---------------------------------------------------------------------------


def test_mutant_union_instead_of_intersection_is_caught(monkeypatch):
    """A depth only ONE follower proved must never be admitted (raenge-nie-uneins)."""
    def union(o):
        told = int(o.told)
        allset = set()
        for rank, own in o.acks.items():
            allset |= {int(own)} | set(o.anchors.get(rank, ()))
        return sorted((d for d in allset if 0 < d < told), reverse=True)

    s = _sched()
    _open(s, "x", 91712, {1: 91712, 2: 80000}, {1: (87296,), 2: ()})
    assert fb.pp0_decide(s, "x", now=100.0, pass_n=10) == (0, fb.REASON_MISMATCH)
    monkeypatch.setattr(fb, "common_depths", union)
    assert fb.pp0_decide(s, "x", now=100.0, pass_n=10) != (0, fb.REASON_MISMATCH)


def test_wiring_store_told_retries_below_a_tf_refusal():
    import inspect

    from sglang.srt.managers import weg2_store_told as st

    src = inspect.getsource(st)
    i_tf = src.index("_tf_told, _ = _tf.pp0_verdict(scheduler, p.req, told_final, p.absolute)")
    seg = src[i_tf:i_tf + 1600]
    assert "_fb.pp0_floor_below(scheduler, rid, p.req, told_final, p.absolute)" in seg
    assert seg.index("pp0_floor_below") < seg.index('"told_fidelity"')
    assert src.index("pp0_floor_below") < src.index("_fb.pp0_note_verdict(scheduler, rid, p.told, told_final")


def test_switch_default_on():
    assert fb.anchor_acks_on()
