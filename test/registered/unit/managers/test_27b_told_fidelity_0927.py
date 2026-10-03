"""TF (27B rc12k27 b1, 27.09. 09:45:58, weg2-10-95): PP0's Admit names the prefix
PP0 itself can admit -- or told=0 for every rank -- and a W27 names which term
split (managers/weg2_told_fidelity.py, pp_admission_congruence.divergence_cause).

The ring is the #1416e harness (a simulated PP3 ring over the shipped module);
PP0's tree gets the read-only match the probe asks."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import test_weg2_store_told_paced_1416e as ring_mod
from sglang.srt.managers import pp_admission_congruence as pac
from sglang.srt.managers import weg2_store_told as m
from sglang.srt.managers import weg2_told_fallback as fb
from sglang.srt.managers import weg2_told_fidelity as tf
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

METAL = ("sender stamp (mb_id=0 seq=613 rows=1024 epoch=-1 fwd_ct=613 "
         "sender_geom=('weg2-10-95', 512, 1536)), receiver fwd_ct=612, "
         "receiver_geom=('weg2-10-95', 16895, 17407)")


# -- the W27 names its cause ---------------------------------------------------


def test_the_metal_w27_is_a_start_split():
    cause = pac.divergence_cause(METAL)
    assert "cause=START-SPLIT" in cause and "512 on the sender and 16895 on the receiver" in cause
    with pytest.raises(pac.PPWidthDivergenceRefused) as e:
        pac.refuse_pp_width_divergence(1024, 512, METAL)
    assert "cause=START-SPLIT" in str(e.value) and "W27 PP WIDTH DIVERGENCE REFUSED" in str(e.value)


def test_width_and_rid_splits_and_no_geometry():
    w = "sender_geom=('r', 100, 1124), receiver_geom=('r', 100, 612)"
    assert "cause=WIDTH-SPLIT" in pac.divergence_cause(w) and "1024 on the sender vs 512" in pac.divergence_cause(w)
    r = "sender_geom=('a', 0, 512), receiver_geom=('b', 0, 512)"
    assert "cause=RID-SPLIT" in pac.divergence_cause(r)
    assert pac.divergence_cause("") == "" and pac.divergence_cause("no provenance") == ""
    assert pac.divergence_cause("sender_geom=('r', 0, 512), receiver_geom=('r', 0, 512)") == ""
    pac.refuse_pp_width_divergence(512, 512, METAL)  # agreeing rows never raise


# -- the probe ------------------------------------------------------------------


def _node(value="v", host=None, mamba=True):
    data = [SimpleNamespace(value="kv", host_value=None)] * 2
    if mamba:
        data = data + [SimpleNamespace(value=value, host_value=host)]
    return SimpleNamespace(component_data=data)


class _Tree:
    def __init__(self, dev, host=0, node=None, eagle=False):
        self.dev, self.host, self.node, self.is_eagle = dev, host, node, eagle
        self.asked = []

    def match_prefix(self, params):
        n = len(params.key)
        self.asked.append(n)
        d = min(self.dev, n)
        h = max(0, min(self.host, n - d))
        return SimpleNamespace(device_indices=[0] * d, host_hit_length=h,
                               last_device_node=self.node, last_host_node=self.node)


def _sched(tree):
    return SimpleNamespace(tree_cache=tree)


def _req(n=18361, head=0):
    return SimpleNamespace(rid="weg2-10-95", origin_input_ids=list(range(n)),
                           full_untruncated_fill_ids=list(range(n)), extra_key=None,
                           _prefetch_registered_prefix_len=head)


def test_node_state_reads_the_928_rule():
    assert tf._node_has_state(_node(value="slot"))
    assert tf._node_has_state(_node(value=None, host="h"))
    assert not tf._node_has_state(_node(value=None, host=None))
    assert tf._node_has_state(_node(mamba=False)), "a dense tree resumes from its KV"
    assert tf._node_has_state(SimpleNamespace())
    assert ComponentType.MAMBA == 2


def test_the_metal_case_is_refused_and_retold_zero(caplog):
    # PP0 at admission: 69 KV tokens matched, the node carries no state (#928 a)
    s = _sched(_Tree(dev=69, node=_node(value=None, host=None)))
    assert tf.pp0_admissible(s, _req(), 16383) == 0
    with caplog.at_level("WARNING", logger=tf.logger.name):
        assert tf.pp0_verdict(s, _req(), 16383, absolute=True) == (0, 0)
    assert any("#TF TOLD-FIDELITY rid=weg2-10-95 told=16383 depth=16383 pp0_admissible=0" in x
               for x in caplog.messages)


def test_a_resumable_told_is_kept_and_asked_with_the_fetch_key():
    s = _sched(_Tree(dev=16383, node=_node(value="slot")))
    assert tf.pp0_verdict(s, _req(), 16383, absolute=True) == (16383, 16383)
    assert s.tree_cache.asked == [16383]
    host = _sched(_Tree(dev=64, host=16319, node=_node(value=None, host="h")))
    assert tf.pp0_verdict(host, _req(), 16383) == (16383, 16383), "host anchor counts"
    big = _sched(_Tree(dev=16383, node=_node(), eagle=True))
    assert tf.pp0_verdict(big, _req(), 16383)[0] == 16383
    assert big.tree_cache.asked == [16383], "bigram: told + 1 raw tokens = told keys"


def test_a_span_relative_told_is_checked_at_head_plus_span():
    """#1400 record (not absolute): the depth PP0 admits is head + told -- the
    probe must not ask for the first `told` keys (mid-path, no anchor there)."""
    s = _sched(_Tree(dev=5000, node=_node(value="slot")))
    assert tf.absolute_depth(_req(head=4000), 1000, absolute=False) == 5000
    assert tf.pp0_verdict(s, _req(head=4000), 1000, absolute=False) == (1000, 5000)
    assert s.tree_cache.asked == [5000]
    short = _sched(_Tree(dev=4500, node=_node(value="slot")))
    assert tf.pp0_verdict(short, _req(head=4000), 1000, absolute=False)[0] == 0


def test_no_probe_switch_off_and_told_zero_keep_the_admit(monkeypatch):
    assert tf.pp0_verdict(SimpleNamespace(), _req(), 16383) == (16383, None)
    boom = SimpleNamespace(tree_cache=SimpleNamespace(match_prefix=lambda p: 1 / 0))
    assert tf.pp0_verdict(boom, _req(), 16383) == (16383, None), "a raising probe gives no verdict"
    s = _sched(_Tree(dev=0))
    assert tf.pp0_verdict(s, _req(), 0) == (0, None)
    monkeypatch.setenv(tf.ENV, "0")
    assert tf.pp0_verdict(s, _req(), 16383) == (16383, None)


# -- the ring: nobody adopts a told PP0 cannot admit ---------------------------


class _TreeWithRelease(ring_mod._TimedTree):
    def release_aborted_request(self, rid):
        self.completed.pop(rid, None)
        self.loaded.pop(rid, None)
        self.done_at.pop(rid, None)
        self._pending_completed.pop(rid, None)


def _ring(monkeypatch, pp0_match):
    ring = ring_mod._Ring(monkeypatch, ring_mod.PROMPTS, ring_mod._read_s(pp0_s=0.2, follower_s=0.2))
    for s in ring.stages:
        t = _TreeWithRelease(ring.clock)
        s.tree_cache = t
    if pp0_match is not None:
        ring.stages[0].tree_cache.match_prefix = pp0_match
    return ring


def _arrive(ring, rid):
    for s in ring.stages:
        r = SimpleNamespace(rid=rid, prefetch_deferred=None, origin_input_ids=list(range(100_001)),
                            extra_key=None)
        s.waiting_queue.append(r)
        m.intake(s, r, lambda gate: None)


def test_ring_pp0_that_lost_its_head_admits_zero_on_every_rank(monkeypatch):
    lost = _Tree(dev=69, node=_node(value=None, host=None))
    ring = _ring(monkeypatch, lost.match_prefix)
    _arrive(ring, "aaaa-told")
    ring.run(60)
    admits = [o for k in sorted(ring.wire) for o in ring.wire[k] if isinstance(o, m.Weg2StoreAdmit)]
    assert len(admits) == 1 and admits[0].told == 0 and getattr(admits[0], fb.WIRE_FALLBACK, 0) == 1
    a = ring.plans("aaaa-told")
    assert a[0] == a[1] == a[2] and len(a[0]) == 1, a  # same PP0 pass, same cap, every rank
    assert a[0][0][2] == 0, "every rank admits at 0: the same prefill on every stage"


def test_ring_resumable_told_is_admitted_unchanged(monkeypatch):
    fine = _Tree(dev=100_000, node=_node(value="slot"))
    ring = _ring(monkeypatch, fine.match_prefix)
    _arrive(ring, "aaaa-told")
    ring.run(60)
    admits = [o for k in sorted(ring.wire) for o in ring.wire[k] if isinstance(o, m.Weg2StoreAdmit)]
    assert len(admits) == 1 and admits[0].told == 100_000
    assert not getattr(admits[0], fb.WIRE_FALLBACK, 0)
    a = ring.plans("aaaa-told")
    assert a[0] == a[1] == a[2] and a[0][0][2] == 100_000


def test_ring_switch_off_is_the_old_admit(monkeypatch):
    monkeypatch.setenv(tf.ENV, "0")
    lost = _Tree(dev=69, node=_node(value=None, host=None))
    ring = _ring(monkeypatch, lost.match_prefix)
    _arrive(ring, "aaaa-told")
    ring.run(60)
    admits = [o for k in sorted(ring.wire) for o in ring.wire[k] if isinstance(o, m.Weg2StoreAdmit)]
    assert len(admits) == 1 and admits[0].told == 100_000 and lost.asked == []


def test_wiring_both_admit_paths_ask_before_the_admit_goes_out():
    import inspect

    src = inspect.getsource(m._pp0_publish_paced)
    fb_site = src.index("_tf.pp0_verdict(scheduler, p.req, told_final, p.absolute)")
    fb_admit = src.index("admit = Weg2StoreAdmit(rid=rid, told=told_final)")
    plain = src.index("_tf_told, _tf_own = _tf.pp0_verdict(scheduler, p.req, p.told, p.absolute)")
    plain_admit = src.index("out.append(Weg2StoreAdmit(rid=rid, told=p.told))")
    assert fb_site < fb_admit and plain < plain_admit
    assert "absolute=bool(absolute))" in src
