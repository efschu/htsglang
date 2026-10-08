"""W27-UNIFORM (NF e124d8f431, P 02.10.): every PP stage admits PP0's told.

The two deaths of that image, same class:

  22:37:12Z pdflip-58-576  PP0 resident=13184 loaded=3456 -> #988 LOADBACK to
                         16640, extend [16640, 16834) = 194 rows; PP1/PP2
                         early read loaded=16640 (record), but their admission
                         match stopped at the device head 13184, extend
                         [13184, 16834) = 3650 -> PPWidthDivergenceRefused.
  23:05:11Z pdflip-6-77    PP0 16896 (174 rows), PP1/PP2 13184 (3886 rows);
                         'FOLLOWER-EARLY-SETTLE told=16896 own=16896 -> equal'.

The followers settled PP0's twin told against their early read's COMPLETION
RECORD; their tree no longer held the span by then (the sibling's read was
released, the sibling prefilled from 0). The fix: a follower admits an
absolute told only when its LIVE tree reaches it, else it re-reads the
missing span (PP0 read the same keys) and waits -- PP0 authoritative, no
refusal.

Three stage views of one rid are simulated over the shipped
``pdflip_store_told.admission``: PP0's tree (device head + loaded span), PP1 and
PP2 (device head only, record says told). Red on e124d8f431: the followers
admit at 13184 while PP0 admits at 16640.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import pdflip_store_told as m  # noqa: E402
from flliper.srt.managers import pdflip_told_fidelity as tf  # noqa: E402
from flliper.srt.pdflip import p_twin_defer as twin  # noqa: E402

RID = "pdflip-58-576"
HEAD = 13184      # the sibling's device rows (twin anchor)
TOLD = 16640      # PP0's absolute twin told (head 13184 + span 3456)
PROMPT = 16834


def _node(state=True):
    mamba = SimpleNamespace(value="slot" if state else None, host_value=None)
    kv = SimpleNamespace(value="kv", host_value=None)
    return SimpleNamespace(component_data=[kv, kv, mamba])


class _StageTree:
    """One PP stage's radix tree, reduced to what admission asks: the device
    head, the host run beyond it and where its recurrent anchor sits."""

    def __init__(self, dev, host=0, anchor=None, record=0):
        self.dev, self.host, self.anchor = dev, host, anchor
        self.record = record
        self.is_eagle = False
        self.prefetch_loaded_tokens_by_reqid = {RID: record}

    # -- the admission's view of its own read --------------------------------
    def check_prefetch_progress(self, rid):
        return True

    def completed_prefetch_tokens(self, rid):
        return self.record

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)

    # -- the read-only match (the probe) -------------------------------------
    def match_prefix(self, params):
        n = len(params.key)
        d = min(self.dev, n)
        h = max(0, min(self.host, n - d))
        return SimpleNamespace(
            device_indices=[0] * d, host_hit_length=h,
            last_device_node=_node(), last_host_node=_node(),
            state_anchor_depth=self.anchor if h > 0 else None,
            key_match_depth=d + h,
        )

    # -- what the stage's admission match + #988 load-back reach --------------
    def admitted_prefix(self, cap):
        """The prefix this stage's admission reaches (the extend start)."""
        reach = self.dev + self.host
        if self.host > 0 and self.anchor is not None:
            reach = min(reach, max(self.dev, self.anchor))
        return min(reach, cap)


def _stage(pp_rank, tree, store_has=TOLD):
    s = SimpleNamespace()
    s.ps = SimpleNamespace(pp_rank=pp_rank)
    s._pdflip_store_held = {}
    s._pdflip_store_told = {}
    s.tree_cache = tree
    s.waiting_queue = []
    s.calls = []

    def _prefetch_kvcache(req, limit_tokens=None):
        # the told-limited store read: from this stage's matched depth to the
        # told, keys PP0 read a pass ago -- they are in the store
        s.calls.append((req.rid, limit_tokens))
        start = tree.dev
        req._prefetch_registered_prefix_len = start
        end = min(int(limit_tokens), store_has)
        tree.host = max(0, end - start)
        tree.anchor = end
        tree.record = tree.host
        tree.prefetch_loaded_tokens_by_reqid[req.rid] = tree.host
        return "issued"

    s._prefetch_kvcache = _prefetch_kvcache
    return s


def _req(head=0):
    ids = list(range(PROMPT))
    return SimpleNamespace(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=ids,
                           extra_key=None, prefix_indices=None, host_hit_length=0,
                           _prefetch_registered_prefix_len=head)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_FOLLOWER_REACH_TOLD", raising=False)
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_SHARE", raising=False)


def _follower_admits(stage, req, early=True):
    stage._pdflip_store_told[RID] = TOLD
    twin.note_follower_twin(stage, RID)        # PdFlipStoreToldTwin absorbed
    if early:
        req._pdflip_early_told = TOLD            # DP-NACHLAUF early read
    credit = m.admission(stage, req, lambda *a: None)
    return credit, stage.tree_cache.admitted_prefix(m.prefix_cap_tokens(stage.tree_cache, TOLD))


def test_the_metal_case_every_stage_admits_pp0s_told():
    """pdflip-58-576: PP0 head 13184 + 3456 loaded; PP1/PP2 early read record
    16640 (head 0) but their tree holds only the device head 13184."""
    pp0 = _StageTree(dev=HEAD, host=TOLD - HEAD, anchor=TOLD, record=TOLD - HEAD)
    pp0_prefix = pp0.admitted_prefix(m.prefix_cap_tokens(pp0, TOLD))
    assert pp0_prefix == TOLD
    widths = {0: PROMPT - pp0_prefix}
    for rank in (1, 2):
        stage = _stage(rank, _StageTree(dev=HEAD, host=0, record=TOLD))
        _credit, prefix = _follower_admits(stage, _req(head=0))
        widths[rank] = PROMPT - prefix
        assert prefix == pp0_prefix, (
            f"PP{rank} admitted at {prefix}, PP0 at {pp0_prefix}: W27 START-SPLIT "
            f"(rows {widths[0]} vs {widths[rank]})")
        assert stage.calls == [(RID, TOLD)], "the told-limited re-read of [13184, 16640)"
    assert set(widths.values()) == {PROMPT - TOLD} == {194}


def test_the_second_death_without_the_early_read():
    """pdflip-6-77 shape through the plain twin path (record = head + span)."""
    told = 16896
    stage = _stage(1, _StageTree(dev=HEAD, host=0, record=told - HEAD), store_has=told)
    stage._pdflip_store_told[RID] = told
    twin.note_follower_twin(stage, RID)
    req = _req(head=HEAD)
    m.admission(stage, req, lambda *a: None)
    assert stage.tree_cache.admitted_prefix(told) == told
    assert stage.calls == [(RID, told)]


def test_host_kv_without_anchor_at_told_is_short_too():
    """The load-back stops at the anchor: host KV to 16640 whose state sits at
    the twin anchor 13184 admits at 13184 -- the probe says so (aligned)."""
    tree = _StageTree(dev=HEAD, host=TOLD - HEAD, anchor=HEAD)
    s = _stage(1, tree)
    assert tf.rank_resumable(s, _req(), TOLD) == HEAD
    assert tf.pp0_admissible(s, _req(), TOLD) == TOLD, "PP0's TF probe is unchanged"


def test_a_follower_whose_tree_reaches_told_reads_nothing_more():
    stage = _stage(1, _StageTree(dev=HEAD, host=TOLD - HEAD, anchor=TOLD, record=TOLD))
    credit, prefix = _follower_admits(stage, _req(head=0))
    assert prefix == TOLD and stage.calls == [] and credit == TOLD


def test_pp0_never_rereads_and_the_switch_restores_the_record(monkeypatch):
    s0 = _stage(0, _StageTree(dev=HEAD, host=0, record=TOLD))
    assert m.follower_reach_told(s0, _req(), TOLD, TOLD) == TOLD and s0.calls == []
    monkeypatch.setenv(m.ENV_REACH_TOLD, "0")
    s1 = _stage(1, _StageTree(dev=HEAD, host=0, record=TOLD))
    assert m.follower_reach_told(s1, _req(), TOLD, TOLD) == TOLD and s1.calls == []


def test_a_reread_that_cannot_reach_told_refuses_nothing(caplog):
    """No new refusal: the store lacks the span -> named ERROR, the record
    stands (the #1233 guard names any split at the forward, as before)."""
    stage = _stage(1, _StageTree(dev=HEAD, host=0, record=TOLD), store_has=HEAD + 64)
    with caplog.at_level("WARNING", logger=m.logger.name):
        credit, prefix = _follower_admits(stage, _req(head=0))
    assert prefix == HEAD + 64 and stage.calls == [(RID, TOLD)]
    assert any("W27-UNIFORM FOLLOWER STILL SHORT" in x for x in caplog.messages)
    assert any("W27-UNIFORM FOLLOWER TREE SHORT" in x and "live=13184" in x for x in caplog.messages)


def test_satisfied_over_read_is_checked_as_well():
    """early 'over' (record past told) -> SATISFIED; the tree must still reach told."""
    stage = _stage(2, _StageTree(dev=HEAD, host=0, record=TOLD + 640))
    credit, prefix = _follower_admits(stage, _req(head=0))
    assert credit == 0 and prefix == TOLD and stage.calls == [(RID, TOLD)]
