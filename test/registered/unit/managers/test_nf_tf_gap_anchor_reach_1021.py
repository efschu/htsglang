"""NF-TF-GAP 1021 (rc27 893d09fa3f, P log ..._1021_020616, pdflip-1-22): PP0's
told-fidelity probe asks the SAME reach the followers ack with.

THE CASE (P log lines):
  21954 PP0 MATCH-CENSUS-DEEP rid=pdflip-1-22 reached=18880 accepted=18688
  22067 PP1 MATCH-CENSUS-DEEP rid=pdflip-1-22 reached=18880 accepted=18880
  22180 PP0 #1423 INSERT-PLACED ... reg_depth=18688 ... matched=192 inserted=120128
  22402 PP0 #TF TOLD-FIDELITY rid=pdflip-1-22 told=139008 depth=139008 pp0_admissible=138816
  22404 PP0 PF TOLD-FLOOR TF-RETRY ... refused=139008 -> 18688
  22405 PP0 PF TOLD-FLOOR ... acks={1: 139008, 2: 139008}
  front 1603 P-REUSE-GAP rid=pdflip-1-22 ... cached=18688 forgone=120320

PP0's tree: device KV to 18880, recurrent state only at 18688 (the 192 tokens
[18688,18880) carry no anchor on PP0), host run [18880,139008) with its anchor at
139008. ``device + host`` = 18688 + 120128 = 138816 -- the gap's device KV is in
neither count -- so TF refused a told PP0's own admission reaches (the #1040
raise: key match reaches the anchor, extent = anchor - device = 120320) and both
followers acked. 120320 tokens were recomputed for a prefix the store held.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from flliper.srt.managers import pdflip_told_fidelity as tf

TOLD = 139008
PROMPT = 139022
DEV_STATE = 18688        # PP0's deepest device node WITH a recurrent state
HOST_RUN = 120128        # the host run [18880, 139008)


def _node(host_state: bool):
    kv = SimpleNamespace(value=None, host_value="host-kv")
    mamba = SimpleNamespace(value=None, host_value="host-state" if host_state else None)
    return SimpleNamespace(component_data=[kv, kv, mamba])


class _GapTree:
    """The PP0 tree of the case: device indices end at the last state node, the
    host count covers only the host run, the anchor and the key match reach told."""

    is_eagle = True

    def __init__(self, dev: int, host: int, anchor: int, key_depth: int):
        self.dev, self.host, self.anchor, self.key_depth = dev, host, anchor, key_depth
        self.node = _node(host_state=True)

    def match_prefix(self, params):
        cap = len(params.key)
        return SimpleNamespace(
            device_indices=torch.arange(min(self.dev, cap), dtype=torch.int64),
            host_hit_length=self.host,
            state_anchor_depth=self.anchor,
            key_match_depth=min(self.key_depth, cap),
            last_host_node=self.node,
            last_device_node=self.node,
        )


def _req():
    ids = list(range(PROMPT))
    return SimpleNamespace(rid="pdflip-1-22", origin_input_ids=ids,
                           full_untruncated_fill_ids=ids, extra_key=None,
                           _prefetch_registered_prefix_len=0)


def test_the_metal_gap_keeps_the_told():
    """pdflip-1-22: PP0's admission reaches 139008 -- the Admit keeps it (was 0 /
    the 18688 floor, 120320 tokens recomputed)."""
    s = SimpleNamespace(tree_cache=_GapTree(DEV_STATE, HOST_RUN, TOLD, TOLD))
    assert tf.pp0_admissible(s, _req(), TOLD) == TOLD
    assert tf.pp0_verdict(s, _req(), TOLD, absolute=True) == (TOLD, TOLD)


def test_pp0_asks_what_the_followers_ack():
    """One expression on every rank: PP0's verdict and the followers' ack
    (``rank_resumable``) agree on the same tree."""
    for anchor in (TOLD, TOLD - 192, DEV_STATE + 64):
        s = SimpleNamespace(tree_cache=_GapTree(DEV_STATE, HOST_RUN, anchor, TOLD))
        assert tf.pp0_admissible(s, _req(), TOLD) == tf.rank_resumable(s, _req(), TOLD)


def test_danger_anchor_below_the_host_end_still_refuses():
    """DANGER DIRECTION (ranks uneins): the host run ends at told but its deepest
    anchor lies BELOW it -- PP0's admission stops at the anchor, so the Admit must
    not keep told (a START-SPLIT against followers adopting told)."""
    anchor = TOLD - 192
    s = SimpleNamespace(tree_cache=_GapTree(DEV_STATE + 192, HOST_RUN, anchor, TOLD))
    own = tf.pp0_admissible(s, _req(), TOLD)
    assert own == anchor, own
    assert tf.pp0_verdict(s, _req(), TOLD, absolute=True) == (0, anchor)


def test_danger_key_short_of_the_anchor_is_not_raised():
    """The raise needs the key match to REACH the anchor: a key match that stops
    short gives no more than device + host."""
    s = SimpleNamespace(tree_cache=_GapTree(DEV_STATE, HOST_RUN, TOLD, TOLD - 64))
    assert tf.pp0_admissible(s, _req(), TOLD) == DEV_STATE + HOST_RUN
    assert tf.pp0_verdict(s, _req(), TOLD, absolute=True)[0] == 0


def test_no_host_run_is_unchanged():
    """A device-only match keeps the #928 node rule byte for byte (aligned only
    acts on a host run)."""
    dev_only = _GapTree(TOLD, 0, TOLD, TOLD)
    dev_only.node = SimpleNamespace(component_data=[
        SimpleNamespace(value="kv", host_value=None),
        SimpleNamespace(value="kv", host_value=None),
        SimpleNamespace(value="slot", host_value=None)])
    s = SimpleNamespace(tree_cache=dev_only)
    assert tf.pp0_verdict(s, _req(), TOLD, absolute=True) == (TOLD, TOLD)
    dead = _GapTree(TOLD, 0, TOLD, TOLD)
    dead.node = SimpleNamespace(component_data=[
        SimpleNamespace(value="kv", host_value=None),
        SimpleNamespace(value="kv", host_value=None),
        SimpleNamespace(value=None, host_value=None)])
    assert tf.pp0_verdict(SimpleNamespace(tree_cache=dead), _req(), TOLD, absolute=True) == (0, 0)
