"""R5 (rc12z2 e8a2cd2dc5, P 03:26:41, rid weg2-44-76): the told-fidelity probe
must ask PP0's tree with the tree's own key form.

THE DEATH. PP0 read 47168 tokens (8x #1433 L3->L2 fill), clamped the told to
the store's anchor at 37952 (#1416 ANCHOR-CLAMP) and put it on the wire. Its
own match then reached 37952 with no recurrent state there
(``#904 ... refused=37952 why=MambaComponent:absent``, ``[#928 anchor]
REFUSING resume``) -> PP0 admitted at 0, PP2 likewise; PP1 resumed at 37952
(``#988 LOADBACK ... mamba_restored``) -> ``#1233 W27 ... cause=START-SPLIT``
(sender 32768.. vs receiver 70720..). TF (``weg2_told_fidelity``, default on)
exists for exactly this -- PP0 asks its own tree before the Admit and retells
0 for every rank -- but every probe on that boot died in ``RadixKey.match``:
``#TF told-fidelity probe skipped ... AssertionError((<class 'array.array'>,
<class 'list'>))`` (10 of 10): ``_probe_key`` built the key from a list, the
tree's keys are ``array('q')``. No verdict -> the Admit stayed 37952.

The fake tree here compares with the REAL ``RadixKey.match`` against an
array-keyed node (the existing TF tests measured ``len(key)`` only and never
reached the comparator). Metal numbers: told 37952, bigram (P, MTP).
"""
from __future__ import annotations

from array import array
from types import SimpleNamespace

from sglang.srt.managers import weg2_store_told as m
from sglang.srt.managers import weg2_told_fidelity as tf
from sglang.srt.mem_cache.radix_cache import RadixKey

TOLD = 37952
PROMPT = 79688


def _node(state: bool):
    kv = SimpleNamespace(value="kv", host_value=None)
    mamba = SimpleNamespace(value="slot" if state else None, host_value=None)
    return SimpleNamespace(component_data=[kv, kv, mamba])


class _ArrayTree:
    """One array-keyed node of ``depth`` keys, matched with RadixKey.match."""

    is_eagle = True

    def __init__(self, depth: int, state: bool):
        self.key = RadixKey(array("q", range(depth + 1)), is_bigram=True)
        self.node = _node(state)

    def match_prefix(self, params):
        n = self.key.match(params.key)  # raises on a list-vs-array key
        return SimpleNamespace(device_indices=[0] * n, host_hit_length=0,
                               last_device_node=self.node, last_host_node=self.node)


def _req():
    return SimpleNamespace(rid="weg2-44-76", origin_input_ids=list(range(PROMPT)),
                           full_untruncated_fill_ids=list(range(PROMPT)), extra_key=None,
                           _prefetch_registered_prefix_len=0)


def test_the_metal_case_retells_zero_for_every_rank(caplog):
    """PP0's anchor at told is gone: the Admit must carry 0 (every stage
    re-prefills alike) -- not 37952 while PP0 itself admits 0."""
    s = SimpleNamespace(tree_cache=_ArrayTree(TOLD, state=False))
    with caplog.at_level("WARNING", logger=tf.logger.name):
        verdict = tf.pp0_verdict(s, _req(), TOLD, absolute=True)
    assert "told-fidelity probe skipped" not in caplog.text, caplog.text
    assert verdict == (0, 0), f"{verdict}: the Admit would stay {TOLD} while PP0 admits 0 (START-SPLIT)"


def test_a_resumable_told_is_kept():
    s = SimpleNamespace(tree_cache=_ArrayTree(TOLD, state=True))
    assert tf.pp0_verdict(s, _req(), TOLD, absolute=True) == (TOLD, TOLD)


def test_the_probe_key_is_the_tree_key_form():
    s = SimpleNamespace(tree_cache=SimpleNamespace(is_eagle=True))
    key, bigram = m._probe_key(s, _req(), list(range(PROMPT)), TOLD)
    assert bigram and isinstance(key.token_ids, array) and key.token_ids.typecode == "q"
    assert len(key.token_ids) == TOLD + 1 and len(key) == TOLD
    # an array origin (the scheduler's own form) goes in unchanged in value
    key2, _ = m._probe_key(s, _req(), array("i", range(PROMPT)), TOLD)
    assert list(key2.token_ids) == list(key.token_ids)


def test_the_anchor_clamp_hashes_the_same_pages():
    """The #1416d clamp shares the probe key: an array key hashes exactly as
    the list key did (the store's page keys do not move)."""
    from sglang.srt.mem_cache.utils import get_hash_str

    ids = list(range(4 * 64 + 1))
    lst = get_hash_str(RadixKey(ids, is_bigram=True), None, page_size=64)
    arr = get_hash_str(RadixKey(array("q", ids), is_bigram=True), None, page_size=64)
    assert lst == arr and len(arr) == 4
