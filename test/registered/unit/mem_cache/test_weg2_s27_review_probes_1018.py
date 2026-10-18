# SPDX-License-Identifier: Apache-2.0
"""Review S27 probe (18.10., Opus): READ-ROOM-D host-only round vs. the S22 nodelete filter and the FORMA L3 cap.

P1 (expected green): a first-level leaf a worker cannot delete stays (spill_host_only's nodel filter).
P2 (expected RED on 4ea4ffec3d): the CASCADE -- after the leaf goes, its parent becomes a host-only leaf on TP0 and is
    taken without the nodel check; a worker whose copy of the parent is device-resident cannot delete it
    (K23 SPILL-DIVERGED).
P3 (expected RED on 4ea4ffec3d): the READ-ROOM host-only round hands more pages to the synchronous L3 write than
    SGLANG_WEG2_W3_FORMA_SPILL_MAX_PAGES (the H16Q A3 / K23 A1 bound of the same broadcast site).
"""
import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache import form_a_host_shadow as R12  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402
from sglang.srt.weg2 import read_room as rr  # noqa: E402

from test_w3_spill_host_leaves_int22_1008 import SPAN, _AnchorPool, _death_shape  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    R12._reset_state_for_tests()
    yield
    R12._reset_state_for_tests()


def _spill(cache, pool, need, nodel=()):
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, False)
    with mock.patch.object(R12, "role", return_value="host"), \
            mock.patch.object(rr, "form_a_role", return_value="host"), \
            mock.patch.object(R12, "worker_nodelete_keys", return_value=set(nodel)), \
            mock.patch.object(R12, "worker_evicted_keys", return_value=set()):
        return rr.form_a_spill(cache, pool, need)


def _sent0():
    return {(e[1], int(e[2])) for e in R12._S.ledger if e[0] == R12.STATE and e[3] == 0}


def test_p1_first_level_leaf_a_worker_cannot_delete_stays():
    cache, top, mid, leaf = _death_shape()
    got = _spill(cache, _AnchorPool(), 2, nodel={R12.node_key(leaf)})
    assert got == 0
    assert leaf in cache._collect_all_nodes() and mid in cache._collect_all_nodes()
    assert _sent0() == set()


def test_p2_cascade_parent_a_worker_cannot_delete_stays():
    """Worker: mid device-resident (key in nd). TP0 may drop the leaf, never mid."""
    cache, top, mid, leaf = _death_shape()
    _spill(cache, _AnchorPool(), 2, nodel={R12.node_key(mid)})
    assert leaf not in cache._collect_all_nodes(), "the leaf every worker can delete goes"
    assert mid in cache._collect_all_nodes(), "TP0 cascaded into a parent a worker cannot delete (S22 class)"
    assert R12.node_key(mid) not in _sent0()


def test_p3_host_round_respects_the_forma_secure_cap():
    cache, top, mid, leaf = _death_shape()
    pool = _AnchorPool()
    with envs.SGLANG_WEG2_W3_FORMA_SPILL_MAX_PAGES.override(SPAN):
        _spill(cache, pool, 2 * SPAN)
    assert sum(pool.secured) <= SPAN, "synchronous L3 write above the FORMA cap at the broadcast site: %r" % pool.secured
