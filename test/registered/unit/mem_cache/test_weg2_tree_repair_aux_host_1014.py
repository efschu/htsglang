"""h17-Tod (NF 38c6d40239, D TP0 15:51:02, D.log Z.57569).

``Sanity check FAILED (3 violations across 18 nodes): node 248/243/251 mamba
host present but Full.host_value=None`` -- AssertionError aus on_idle ->
_check_tree_cache_bounded -> sanity_check; der Server war tot.

Die Knoten trugen Device-KV (Full.value), KEINE Full-Host-Kopie und einen
Mamba-Host-Zustand (MAMBA-HOST-RESUME depth=58112/94080 unmittelbar davor).
Das Netz im Verdraengungs-Trichter (y5h) griff nur, wenn die Full-Host-Kopie
einen Knoten MIT Device-KV verliess. Verliess sie einen host-only-Knoten
(Full.value None, innerer Knoten lebt fuer seine Kinder weiter), blieb der
Mamba-Host-Zustand haengen; ein spaeterer Insert belebte Full.value darueber
-> genau die Verletzung.

Gebaut (Schalter SGLANG_WEG2_TREE_REPAIR_AUX_HOST, Vorgabe an):
* Trichter: verlaesst die Full-Host-Kopie einen Knoten (HOST oder ALL), gehen
  seine Aux-Host-Zustaende zuerst -- auch ohne Device-KV.
* PREFETCH-Commit: der Read-Anker (Arena-Slot oder ANCHOR-BYPASS-Staging-
  Zeile) landet nie an einem Knoten ohne Full-Host-Kopie; er geht an den Pool.
* sanity_check: genau diese Verletzung (Mamba-Host ohne Full-Host, Knoten mit
  Device-KV) wird repariert (Zustand freigegeben, ``WEG2 TREE-REPAIR``), statt
  den Server zu toeten. Jede andere Verletzung bleibt fatal.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
    EvictLayer,
)

_spec = importlib.util.spec_from_file_location(
    "_t_attach_h17", os.path.join(os.path.dirname(__file__), "test_weg2_prefetch_anchor_attach_0930.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
MC = ComponentType.MAMBA
FULL = ComponentType.FULL


def _attached():
    """Knoten bei 96 mit Device-KV, Full-Host-Kopie und Mamba-Host-Anker."""
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    A._commit(cache, res)
    node = A._node_at(cache, 96)
    assert node.component_data[FULL].host_value is not None
    assert node.component_data[MC].host_value is not None
    cache.sanity_check()
    return cache, node


def _break_law(cache, node):
    """Die h17-Lage herstellen: Full-Host-Kopie weg, ohne Trichter-Netz."""
    base = cache.components[FULL]
    base.evict_component(node, target=EvictLayer.HOST)
    cache.evictable_host_leaves.discard(node)
    assert node.component_data[FULL].host_value is None
    assert node.component_data[FULL].value is not None
    assert node.component_data[MC].host_value is not None


def test_h17_violation_is_repaired_not_fatal(caplog):
    cache, node = _attached()
    _break_law(cache, node)
    with caplog.at_level(logging.ERROR):
        cache.sanity_check()   # h17: AssertionError, Server tot
    assert node.component_data[MC].host_value is None, "Zeile zurueck an den Pool"
    assert not cache.host_lru_lists[MC].in_list(node)
    assert "WEG2 TREE-REPAIR aux_host_without_full repaired=1" in caplog.text
    assert node.component_data[FULL].value is not None, "Device-KV bleibt"
    cache.sanity_check()   # der reparierte Baum ist gesetzestreu


def test_switch_off_keeps_it_fatal():
    cache, node = _attached()
    _break_law(cache, node)
    with envs.SGLANG_WEG2_TREE_REPAIR_AUX_HOST.override(False):
        with pytest.raises(AssertionError, match="mamba host present but Full.host_value=None"):
            cache.sanity_check()


def test_other_violations_stay_fatal_even_with_a_repairable_node():
    cache, node = _attached()
    _break_law(cache, node)
    cache.component_evictable_size_[FULL] += 7   # Groessenbuchhaltung kaputt
    with pytest.raises(AssertionError):
        cache.sanity_check()


def test_a_locked_state_is_never_freed_under_its_reader(caplog):
    cache, node = _attached()
    _break_law(cache, node)
    node.component_data[MC].host_lock_ref += 1
    try:
        with caplog.at_level(logging.ERROR):
            assert cache._weg2_tree_repair_aux_host([node]) == 0
        assert node.component_data[MC].host_value is not None
        assert "deferred_locked=1" in caplog.text
    finally:
        node.component_data[MC].host_lock_ref -= 1


def test_funnel_takes_aux_host_when_full_host_leaves_a_host_only_node():
    """Die Wurzel: Full-Host verlaesst einen Knoten OHNE Device-KV. Vorher blieb
    der Mamba-Host-Zustand; jetzt geht er zuerst."""
    cache, node = _attached()
    base = cache.components[FULL]
    # Device-KV weg (host-only-Knoten), wie bei einer Device-Verdraengung
    node.component_data[FULL].value = None
    node.component_data[MC].value = None
    with envs.SGLANG_WEG2_TREE_REPAIR_AUX_HOST.override(False):
        cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
    assert node.component_data[MC].host_value is not None, "alter Pfad: Zustand bleibt haengen"

    cache, node = _attached()
    base = cache.components[FULL]
    node.component_data[FULL].value = None
    node.component_data[MC].value = None
    cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
    assert node.component_data[FULL].host_value is None
    assert node.component_data[MC].host_value is None, "neu: Aux-Host geht zuerst"
    assert not cache.host_lru_lists[MC].in_list(node)


def test_prefetch_anchor_never_lands_on_a_node_without_full_host(caplog):
    """Bypass-/Arena-Anker ueber inserted_host_node, dessen Full-Host fehlt:
    freigegeben statt angehaengt."""
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    node = A._node_at(cache, 96)
    assert node.component_data[FULL].host_value is None
    res.inserted_host_node = node
    with caplog.at_level(logging.ERROR):
        A._commit(cache, res)
    assert node.component_data[MC].host_value is None
    assert len(cache.cache_controller.released) == 1
    assert "WEG2 AUX-HOST-LAW refused site=prefetch" in caplog.text
    cache.sanity_check()
