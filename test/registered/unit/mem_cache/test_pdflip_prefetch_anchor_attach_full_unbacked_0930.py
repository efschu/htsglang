"""PREFETCH-ANCHOR-ATTACH ohne Full-Host (NF y5h 30.09., rc12z30y5h, a54a22e54d).

Die Metall-Kette (D-Log ..._a54a22e54d_0930_201824, TP1/TP2, Form-A-Worker):
* 20:29:37 ``PDFLIP PREFETCH-ANCHOR-ATTACH n=6 node=540 depth=24320`` -- der
  Store-Read von pdflip-32-82 (``matched=24320 loaded=0``) fand seinen ganzen
  Span schon im Baum (Device-KV, KEINE Full-Host-Kopie), und d5d75f328a haengte
  den gelesenen Mamba-Anker als host_value an den Endknoten.
* ATTACH feuert genau dann, wenn ``_insert_helper_host`` keinen
  ``inserted_host_node`` nennt -- also genau dann, wenn der Endknoten KEINE
  Full-Host-Kopie hat. Jeder ATTACH schrieb also "Aux-Host ohne Full-Host".
  Bisher heilte das die naechste Sicherung (PUBLISH-SWEEP -> write_backup setzt
  Full.host_value); am Ende des Boots verweigerte der Sweep den Knoten
  (``unbacked=1 issued=0 refused=1``), und die Idle-Pruefung toetete TP1/TP2:
  ``Sanity check FAILED (1 violations across 45 nodes): node 540 mamba host
  present but Full.host_value=None``.

Jetzt: ATTACH haengt den Anker nur an einen Knoten mit Full-Host-Kopie. Hat
der Endknoten keine, uebernimmt er die KV-Zeilen, die derselbe Read fuer genau
seinen Span geliefert hat (sie wurden bisher als "schon im Baum" freigegeben);
der Aufrufer gibt diese Zeilen dann nicht frei. Geht das nicht (write_through
und der Elternknoten ohne Host-Kopie, keine Zeilen), wird der Anker rejected
und an den Pool zurueckgegeben, mit ``PDFLIP PREFETCH-ANCHOR-ATTACH dropped``.
Dazu das Netz im Verdraengungs-Trichter: verlaesst die Full-Host-Kopie einen
Knoten, gehen seine Aux-Host-Zustaende zuerst.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
    EvictLayer,
)

_spec = importlib.util.spec_from_file_location(
    "_t_attach_aa", os.path.join(os.path.dirname(__file__), "test_pdflip_prefetch_anchor_attach_0930.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
MC = ComponentType.MAMBA
FULL = ComponentType.FULL


class _CtlWT(A._Ctl):
    write_policy = "write_through"


def _two_node_path(ctl_cls=A._Ctl):
    """Device path of two nodes (64 + 64 tokens), no state, no host copy."""
    cache, pool, toks = A._device_path(64, anchor=None)
    from flliper.srt.mem_cache.base_prefix_cache import InsertParams

    toks2 = toks + list(range(7000, 7064))
    kalloc = cache.token_to_kv_pool_allocator
    cache.insert(InsertParams(key=RadixKey(array("q", toks2)), value=torch.cat(
        [cache.root_node.children[next(iter(cache.root_node.children))].component_data[FULL].value,
         kalloc.alloc(64)])))
    cache.cache_controller = ctl_cls()
    return cache, pool, toks2


def test_y5h_attach_on_a_node_without_full_host_keeps_the_tree_law():
    """Die Metall-Kette: Read-Span liegt schon als Device-Pfad im Baum (ohne
    Full-Host), ATTACH, dann die Idle-Pruefung. Auf a54a22e54d:
    "node N mamba host present but Full.host_value=None"."""
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    assert res.inserted_host_node is None and res.matched_end_node is not None
    A._commit(cache, res)
    cache.sanity_check()   # die Idle-Pruefung, die TP1/TP2 getoetet hat
    node = A._node_at(cache, 96)
    assert node.component_data[MC].host_value is not None, "the anchor stays (y5a)"
    assert node.component_data[FULL].host_value is not None, "Aux-Host verlangt Full-Host"


def test_the_node_takes_exactly_its_own_rows_of_the_read():
    """Die Full-Host-Kopie des Endknotens sind die KV-Zeilen, die der Read fuer
    SEINEN Span geliefert hat (Read-Zeilen 10000.., Span [0, 96)), und der
    Aufrufer gibt genau diese Zeilen nicht frei."""
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    A._commit(cache, res)
    node = A._node_at(cache, 96)
    assert node.component_data[FULL].host_value.tolist() == list(range(10000, 10096))
    assert res.anchor_adopted_tokens == 96
    assert cache._prefetch_head_free_to(res) == 0
    assert node.hash_value == ["h%d" % i for i in range(96)]
    assert node.l3_present, "the rows came from the Store"
    assert node in cache.evictable_host_leaves or not cache._is_host_leaf(node)


def test_deeper_end_node_adopts_only_its_own_span_and_the_head_is_released():
    """Read endet im ZWEITEN Knoten (64 + 32): nur der Endknoten [64, 96)
    uebernimmt Zeilen; der Kopf [0, 64) gehoert dem Baum nicht und wird wie
    bisher freigegeben. write_back: ein Kind mit Host unter einem Elternknoten
    ohne Host ist erlaubt (sanity_check prueft das nur ohne write_back)."""
    cache, pool, toks = _two_node_path()
    res = A._read_span(cache, toks, 96)
    A._commit(cache, res)
    node = A._node_at(cache, 96)
    assert node.component_data[FULL].host_value.tolist() == list(range(10064, 10096))
    assert res.anchor_adopted_tokens == 32
    assert cache._prefetch_head_free_to(res) == 64
    assert node.parent.component_data[FULL].host_value is None
    cache.sanity_check()


def test_write_through_with_unbacked_parent_drops_and_releases_the_anchor(caplog):
    """write_through: der Elternknoten ohne Host-Kopie verbietet die Uebernahme
    (#841-Gesetz) -> Anker rejected, Slot an den Pool (append_host_mem_release),
    benannte Zeile, der Baum bleibt gesetzestreu."""
    cache, pool, toks = _two_node_path(_CtlWT)
    res = A._read_span(cache, toks, 96)
    with caplog.at_level(logging.INFO):
        A._commit(cache, res)
    node = A._node_at(cache, 96)
    assert node.component_data[MC].host_value is None
    assert node.component_data[FULL].host_value is None
    assert len(cache.cache_controller.released) == 1, "the Mamba-slot goes to the Pool"
    assert int(getattr(res, "anchor_adopted_tokens", 0) or 0) == 0
    assert cache._prefetch_head_free_to(res) == 96
    assert "PDFLIP PREFETCH-ANCHOR-ATTACH dropped" in caplog.text
    assert "why=parent_unbacked" in caplog.text
    cache.sanity_check()


def test_write_through_under_the_root_adopts():
    cache, pool, toks = A._device_path(128, anchor=None)
    cache.cache_controller = _CtlWT()
    res = A._read_span(cache, toks, 96)
    A._commit(cache, res)
    node = A._node_at(cache, 96)
    assert node.component_data[MC].host_value is not None
    assert node.component_data[FULL].host_value is not None
    cache.sanity_check()


def test_no_read_rows_for_the_node_drops(caplog):
    """Ohne Zeilen des Reads fuer den Endknoten gibt es keine Full-Host-Kopie:
    rejected, nicht angehaengt."""
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    res.matched_end_host_kv = None
    with caplog.at_level(logging.INFO):
        A._commit(cache, res)
    node = A._node_at(cache, 96)
    assert node.component_data[MC].host_value is None
    assert len(cache.cache_controller.released) == 1
    assert "why=no_read_rows" in caplog.text
    cache.sanity_check()


def test_the_next_match_still_resumes_at_the_attached_anchor():
    from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams

    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    A._commit(cache, res)
    m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", toks[:120]))))
    assert m.state_anchor_depth == 96
    assert m.best_match_node is res.matched_end_node


def test_full_host_leaving_a_node_takes_its_aux_host_first(caplog):
    """Das Netz im Trichter: jeder Weg, der nur die Full-Host-Zeilen eines
    Knotens freigibt (W3-Spill, PUBLISH-CHAIN-Release, ...), nimmt die
    Aux-Host-Zustaende vorher mit -- nie ein Aux-Host ohne Full-Host."""
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    A._commit(cache, res)
    node = A._node_at(cache, 96)
    assert node.component_data[MC].host_value is not None
    base = cache.components[FULL]
    type(cache)._aux_host_with_full_n = 0   # the line is rate-capped per process
    with caplog.at_level(logging.INFO):
        cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
    cache.evictable_host_leaves.discard(node)
    assert node.component_data[FULL].host_value is None
    assert node.component_data[MC].host_value is None
    assert not cache.host_lru_lists[MC].in_list(node)
    assert "PDFLIP AUX-HOST-WITH-FULL" in caplog.text
    cache.sanity_check()


def test_a_node_that_already_has_a_full_host_copy_keeps_the_old_path():
    """Endknoten MIT Full-Host-Kopie: ``inserted_host_node`` ist er selbst,
    ATTACH greift nicht, der Anker haengt wie immer, nichts wird uebernommen."""
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    A._commit(cache, res)
    # zweiter Read desselben Spans: jetzt traegt der Knoten eine Host-Kopie
    cache.cache_controller = A._Ctl()
    node = A._node_at(cache, 96)
    comp = cache.components[MC]
    comp.evict_component(node, target=EvictLayer.HOST)
    cache.host_lru_lists[MC].remove_node(node) if cache.host_lru_lists[MC].in_list(node) else None
    res2 = A._read_span(cache, toks, 96)
    assert res2.inserted_host_node is node
    A._commit(cache, res2)
    assert int(getattr(res2, "anchor_adopted_tokens", 0) or 0) == 0
    assert node.component_data[MC].host_value is not None
    cache.sanity_check()
