"""h17-Tod (NF 38c6d40239, D TP0 15:51:02, D.log Z.57569).

``Sanity check FAILED (3 violations across 18 nodes): node 248/243/251 mamba
host present but Full.host_value=None`` -- AssertionError aus on_idle.

Die Knoten trugen Device-KV, KEINE Full-Host-Kopie und einen Mamba-Host-
Zustand. Das Netz im Verdraengungs-Trichter (y5h) nahm die Aux-Host-Zustaende
nur mit, wenn die Full-Host-Kopie einen Knoten MIT Device-KV verliess.
Verliess sie einen host-only-Knoten (innerer Knoten, lebt fuer seine Kinder
weiter), blieb der Mamba-Host-Zustand haengen; ein spaeterer Insert belebte
Full.value darueber -> genau die Verletzung.

Konstruktion (FLLIPER_PDFLIP_AUX_HOST_LAW, Vorgabe an), keine Reparatur: der
Trichter nimmt die Aux-Host-Zustaende bei JEDEM Full-Host-Abgang zuerst mit,
und der PREFETCH-Commit haengt einen Read-Anker (Arena-Slot/ANCHOR-BYPASS) nie
an einen Knoten ohne Full-Host-Kopie. Der Sanity-Check bleibt toedlich
(raenge-nie-uneins-crash-stop). Mutant: der Trichter ohne die neue Klausel
erzeugt die h17-Lage wieder, und der Sanity-Check stirbt daran.
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache import unified_radix_cache as URC  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
    EvictLayer,
)

_spec = importlib.util.spec_from_file_location(
    "_t_attach_h17", os.path.join(os.path.dirname(__file__), "test_pdflip_prefetch_anchor_attach_0930.py"))
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


def _h17_sequence(cache, node):
    """Full-Host verlaesst den host-only-Knoten (Trichter), dann belebt ein
    Insert die Device-KV wieder (hier: der alte Device-Wert kommt zurueck)."""
    base = cache.components[FULL]
    dev = node.component_data[FULL].value
    node.component_data[FULL].value = None          # host-only: Device verdraengt
    cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
    cache.evictable_host_leaves.discard(node)
    node.component_data[FULL].value = dev           # spaeterer Insert belebt Full.value


def test_h17_full_host_leaving_a_host_only_node_takes_the_mamba_host_first():
    cache, node = _attached()
    _h17_sequence(cache, node)
    assert node.component_data[FULL].host_value is None
    assert node.component_data[MC].host_value is None, "the aux host goes first"
    assert not cache.host_lru_lists[MC].in_list(node)
    cache.sanity_check()


def test_switch_off_reproduces_the_h17_death():
    cache, node = _attached()
    with envs.FLLIPER_PDFLIP_AUX_HOST_LAW.override(False):
        _h17_sequence(cache, node)
    assert node.component_data[MC].host_value is not None
    with pytest.raises(AssertionError, match="mamba host present but Full.host_value=None"):
        cache.sanity_check()


def test_the_sanity_check_stays_fatal_on_this_violation():
    """raenge-nie-uneins-crash-stop: erkannt heisst Stopp, nie Reparatur."""
    cache, node = _attached()
    cache.components[FULL].evict_component(node, target=EvictLayer.HOST)   # am Trichter vorbei
    cache.evictable_host_leaves.discard(node)
    with pytest.raises(AssertionError, match="mamba host present but Full.host_value=None"):
        cache.sanity_check()
    assert node.component_data[MC].host_value is not None, "nothing repaired"


def test_mutant_funnel_without_the_clause_is_killed():
    """Quelltext-Mutant: der Trichter mit der alten Bedingung (nur mit
    Device-KV). Die h17-Folge muss daran sterben -- der Test oben toetet ihn."""
    src = textwrap.dedent(inspect.getsource(URC.UnifiedRadixCache._evict_component_and_detach_lru))
    import re
    mut, k = re.subn(r"\n\s*or \(\n\s*EvictLayer\.HOST in target\n\s*and envs\.FLLIPER_PDFLIP_AUX_HOST_LAW\.get\(\)\n\s*\)", "", src)
    assert k == 1, "the clause must exist exactly once"
    ns = dict(vars(URC))
    exec(compile(mut, URC.__file__, "exec"), ns)
    cache, node = _attached()
    cache._evict_component_and_detach_lru = types.MethodType(ns["_evict_component_and_detach_lru"], cache)
    _h17_sequence(cache, node)
    assert node.component_data[MC].host_value is not None, "the mutant leaves the mamba host hanging"
    with pytest.raises(AssertionError, match="mamba host present but Full.host_value=None"):
        cache.sanity_check()


def test_full_host_leaving_a_device_node_is_unchanged():
    cache, node = _attached()
    base = cache.components[FULL]
    cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
    cache.evictable_host_leaves.discard(node)
    assert node.component_data[MC].host_value is None
    cache.sanity_check()


def test_prefetch_anchor_never_lands_on_a_node_without_full_host(caplog):
    """Bypass-/Arena-Anker ueber inserted_host_node, dessen Full-Host fehlt:
    an den Pool statt angehaengt."""
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    node = A._node_at(cache, 96)
    assert node.component_data[FULL].host_value is None
    res.inserted_host_node = node
    with caplog.at_level(logging.ERROR):
        A._commit(cache, res)
    assert node.component_data[MC].host_value is None
    assert len(cache.cache_controller.released) == 1
    assert "PDFLIP AUX-HOST-LAW refused site=prefetch" in caplog.text
    cache.sanity_check()


def test_prefetch_guard_off_attaches_and_the_check_dies():
    cache, pool, toks = A._device_path(128, anchor=None)
    res = A._read_span(cache, toks, 96)
    node = A._node_at(cache, 96)
    res.inserted_host_node = node
    with envs.FLLIPER_PDFLIP_AUX_HOST_LAW.override(False):
        A._commit(cache, res)
    assert node.component_data[MC].host_value is not None
    with pytest.raises(AssertionError):
        cache.sanity_check()
