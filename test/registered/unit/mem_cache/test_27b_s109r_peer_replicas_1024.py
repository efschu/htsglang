"""S109R-PEER: a peer_replicas refusal is a defect only where an anchor existed to be refused.

THE METAL (27B rc35i4ar3 fe572f3224, boot ..._fe572f3224_1024_003732.D.log, D = TP2 on 2x3080, no Form A):
41 cases ``S109R ANCHOR-DEVICE-READ ... how=peer_replicas attached=0 defect=1`` on TP0 and TP1 each, 00:42:44 to
01:18 (pdflip-1-11 depth=18139, pdflip-1-12 depth=50635, pdflip-1-13 depth=20661, pdflip-2-14 depth=46, ...). The
branch answered ``peer_replicas`` BEFORE it asked whether any tier held an anchor at that node: none of the 41 depths
is a depth at which this boot wrote, saved, secured or lost a state (416 distinct state depths on D and P: HANDBACK
anchor, END-ANCHOR, ANCHOR-STEP, ANCHOR-ONLY-BACKUP, K15 ANCHOR-SAVED, S25, ANCHOR-LOST, FINISH-TRACK/RESUMABLE),
the boot has no ``S109c W3-AUX-DEVICE SPILL`` at all, and all 27 e1 host-path probes of the same boot answered
``held=0 written_here=0 how=store_miss_unsecured``. The depths are KV-walk ends (prompt divergence points, e.g.
pdflip-1-11 ``PROMPT-DIVERGE at=18140``; sequence tails past the 64-grid). The monitor counted every line as
"anchor only on L3 and not read -> avoidable tokens" from the line's own ``defect=1``.

WHAT MUST HOLD.
(1) A TP group whose peers hold mamba replicas still never reads (#639b: no rank-local device slot) -- no store call.
(2) Nothing says an anchor existed (held=0, not secured, not written here): ``how=peer_replicas_unanchored
    defect=0`` with ``held=0 written_here=0`` (RED on fe572f3224/dc1d31ad4c: ``how=peer_replicas defect=1``).
(3) The index holds the key, or the node was secured, or this rank wrote it: ``how=peer_replicas defect=1`` -- a
    real anchor the group form refused (danger direction: hiding a real loss).
(4) The index cannot tell (held=None) and nothing else proves it: still ``how=peer_replicas defect=1``
    (an unanswerable index never hides a loss).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import host_only_spill as HOS  # noqa: E402

_UNIT = pathlib.Path(__file__).resolve().parent.parent
LOGGER = "flliper.srt.mem_cache.unified_radix_cache"


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


S90B = _load("_s90b_fixture_for_s109r_peer", _UNIT / "pdflip" / "test_s90b_anchor_l3_1020.py")
DEPTH = S90B.D_TOLD


def _world(monkeypatch, *, secured=False, held=None, written=False):
    be = S90B._Backend(ok=True)   # a store that WOULD serve: the refusal, not the store, decides
    t, n1, _n2, hp, _dev = S90B._ftree(monkeypatch, be)   # n1: device KV, no Full host, no mamba state
    n1.pdflip_anchor_secured = secured
    hp.anchor_held = lambda last_hash: held
    if written:
        UnifiedRadixCache._s109r_note_written(t, "n19200-last", object())
    monkeypatch.setattr(HOS, "form_a_host", lambda cache: False)   # TP group with peer mamba replicas
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)
    return t, n1, be


def _req():
    return types.SimpleNamespace(rid="pdflip-1-11", key_match_depth=DEPTH, state_anchor_depth=DEPTH - 2 * 64,
                                 full_untruncated_fill_ids=S90B.TOKENS, extra_key=None,
                                 mamba_branching_seqlen=None)


def _line(caplog):
    lines = [m for m in caplog.messages if "S109R ANCHOR-DEVICE-READ" in m]
    assert lines, caplog.messages
    return lines[-1]


def _read(t, req):
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        return t.pdflip_anchor_only_read(req)


def test_metal_peer_refusal_without_any_anchor_is_no_defect(monkeypatch, caplog):
    caplog.set_level(logging.INFO, LOGGER)
    t, _n1, be = _world(monkeypatch, held=False)
    assert _read(t, _req()) == 0
    assert be.calls == [], "a peer-replica group never reads rank-locally (#639b)"
    line = _line(caplog)
    assert " how=peer_replicas_unanchored " in line and "attached=0 defect=0 " in line, line
    assert "held=0 written_here=0" in line, line


@pytest.mark.parametrize("kw,witness", [
    (dict(held=True), "held=1 written_here=0"),
    (dict(held=False, secured=True), "held=0 written_here=0"),
    (dict(held=False, written=True), "held=0 written_here=1"),
    (dict(held=None), "held=- written_here=0"),
])
def test_peer_refusal_of_an_anchor_that_existed_stays_the_defect(monkeypatch, caplog, kw, witness):
    caplog.set_level(logging.INFO, LOGGER)
    t, _n1, be = _world(monkeypatch, **kw)
    assert _read(t, _req()) == 0
    assert be.calls == []
    line = _line(caplog)
    assert " how=peer_replicas " in line and "attached=0 defect=1 " in line, line
    assert witness in line, line


def test_defect_set_names_the_refusal_not_the_unanchored_class():
    assert "peer_replicas" in UnifiedRadixCache.S109R_DEFECT_HOWS
    assert "peer_replicas_unanchored" not in UnifiedRadixCache.S109R_DEFECT_HOWS


@pytest.mark.parametrize("secured,written,held,want", [
    (False, False, False, "peer_replicas_unanchored"),
    (False, False, None, "peer_replicas"),
    (False, False, True, "peer_replicas"),
    (True, False, False, "peer_replicas"),
    (False, True, False, "peer_replicas"),
])
def test_peer_class_table(secured, written, held, want):
    node = types.SimpleNamespace(pdflip_anchor_secured=secured)
    assert UnifiedRadixCache._s109r_peer_class(node, held, written) == want
