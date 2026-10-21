"""S109R-MISS: a store miss of the S109R read is a defect only where a SECURED anchor was not served.

THE METAL (NF rc30 1d822bd150, boot ..._1d822bd150_1021_063252.D.log, D TP0):
``S109R ANCHOR-DEVICE-READ n=3 rid=pdflip-2-15 depth=62656 how=store_miss attached=0`` (06:37:29), n=4 pdflip-2-21
depth=77376 (06:37:58), n=5 pdflip-2-26 depth=62912 (06:38:17) -- in a boot with NO ``S109c W3-AUX-DEVICE SPILL``.
Each one: ``MATCH-CENSUS-DEEP ... reached=<depth> accepted=<depth-128/320/128> branching=<depth>``: the KV walk ended
at a BRANCH POINT, the parent half of a split, which gets no state (``MambaComponent.redistribute_on_node_split``) and
under whose page hash no anchor was ever written. The line said "S109c released it after its L3 copy" and the monitor
counted a defect that cannot exist. n=1 (62528) and n=2 (18752) of the same boot attached: store and keys work.

WHAT MUST HOLD.
(1) A miss at the match's branching depth on a node never secured: ``how=branch_point_never_anchored defect=0``
    (RED on 1d822bd150: ``how=store_miss``).
(2) A miss on a node whose anchor WAS released with its copy proven (``pdflip_anchor_secured``, the S25 mark the
    S109c release sets through the aux funnel): ``how=store_miss defect=1`` -- also at a branching depth
    (precedence: a secured anchor is never explained away).
(3) Neither secured nor branching: ``how=store_miss_unsecured defect=0``.
(4) The read is still attempted in every case (a branch point may carry a tracked state, #59); an attach is defect=0.
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
from flliper.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
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


S90B = _load("_s90b_fixture_for_s109r_miss", _UNIT / "pdflip" / "test_s90b_anchor_l3_1020.py")
DEPTH = S90B.D_TOLD


def _world(monkeypatch, *, ok=False, secured=False, held=None, written=False):
    be = S90B._Backend(ok=ok)
    t, n1, _n2, hp, _dev = S90B._ftree(monkeypatch, be)   # n1: device KV, no Full host, no mamba state
    n1.pdflip_anchor_secured = secured
    hp.anchor_held = lambda last_hash: held                 # the arena / #1459 L3 index answer (memory only)
    if written:
        UnifiedRadixCache._s109r_note_written(t, "n19200-last", object())
    monkeypatch.setattr(HOS, "form_a_host", lambda cache: True)
    return t, n1, be


def _req(branching):
    return types.SimpleNamespace(rid="pdflip-2-15", key_match_depth=DEPTH, state_anchor_depth=DEPTH - 2 * 64,
                                 full_untruncated_fill_ids=S90B.TOKENS, extra_key=None,
                                 mamba_branching_seqlen=branching)


def _line(caplog):
    lines = [m for m in caplog.messages if "S109R ANCHOR-DEVICE-READ" in m]
    assert lines, caplog.messages
    return lines[-1]


def _read(t, req):
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        return t.pdflip_anchor_only_read(req)


def test_rc30_branch_point_miss_is_no_defect(monkeypatch, caplog):
    caplog.set_level(logging.INFO, LOGGER)
    t, n1, be = _world(monkeypatch)
    assert _read(t, _req(branching=DEPTH)) == 0
    assert be.calls == [(PoolName.MAMBA, ["n19200-last"])], "the read is still attempted"
    line = _line(caplog)
    assert "how=branch_point_never_anchored" in line and "defect=0" in line, line


def test_a_secured_anchor_that_is_gone_is_the_defect_even_at_a_branch_point(monkeypatch, caplog):
    """Precedence (danger direction: hiding a real loss behind the branch explanation)."""
    caplog.set_level(logging.INFO, LOGGER)
    t, _n1, _be = _world(monkeypatch, secured=True)
    assert _read(t, _req(branching=DEPTH)) == 0
    line = _line(caplog)
    assert "how=store_miss " in line and "defect=1" in line, line


def test_unsecured_non_branch_miss_is_named_apart(monkeypatch, caplog):
    caplog.set_level(logging.INFO, LOGGER)
    t, _n1, _be = _world(monkeypatch)
    assert _read(t, _req(branching=None)) == 0
    line = _line(caplog)
    assert "how=store_miss_unsecured" in line and "defect=0" in line, line


def test_an_attach_is_no_defect(monkeypatch, caplog):
    caplog.set_level(logging.INFO, LOGGER)
    t, n1, _be = _world(monkeypatch, ok=True)
    assert _read(t, _req(branching=DEPTH)) == DEPTH
    line = _line(caplog)
    assert "how=device_attach" in line and "attached=1" in line and "defect=0" in line, line


def test_an_anchor_this_rank_wrote_and_lost_is_the_defect_with_its_witness(monkeypatch, caplog):
    """The rc30 n=3 shape (pdflip-2-15 06:37:29: ``ANCHOR-ONLY-BACKUP node=32 depth=62656`` 35 s before the miss): the
    key was claimed by this rank in this boot, now no tier holds it -> lost, ``written_here=1 held=0``."""
    caplog.set_level(logging.INFO, LOGGER)
    t, _n1, be = _world(monkeypatch, held=False, written=True)
    assert _read(t, _req(branching=DEPTH)) == 0
    assert be.calls, "the read is attempted whatever the index says (an index gap must never cost tokens)"
    line = _line(caplog)
    assert "how=store_miss " in line and "defect=1" in line and "held=0 written_here=1" in line, line


def test_a_held_key_the_read_cannot_serve_is_a_read_path_defect(monkeypatch, caplog):
    caplog.set_level(logging.INFO, LOGGER)
    t, _n1, _be = _world(monkeypatch, held=True)
    assert _read(t, _req(branching=DEPTH)) == 0
    line = _line(caplog)
    assert "how=held_unread" in line and "defect=1" in line and "held=1" in line, line


def test_the_claim_records_the_key_it_wrote():
    t = object.__new__(UnifiedRadixCache)
    UnifiedRadixCache._s109r_note_written(t, "k1", object())
    UnifiedRadixCache._s109r_note_written(t, "k2", None)        # a refused claim wrote nothing
    assert "k1" in t._s109r_written and "k2" not in t._s109r_written


@pytest.mark.parametrize("secured,written,held,branching,want", [
    (True, False, None, DEPTH, "store_miss"), (True, False, None, None, "store_miss"),
    (False, True, False, DEPTH, "store_miss"), (False, False, True, DEPTH, "held_unread"),
    (True, True, True, None, "held_unread"),
    (False, False, None, DEPTH, "branch_point_never_anchored"), (False, False, False, DEPTH, "branch_point_never_anchored"),
    (False, False, None, DEPTH + 64, "store_miss_unsecured"),
    (False, False, None, None, "store_miss_unsecured"), (False, False, None, True, "store_miss_unsecured"),
])
def test_miss_class_table(secured, written, held, branching, want):
    node = types.SimpleNamespace(pdflip_anchor_secured=secured)
    req = types.SimpleNamespace(mamba_branching_seqlen=branching)
    assert UnifiedRadixCache._s109r_miss_class(node, req, DEPTH, held, written) == want
