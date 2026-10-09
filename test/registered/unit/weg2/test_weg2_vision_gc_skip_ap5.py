"""VISION-WEIGHTS AP5: the teardown's full ``gc.collect`` (27B 02.10.: 496 of
498 ms teardown) is skipped on a weights stage when no tower tensor survives
the strip -- ported from NF VISION-GC-SKIP-1002 (c651892375).

Pinned:
  * a clean weights stage skips the collect and says so in W102 (gc=skipped);
  * a tower tensor that is still alive after the strip (held elsewhere), or a
    failed stage, takes the full collect (gc=full, gc_alive counted);
  * the default places keep the full collect exactly as before (no gc field).
"""

import types

import pytest
import torch

from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs
from test_weg2_vision_rank_runner import _Alloc, _build, _Item, _kv, _req, _write_model  # noqa: E402
from test_weg2_vision_victim_core import _Victims  # noqa: E402

KEEP = []


@pytest.fixture
def collects(monkeypatch):
    calls = []
    monkeypatch.setattr(vrr.gc, "collect", lambda *a, **k: calls.append(1) or 0)
    yield calls
    KEEP.clear()


def _run(tmp_path, place, **kw):
    return vrr.run_rank_stage(types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(_kv())),
                              [_req("r", [_Item()])], model_dir=str(tmp_path), hf_config=None,
                              device=torch.device("cpu"), build=_build(), place=place, **kw)


def _keeps_a_tower_tensor(module, items):
    KEEP.append(module.merger.linear_fc1.weight)   # survives the strip
    return vrr.encode_items(module, items)


@pytest.mark.parametrize("case,want_mode,want_calls", [
    ("clean", "skipped", 0),
    ("survivor", "full", 1),
    ("encode_error", "full", 1),
])
def test_a_weights_stage_skips_the_full_collect_only_when_nothing_survives(tmp_path, collects, caplog,
                                                                           case, want_mode, want_calls):
    _write_model(tmp_path)
    kw = {}
    if case == "survivor":
        kw["encode"] = _keeps_a_tower_tensor
    elif case == "encode_error":
        kw["encode"] = lambda m, i: (_ for _ in ()).throw(RuntimeError("boom"))
    out = _run(tmp_path, vrs.PLACE_WEIGHTS, victims=_Victims(), **kw)
    assert out.gc_mode == want_mode and len(collects) == want_calls
    assert (out.gc_alive >= 1) == (case == "survivor")
    with caplog.at_level("INFO"):
        vrr.log_outcome(out, ["r"], 1)
    assert f"gc={want_mode}" in caplog.text


def test_the_default_places_keep_the_full_collect(tmp_path, collects, caplog):
    _write_model(tmp_path)
    out = _run(tmp_path, vrs.PLACE_KVTAIL)
    assert out.ok and out.gc_mode == "" and len(collects) == 1
    with caplog.at_level("INFO"):
        vrr.log_outcome(out, ["r"], 1)
    assert " gc=" not in caplog.text
