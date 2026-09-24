"""weg2xsn417 (24.09.): W11b refused the DFlash producer's build.

Metal: ``W11b Weg2DraftBuildUnaccounted: nvml_delta_mib=3072.0 is not explained
by resident_mib=2174.5 + ... + tag_pool_inactive_mib=-1.0 + outside_torch_mib=-1.0
+ default_pool_inactive_mib=-1.0 + other_live_mib=-1.0 (unaccounted 897.5 MiB,
tolerance 256)``. The DFlash producer carried its own pre-#66 two-term copy
(context growth + allocator cache) and printed every #66 term as -1. xsn411
(eb5d04453f, the same DFlash2 build) measured with that copy:
``allocator_cache_mib=891.8 context_growth_mib=0.0`` -> 3072.0 - 2174.5 - 891.8
= 5.7 MiB live non-model bytes.

Hermetic, CPU. Pins:
* ``DraftBuildMeter`` -- the ONE implementation both producers use -- turns the
  scripted xsn411 card states into the #66 terms, and W11b (unchanged,
  tolerance 256 MiB) accepts the resulting armed line;
* the xsn417 placeholder line still refuses (the gate is not softened);
* ``DFlashDraftKvProducer`` builds the meter before its draft and finishes it
  in ``load_resident_embedding``; neither producer keeps a private copy of the
  instrument logic;
* a handle built without ``__init__`` reads -1 everywhere (never a guess).
"""

import inspect
import re
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.weg2.launcher as L
from sglang.srt.speculative import draft_kv_producer as dkp
from sglang.srt.speculative import dflash_draft_kv_producer as dfp

#: boot_weg2_weg2xsn417_..._095627.P.log line 1387 (the armed line W11b read).
XSN417_ARMED = (
    "[2026-09-24 09:57:14 PP2] WEG2 DRAFT-KV-PRODUCER armed stage=2/3 "
    "drafter=f0c73316257fec1d layout=v1 heads=8 head_dim=128 page_bytes=2048 "
    "embed=resident mtp_mib=2046.3 embed_mib=0.0 resident_mib=2174.5 "
    "head_released_mib=0.0 head_deferred=False nvml_delta_mib=3072.0 "
    "tag_pool_inactive_mib=-1.0 outside_torch_mib=-1.0 "
    "default_pool_inactive_mib=-1.0 card_free_mib=-1.0 other_live_mib=-1.0 "
    "embed_dtype=n/a build_s=2.5\n"
)


class _Script:
    """Scripted card states: each helper returns its BEFORE value on the first
    call and its AFTER value on every later one."""

    def __init__(self, monkeypatch, **pairs):
        self.calls = {}
        for name, (before, after) in pairs.items():
            self._install(monkeypatch, name, before, after)

    def _install(self, monkeypatch, name, before, after):
        def fn():
            n = self.calls.get(name, 0)
            self.calls[name] = n + 1
            return before if n == 0 else after

        monkeypatch.setattr(dkp, name, fn)


def _xsn411_states(monkeypatch):
    # xsn411's DFlash2 build on P's last stage (3080, 20480 MiB): NVML delta
    # 3072.0, allocator cache added 891.8 (all of it in the draft's tag pool
    # here), nothing outside torch, allocator growth 2180.2 = 2174.5 model
    # (params + the 128.2 MiB rotary buffer) + 5.7 live non-model.
    return _Script(
        monkeypatch,
        _cuda_free_mib=(13383.0, 10311.0),
        _tag_pool_inactive_mib=(50.0, 941.8),
        _outside_torch_mib=(512.0, 512.0),
        _default_pool_inactive_mib=(100.0, 991.8),
        _live_allocated_mib=(6000.0, 8180.2),
    )


def _armed_line(h) -> str:
    """The key=value terms the launcher's W11/W11b regexes read."""
    terms = " ".join(
        f"{name}={getattr(h, name):.1f}"
        for name in ("resident_mib", "head_released_mib") + dkp.DRAFT_BUILD_FIELDS[1:]
    )
    return f"[t PP2] WEG2 DRAFT-KV-PRODUCER armed stage=2/3 {terms} build_s=2.5\n"


def _w11(tmp_path, line, *, budget=2072.1):
    p = tmp_path / "P.log"
    p.write_text(line)
    return L.check_draft_resident(str(p), budget_mib=budget)


def test_meter_turns_the_xsn411_states_into_the_66_terms(monkeypatch):
    _xsn411_states(monkeypatch)
    h = SimpleNamespace()
    meter = dkp.DraftBuildMeter()
    dkp.DraftBuildMeter.init_fields(h)
    meter.finish(h, resident_mib=lambda: 2174.5)
    assert h.nvml_delta_mib == pytest.approx(3072.0)
    assert h.default_pool_inactive_mib == pytest.approx(891.8)
    assert h.tag_pool_inactive_mib == pytest.approx(891.8)
    assert h.outside_torch_mib == pytest.approx(0.0)
    assert h.other_live_mib == pytest.approx(5.7)
    assert h.card_free_mib == pytest.approx(10311.0)
    assert h.resident_mib == pytest.approx(2174.5)


def test_w11b_accepts_the_measured_line_and_still_refuses_xsn417(monkeypatch, tmp_path):
    _xsn411_states(monkeypatch)
    h = SimpleNamespace(head_released_mib=0.0)
    meter = dkp.DraftBuildMeter()
    dkp.DraftBuildMeter.init_fields(h)
    meter.finish(h, resident_mib=lambda: 2174.5)
    good = _w11(tmp_path, _armed_line(h))
    assert good["cache_term"] == "torch_total"
    assert good["unaccounted_mib"] == pytest.approx(0.0, abs=0.05)
    assert good["accounted"] and good["ok"]
    bad = _w11(tmp_path, XSN417_ARMED)
    assert bad["unaccounted_mib"] == pytest.approx(897.5)
    assert not bad["accounted"]
    # the tolerance is untouched
    assert L.P_DRAFT_BUILD_ACCOUNTING_TOL_MIB == 256.0


def test_gate_w11_refuses_xsn417_by_name(tmp_path, monkeypatch):
    monkeypatch.setitem(L._SPEC_FORM, "form", "NEXTN")  # budget constant path
    p = tmp_path / "P.log"
    p.write_text(XSN417_ARMED)
    lines = []
    with pytest.raises(L.Weg2LaunchRefused, match="W11b Weg2DraftBuildUnaccounted"):
        L.check_draft_resident  # noqa: B018 -- the reader below is the gate's
        L.gate_w11(str(p), lines.append)


def test_dflash_producer_builds_and_finishes_the_one_meter(monkeypatch):
    _xsn411_states(monkeypatch)
    prod = dfp.DFlashDraftKvProducer.__new__(dfp.DFlashDraftKvProducer)
    prod._build_meter = dkp.DraftBuildMeter()
    dkp.DraftBuildMeter.init_fields(prod)
    prod.head_released_mib = 0.0
    model = torch.nn.Linear(4, 4, bias=False)
    prod.draft_runner = SimpleNamespace(model=model)
    assert prod.load_resident_embedding("/nonexistent") == 0.0
    assert prod.default_pool_inactive_mib == pytest.approx(891.8)
    assert prod.nvml_delta_mib == pytest.approx(3072.0)
    # resident is the handle's own live-weight reading of its draft model
    assert prod.resident_mib == pytest.approx(4 * 4 * 4 / float(2**20))


def _code_names(cls) -> set:
    """Every identifier the CODE of ``cls`` uses (names and attributes);
    comments and docstrings are not code."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def test_one_meter_no_second_copy():
    init = inspect.getsource(dfp.DFlashDraftKvProducer.__init__)
    load = inspect.getsource(dfp.DFlashDraftKvProducer.load_resident_embedding)
    assert "DraftBuildMeter()" in init and "DraftBuildMeter.init_fields(self)" in init
    assert "meter.finish(" in load
    dflash = _code_names(dfp.DFlashDraftKvProducer)
    for private_copy in ("_cuda_free_mib", "memory_reserved", "context_growth_mib",
                         "allocator_cache_mib", "mem_get_info", "_reserved_before_mib"):
        assert private_copy not in dflash, private_copy
    nextn_src = inspect.getsource(dkp.DraftKvProducer)
    assert "DraftBuildMeter()" in nextn_src and "meter.finish(" in nextn_src
    assert "mark_pool_before_release()" in nextn_src
    nextn = _code_names(dkp.DraftKvProducer)
    for helper in ("_tag_pool_inactive_mib", "_outside_torch_mib",
                   "_default_pool_inactive_mib", "_live_allocated_mib", "_cuda_free_mib"):
        assert helper not in nextn, helper


def test_a_handle_without_init_reads_unmeasured(monkeypatch):
    for name in ("_cuda_free_mib", "_tag_pool_inactive_mib", "_outside_torch_mib",
                 "_default_pool_inactive_mib", "_live_allocated_mib"):
        monkeypatch.setattr(dkp, name, lambda: 777.0)
    prod = dfp.DFlashDraftKvProducer.__new__(dfp.DFlashDraftKvProducer)
    prod.draft_runner = SimpleNamespace(model=torch.nn.Linear(2, 2, bias=False))
    prod.load_resident_embedding("/nonexistent")
    for name in ("nvml_delta_mib", "tag_pool_inactive_mib", "outside_torch_mib",
                 "default_pool_inactive_mib", "other_live_mib"):
        assert getattr(prod, name) == -1.0, name


def test_armed_line_fields_are_the_launcher_regex_keys():
    for name in dkp.DRAFT_BUILD_FIELDS:
        rx = getattr(L, {
            "resident_mib": "_RESIDENT_RE",
            "nvml_delta_mib": "_NVML_DELTA_RE",
            "tag_pool_inactive_mib": "_TAG_POOL_INACTIVE_RE",
            "outside_torch_mib": "_OUTSIDE_TORCH_RE",
            "default_pool_inactive_mib": "_DEFAULT_POOL_RE",
            "card_free_mib": "_CARD_FREE_RE",
            "other_live_mib": "_OTHER_LIVE_RE",
        }[name])
        assert re.escape(name) in rx.pattern or name in rx.pattern
