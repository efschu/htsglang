"""MEMRANK-1015 Nachbesserung (review verdict JA_MIT_AUFLAGE): A1, A3, A4, A5, A7.

A1  the pinned-host reader uses the REAL torch 2.11 host-allocator keys
    (``active_bytes.current`` = used, ``allocated_bytes.current`` = held); a missing
    key prints -1, never a default 0; the key names are checked against the
    installed torch's own ``host_memory_stats`` docstring.
A3  the front's census sampler logs the live probe (per role) as its own line.
A4  retire points at the request end: guard ``retire_request`` (PP0 table),
    front ``retire_x_requeues``; the caps stay as backstop.
A5  the probe enters no collective -- also not through sglang's parallel_state /
    GroupCoordinator (recorded, so a swallowing try/except cannot hide it).
A7  held-sum without a torch OMP team.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import sys
import threading
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_census as hc  # noqa: E402
from sglang.srt.weg2 import mem_rank as mr  # noqa: E402

ROOT = os.path.join(os.path.dirname(mr.__file__), "..")
MIB = 1 << 20


def _kv(line: str) -> dict:
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


@pytest.fixture(autouse=True)
def _clean():
    mr._reset_for_tests()
    yield
    mr._reset_for_tests()


# ------------------------------------------------------------------ A1
def _fake_torch(stats):
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(memory=types.SimpleNamespace(host_memory_stats=lambda: stats))
    return t


def test_a1_real_keys_are_read_and_named(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch({
        "allocated_bytes.current": 512 * MIB, "active_bytes.current": 128 * MIB,
        "reserved_bytes.current": 999 * MIB,  # a device-allocator key: must NOT be read
    }))
    assert mr.read_pinned_torch() == (128 * MIB, 512 * MIB)
    kv = _kv(mr.MemRankProbe("PP0", group="P", rank=0).line())
    assert kv["pin_torch_mib"] == "128.0"
    assert kv["pin_torch_held_mib"] == "512.0"
    assert "pin_torch_rsvd_mib" not in kv


@pytest.mark.parametrize("stats,expect", [
    ({}, (-1, -1)),                                          # no pinned allocation yet / CPU torch
    ({"allocated_bytes.current": 4 * MIB}, (-1, 4 * MIB)),   # only held present
    ({"active_bytes.current": 4 * MIB}, (4 * MIB, -1)),      # only active present
    ({"reserved_bytes.current": 4 * MIB}, (-1, -1)),         # the old (wrong) key alone
])
def test_a1_missing_key_prints_minus_one_never_zero(monkeypatch, stats, expect):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(stats))
    assert mr.read_pinned_torch() == expect
    kv = _kv(mr.MemRankProbe("PP0", group="P", rank=0).line())
    want = tuple("-1" if v < 0 else "%.1f" % (v / MIB) for v in expect)
    assert (kv["pin_torch_mib"], kv["pin_torch_held_mib"]) == want


def test_a1_failing_reader_prints_minus_one(monkeypatch):
    def _boom():
        raise RuntimeError("no cuda")

    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(memory=types.SimpleNamespace(host_memory_stats=_boom))
    monkeypatch.setitem(sys.modules, "torch", t)
    assert mr.read_pinned_torch() is None
    kv = _kv(mr.MemRankProbe("PP0", group="P", rank=0).line())
    assert kv["pin_torch_mib"] == "-1" and kv["pin_torch_held_mib"] == "-1"


def test_a1_key_names_exist_in_the_installed_torch_docstring():
    torch = pytest.importorskip("torch")
    doc = torch.cuda.memory.host_memory_stats.__doc__ or ""
    for key in (mr.PIN_TORCH_ACTIVE_KEY, mr.PIN_TORCH_HELD_KEY):
        base, _, field = key.partition(".")
        m = re.search(r'"%s\.\{([^}]*)\}"' % re.escape(base), doc)
        assert m, "torch %s documents no %r key family -- rename the reader" % (torch.__version__, base)
        assert field in [f.strip() for f in m.group(1).split(",")], (key, m.group(1))
    # the host allocator has no reserved_bytes (that one is the device allocator's)
    assert not re.search(r'"reserved_bytes\.', doc)


# ------------------------------------------------------------------ A3
def test_a3_live_line_carries_roles_and_cgroup_terms():
    census = {"at": "2026-10-15T12:00:00Z", "cg_anon_gib": 30.25, "cg_shmem_gib": 61.4, "skipped_pids": 1,
              "roles_anon_gib": {"server_main": 2.5, "inductor_compile_worker": 1.95, "ple_pread_worker": 0.3,
                                 "rank": 11.0}}
    line = hc.live_line("k|form", census)
    assert line.startswith(hc.LIVE_MARKER + " key=k|form")
    assert "cg_anon_gib=30.250" in line and "cg_shmem_gib=61.400" in line and "skipped_pids=1" in line
    for r, v in (("server_main", "2.500"), ("inductor_compile_worker", "1.950"),
                 ("ple_pread_worker", "0.300"), ("rank", "11.000")):
        assert "%s=%s" % (r, v) in line
    # an empty / odd census never raises
    assert "roles_anon_gib[none]" in hc.live_line("k", {})


def test_a3_sampler_logs_the_live_probe_each_tick(monkeypatch, caplog, tmp_path):
    from sglang.srt.weg2 import front

    census = {"at": "t", "cg_anon_gib": 1.0, "cg_shmem_gib": 2.0, "skipped_pids": 0,
              "roles_anon_gib": {"server_main": 0.5}}
    ticks = []
    first_done = threading.Event()
    park = threading.Event()  # never set: the daemon thread stays parked after tick 1 (it must not run on
    #                           with the REAL sample_live / merge_into_record once the monkeypatch is undone)

    def _sample(**kw):
        ticks.append(1)
        if len(ticks) > 1:
            park.wait(3600)
        return census

    monkeypatch.setattr(hc, "sample_live", _sample)
    monkeypatch.setattr(hc, "merge_into_record", lambda *a, **k: {"merged": True})
    monkeypatch.setattr(hc, "ledger_terms", lambda ent: ent)

    def _census_line(key, terms):
        first_done.set()
        return "WEG2-HOST-CENSUS merged-line"

    monkeypatch.setattr(hc, "census_line", _census_line)
    with caplog.at_level(logging.INFO):
        th = front.start_host_census_sampler(str(tmp_path / "rec.json"), "k", str(tmp_path), [], period_s=0.01)
        assert first_done.wait(5)
    assert th.daemon
    msgs = [r.getMessage() for r in caplog.records]
    live = [m for m in msgs if m.startswith(hc.LIVE_MARKER)]
    assert live and "roles_anon_gib[server_main=0.500]" in live[0]
    # the merged (max) line is still logged, after the live one of the same tick
    assert msgs.index(live[0]) < msgs.index("WEG2-HOST-CENSUS merged-line")


def test_a3_a_failing_live_line_does_not_skip_the_merge(monkeypatch, caplog, tmp_path):
    from sglang.srt.weg2 import front

    merged = threading.Event()
    park = threading.Event()
    n = []

    def _sample(**kw):
        n.append(1)
        if len(n) > 1:
            park.wait(3600)  # parked for good (see the sibling test)
        return {}

    def _merge(*a, **k):
        merged.set()
        return {}

    monkeypatch.setattr(hc, "sample_live", _sample)
    monkeypatch.setattr(hc, "merge_into_record", _merge)
    monkeypatch.setattr(hc, "ledger_terms", lambda ent: ent)
    monkeypatch.setattr(hc, "census_line", lambda k, t: "x")

    def _bad(*a, **k):
        raise ValueError("format")

    monkeypatch.setattr(hc, "live_line", _bad)
    front.start_host_census_sampler(str(tmp_path / "rec.json"), "k", str(tmp_path), [], period_s=0.01)
    assert merged.wait(5)


# ------------------------------------------------------------------ A4
def test_a4_guard_retire_request_drops_only_that_rid():
    from sglang.srt.managers.pp_admission_congruence import PPAdmissionCongruenceGuard as G

    g = G()
    for r, v in (("a", 5), ("b", 6), ("ab1", 7), ("c", 8)):
        g.note_observed_coverage(r, v)
    assert g.retire_request("b") == 1
    assert g.uniform_prefix_for("b") is None and g.uniform_prefix_for("a") == 5
    assert g.retire_request("b") == 0  # idempotent (an overlap batch finishes twice)
    assert g.retire_request("a", prefix=True) == 2  # abort prefix semantics: a, ab1
    assert g.uniform_prefix_for("c") == 8
    assert g.retire_request(None) == 1 and g.uniform_prefix_for("c") is None


def test_a4_follower_without_a_table_is_a_noop_and_creates_none():
    from sglang.srt.managers.pp_admission_congruence import PPAdmissionCongruenceGuard as G

    g = G()
    assert not hasattr(g, "_1059_observed")
    assert g.retire_request("x") == 0 and g.retire_request(None) == 0 and g.retire_request("x", prefix=True) == 0
    assert not hasattr(g, "_1059_observed")  # decides nothing, writes nothing a rank reads


def test_a4_retired_rid_reads_no_fact_new_instance_starts_clean():
    from sglang.srt.managers.pp_admission_congruence import PPAdmissionCongruenceGuard as G

    g = G()
    g.note_observed_coverage("r", 10)
    g.retire_request("r")
    g.note_observed_coverage("r", 40)  # the same rid, a NEW instance (lane resume): its own promise
    assert g.uniform_prefix_for("r") == 40


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel)) as fh:
        return fh.read()


def _func_body(src: str, header: str) -> str:
    i = src.index(header)
    j = src.find("\n    def ", i + len(header))
    return src[i: j if j > 0 else len(src)]


def test_a4_scheduler_retires_on_finish_and_on_abort():
    src = _src("managers/scheduler.py")
    fin = _func_body(src, "    def process_batch_result(")
    assert "_pp_admission_guard" in fin and ".retire_request(_fin.rid)" in fin and "_fin.finished()" in fin
    ab = _func_body(src, "    def _abort_request_now(")
    assert "retire_request(None if recv_req.abort_all else recv_req.rid, prefix=True)" in ab


def test_a4_front_retire_helper_and_handler_end_site():
    from sglang.srt.weg2 import front

    t = {"a": 1, "b": 2}
    assert front.retire_x_requeues(t, "a") is True and t == {"b": 2}
    assert front.retire_x_requeues(t, "a") is False
    src = _src("weg2/front.py")
    assert "retire_x_requeues(self._x_requeues, rid)" in src
    # the retire sits in the generate handler's finally, next to the other rid-end hooks
    i = src.index("retire_x_requeues(self._x_requeues, rid)")
    assert "Front._lane_end(self, rid)" in src[i - 400: i]


# ------------------------------------------------------------------ A5
_FORBIDDEN = re.compile(r"parallel_state|get_\w*group\w*|GroupCoordinator|torch\.distributed|\bdistributed\b")


def _code_names(src: str):
    """Every import, name, attribute and non-docstring string constant of the module."""
    tree = ast.parse(src)
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                doc_nodes.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append(node.module or "")
            out += [a.name for a in node.names]
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_nodes:
            out.append(node.value)
    return out


def test_a5_probe_makes_no_collective_through_sglang_groups(monkeypatch):
    """Recorded, not raised: a ``try/except`` around a collective (mutant MB) would swallow a raise."""
    pytest.importorskip("torch")
    ps = pytest.importorskip("sglang.srt.distributed.parallel_state")
    calls = []

    def _rec(name):
        def _f(*a, **k):
            calls.append(name)
            raise AssertionError("WEG2-MEM-RANK must not enter a collective: " + name)
        return _f

    for name in dir(ps.GroupCoordinator):
        if not name.startswith("_") and callable(getattr(ps.GroupCoordinator, name, None)):
            monkeypatch.setattr(ps.GroupCoordinator, name, _rec("GroupCoordinator." + name))
    for name in dir(ps):
        if re.match(r"get_\w*group", name) and callable(getattr(ps, name)):
            monkeypatch.setattr(ps, name, _rec("parallel_state." + name))
    import torch.distributed as dist

    for name in ("all_reduce", "all_gather", "all_gather_object", "broadcast", "barrier", "reduce", "gather",
                 "scatter", "send", "recv", "all_to_all", "broadcast_object_list", "new_group"):
        if hasattr(dist, name):
            monkeypatch.setattr(dist, name, _rec("torch.distributed." + name))
    p = mr.MemRankProbe("PP2", group="P", rank=2)
    assert p.line().startswith(mr.MARKER)
    assert calls == [], calls


def test_a5_module_source_names_no_collective_api():
    with open(mr.__file__) as fh:
        src = fh.read()
    bad = [n for n in _code_names(src) if _FORBIDDEN.search(n)]
    assert bad == [], bad
    assert "import zmq" not in src and "requests." not in src


def test_a5_the_static_check_would_catch_mutant_mb():
    mutant = ("def _gauges(self):\n    try:\n        sglang.srt.distributed.parallel_state.get_tp_group().barrier()\n"
              "    except Exception:\n        pass\n")
    assert [n for n in _code_names(mutant) if _FORBIDDEN.search(n)]
    mutant2 = "from sglang.srt.distributed.parallel_state import get_tp_group\n"
    assert [n for n in _code_names(mutant2) if _FORBIDDEN.search(n)]


# ------------------------------------------------------------------ A7
def test_a7_held_sum_is_numpy_not_a_torch_reduction(monkeypatch):
    torch = pytest.importorskip("torch")
    ledg = types.ModuleType("sglang.srt.mem_cache.storage.file.hicache_arena")

    class _Held:
        """A tensor whose ``.sum()`` must not be called (it may start an OMP team)."""

        def __init__(self, t):
            self._t = t

        def numpy(self):
            return self._t.numpy()

        def sum(self):
            raise AssertionError("torch .sum() in the probe thread")

    class _L:
        def __init__(self):
            t = torch.zeros(100000, dtype=torch.int32)
            t[5] = 3
            t[99999] = 4
            self.held = _Held(t)
            self.lock = threading.Lock()

    ledg._LEDGERS = {("p", 1): _L()}
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.storage.file.hicache_arena", ledg)
    assert mr._ledger_held() == 7
