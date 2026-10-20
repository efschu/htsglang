"""S104 (20.10.): the sleep flush's leftovers and the stop's L2 tail.

F1 (rc22 D 09:54:35Z, ``#1470 FLUSH-UNBACKED site=flush nodes=14 tokens=10432``): an IDLE quiesce ``/flush_cache``
whose #1470 loop left un-backed device KV (KV arena held in full by the trees) reset the tree itself, before the
sleep leg's S86 SLEEP-L3-RELEASE could free the room. Now the group asks once (one MAX) and hands the reset to the
sleep leg. Form A: TP0 (the share-0 host) and the workers hold DIFFERENT trees -- the three trees below are cut at
different depths, and only the group-reduced answer may decide (lesson of the rc23 W86 death, 11:05Z).

F2 (rc24 9/9 sleeps TP1/TP2 ``unbacked_left`` 6..27 ``refused_by=anchor_only_no_arena``, TP0 0): a Form A worker's
anchor is byteless; its KV-backed nodes are no ANCHOR-ONLY candidates.

F3 (rc22 10:50 stop -> rc23, ``PREFIX-DIVERGE verdict=store_short lost=1216``): the pages L2 completed since the
last write-behind pass get their L3 copy at SIGTERM (real C arena + real L3P store, restart probe).

Real UnifiedRadixCache trees over real HybridReqToTokenPools; no GPU, no collective backend (the group MAX is a
three-thread barrier)."""
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
from array import array
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.managers import pdflip_flush_nonblock as fnb  # noqa: E402
from flliper.srt.mem_cache import l3_final_drain as fd  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import InsertParams  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("_s104_compact", "test_pdflip_d_seat_compact_0930.py")
AO = _load("_s104_anchor_only", os.path.join("..", "mem_cache", "test_pdflip_anchor_only_backup_0930.py"))
MC = ComponentType.MAMBA
KV = ComponentType.FULL
BASE = list(range(3000, 3256))


# ---------------------------------------------------------------- three trees, cut differently ----------
def _tree(cuts, mamba_on=True):
    """A real tree holding BASE[:256] whose node boundaries are ``cuts`` (each prefix inserted first, so the
    tree splits there -- TP0 and the workers end up with different node sets, as on Form A)."""
    cache, pool, kalloc = T._build()
    for c in list(cuts) + [len(BASE)]:
        toks = BASE[:c]
        kw = {}
        if mamba_on:
            kw["mamba_value"] = pool.mamba_allocator.alloc(1).reshape(1)
        cache.insert(InsertParams(key=RadixKey(array("q", toks)), value=kalloc.alloc(len(toks)), **kw))
    nodes = [n for n in cache._collect_all_nodes() if n is not cache.root_node]
    for n in nodes:
        n.hash_value = ["h%d_%d" % (n.id, i) for i in range(len(n.key))]
    return cache, nodes


def _depth_end(cache, node):
    return cache.pdflip_node_depth(node)


def _back_all(nodes, *, l3=True):
    for n in nodes:
        if l3:
            n.l3_present = True
        else:
            n.component_data[KV].host_value = torch.arange(len(n.key), dtype=torch.int64)


class _GroupMax:
    """MAX over three ranks behind a barrier: a rank that posts a collective its peers do not post (or posts a
    different number) breaks the barrier -- the test fails instead of hanging."""

    def __init__(self, n=3):
        self.n = n
        self.bar = threading.Barrier(n, timeout=10)
        self.lock = threading.Lock()
        self.vals = {}
        self.labels = {}

    def for_rank(self, r):
        def gm(values, label=""):
            with self.lock:
                self.vals[r] = list(values)
                self.labels.setdefault(r, []).append(label)
            self.bar.wait()
            with self.lock:
                out = [max(v[i] for v in self.vals.values()) for i in range(len(values))]
            self.bar.wait()
            return out
        return gm


def _sched(tree):
    return types.SimpleNamespace(enable_hierarchical_cache=True, ps=types.SimpleNamespace(pp_size=1),
                                 tree_cache=tree)


def _run_three(trees, monkeypatch, *, group_idle=True, tp_group_verdict=True, local=False, flip_quiesce=True):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    g = _GroupMax(len(trees))
    out, errs = {}, {}

    def run(r, tree):
        try:
            if local:   # the mutant: a rank-local decision, no group reduction
                tree.hicache_group_max = lambda values, label="": list(values)
            else:
                tree.hicache_group_max = g.for_rank(r)
            out[r] = fnb.s104_defer_reset_verdict(_sched(tree), group_idle, tp_group_verdict, 0,
                                                  flip_quiesce=flip_quiesce)
        except Exception as e:  # noqa: BLE001
            errs[r] = e

    ths = [threading.Thread(target=run, args=(r, t)) for r, t in enumerate(trees)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(15)
    assert not errs, errs
    return out, g


def _three():
    """TP0 cut at 64 and 192, TP1 one node, TP2 cut at 96 -- three different node sets of the same chain."""
    return _tree([64, 192]), _tree([]), _tree([96])


def test_f1_three_trees_cut_differently_really_differ():
    (t0, n0), (t1, n1), (t2, n2) = _three()
    ends = [sorted(_depth_end(t, n) for n in ns) for t, ns in ((t0, n0), (t1, n1), (t2, n2))]
    assert ends == [[64, 192, 256], [256], [96, 256]]


def test_f1_nothing_to_drop_on_any_rank_no_defer_and_one_collective_each(monkeypatch):
    (t0, n0), (t1, n1), (t2, n2) = _three()
    for ns in (n0, n1, n2):
        _back_all(ns)
    out, g = _run_three([t0, t1, t2], monkeypatch)
    assert out == {0: None, 1: None, 2: None}
    assert {r: len(v) for r, v in g.labels.items()} == {0: 1, 1: 1, 2: 1}


def test_f1_byteless_worker_anchors_and_host_copies_never_defer(monkeypatch):
    """The workers' KV is l3_present with a device anchor (the rc24 shape, no bytes behind the anchor); TP0's KV
    has host copies. The census is the FLUSH-UNBACKED predicate -- nothing would be dropped."""
    (t0, n0), (t1, n1), (t2, n2) = _three()
    _back_all(n0, l3=False)
    _back_all(n1)
    _back_all(n2)
    assert any(n.component_data[MC].value is not None for n in n1 + n2), "device anchors on the workers"
    out, _g = _run_three([t0, t1, t2], monkeypatch)
    assert out == {0: None, 1: None, 2: None}


@pytest.mark.parametrize("who", [0, 1, 2])
def test_f1_one_rank_with_unbacked_kv_defers_every_rank(monkeypatch, who):
    """rc22 09:54:35: TP0 14 un-backed nodes. Whichever rank holds the un-backed node -- TP0's 64..192 piece, TP1's
    whole node, TP2's 96..256 piece -- every rank answers 'deferred' in the same pass."""
    trees = _three()
    for r, (t, ns) in enumerate(trees):
        _back_all(ns)
    t, ns = trees[who]
    victim = sorted(ns, key=lambda n: _depth_end(t, n))[-1 if who else 1]
    victim.l3_present = False
    out, g = _run_three([x[0] for x in trees], monkeypatch)
    assert all(v is not None and v.startswith(fnb.LINE_S104_DEFER) for v in out.values()), out
    assert "nodes=1" in out[who]
    assert {r: len(v) for r, v in g.labels.items()} == {0: 1, 1: 1, 2: 1}


def test_f1_mutant_rank_local_decision_splits_the_group(monkeypatch):
    """Killed mutant: without the group MAX only the holding rank defers -- TP0 would keep its tree while the
    workers reset theirs (RAENGE-NIE-UNEINS)."""
    trees = _three()
    for t, ns in trees:
        _back_all(ns)
    t2, n2 = trees[2]
    sorted(n2, key=lambda n: _depth_end(t2, n))[-1].l3_present = False
    out, _g = _run_three([x[0] for x in trees], monkeypatch, local=True)
    assert out[2] is not None and out[0] is None and out[1] is None


@pytest.mark.parametrize("kw", [dict(group_idle=False), dict(tp_group_verdict=False)])
def test_f1_no_collective_outside_the_idle_tp_group_path(monkeypatch, kw):
    """A busy group (B1's own path) or a rank-local verdict (the sleep leg's own flush) posts nothing here."""
    trees = _three()
    sorted(trees[0][1], key=lambda n: _depth_end(trees[0][0], n))[0].l3_present = False
    out, g = _run_three([x[0] for x in trees], monkeypatch, **kw)
    assert out == {0: None, 1: None, 2: None} and g.labels == {}


def test_f1_other_groups_and_pp_untouched(monkeypatch):
    t, ns = _tree([64])
    t.hicache_group_max = lambda values, label="": pytest.fail("no collective outside group D / pp=1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert fnb.s104_defer_reset_verdict(_sched(t), True, True, 3, flip_quiesce=True) is None
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    s = _sched(t)
    s.ps.pp_size = 3
    assert fnb.s104_defer_reset_verdict(s, True, True, 3, flip_quiesce=True) is None


# ---------------------------------------------------------------- review A1: only the flip quiesce defers -----
def test_a1_an_operator_flush_never_defers_and_posts_no_collective(monkeypatch):
    """A1 (nf-review-s104-1020): the same idle D with un-backed KV (full arena) -- a /flush_cache that is NOT the
    front's flip quiesce (operator, test, removing wrong data) resets as before on every rank."""
    trees = _three()
    for t, ns in trees:
        _back_all(ns)
    sorted(trees[0][1], key=lambda n: _depth_end(trees[0][0], n))[1].l3_present = False
    out, g = _run_three([x[0] for x in trees], monkeypatch, flip_quiesce=False)
    assert out == {0: None, 1: None, 2: None} and g.labels == {}
    out, g = _run_three([x[0] for x in trees], monkeypatch, flip_quiesce=True)
    assert all(v is not None for v in out.values())


def test_a1_the_default_request_is_no_flip_quiesce():
    from flliper.srt.managers.io_struct import FlushCacheReqInput

    assert FlushCacheReqInput(timeout_s=0.0).pdflip_flip_quiesce is False
    import inspect

    assert inspect.signature(fnb.s104_defer_reset_verdict).parameters["flip_quiesce"].default is False


def _wrapper(calls):
    from flliper.srt.managers.scheduler_components.flush_wrapper import SchedulerFlushWrapper

    def flush(**kw):
        calls.append(kw)
        return True

    return SchedulerFlushWrapper(flush_cache=flush, is_fully_idle=lambda: True, ipc_channels=None)


@pytest.mark.parametrize("mark", [False, True])
def test_a1_the_wrapper_carries_the_mark_only_when_set(mark):
    from flliper.srt.managers.io_struct import FlushCacheReqInput

    calls = []
    out = _wrapper(calls).handle(FlushCacheReqInput(timeout_s=0.0, pdflip_flip_quiesce=mark))
    assert out.success
    assert calls == ([dict(tp_group_verdict=True, pdflip_flip_quiesce=True)] if mark
                     else [dict(tp_group_verdict=True)])


def test_a1_the_front_quiesce_marks_itself_and_the_endpoint_reads_only_a_post_body():
    import inspect

    from flliper.srt.pdflip.front import Front

    assert '"/flush_cache", {"pdflip_flip_quiesce": True}' in inspect.getsource(Front.quiesce)
    src = open(os.path.join(HERE, "..", "..", "..", "..", "python", "flliper", "srt", "entrypoints",
                            "http_server.py")).read()
    i = src.index("async def _s104_flip_quiesce_flag")
    body = src[i:src.index("@app.api_route", i)]
    assert 'getattr(request, "method", "GET") != "POST"' in body and 'body.get("pdflip_flip_quiesce") is True' in body
    j = src.index('@app.api_route("/flush_cache"')
    assert "flip_quiesce=await _s104_flip_quiesce_flag(request)" in src[j:j + 900]


def test_a1_the_flag_parser_on_fake_requests():
    """The endpoint helper, run on stand-in requests: only POST + JSON body {"pdflip_flip_quiesce": true}."""
    import asyncio
    import ast

    src = open(os.path.join(HERE, "..", "..", "..", "..", "python", "flliper", "srt", "entrypoints",
                            "http_server.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_s104_flip_quiesce_flag")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "x", "exec"), ns)
    flag = ns["_s104_flip_quiesce_flag"]

    class R:
        def __init__(self, method, body):
            self.method, self._b = method, body

        async def json(self):
            if isinstance(self._b, Exception):
                raise self._b
            return self._b

    cases = [(R("POST", {"pdflip_flip_quiesce": True}), True), (R("POST", {}), False),
             (R("POST", {"pdflip_flip_quiesce": 1}), False), (R("POST", ValueError("no body")), False),
             (R("GET", {"pdflip_flip_quiesce": True}), False), (R("POST", [1]), False)]
    assert [asyncio.run(flag(r)) for r, _ in cases] == [w for _, w in cases]


# ---------------------------------------------------------------- review A2: teardown kills a stuck owner ------
_STUCK = textwrap.dedent("""
    import signal, sys, time, setproctitle
    setproctitle.setproctitle({title!r})
    if {stuck}:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # an owner whose handler never runs (main thread in C)
    print("ready", flush=True)
    time.sleep(60)
""")


def _child(title, stuck):
    p = subprocess.Popen([sys.executable, "-c", _STUCK.format(title=title, stuck=stuck)], stdout=subprocess.PIPE)
    assert p.stdout.readline().strip() == b"ready"
    return p


def test_a2_a_stuck_scheduler_is_killed_after_the_bound_and_named():
    from flliper.srt.pdflip import launcher as L

    stuck = _child("flliper::scheduler_TP0", True)
    healthy = _child("flliper::scheduler_TP1", False)
    other = _child("flliper.srt.pdflip.front-stand-in", True)
    try:
        pids = {stuck.pid, healthy.pid, other.pid}
        sched = L.s104_scheduler_pids(pids)
        assert sched == sorted([stuck.pid, healthy.pid]), "named before the TERM, by process title"
        for p in pids:
            os.kill(p, signal.SIGTERM)
        healthy.wait(5)
        lines = []
        t0 = time.monotonic()
        killed = L.s104_escalate_schedulers(sched, bound_s=1.0, log=lambda m, **k: lines.append(m))
        assert killed == [stuck.pid] and time.monotonic() - t0 < 3.0
        assert stuck.wait(5) == -signal.SIGKILL
        assert healthy.returncode == -signal.SIGTERM
        assert lines and lines[0].startswith("S104 TEARDOWN-SCHED-KILL pids=[%d]" % stuck.pid)
        assert other.poll() is None, "a non-scheduler keeps the teardown's 60 s bound"
    finally:
        for p in (stuck, healthy, other):
            if p.poll() is None:
                p.kill()
                p.wait(5)


def test_a2_the_bound_is_the_drain_plus_two_and_wired_before_the_60s_wait():
    import inspect

    from flliper.srt.pdflip import launcher as L

    assert L.S104_SCHED_KILL_S == fd.FINAL_DRAIN_S + 2.0
    src = inspect.getsource(L.teardown)
    assert src.index("s104_scheduler_pids(pids)") < src.index('print(f"TERM {sorted(pids)}")') \
        < src.index("s104_escalate_schedulers(s104_sched)") < src.index("while time.time() - t0 < 60")


def test_a2_nothing_to_wait_for_returns_at_once():
    from flliper.srt.pdflip import launcher as L

    assert L.s104_escalate_schedulers([], alive=lambda p: pytest.fail("no wait")) == []


def test_f1_wired_between_b1_and_the_reset():
    import inspect

    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.flush_cache)
    i_b1 = src.index("_b1_quiesced = _pdflip_flush_nonblock.quiesce_verdict(")
    i_s104 = src.index("_pdflip_flush_nonblock.s104_defer_reset_verdict(")
    i_reset = src.index("self.tree_cache.reset()")
    i_idle = src.index("if group_idle:", i_s104)
    assert i_b1 < i_s104 < i_idle < i_reset
    assert "tp_group_verdict, _s104_unbacked_left" in src
    assert '_s104_unbacked_left = int(_stats.get("unbacked", 0) or 0)' in src


def test_f1_the_sleep_leg_releases_before_its_flush():
    """The deferred reset lands where S86 runs first (unchanged order, pinned here)."""
    import inspect

    from flliper.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager

    src = inspect.getsource(SchedulerWeightUpdaterManager._pdflip_sleep_flush)
    assert src.index("self._pdflip_sleep_l3_release(sch, tc)") < src.index("return sleep_flush_until_reset(")


# ---------------------------------------------------------------- F2: byteless worker anchors -----------
def _ao_sweep(cache, arena, *, worker):
    with mock.patch("flliper.srt.rank_role.this_rank_is_form_a_worker", lambda: worker), \
            mock.patch.object(UnifiedRadixCache, "_pdflip_mamba_pool", lambda self: arena), \
            mock.patch.object(UnifiedRadixCache, "_mamba_write_through_pin_admissible",
                              lambda self, node, write_back=False: True):
        return cache.publish_unbacked_sweep(max_issue=64)


def test_f2_form_a_worker_without_arena_counts_no_byteless_anchor():
    """RED on e1dea5463a: unbacked=1 refused=1 refused_by={'anchor_only_no_arena': 1} (rc24 TP1/TP2)."""
    cache, node = AO._tree_with_backed_kv_and_device_anchor(l3=True)
    stats = _ao_sweep(cache, None, worker=True)
    assert (stats["unbacked"], stats["issued"], stats["refused"]) == (0, 0, 0), stats
    assert stats["refused_by"] == {}
    assert cache.cache_controller.writes == []


def test_f2_the_host_still_backs_its_anchor():
    cache, node = AO._tree_with_backed_kv_and_device_anchor(l3=True)
    stats = _ao_sweep(cache, AO._Arena(), worker=False)
    assert stats["issued"] == 1 and stats.get("anchor_only") == 1


def test_f2_a_host_without_arena_still_names_its_refusal():
    """Only the byteless worker is exempt: a rank that is no Form A worker keeps the named refusal."""
    cache, node = AO._tree_with_backed_kv_and_device_anchor(l3=True)
    stats = _ao_sweep(cache, None, worker=False)
    assert stats["unbacked"] == 1 and stats["refused_by"] == {"anchor_only_no_arena": 1}


def test_f2_a_worker_with_an_arena_is_not_exempt():
    """Mutant guard: the exemption needs BOTH conditions (worker role AND no mamba arena pool)."""
    cache, node = AO._tree_with_backed_kv_and_device_anchor(l3=True)
    stats = _ao_sweep(cache, AO._Arena(), worker=True)
    assert stats["issued"] == 1


def test_f2_three_trees_cut_differently_host_and_workers():
    """The three Form A trees: TP0 (host, arena) backs the anchor of each KV-backed node, the workers (no arena)
    count nothing un-backed -- whatever their node cut."""
    res = {}
    for r, cuts in enumerate(([64, 192], [], [96])):
        t, ns = _tree(cuts)
        _back_all(ns)
        t.cache_controller = AO._Ctl()
        res[r] = _ao_sweep(t, AO._Arena() if r == 0 else None, worker=r != 0)
    assert res[0]["anchor_only"] == res[0]["issued"] == 3
    assert [(res[r]["unbacked"], res[r]["refused"]) for r in (1, 2)] == [(0, 0), (0, 0)]


# ---------------------------------------------------------------- F3: L3 final drain at SIGTERM ---------
L3 = _load("_s104_l3reuse", os.path.join("..", "mem_cache", "test_l3_reuse_after_restart_0928.py"))


@pytest.fixture
def _gate():
    from flliper.srt.mem_cache import l3_write_behind as gate

    gate._reset_for_tests()
    yield
    gate._reset_for_tests()


@pytest.mark.skipif(L3.shutil.which("gcc") is None, reason="needs gcc")
def test_f3_stop_shape_without_drain_the_restart_reads_nothing(tmp_path, monkeypatch, _gate):
    """The rc22 10:50 shape: pages COMPLETE in L2, no write-behind pass reached them, the boot ends -- the next
    boot's probe finds 0 of them (rc23 ``L3-INDEX-PRESENCE depth=35904`` for a chain D held to 37120)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_WRITE_BEHIND_S", "0")
    be = L3._boot(tmp_path, monkeypatch, "boot1")
    hs = L3._hashes("s")
    L3._prefill(be, hs)
    _be2, res = L3._probe_after_restart(tmp_path, monkeypatch, hs)
    assert res.kv_uncapped == 0


@pytest.mark.skipif(L3.shutil.which("gcc") is None, reason="needs gcc")
def test_f3_final_drain_copies_the_l2_tail_and_the_restart_reads_it(tmp_path, monkeypatch, _gate):
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_WRITE_BEHIND_S", "0")
    be = L3._boot(tmp_path, monkeypatch, "boot1")
    hs = L3._hashes("t")
    L3._prefill(be, hs)
    r = fd.drain_backend(be, time.monotonic() + 5.0)
    assert r["written"] == 2 * L3.PAGES and r["stopped"] == "done", r
    _be2, res = L3._probe_after_restart(tmp_path, monkeypatch, hs)
    assert (res.kv_uncapped, res.kv_hit_pages) == (L3.PAGES, L3.PAGES)


@pytest.mark.skipif(L3.shutil.which("gcc") is None, reason="needs gcc")
def test_f3_drain_ignores_the_byte_budget_of_a_cycle(tmp_path, monkeypatch, _gate):
    """The regular pass stops at its 256 MiB budget per arena (four mamba slots); the final drain is bounded by
    the wall clock only -- a budget of 1 page per cycle still drains everything."""
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_WRITE_BEHIND_S", "0")
    be = L3._boot(tmp_path, monkeypatch, "boot1")
    hs = L3._hashes("u")
    L3._prefill(be, hs)
    monkeypatch.setattr(type(be), "_l3_write_behind_budget", staticmethod(lambda: L3.KV_TOTAL))
    assert be.l3_write_behind_pass()["pending"] > 0, "the regular pass leaves the tail pending"
    r = fd.drain_backend(be, time.monotonic() + 5.0)
    _be2, res = L3._probe_after_restart(tmp_path, monkeypatch, hs)
    assert res.kv_hit_pages == L3.PAGES and r["stopped"] == "done"


class _FakeBackend:
    def __init__(self, cycles):
        self.cycles = list(cycles)   # one dict per pass call
        self.calls = []
        self._l3wb_stop = threading.Event()
        self._l3wb_thread = None

    def l3_write_behind_pass(self, budget_bytes=None, quiet=None, cont=False, slice_s=None):
        self.calls.append(dict(budget=budget_bytes, quiet=quiet(), cont=cont, slice_s=slice_s))
        return self.cycles.pop(0) if self.cycles else {"written": 0}


def test_f3_drain_loop_continues_slices_and_ends_on_an_empty_cycle():
    b = _FakeBackend([{"written": 3, "sliced": True}, {"written": 2}, {"written": 0}])
    r = fd.drain_backend(b, time.monotonic() + 5.0)
    assert r["written"] == 5 and r["passes"] == 3 and r["cycles"] == 2 and r["stopped"] == "done"
    assert [c["cont"] for c in b.calls] == [False, True, False]
    assert all(c["quiet"] is None and c["budget"] >= 1 << 40 for c in b.calls)
    assert b._l3wb_stop.is_set(), "the write-behind thread is stopped first"


def test_f3_drain_respects_its_deadline():
    b = _FakeBackend([{"written": 1}] * 1000)
    t = [0.0]

    def clock():
        t[0] += 0.4
        return t[0]

    r = fd.drain_backend(b, 2.0, clock=clock)
    assert r["stopped"] == "deadline" and r["passes"] < 10


def test_f3_a_busy_write_behind_thread_is_never_doubled():
    b = _FakeBackend([{"written": 1}])
    hold = threading.Event()
    th = threading.Thread(target=hold.wait, args=(5,), daemon=True)
    th.start()
    b._l3wb_thread = th
    r = fd.drain_backend(b, time.monotonic() + 0.3)
    hold.set()
    assert r["stopped"] == "thread_busy" and b.calls == []


_CHILD = textwrap.dedent("""
    import os, sys, threading, time
    sys.path.insert(0, {py!r})
    from flliper.srt.mem_cache import l3_final_drain as fd
    marker = {marker!r}
    class B:
        def __init__(self):
            self._l3wb_stop = threading.Event(); self._l3wb_thread = None; self.file_path = "x"
        def l3_write_behind_pass(self, budget_bytes=None, quiet=None, cont=False, slice_s=None):
            if not os.path.exists(marker):
                open(marker, "w").write("drained"); return {{"written": 1}}
            return {{"written": 0}}
    b = B()
    if {register}:
        fd.register(b)
    assert fd.install() is {register}
    print("ready", flush=True)
    time.sleep(30)
    sys.exit(7)
""")


@pytest.mark.parametrize("register", [True, False])
def test_f3_sigterm_drains_then_ends_by_sigterm(tmp_path, register):
    """A real process: SIGTERM runs the drain (owner) or nothing (non-owner) and the process still ends BY
    SIGTERM (rc -15), as before -- only the owner's L2 tail is copied first."""
    marker = str(tmp_path / "m")
    py = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python"))
    code = _CHILD.format(py=py, marker=marker, register=register)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, env=dict(os.environ))
    assert p.stdout.readline().strip() == b"ready"
    t0 = time.monotonic()
    p.send_signal(signal.SIGTERM)
    rc = p.wait(15)
    assert rc == -signal.SIGTERM
    assert time.monotonic() - t0 < fd.FINAL_DRAIN_S + 2.0
    assert os.path.exists(marker) is register
