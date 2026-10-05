"""#968 RANK-TRACE (desk 1951): pure instrumentation, ENV SGLANG_WEG2_968_RANK_TRACE,
default OFF, dual group P only.

Pinned:
  * the gate: off by default, off outside dual P, on only with the switch AND dual P;
  * the line shape (one key=value line, fixed key order, full rid, pp label);
  * every decision site writes its line when armed and NOTHING when not;
  * NO BEHAVIOUR CHANGE: the same scenario armed and unarmed ends the same way
    (return value / exception text / request state);
  * no collective, no torch.distributed call, no exception out of the instrument.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

import sglang.srt.managers.pp_admission_congruence as congruence  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.managers.pp_admission_congruence import (  # noqa: E402
    execute_scheduled_prefix,
    state_aligned_load_back_len,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
)
from sglang.srt.weg2 import rank_trace_968 as rt  # noqa: E402

DECISION = 49152
DEEP = 53248


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def lines():
    h = _Lines()
    lg = logging.getLogger(rt.__name__)
    old = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    yield h.lines
    lg.removeHandler(h)
    lg.setLevel(old)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in (rt.ENV, "SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)
    rt._once_seen.clear()
    monkeypatch.setattr(congruence, "MATERIALISE_BASE_S", 0.0)
    monkeypatch.setattr(congruence, "MATERIALISE_MAX_S", 0.05)


def _arm(monkeypatch):
    monkeypatch.setenv(rt.ENV, "1")
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")


def _rt_lines(lines, site=None):
    out = [x for x in lines if x.startswith(rt.MARKER)]
    if site:
        out = [x for x in out if (" site=%s " % site) in x]
    return out


# ---------------------------------------------------------------- the gate
def test_default_off(lines):
    assert rt.armed() is False
    assert rt.emit("told_verdict", "weg2-0-4", told=1) is False
    assert lines == []


def test_switch_alone_is_not_enough(monkeypatch):
    monkeypatch.setenv(rt.ENV, "1")
    assert rt.armed() is False  # not the dual layout
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    assert rt.armed() is False  # dual, but group D


def test_dual_p_alone_is_not_enough(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert rt.armed() is False  # default OFF even in dual P
    monkeypatch.setenv(rt.ENV, "0")
    assert rt.armed() is False


def test_armed_in_dual_p(monkeypatch):
    _arm(monkeypatch)
    assert rt.armed() is True


# --------------------------------------------------------------- line shape
def test_line_shape(monkeypatch, lines):
    _arm(monkeypatch)
    sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=1))
    assert rt.emit(
        "state_align", "weg2-0-4", told=DEEP, matched_prefix_len=DEEP,
        resident_rows=DECISION, src="radix", decision="extent=49152",
        scheduler=sched, anchor_depth=DEEP,
    )
    (ln,) = lines
    assert ln == (
        "#968-RT site=state_align pp=1 rid=weg2-0-4 told=53248 matched_prefix_len=53248 "
        "resident_rows=49152 src=radix decision=extent=49152 anchor_depth=53248"
    )


def test_missing_fields_are_dashes_and_rid_is_full(monkeypatch, lines):
    _arm(monkeypatch)
    rt.emit("prefix_exec", "weg2-0-4-with-a-long-rid-0123456789")
    (ln,) = lines
    assert "rid=weg2-0-4-with-a-long-rid-0123456789 told=- matched_prefix_len=- resident_rows=- src=- decision=-" in ln


def test_pp_label_without_group_is_a_question_mark_not_an_error():
    assert rt.pp_label(None) == "?"
    assert rt.pp_label(SimpleNamespace(ps=None)) == "?"


def test_emit_never_raises(monkeypatch):
    _arm(monkeypatch)

    class Bad:
        def __str__(self):
            raise RuntimeError("boom")

    assert rt.emit("told_verdict", Bad()) is False
    assert rt.resident_rows_of(object()) == 0
    assert rt.node_depth_rel(object(), object()) is None
    assert rt.mamba_flags(object()) == "na"


def test_no_collective_is_ever_reached(monkeypatch, lines):
    """pp label and emit never touch torch.distributed."""
    _arm(monkeypatch)
    import torch.distributed as dist

    def _boom(*a, **k):
        raise AssertionError("collective / dist call from the instrument")

    for name in ("all_reduce", "broadcast", "barrier", "all_gather", "get_rank",
                 "get_world_size", "is_initialized"):
        monkeypatch.setattr(dist, name, _boom, raising=False)
    assert rt.emit("told_verdict", "r", scheduler=None, told=1)
    assert rt.emit("told_verdict", "r", scheduler=SimpleNamespace(ps=SimpleNamespace(pp_rank=2)))
    assert "pp=?" in lines[0] and "pp=2" in lines[1]


def test_once_is_a_distinct_fact_filter(monkeypatch):
    assert rt.once("a", 1) is False  # off
    _arm(monkeypatch)
    assert rt.once("a", 1) is True
    assert rt.once("a", 1) is False
    assert rt.once("a", 2) is True  # changed fact = new line


# ------------------------------------------------------------ site: told verdict
def _sched_pp(pp_rank=0, pp_size=3):
    return SimpleNamespace(ps=SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size))


def test_site_told_verdict_fallback_and_acked(monkeypatch, lines):
    _arm(monkeypatch)
    s = _sched_pp(0)
    fb.pp0_note_verdict(s, "weg2-0-3", 51788, 0, fb.REASON_FRIST, now=10.5, published_at=10.0)
    fb.pp0_note_verdict(s, "weg2-0-4", 53248, 53248, fb.REASON_ACKS, now=10.5, published_at=10.0)
    got = _rt_lines(lines, "told_verdict")
    assert len(got) == 2
    assert "pp=0 rid=weg2-0-3 told=51788 matched_prefix_len=0" in got[0]
    assert "decision=%s" % fb.REASON_FRIST in got[0]
    assert "pp=0 rid=weg2-0-4 told=53248 matched_prefix_len=53248" in got[1]
    assert "decision=%s" % fb.REASON_ACKS in got[1]


def test_site_told_verdict_silent_when_off(lines):
    fb.pp0_note_verdict(_sched_pp(0), "weg2-0-3", 51788, 0, fb.REASON_FRIST, now=10.5, published_at=10.0)
    assert _rt_lines(lines) == []


# ------------------------------------------------------------ site: told release
class _Tree:
    def __init__(self, completed=None):
        self.completed = completed or {}
        self.released = []

    def completed_prefetch_tokens(self, rid):
        return self.completed.get(rid)

    def release_aborted_request(self, rid):
        self.released.append(rid)


def test_site_told_release_logs_the_released_read_and_changes_nothing(monkeypatch, lines):
    _arm(monkeypatch)
    tree = _Tree({"weg2-0-3": 0})
    s = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=3), tree_cache=tree)
    s._weg2_store_told_satisfied = {"weg2-0-3": 51788}
    fb.follower_release(s, "weg2-0-3")
    got = _rt_lines(lines, "told_release")
    assert len(got) == 1
    assert "pp=1 rid=weg2-0-3 told=51788 matched_prefix_len=0" in got[0]
    assert "src=store_read" in got[0] and "decision=fallback_release" in got[0]
    assert tree.released == ["weg2-0-3"]  # behaviour untouched
    assert "weg2-0-3" not in s._weg2_store_told_satisfied


def test_site_told_release_silent_when_off(lines):
    tree = _Tree()
    s = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=3), tree_cache=tree)
    fb.follower_release(s, "weg2-0-3")
    assert _rt_lines(lines) == [] and tree.released == ["weg2-0-3"]


# ------------------------------------------------------------ site: state_align
def _req_sa(kv=4096, anchor=DEEP, device=0, key=DEEP):
    return SimpleNamespace(
        rid="weg2-0-4", host_hit_length=kv, state_anchor_depth=anchor,
        prefix_indices=torch.arange(device), key_match_depth=key,
    )


def test_site_state_align_same_value_armed_and_not(monkeypatch, lines):
    off = state_aligned_load_back_len(_req_sa())
    assert _rt_lines(lines) == []
    _arm(monkeypatch)
    on = state_aligned_load_back_len(_req_sa())
    assert on == off
    (ln,) = _rt_lines(lines, "state_align")
    assert "rid=weg2-0-4" in ln and "src=radix" in ln
    assert "matched_prefix_len=%d" % DEEP in ln
    assert "anchor_depth=%d" % DEEP in ln and "extent=" in ln
    # the very same facts again: one line per distinct fact
    state_aligned_load_back_len(_req_sa())
    assert len(_rt_lines(lines, "state_align")) == 1


def test_site_state_align_without_host_hit_stays_silent(monkeypatch, lines):
    _arm(monkeypatch)
    assert state_aligned_load_back_len(_req_sa(kv=0)) is None
    assert _rt_lines(lines) == []


# ------------------------------------------------------------ site: loadback #988
def test_site_loadback(monkeypatch, lines):
    from sglang.srt.managers import schedule_policy as sp

    req = SimpleNamespace(
        rid="weg2-0-4", mamba_loadback_anchor_adopted=True, state_anchor_depth=DECISION,
        pp_load_back_extent=DECISION, prefix_indices=torch.arange(DECISION), host_hit_length=0,
    )
    sp._note_988_loadback(req, DECISION)
    assert _rt_lines(lines) == []
    _arm(monkeypatch)
    sp._note_988_loadback(req, DECISION)
    (ln,) = _rt_lines(lines, "loadback")
    assert "rid=weg2-0-4" in ln and "matched_prefix_len=%d" % DECISION in ln
    assert "mamba_adopted=1" in ln and "anchor_depth=%d" % DECISION in ln


# ------------------------------------------------------------ site: anchor hold take
class _N:
    def __init__(self, parent, n, host=True, device=False, nid=1):
        self.id, self.parent, self.key = nid, parent, list(range(n))
        self.component_data = {
            ComponentType.MAMBA: SimpleNamespace(
                value=object() if device else None,
                host_value=object() if host else None,
            )
        }


def test_site_anchor_hold_take(monkeypatch, lines):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U

    root = _N(None, 0, nid=0)
    a = _N(root, 49152, nid=31)
    hold = SimpleNamespace(
        note_take=lambda *x: None, told_depths=lambda: {49152: ["weg2-0-4"]},
        armed=lambda: True,
    )
    fake = SimpleNamespace(root_node=root, _weg2_told_hold=hold)
    fake._weg2_node_end_depth = lambda n: U._weg2_node_end_depth(fake, n)
    U._weg2_told_note_take(fake, "INNER", a)
    assert _rt_lines(lines) == []
    _arm(monkeypatch)
    U._weg2_told_note_take(fake, "INNER", a)
    (ln,) = _rt_lines(lines, "anchor_hold_take")
    assert "decision=TAKE-INNER" in ln and "node=31" in ln and "depth=49152" in ln
    assert "mamba=host=1,device=0" in ln and "standing_told=1" in ln and "told=49152" in ln
    assert "rid=weg2-0-4" in ln


# ------------------------------------------------------------ site: prefix_exec
class _PTree:
    enable_storage = True

    def __init__(self):
        self.loaded_from = []

    def init_load_back(self, params):
        node, req = params.best_match_node, params.req
        self.loaded_from.append(node)
        n, cur = 0, node
        while cur is not req.last_node:
            n += len(cur.key)
            cur = cur.parent
        comp = node.component_data[ComponentType.MAMBA]
        if comp.host_value is not None and comp.value is None:
            req.mamba_loadback_anchor_adopted = True
        return torch.arange(n), node


def _follower():
    root = _N(None, 0, nid=10)
    a = _N(root, 45056, nid=11)
    b = _N(a, DECISION - 45056, nid=12)
    c = _N(b, DEEP - DECISION, nid=13)
    req = SimpleNamespace(
        rid="weg2-0-4", prefix_indices=torch.arange(0), cache_protected_len=0,
        host_hit_length=DECISION, best_match_node=c, last_node=root,
        mamba_loadback_anchor_adopted=False, full_untruncated_fill_ids=list(range(53890)),
    )
    return req, c


def _run_shortfall():
    req, c = _follower()
    tree = _PTree()
    try:
        execute_scheduled_prefix(req, tree, DECISION)
    except RuntimeError as exc:
        # the elapsed seconds differ run to run
        import re

        return ("died", re.sub(r"\d+\.\d+s", "Xs", str(exc)), len(tree.loaded_from), len(req.prefix_indices))
    return ("ok", len(tree.loaded_from), len(req.prefix_indices))


def test_site_prefix_exec_shortfall_logged_and_behaviour_identical(monkeypatch, lines):
    off = _run_shortfall()
    assert off[0] == "died" and _rt_lines(lines) == []
    _arm(monkeypatch)
    on = _run_shortfall()
    assert on == off  # the instrument changed nothing
    got = _rt_lines(lines, "prefix_exec")
    decisions = [x.split("decision=")[1].split(" ")[0] for x in got]
    assert decisions[0] == "materialise_enter"
    assert "loadback" in decisions and decisions[-1] == "SHORTFALL"
    lb = next(x for x in got if "decision=loadback" in x)
    assert "best_node=13" in lb and "best_rel=%d" % DEEP in lb
    assert "scheduled=%d" % DECISION in lb and "applied=%d" % DEEP in lb and "adopted=1" in lb
    assert "mamba=host=1,device=0" in lb
    # same facts each poll: not one line per poll
    assert decisions.count("loadback") == 1


def test_site_prefix_exec_noop_and_truncate_and_materialised(monkeypatch, lines):
    _arm(monkeypatch)
    req = SimpleNamespace(rid="r1", prefix_indices=torch.arange(100), cache_protected_len=100,
                          host_hit_length=0)
    assert execute_scheduled_prefix(req, None, 100) == 0
    assert execute_scheduled_prefix(req, None, 64) == 0
    assert len(req.prefix_indices) == 64
    d = [x.split("decision=")[1].split(" ")[0] for x in _rt_lines(lines, "prefix_exec")]
    assert d == ["noop", "truncate"]
    # a clean materialisation
    root = _N(None, 0, nid=20)
    b = _N(root, DECISION, nid=21)
    req2 = SimpleNamespace(
        rid="r2", prefix_indices=torch.arange(0), cache_protected_len=0,
        host_hit_length=DECISION, best_match_node=b, last_node=root,
        mamba_loadback_anchor_adopted=False, full_untruncated_fill_ids=list(range(60000)),
    )
    assert execute_scheduled_prefix(req2, _PTree(), DECISION) == DECISION
    d2 = [x.split("decision=")[1].split(" ")[0] for x in _rt_lines(lines, "prefix_exec")]
    assert d2[-1] == "materialised"


# ------------------------------------------------------------ site: told admission
def test_site_told_admission_wired_before_the_comparison():
    """admission() needs a full scheduler; pin the wiring in the source: the
    emit stands before `if own != told`, in the satisfied branch and no
    return value is taken from it."""
    import inspect

    from sglang.srt.managers import weg2_store_told as st

    import re

    src = inspect.getsource(st.admission)
    pat = re.compile(r'_rt968\.emit\(\s*"told_admission"')
    marks = [m.start() for m in pat.finditer(src)]
    assert len(marks) == 2  # satisfied branch + comparison
    assert marks[1] < src.index("if own != told:")
    assert "= _rt968" not in src  # nothing is taken from the instrument


def test_every_site_is_wired_exactly_where_the_docstring_says():
    import inspect

    from sglang.srt.managers import schedule_policy as sp
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U

    wired = {
        "told_verdict": inspect.getsource(fb.pp0_note_verdict),
        "told_release": inspect.getsource(fb.follower_release),
        "state_align": inspect.getsource(state_aligned_load_back_len),
        "loadback": inspect.getsource(sp._note_988_loadback),
        "anchor_hold_take": inspect.getsource(U._weg2_told_note_take),
        "prefix_exec": inspect.getsource(execute_scheduled_prefix),
    }
    for site, src in wired.items():
        assert ('"%s"' % site) in src, site
    assert set(wired) | {"told_admission"} == set(rt.SITES)
