# SPDX-License-Identifier: Apache-2.0
"""L15-PARK-AFTER-FLUSH + L15-TREE-CAND-DIAG (desk 2023).

Cause (18:35-18:47Z short run, image l15c): in 7 of 9 D sleeps nothing was parked because
``park_at_release`` ran BEFORE ``_weg2_sleep_flush()`` in the release RPC; on an idle D the front's
/flush_cache answers WEG2-FLUSH-NONBLOCK quiesced (no retain), the retain runs only in the release
flush, so the park found "no manifest on this rank".

Fix: ``SGLANG_WEG2_L15_PARK_AFTER_FLUSH`` (default off). On + L15 master + group D + not dual: the
park runs after the flush, before the kv pause (and not at the old position). Everything else is the
old order byte for byte. The tests run the REAL source of both park sites (sliced out of
weight_updater.py and exec'd with a recorder) so that the mutants -- gate removed, default ON, order
swapped -- change what is under test.

Hermetic, no CUDA. Run (own worktree, capped):
  cd /spinning/wt-27b-l15-parkorder-1005 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_l15_park_order_1005.py
"""
from __future__ import annotations

import inspect
import os
import pathlib
import sys
import textwrap
import types
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest  # noqa: E402

from sglang.srt.weg2 import l15_park, l15_plan, l15_tree_cand  # noqa: E402

ROOT = pathlib.Path(l15_plan.__file__).resolve().parents[1]  # python/sglang/srt
WU = (ROOT / "managers" / "scheduler_components" / "weight_updater.py").read_text()
ENVIRON = (ROOT / "environ.py").read_text()
SCHED = (ROOT / "managers" / "scheduler.py").read_text()

BASE = {
    "SGLANG_WEG2_L15": "1",
    "SGLANG_WEG2_L15_PARK": "1",
    "SGLANG_WEG2_L15_PARK_AFTER_FLUSH": "1",
    "SGLANG_WEG2_GROUP": "D",
}

PRE_START = '        if ("kv_cache" in tags and weg2_memory_saver_on\n'
PRE_END = "        # Task #47 Scheibe 6a"
FLUSH = '            self._weg2_sleep_flush()\n            _kvsub.mark("flush")\n'
POST_START = "            # L15-PARK-AFTER-FLUSH (SGLANG_WEG2_L15_PARK_AFTER_FLUSH, default off,"
POST_END = "            # AH (--p-attn-head-split)"
PAUSE = "            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)\n"
DORMANT = "                scheduler.weg2_dormant = True\n"


# ------------------------------------------------------------------ the predicate

def _pred_matrix(fn):
    bad = []
    if fn(BASE) is not True:
        bad.append("all on -> True")
    for k in ("SGLANG_WEG2_L15", "SGLANG_WEG2_L15_PARK_AFTER_FLUSH", "SGLANG_WEG2_GROUP"):
        env = dict(BASE)
        env.pop(k)
        if fn(env) is not False:
            bad.append("%s missing -> False" % k)
    for sw in ("0", "", "no", "off"):
        if fn(dict(BASE, SGLANG_WEG2_L15_PARK_AFTER_FLUSH=sw)) is not False:
            bad.append("switch %r -> False" % sw)
    if fn(dict(BASE, SGLANG_WEG2_L15="0")) is not False:
        bad.append("master 0 -> False")
    for g in ("P", "", "d?"):
        if fn(dict(BASE, SGLANG_WEG2_GROUP=g)) is not False:
            bad.append("group %r -> False" % g)
    if fn(dict(BASE, SGLANG_WEG2_DUAL_LAYOUT="1")) is not False:
        bad.append("dual layout -> False")
    if fn({}) is not False:
        bad.append("empty env -> False")
    return bad


def _pred_mutant(old, new):
    # mutate ONLY park_after_flush_active (noparkinit_apply carries the same group/dual lines)
    fn_src = inspect.getsource(l15_plan.park_after_flush_active)
    assert fn_src.count(old) == 1, old
    src = inspect.getsource(l15_plan)
    assert src.count(fn_src) == 1
    src = src.replace(fn_src, fn_src.replace(old, new))
    old = new = ""
    mod = types.ModuleType("l15_plan_mutant")
    mod.__file__ = l15_plan.__file__
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(src, "l15_plan_mutant.py", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    return mod.park_after_flush_active


def test_predicate_matrix_real():
    assert _pred_matrix(l15_plan.park_after_flush_active) == []


PRED_MUTANTS = {
    "gate master removed": ("        master_on(env)\n        and _switch(env, PARK_AFTER_FLUSH_ENV)",
                            "        _switch(env, PARK_AFTER_FLUSH_ENV)"),
    "gate group D removed": ('        and group == "D"\n', ""),
    "gate dual removed": ('        and dual != "1"\n', ""),
    "default ON": ("_switch(env, PARK_AFTER_FLUSH_ENV)",
                   "(str(env.get(PARK_AFTER_FLUSH_ENV, '1')).strip().lower() in _ON_VALUES)"),
    "switch ignored": ("        and _switch(env, PARK_AFTER_FLUSH_ENV)\n", ""),
}


@pytest.mark.parametrize("name", sorted(PRED_MUTANTS))
def test_predicate_mutant_goes_red(name):
    old, new = PRED_MUTANTS[name]
    assert _pred_matrix(_pred_mutant(old, new)), "mutant %r survived" % name


def test_env_defaults_off_and_declared():
    from sglang.srt.environ import envs

    assert "SGLANG_WEG2_L15_PARK_AFTER_FLUSH = EnvBool(False)" in ENVIRON
    assert "SGLANG_WEG2_L15_TREE_CAND_DIAG = EnvBool(False)" in ENVIRON
    assert l15_plan.PARK_AFTER_FLUSH_ENV == "SGLANG_WEG2_L15_PARK_AFTER_FLUSH"
    assert l15_tree_cand.TREE_CAND_DIAG_ENV == "SGLANG_WEG2_L15_TREE_CAND_DIAG"
    saved = {k: os.environ.pop(k, None) for k in (
        "SGLANG_WEG2_L15_PARK_AFTER_FLUSH", "SGLANG_WEG2_L15_TREE_CAND_DIAG")}
    try:
        assert envs.SGLANG_WEG2_L15_PARK_AFTER_FLUSH.get() is False
        assert envs.SGLANG_WEG2_L15_TREE_CAND_DIAG.get() is False
        assert l15_tree_cand.diag_on({}) is False
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ------------------------------------------------------------------ the two sites, executed

def _slice(src, a, b):
    i = src.index(a)
    j = src.index(b, i)
    return src[i:j]


def _compile_sites(src):
    pre = textwrap.dedent(_slice(src, PRE_START, PRE_END))
    post = textwrap.dedent(_slice(src, POST_START, POST_END))
    return compile(pre, "pre_site", "exec"), compile(post, "post_site", "exec")


def _run_sequence(src, env, *, group="D", saver_on=True, boom=False):
    """pre site -> flush -> post site, over the real source text. Returns the call order."""
    pre, post = _compile_sites(src)
    calls = []

    def fake_park(sched, e, log):
        calls.append("park")
        if boom:
            raise RuntimeError("boom")
        return 1

    real = l15_park.park_at_release
    l15_park.park_at_release = fake_park
    try:
        logger = SimpleNamespace(info=lambda *a, **k: None,
                                 warning=lambda *a, **k: calls.append("warn"))
        ns = {
            "os": SimpleNamespace(environ=env),
            "logger": logger,
            "tags": ["kv_cache"],
            "weg2_memory_saver_on": saver_on,
            "self": SimpleNamespace(scheduler=object(), _weg2_group_name=lambda: group),
        }
        exec(pre, ns)
        calls.append("flush")
        exec(post, ns)
    finally:
        l15_park.park_at_release = real
    return calls


def _site_matrix(src):
    bad = []

    def chk(label, got, want):
        if got != want:
            bad.append("%s: %s != %s" % (label, got, want))

    # switch OFF = the old order (park, then flush); the new site never runs
    for sw in (None, "0", ""):
        env = dict(BASE)
        if sw is None:
            env.pop("SGLANG_WEG2_L15_PARK_AFTER_FLUSH")
        else:
            env["SGLANG_WEG2_L15_PARK_AFTER_FLUSH"] = sw
        chk("off %r" % (sw,), _run_sequence(src, env), ["park", "flush"])
    # ON + master + group D (+ park on) = after the flush, exactly once
    chk("on", _run_sequence(src, dict(BASE)), ["flush", "park"])
    chk("on, park via POOL", _run_sequence(
        src, {k: v for k, v in BASE.items() if k != "SGLANG_WEG2_L15_PARK"}
        | {"SGLANG_WEG2_L15_POOL": "1"}), ["flush", "park"])
    # dual layout / group P / no master: the old order (or nothing), never the new position
    chk("dual", _run_sequence(src, dict(BASE, SGLANG_WEG2_DUAL_LAYOUT="1")),
        ["park", "flush"])
    chk("group P", _run_sequence(src, dict(BASE, SGLANG_WEG2_GROUP="P"), group="P"), ["flush"])
    chk("no master", _run_sequence(src, dict(BASE, SGLANG_WEG2_L15="0")), ["flush"])
    chk("master absent", _run_sequence(
        src, {k: v for k, v in BASE.items() if k != "SGLANG_WEG2_L15"}), ["flush"])
    # the park itself off: nothing at either position
    chk("park off", _run_sequence(
        src, {k: v for k, v in BASE.items() if k != "SGLANG_WEG2_L15_PARK"}), ["flush"])
    chk("saver off (pre)", _run_sequence(
        src, {k: v for k, v in BASE.items() if k != "SGLANG_WEG2_L15_PARK_AFTER_FLUSH"},
        saver_on=False), ["flush"])
    chk("saver off (post)", _run_sequence(src, dict(BASE), saver_on=False), ["flush"])
    # a failing park is logged, never raised, in both positions
    chk("boom on", _run_sequence(src, dict(BASE), boom=True), ["flush", "park", "warn"])
    chk("boom off", _run_sequence(
        src, dict(BASE, SGLANG_WEG2_L15_PARK_AFTER_FLUSH="0"), boom=True),
        ["park", "warn", "flush"])
    return bad


def test_sites_matrix_real_source():
    assert _site_matrix(WU) == []


def test_switch_off_old_order_is_the_literal_old_code():
    """The `pre` site differs from the pre-change code ONLY by the extra `and not
    park_after_flush_active` conjunct: same gate, same call, same warning."""
    pre = _slice(WU, PRE_START, PRE_END)
    assert pre.count("_l15_pk.park_at_release(self.scheduler, os.environ, logger.info)") == 1
    assert '"L15-PARK at=sleep failed (%s: %s)"' in pre
    assert "_l15_pl2.master_on(os.environ) and _l15_pk.park_on(os.environ)" in pre
    assert "not _l15_pl2.park_after_flush_active(os.environ)" in pre


# ------------------------------------------------------------------ position in the release RPC

def _order_problems(src):
    bad = []
    pre_i = src.index(PRE_START)
    fl_i = src.index(FLUSH, pre_i)
    post_i = src.index(POST_START)
    pause_i = src.index(PAUSE, pre_i)
    dorm_i = src.index(DORMANT, pre_i)
    if not pre_i < fl_i:
        bad.append("pre site must precede the flush")
    if not fl_i < post_i:
        bad.append("post site must follow the flush")
    if not post_i < pause_i:
        bad.append("post site must precede the kv pause")
    if not pause_i < dorm_i:
        bad.append("kv pause precedes the dormant marker")
    # exactly two park calls in the whole file's release RPC area, one per site
    seg = src[pre_i:dorm_i]
    if seg.count("park_at_release(") != 2:
        bad.append("expected exactly 2 park_at_release calls, got %d" % seg.count("park_at_release("))
    # nothing between the post site and the pause touches the pools' mapping
    mid = src[post_i:pause_i]
    for forbidden in ("flush_cache(", ".resume(", "offload_tags"):
        if forbidden in mid:
            bad.append("%s between post site and pause" % forbidden)
    return bad


def test_order_real_source():
    assert _order_problems(WU) == []


def test_order_mutant_post_site_before_flush_goes_red():
    moved = WU.replace(FLUSH, "", 1)
    k = moved.index(POST_END)
    moved = moved[:k] + FLUSH + moved[k:]  # flush now AFTER the post site
    assert _order_problems(moved), "swap survived"


def test_order_mutant_post_site_after_pause_goes_red():
    post = _slice(WU, POST_START, POST_END)
    moved = WU.replace(post, "", 1)
    k = moved.index(PAUSE) + len(PAUSE)
    moved = moved[:k] + post + moved[k:]
    assert _order_problems(moved)


# ------------------------------------------------------------------ site mutants through the matrix

SITE_MUTANTS = {
    # pre site keeps parking although the switch moved it: parks twice
    "pre gate (not active) removed": (
        "\n                        and not _l15_pl2.park_after_flush_active(os.environ)):", "):"),
    # post site parks although the switch is off
    "post gate (active) removed": (
        "if _l15_pl2.park_after_flush_active(os.environ) and _l15_pk.park_on(\n"
        "                            os.environ):",
        "if _l15_pk.park_on(\n                            os.environ):"),
    # post site parks without the park switch
    "post park_on removed": (
        "if _l15_pl2.park_after_flush_active(os.environ) and _l15_pk.park_on(\n"
        "                            os.environ):",
        "if _l15_pl2.park_after_flush_active(os.environ):"),
    # post site forgets the saver gate
    "post saver gate removed": (
        '            if weg2_memory_saver_on and self._weg2_group_name() == "D":\n                try:\n'
        "                    from sglang.srt.weg2 import l15_park as _l15_pk\n"
        "                    from sglang.srt.weg2 import l15_plan as _l15_pl2\n",
        "            if True:\n                try:\n"
        "                    from sglang.srt.weg2 import l15_park as _l15_pk\n"
        "                    from sglang.srt.weg2 import l15_plan as _l15_pl2\n"),
    # post site swallows nothing: exception escapes
    "post try removed": (
        "                except Exception as exc:  # noqa: BLE001 -- the wake refills from L2\n"
        '                    logger.warning("L15-PARK at=sleep failed (%s: %s)",\n'
        "                                   type(exc).__name__, exc)\n"
        "            # AH",
        "                except ZeroDivisionError as exc:\n"
        '                    logger.warning("L15-PARK at=sleep failed (%s: %s)",\n'
        "                                   type(exc).__name__, exc)\n"
        "            # AH"),
}


@pytest.mark.parametrize("name", sorted(SITE_MUTANTS))
def test_site_mutant_goes_red(name):
    old, new = SITE_MUTANTS[name]
    assert WU.count(old) == 1, "mutation anchor moved: %r" % name
    mutated = WU.replace(old, new)
    try:
        bad = _site_matrix(mutated)
    except Exception:  # noqa: BLE001 -- a mutant that blows up the harness is red as well
        return
    assert bad, "mutant %r survived the matrix" % name


# ------------------------------------------------------------------ TREE-CAND-DIAG

from test_weg2_l15_tree_cand_1003 import FULL, MAMBA, N, _tree  # noqa: E402


def _walk(tree, require_l2=False, env=None, other_vote=None):
    """build() with two ranks: this one and a peer whose vote is ``other_vote`` (digests)."""
    logs = []

    def gather(obj):
        return [obj, list(other_vote or [])]

    out = l15_tree_cand.build(tree, gather, 0, env or {}, logs.append,
                              require_l2=require_l2, rank=2)
    return out, logs


def test_loss_census_buckets():
    tree, (a, b, c, d) = _tree()  # anchors on B, C, D, nothing on host
    cen = l15_tree_cand.loss_census(tree)
    assert cen["tips"] == 3 and cen["no_mamba_host"] == 3 and cen["l2_ok"] == 0
    assert cen["kept"] == 3  # no require_l2: every tip that digests is voted
    assert l15_tree_cand.loss_census(tree, require_l2=True)["kept"] == 0

    tree, (a, b, c, d) = _tree(host_mamba=True)  # mamba host rows, KV host rows missing
    cen = l15_tree_cand.loss_census(tree, require_l2=True)
    assert (cen["no_mamba_host"], cen["no_kv_host"], cen["l2_ok"], cen["kept"]) == (0, 3, 0, 0)
    assert cen["kv_pending"] == 0

    a.host_ref_counter = 1  # a write-through still pins the shared prefix node A (in every chain)
    cen = l15_tree_cand.loss_census(tree, require_l2=True)
    assert cen["no_kv_host"] == 3 and cen["kv_pending"] == 2  # B and C sit below A, D does not
    d.host_ref_counter = 1
    assert l15_tree_cand.loss_census(tree, require_l2=True)["kv_pending"] == 3

    tree, (a, b, c, d) = _tree(host_mamba=True, host_kv=True)  # fully backed
    cen = l15_tree_cand.loss_census(tree, require_l2=True)
    assert (cen["tips"], cen["l2_ok"], cen["kept"]) == (3, 3, 3)
    # a recorded L2 shadow counts as host-backed
    tree, (a, b, c, d) = _tree(host_mamba=True)
    for n in (a, b, c, d):
        n._weg2_l2_shadow = ([1], [2])
    assert l15_tree_cand.loss_census(tree, require_l2=True)["l2_ok"] == 3


def test_loss_census_where_in_the_chain():
    """Shared head lost (the split-parent hypothesis): A (2 tokens) has neither host value
    nor shadow, B/C below it are backed -> every tip below A loses it at offset 2."""
    tree, (a, b, c, d) = _tree(host_mamba=True, host_kv=True)
    a.component_data[FULL].host_value = None
    cen = l15_tree_cand.loss_census(tree, require_l2=True)
    assert (cen["no_kv_host"], cen["l2_ok"], cen["kept"]) == (2, 1, 1)  # B, C lost, D fine
    assert cen["first_miss_tok"] == [2] and cen["head_miss"] == 2
    # a shadow of the wrong length is flagged (l15_bind would refuse it), not a l2_backed loss
    tree, (a, b, c, d) = _tree(host_mamba=True, host_kv=True)
    b._weg2_l2_shadow = ([1, 2, 3], [0, 0, 0])  # B has 2 tokens
    cen = l15_tree_cand.loss_census(tree, require_l2=True)
    assert cen["shadow_len_mismatch"] == 1 and cen["l2_ok"] == 3
    # first_miss_tok is deep when only the tip is missing: A 2 + B 2 = 4 for tip B
    tree, (a, b, c, d) = _tree(host_mamba=True, host_kv=True)
    b.component_data[FULL].host_value = None
    cen = l15_tree_cand.loss_census(tree, require_l2=True)
    assert cen["first_miss_tok"] == [4] and cen["head_miss"] == 1
    line = l15_tree_cand.loss_line(1, cen, 0, 0, True)
    assert line.endswith("head_miss=1 shadow_len_mismatch=0 tip_miss=1 anc_miss=0 first_miss_tok=4")


def test_diag_line_only_when_switch_on_and_walk_ends_empty():
    tree, _ = _tree()  # three tips, peer votes nothing -> agreed=0
    # switch off (default): no LOSS line, same result
    out_off, logs_off = _walk(tree)
    assert not [x for x in logs_off if "TREE-CAND-LOSS" in x]
    # switch on: exactly one LOSS line naming the rank and the counts
    out_on, logs_on = _walk(tree, env={"SGLANG_WEG2_L15_TREE_CAND_DIAG": "1"})
    loss = [x for x in logs_on if "L15-TREE-CAND-LOSS" in x]
    assert len(loss) == 1
    assert loss[0].startswith("L15-TREE-CAND-LOSS rank=2 local=3 agreed=0 require_l2=0 tips=3 "
                              "no_mamba_host=3 ")
    assert loss[0].endswith("head_miss=0 shadow_len_mismatch=0 tip_miss=0 anc_miss=0 first_miss_tok=-")
    # log-only: the result and every other line are identical
    assert out_on == out_off == []
    assert [x for x in logs_on if "TREE-CAND-LOSS" not in x] == logs_off
    # cap-0 rank (require_l2) with nothing backed: local=0
    _, logs = _walk(tree, require_l2=True, env={"SGLANG_WEG2_L15_TREE_CAND_DIAG": "on"})
    loss = [x for x in logs if "L15-TREE-CAND-LOSS" in x]
    assert len(loss) == 1 and "local=0 agreed=0 require_l2=1 tips=3" in loss[0]


def test_diag_silent_on_a_healthy_walk():
    tree, _ = _tree(host_mamba=True, host_kv=True)
    mine = [(c.digest, c.n_tokens) for c in l15_tree_cand.local_candidates(
        tree, None, lazy_tokens=True)]
    out, logs = _walk(tree, env={"SGLANG_WEG2_L15_TREE_CAND_DIAG": "1"}, other_vote=mine)
    assert out and not [x for x in logs if "TREE-CAND-LOSS" in x]


def test_diag_failure_never_escapes():
    tree, _ = _tree()
    logs = []
    orig = l15_tree_cand.loss_census
    l15_tree_cand.loss_census = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    try:
        out = l15_tree_cand.build(tree, lambda o: [o, []], 0,
                                  {"SGLANG_WEG2_L15_TREE_CAND_DIAG": "1"}, logs.append, rank=0)
    finally:
        l15_tree_cand.loss_census = orig
    assert out == [] and any("TREE-CAND-LOSS failed" in x for x in logs)


def test_scheduler_passes_the_rank_to_build():
    i = SCHED.index("l15_tree_cand.build(")
    assert 'rank=int(getattr(getattr(self, "ps", None),' in SCHED[i:i + 1800]
