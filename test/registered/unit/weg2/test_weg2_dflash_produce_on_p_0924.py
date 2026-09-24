"""--dflash-produce-on-p (user decision 2026-09-24: "P ohne draft rechnen").

Under --spec-form DFLASH group P keeps LOADING the DFlash2 draft on its last
PP stage (cold-resident bytes: same VRAM, planner cut, census, flip) but
COMPUTES nothing with it. Hermetic, CPU. Pins:

Launcher
  * the switch defaults to ``off`` and reaches group P ONLY through its
    environment: ``spec_form_env("P") == {SGLANG_WEG2_DFLASH_PRODUCE: "0"}``
    by default, ``"1"`` under ``on``; NEXTN and group D are untouched;
  * ``argv_p`` and the P FORM key are byte-identical between on and off;
  * the launcher names the form in one line.
Rank
  * ``Scheduler._draft_kv_producer_wants`` is False for the DFlash producer
    under ``=0`` (so no FULL capture, no ``produce()``, no
    ``publish_draft_rows_direct``), True under ``=1`` and when unset, and
    the NEXTN producer is not affected;
  * ``produce()`` has exactly one call site and it sits behind that answer;
  * the target ``ModelRunner`` arms no DFlash aux capture on a draft-KV-only
    target under ``=0`` (no ``set_dflash_layers_to_capture`` marks on any PP
    stage), and a proposing target (group D) always captures.
"""

import ast
import inspect
import json
import textwrap
from types import SimpleNamespace

import pytest

import sglang.srt.weg2.launcher as L
from sglang.srt.speculative.dflash_draft_kv_producer import (
    DFLASH_PRODUCE_ENV,
    dflash_produce_on_p,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

ENV = "SGLANG_WEG2_DFLASH_PRODUCE"


@pytest.fixture
def restore_form():
    saved = dict(L._SPEC_FORM)
    try:
        yield
    finally:
        L._SPEC_FORM.clear()
        L._SPEC_FORM.update(saved)


def _dflash(tmp_path, **kw):
    ns = SimpleNamespace(
        spec_form="DFLASH", dflash_draft_path=str(tmp_path), dflash_block=8,
        dflash_window=2048, **kw,
    )
    L.apply_spec_form(ns)


# ----------------------------------------------------------------- launcher


def test_env_name_is_one_name():
    assert L.DFLASH_PRODUCE_ENV == DFLASH_PRODUCE_ENV == ENV


def test_cli_default_is_off():
    ap = L.build_parser()
    assert ap.get_default("dflash_produce_on_p") == "off" == L.DFLASH_PRODUCE_ON_P_DEFAULT
    action = next(a for a in ap._actions if a.dest == "dflash_produce_on_p")
    assert list(action.choices) == ["off", "on"]


def test_p_env_carries_the_switch_by_default(restore_form, tmp_path):
    _dflash(tmp_path)  # no dflash_produce_on_p attribute at all -> default
    assert L.spec_form_env("P") == {ENV: "0"}
    _dflash(tmp_path, dflash_produce_on_p="off")
    assert L.spec_form_env("P") == {ENV: "0"}
    _dflash(tmp_path, dflash_produce_on_p="on")
    assert L.spec_form_env("P") == {ENV: "1"}


def test_d_env_and_nextn_are_untouched(restore_form, tmp_path):
    for value in ("off", "on"):
        _dflash(tmp_path, dflash_produce_on_p=value)
        assert L.spec_form_env("D") == {"SGLANG_DFLASH_WINDOW_POOL": "1"}
        assert ENV not in L.spec_form_env("D")
    L.apply_spec_form(SimpleNamespace(spec_form="NEXTN", dflash_produce_on_p="off"))
    assert L.spec_form_env("P") == {} and L.spec_form_env("D") == {}


def test_build_env_hands_the_switch_to_p_only(restore_form, tmp_path):
    _dflash(tmp_path)
    src = inspect.getsource(L.build_env)
    assert "env.update(spec_form_env(group))" in src


STORE_CFG = json.dumps(
    {"max_size": str(30 * 1024 ** 3), "min_free_space": str(32 * 1024 ** 3),
     "max_size_scope": "shared"},
    separators=(",", ":"),
)


def _argv_p():
    return L.argv_p(
        py="/nonexistent/python",
        model="/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed",
        budgets=[27960, 17064, 16552], s_gb=48, m_mib=2400, store_cfg=STORE_CFG,
        extra=[], p_max_total_tokens=463763, draft_kv_on_p=True,
        spec_flags=L.spec_flags(producer=True),
    )


def test_argv_p_and_form_key_identical_between_on_and_off(restore_form, tmp_path):
    from sglang.srt.weg2 import ring_table

    _dflash(tmp_path, dflash_produce_on_p="off")
    off = _argv_p()
    _dflash(tmp_path, dflash_produce_on_p="on")
    on = _argv_p()
    assert off == on
    assert ring_table.p_form_key(off) == ring_table.p_form_key(on)
    # and the producer flags are still shipped: P keeps loading the draft
    assert "--speculative-draft-kv-only" in off
    assert off[off.index("--speculative-algorithm") + 1] == "DFLASH"


def test_launcher_names_the_form_in_one_line(restore_form, tmp_path):
    _dflash(tmp_path)
    line = L.dflash_produce_line()
    assert line.startswith("WEG2 DFLASH-PRODUCE-ON-P: off")
    assert f"{ENV}=0" in line and "\n" not in line
    _dflash(tmp_path, dflash_produce_on_p="on")
    line = L.dflash_produce_line()
    assert line.startswith("WEG2 DFLASH-PRODUCE-ON-P: on") and f"{ENV}=1" in line
    assert "log(dflash_produce_line())" in inspect.getsource(L.main)


# --------------------------------------------------------------------- rank


def test_rank_default_when_unset_is_the_old_form(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert dflash_produce_on_p() is True
    monkeypatch.setenv(ENV, "0")
    assert dflash_produce_on_p() is False
    monkeypatch.setenv(ENV, "1")
    assert dflash_produce_on_p() is True


def _wants(algo, mode_extend=True):
    from sglang.srt.managers.scheduler import Scheduler

    sched = SimpleNamespace(
        draft_kv_producer=object(),
        draft_kv_producer_algorithm=SpeculativeAlgorithm.from_string(algo),
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: mode_extend)
    )
    return Scheduler._draft_kv_producer_wants(sched, batch)


def test_off_means_the_dflash_producer_is_never_asked(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    assert _wants("DFLASH") is False
    monkeypatch.setenv(ENV, "1")
    assert _wants("DFLASH") is True
    monkeypatch.delenv(ENV, raising=False)
    assert _wants("DFLASH") is True
    # decode chunks were never asked, either way
    assert _wants("DFLASH", mode_extend=False) is False


def test_the_nextn_producer_is_not_affected(monkeypatch):
    # the NEXTN (MTP) producer's algorithm is EAGLE-family; the switch is
    # DFlash-only
    monkeypatch.setenv(ENV, "0")
    assert _wants("EAGLE") is True


def test_no_producer_no_wants(monkeypatch):
    from sglang.srt.managers.scheduler import Scheduler

    monkeypatch.setenv(ENV, "1")
    sched = SimpleNamespace(
        draft_kv_producer=None,
        draft_kv_producer_algorithm=SpeculativeAlgorithm.from_string("DFLASH"),
    )
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
    assert Scheduler._draft_kv_producer_wants(sched, batch) is False


def test_produce_has_one_call_site_behind_wants():
    from sglang.srt.managers import scheduler as sched_mod

    tree = ast.parse(textwrap.dedent(inspect.getsource(sched_mod.Scheduler)))
    calls = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "produce"
            ):
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "_draft_kv_produce"
                    ):
                        calls.append(fn.name)
    all_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_draft_kv_produce"
    ]
    assert len(all_calls) == 1 and len(calls) == 1
    src = inspect.getsource(getattr(sched_mod.Scheduler, calls[0]))
    assert "produce = self._draft_kv_producer_wants(batch)" in src


class _Model:
    def __init__(self):
        self.dflash_calls = []

    def set_dflash_layers_to_capture(self, ids):
        self.dflash_calls.append(list(ids))


def _runner(*, draft_kv_only, algo="DFLASH"):
    from sglang.srt.model_executor.model_runner import ModelRunner

    mr = ModelRunner.__new__(ModelRunner)
    mr.server_args = SimpleNamespace(speculative_draft_kv_only=draft_kv_only)
    mr.spec_algorithm = SpeculativeAlgorithm.from_string(algo)
    mr.eagle_use_aux_hidden_state = False
    mr.dflash_family_use_aux_hidden_state = True
    mr.dflash_family_target_layer_ids = [6, 20, 34, 48, 62]
    mr.tp_rank = 0
    mr.pp_rank = 1
    mr.model = _Model()
    return mr


def test_p_target_arms_no_aux_capture_under_off(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    mr = _runner(draft_kv_only=True)
    mr.init_aux_hidden_state_capture()
    assert mr.model.dflash_calls == []
    # the pricing fields stay set -> pool_configurator scales the cell alike
    assert mr.dflash_family_use_aux_hidden_state is True


def test_p_target_arms_capture_under_on(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    mr = _runner(draft_kv_only=True)
    mr.init_aux_hidden_state_capture()
    assert mr.model.dflash_calls == [[6, 20, 34, 48, 62]]


def test_d_target_always_captures(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    mr = _runner(draft_kv_only=False)
    mr.init_aux_hidden_state_capture()
    assert mr.model.dflash_calls == [[6, 20, 34, 48, 62]]


# ------------------------------------------------------------ W10 (08:56Z)

_P_LINE = ("HiCache draft KV registered: pool=x drafter=b53f8c336da51155\n"
           "#706 canonical DRAFT page active: layout=v1 drafter=b53f8c336da51155\n")
_D_LINE = ("HiCache draft KV registered: pool=y drafter=f0c73316257fec1d\n"
           "#706 canonical DRAFT page active: layout=v1 drafter=f0c73316257fec1d\n")


def _logs(tmp_path):
    lp, ld = tmp_path / "P.log", tmp_path / "D.log"
    lp.write_text(_P_LINE)
    ld.write_text(_D_LINE)
    return str(lp), str(ld)


def test_w10_is_skipped_by_name_when_p_writes_no_draft_pages(tmp_path):
    lp, ld = _logs(tmp_path)
    lines = []
    w10 = L.gate_w10(lp, ld, lines.append, p_produces_draft_pages=False)
    assert w10["match"] is False  # the weg2xsn414 identities really disagree
    assert len(lines) == 1
    assert lines[0].startswith(
        "W10 SKIPPED: P produces no draft pages (--dflash-produce-on-p off)"
    )
    assert "b53f8c336da51155" in lines[0] and "f0c73316257fec1d" in lines[0]


def test_w10_still_refuses_when_p_writes_draft_pages(tmp_path):
    lp, ld = _logs(tmp_path)
    with pytest.raises(L.Weg2LaunchRefused, match="W10 Weg2DrafterIdentityMismatch"):
        L.gate_w10(lp, ld, lambda s: None, p_produces_draft_pages=True)


def test_p_produces_draft_pages_follows_the_form(restore_form, tmp_path):
    _dflash(tmp_path)
    assert L.p_produces_draft_pages() is False
    _dflash(tmp_path, dflash_produce_on_p="on")
    assert L.p_produces_draft_pages() is True
    L.apply_spec_form(SimpleNamespace(spec_form="NEXTN"))
    assert L.p_produces_draft_pages() is True  # the NEXTN producer writes


def test_main_routes_w10_through_the_gate():
    src = inspect.getsource(L.main)
    assert "gate_w10(spec_p.log, spec_d.log, log," in src
    assert "p_produces_draft_pages=p_produces_draft_pages()" in src
    assert "check_drafter_identity(" not in src  # one grader, not two
