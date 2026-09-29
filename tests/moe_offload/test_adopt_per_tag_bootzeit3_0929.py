"""BOOTZEIT 3 (A), 29.09.: #108 (D adopts P's bytes at the first flip, no
disk) under the per-tag collect of #1374.

The prior art is complete except for one seam: the per-tag wake returned
BEFORE the #108 settle, and the cover field held only the LAST tag. Under
``--pdflip-d-adopt on`` a placeholder rank could therefore never answer. Here:
the cover accumulates over every tag of the first wake, both wake shapes run
the same settle, and a tensor no leg wrote keeps the guard up (it would still
hold the dummy load).
"""

import inspect
from types import SimpleNamespace

import pytest

from flliper.srt.managers.scheduler_components import weight_updater as wu
from flliper.srt.pdflip import adopt

M = wu.SchedulerWeightUpdaterManager


@pytest.fixture(autouse=True)
def _placeholder(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_D_ADOPT_UNCOVERED", raising=False)
    adopt.arm_placeholder("test")
    yield
    adopt._PLACEHOLDER["pending"] = False
    adopt._PLACEHOLDER["reason"] = ""


class _Rank:
    _pdflip_adopt_acc = None
    _pdflip_last_inject_cover = None
    _pdflip_adopt_accumulate = M._pdflip_adopt_accumulate
    _pdflip_adopt_cover = M._pdflip_adopt_cover
    _pdflip_adopt_uncovered = M._pdflip_adopt_uncovered
    _pdflip_adopt_settle = M._pdflip_adopt_settle

    def __init__(self, live):
        self._live = live

    def _pdflip_rank_param_table(self):
        return {("weights", n): object() for n in self._live}


def _d(name, src=0):
    return SimpleNamespace(param_name=name, src_rank=src, dst_rank=1)


def _plan(descs):
    return SimpleNamespace(descs=descs)


def test_cover_accumulates_over_the_tags_of_the_first_wake():
    descs = [_d("a"), _d("b"), _d("c")]
    r = _Rank(["a", "b", "c"])
    r._pdflip_adopt_accumulate("weights_0", _plan(descs), descs[:2])
    r._pdflip_adopt_accumulate("weights_1", _plan(descs), descs[2:])
    assert r._pdflip_adopt_cover() == (3, 3)
    r._pdflip_adopt_settle()
    assert adopt.weights_are_placeholder() is False
    assert r._pdflip_adopt_acc is None


def test_a_tag_collected_twice_is_not_counted_twice():
    descs = [_d("a"), _d("b")]
    r = _Rank(["a", "b"])
    r._pdflip_adopt_accumulate("weights_0", _plan(descs), descs[:1])
    r._pdflip_adopt_accumulate("weights_0", _plan(descs), descs[:1])
    assert r._pdflip_adopt_cover() == (1, 2)
    r._pdflip_adopt_settle()
    assert adopt.weights_are_placeholder() is True


def test_a_tensor_no_leg_wrote_keeps_the_guard_up():
    descs = [_d("a"), _d("b")]
    r = _Rank(["a", "b", "vision.tower.w"])
    r._pdflip_adopt_accumulate(None, _plan(descs), descs)
    assert r._pdflip_adopt_uncovered() == ["vision.tower.w"]
    r._pdflip_adopt_settle()
    assert adopt.weights_are_placeholder() is True


def test_log_mode_names_it_but_lets_the_count_decide(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_D_ADOPT_UNCOVERED", "log")
    descs = [_d("a")]
    r = _Rank(["a", "extra"])
    r._pdflip_adopt_accumulate(None, _plan(descs), descs)
    r._pdflip_adopt_settle()
    assert adopt.weights_are_placeholder() is False


def test_zerofill_descs_do_not_count_as_written_bytes():
    descs = [_d("a"), _d("pad", src=-1)]
    r = _Rank(["a", "pad"])
    r._pdflip_adopt_accumulate(None, _plan(descs), descs)
    assert r._pdflip_adopt_uncovered() == ["pad"]


def test_nothing_is_accumulated_outside_a_placeholder_wake():
    adopt._PLACEHOLDER["pending"] = False
    r = _Rank(["a"])
    r._pdflip_adopt_accumulate("weights_0", _plan([_d("a")]), [_d("a")])
    assert r._pdflip_adopt_acc is None


def test_both_wake_shapes_run_the_same_settle():
    src = inspect.getsource(M._pdflip_wake_reload_weights)
    per_tag = src.split("if self._pdflip_xchg_collected_per_tag:")[1].split("return")[0]
    assert "_pdflip_adopt_settle()" in per_tag
    assert src.count("_pdflip_adopt_settle()") == 2
    inject = inspect.getsource(M._pdflip_xchg_inject_from_peer)
    assert "_pdflip_adopt_accumulate(" in inject
