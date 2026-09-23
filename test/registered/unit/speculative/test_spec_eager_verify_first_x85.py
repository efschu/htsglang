"""fnFL2x85 (23.09.): ``SGLANG_SPEC_EAGER_VERIFY=first`` -- every request's
FIRST verify round runs eager, every later round replays the graph.

Bug regression (x44-x50, Next Flash Form A, D group): the first graph verify
of a NEW request after its extend dies on TP0 (~4,3 s, deterministic,
illegal memory access); an eager round 1 lives and the graph rounds after
it live (x46 CODE MATCH). The boot-wide budget ``=N`` was the instrument
that found this; the ``first`` form keeps the graph decode for every round
but that one. Rank-uniform by construction: the decision is a pure function
of the batch's rids, identical on every rank. Hermetic.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.speculative import spec_stage_sync as ss  # noqa: E402


def _reset(monkeypatch, value):
    monkeypatch.setenv(ss.EAGER_ENV, value)
    monkeypatch.setitem(ss._EAGER, "budget", None)
    monkeypatch.setitem(ss._EAGER, "round_ct", 0)


def test_first_mode_spends_one_eager_round_per_request_then_replays(monkeypatch):
    _reset(monkeypatch, "first")
    assert ss.eager_verify_round(4521, rids=["a"]) is True   # a's first verify
    assert ss.eager_verify_round(4522, rids=["a"]) is False  # a's second: graph
    assert ss.eager_verify_round(4523, rids=["a"]) is False
    assert ss.eager_verify_round(100, rids=["b"]) is True    # a new request, whatever its length
    assert ss.eager_verify_round(101, rids=["a", "b"]) is False
    assert ss.eager_verify_round(1, rids=["c", "a"]) is True  # one fresh rid makes the round eager


def test_first_mode_ignores_the_minimum_length_and_an_empty_batch_is_never_eager(monkeypatch):
    _reset(monkeypatch, "first")
    assert ss.eager_verify_round(1, rids=["short"]) is True
    assert ss.eager_verify_round(4521, rids=[]) is False


def test_the_numeric_budget_keeps_its_boot_wide_form(monkeypatch):
    """x45's instrument: N rounds over the boot, spent only on contexts of
    the minimum length -- untouched by the per-request form."""
    _reset(monkeypatch, "1")
    assert ss.eager_verify_round(10, rids=["x"]) is False    # below EAGER_MIN_LEN_DEFAULT
    assert ss.eager_verify_round(4521, rids=["x"]) is True
    assert ss.eager_verify_round(4521, rids=["y"]) is False  # budget spent boot-wide


def test_unset_means_graph_every_round(monkeypatch):
    _reset(monkeypatch, "")
    assert ss.eager_verify_round(4521, rids=["a"]) is False
