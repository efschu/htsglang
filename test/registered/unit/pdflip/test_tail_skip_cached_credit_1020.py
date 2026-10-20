"""ANKER-D (rc28 boot ..._297198745d_1020_174205, pdflip-18-315, 18:41:37).

Metal: P prefilled 78464 tokens and published its END anchor (``STORE-PRESENCE src=p_flush ... 'pdflip-18-315:78464'``
18:41:36). D took it over with the E2 skip -- ``PDFLIP-HANDBACK N=78464 d_prefix=78464 d_compute=0 path=skip``,
``PDFLIP-TAIL-SKIP-EXTEND prefix=78464 ... (no target forward: P's END state + token)`` -- but the admission grows the
prefix only to c = floor_4(N-1) = 78460 and books ``len(prefix_indices)`` as cached: ``#969 EXTENT ('pdflip-18-315',
78460, 78464, 78460, 4)``, ``cached_tokens=78460``, front ``PDFLIP-USAGE-TRUE ... d_computed=4``, monitor
``ANKER-VERFEHLT-D Start-Cache 78460 < Anker 78464 -> 4 Token``. No target token ran: an accounting defect.

Now the admission commit credits the skipped [c, N) to the counter ``prepare_for_extend`` books
(``FLLIPER_PDFLIP_TAIL_SKIP_CACHED_CREDIT``, default on). RED on rc28: ``test_*credit*``.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import tail_adopt as TA  # noqa: E402
from flliper.srt.pdflip import tail_handoff as th  # noqa: E402

PAGE = 64
GRAIN = 4
ENV = "FLLIPER_PDFLIP_TAIL_SKIP_CACHED_CREDIT"


def _spec(n, rid="pdflip-18-315"):
    cut = th.tail_cut(n, GRAIN)
    return th.TailSpec(rid=rid, n_tokens=n, page_prefix=th.page_floor(cut, PAGE), cut=cut, key="k")


def _entry(spec, park=False):
    st = SimpleNamespace(spec=spec, park=park, headers=[], n_parts=3, verdict="ready", end_verdict="absent",
                         first_token=5666, token_src="publish", e1=False)
    return SimpleNamespace(staged=st, waited_ms=0.0)


def _req(prefix_len, retracted=False, n_out=0):
    return SimpleNamespace(rid="pdflip-18-315", prefix_indices=torch.zeros(prefix_len, dtype=torch.int64),
                           cached_tokens=0, already_computed=0, retracted_stain=retracted,
                           readmit_cached_tokens=0, readmit_already_computed=0,
                           origin_input_ids=list(range(10)), output_ids=list(range(n_out)))


@pytest.fixture(autouse=True)
def _iso(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    import flliper.srt.mem_cache.common as common

    monkeypatch.setattr(common, "alloc_token_slots", lambda tree, n: torch.arange(n, dtype=torch.int64))
    TA.SKIP_PLANS.clear()
    yield
    TA.SKIP_PLANS.clear()


def _book(req, seq_len):
    """``prepare_for_extend``'s booking, as the source has it (guarded by ``test_booking_is_what_the_credit_feeds``)."""
    pre_len = len(req.prefix_indices)
    if not req.retracted_stain:
        req.cached_tokens += pre_len - req.already_computed
        req.already_computed = seq_len
    else:
        new_cached = max(0, pre_len - req.readmit_already_computed)
        req.readmit_cached_tokens += new_cached
        req.readmit_already_computed = seq_len
        req.cached_tokens = req.readmit_cached_tokens


def test_metal_handoff_skip_credits_the_skipped_tokens_as_cached():
    """18-315: N=78464, c=78460, the load-back delivered 78400 -> booked 78460 on rc28, 78464 now."""
    spec = _spec(78464)
    assert (spec.cut, spec.extend) == (78460, 4)
    req = _req(78400)
    assert TA._commit_skip(req, _entry(spec), tree_cache=None, page_size=PAGE) == 78460
    assert len(req.prefix_indices) == 78460, "the admission geometry is unchanged: prefix c, extend [c, N)"
    _book(req, 78464)
    assert req.cached_tokens == 78464, "no target forward ran over [c, N): every token was cached"


@pytest.mark.parametrize("n", [78464, 78465, 78466, 78467, 78628, 67616, 100761])
def test_every_skip_books_exactly_n(n):
    spec = _spec(n)
    req = _req(spec.page_prefix)
    TA._commit_skip(req, _entry(spec), tree_cache=None, page_size=PAGE)
    _book(req, n)
    assert req.cached_tokens == n


def test_park_resume_credit_goes_to_the_readmission_counter():
    """F4 park resume: the request was retracted (stain); the booking ASSIGNS readmit_cached_tokens -- a credit into the
    lifetime counter would be overwritten, one into readmit survives."""
    spec = _spec(100761, rid="pdflip-4-223")
    req = _req(spec.cut, retracted=True, n_out=3)
    TA._commit_skip(req, _entry(spec, park=True), tree_cache=None, page_size=PAGE)
    _book(req, 100761)
    assert req.cached_tokens == 100761


def test_switch_off_is_the_rc28_count():
    os.environ[ENV] = "0"
    try:
        spec = _spec(78464)
        req = _req(78400)
        TA._commit_skip(req, _entry(spec), tree_cache=None, page_size=PAGE)
        _book(req, 78464)
        assert req.cached_tokens == 78460
    finally:
        del os.environ[ENV]


def test_credit_never_negative_or_double():
    req = _req(0)
    assert TA.skip_cached_credit(req, 0) == 0 and TA.skip_cached_credit(req, -3) == 0 and req.cached_tokens == 0
    assert TA.skip_cached_credit(req, 4) == 4 and req.cached_tokens == 4


def test_booking_is_what_the_credit_feeds():
    """The credit relies on prepare_for_extend's two bookings; if they change, this test names it."""
    from flliper.srt.managers.schedule_batch import ScheduleBatch

    src = inspect.getsource(ScheduleBatch.prepare_for_extend)
    assert "new_cached = pre_len - req.already_computed" in src
    assert "req.cached_tokens += new_cached" in src
    assert "req.readmit_cached_tokens = (" in src and "req.cached_tokens = req.readmit_cached_tokens" in src


def test_extent_probe_names_the_skip():
    from flliper.srt.managers.schedule_batch import ScheduleBatch

    src = inspect.getsource(ScheduleBatch.prepare_for_extend)
    assert "skip_no_target_fwd=" in src and "_ta.SKIP_PLANS" in src
