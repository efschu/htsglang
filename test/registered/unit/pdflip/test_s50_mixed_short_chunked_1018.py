"""S50 (18.10.): MIXED-D -- a SHORT request that D chunks still mixes; a BIG one keeps every chunk unmixed.

User order 18.10. ~09:10Z: "mixed chunk fuer kurze D-Prefills wieder an" -- short D prefills (u < halbX, SOLL R2) must not
halt the decodes; big ones (R3 timer end, R6 rest) stay unmixed. The S32 chunk stamp made "chunked" mean "big", but D chunks by
its own budget: rc13 D log 1018_200305, pdflip-14-180 (745 uncached, front halbX 3116) ran as 512 + 233 and neither chunk mixed;
pdflip-20-226 (5983 uncached, halbX 3787) is big and its 1728/1720/1720/815 chunks rightly stayed unmixed.

Drives the real ``Scheduler._mixed_d_agreed`` (gate + group MIN + census) and the real chunk stamp; the group MIN is a recorder.
"""
from __future__ import annotations

import ast
import logging
import os
import pathlib
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import mixed_d_short as mds  # noqa: E402

SRT = pathlib.Path(__import__("flliper.srt", fromlist=["x"]).__path__[0])
RIEGEL = 12288   # NF --tp-prefill-max-tokens: only says "group D"


def _sched_cls():
    from flliper.srt.managers.scheduler import Scheduler

    return Scheduler


class _Group:
    """Recorder for ``_pdflip_group_min_flags``: the other ranks' flags are given, the result is the MIN."""

    def __init__(self, others=()):
        self.others = list(others)
        self.calls = 0

    def __call__(self, flags):
        self.calls += 1
        return [min([int(flags[0])] + [int(o) for o in self.others])]


def _fake_sched(chunked_req=None, others=()):
    return types.SimpleNamespace(
        server_args=types.SimpleNamespace(tp_prefill_max_tokens=RIEGEL),
        chunked_req=chunked_req,
        _mixed_verify=False,   # MIXED-VERIFY off: the gate's collective stays the single ``[mine]`` flag
        _pdflip_group_min_flags=_Group(others),
    )


def _req(rid, halfx, seqlen=None, start=None):
    r = types.SimpleNamespace(rid=rid, weg2_front_halfx=halfx, origin_input_ids=[], output_ids=[], prefix_indices=None)
    if seqlen is not None:
        r.seqlen = seqlen
        r.extend_range = types.SimpleNamespace(start=start, end=None)
    return r


def _batch(reqs, u):
    return types.SimpleNamespace(reqs=list(reqs), extend_num_tokens=u)


def _running(n):
    return types.SimpleNamespace(reqs=[object()] * n)


def _agreed(sched, batch, n_running=5, local_ok=True):
    return _sched_cls()._mixed_d_agreed(sched, batch, _running(n_running), local_ok)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_MIXED_D_SHORT_CHUNKED", raising=False)
    from flliper.srt.pdflip import tail_adopt

    monkeypatch.setattr(tail_adopt, "SKIP_PLANS", {})


# --------------------------------------------------------------------------------------------- metal shapes (rc13)
def test_metal_short_request_chunked_by_d_mixes_both_chunks():
    """pdflip-14-180, 20:25:12: uncached 745 under front halbX 3116, D chunked 41728..42240 (512) + 42240..42473 (233)."""
    r = _req("pdflip-14-180", 3116)
    r._mixed_d_was_chunked = True
    r._mixed_d_u_total = 745
    s = _fake_sched(chunked_req=r)            # first chunk: the request is the chunked_req of this pass
    assert _agreed(s, _batch([r], 512)) is True
    s = _fake_sched(chunked_req=None)         # last chunk: no longer chunked_req, the stamp stays
    assert _agreed(s, _batch([r], 233)) is True


def test_metal_big_request_keeps_every_chunk_unmixed_r6_rest():
    """pdflip-20-226, 20:30:25: uncached 5983 >= halbX 3787 -- 1728, 1720, 1720 and the small last 815 stay unmixed."""
    r = _req("pdflip-20-226", 3787)
    r._mixed_d_was_chunked = True
    r._mixed_d_u_total = 5983
    for u, cr in ((1728, r), (1720, r), (1720, r), (815, None)):
        assert _agreed(_fake_sched(chunked_req=cr), _batch([r], u)) is False


def test_stamp_takes_the_whole_uncached_amount_at_the_first_chunk_once():
    r = _req("pdflip-14-180", 3116, seqlen=42473, start=41728)
    mds.stamp_chunked(r)
    assert r._mixed_d_was_chunked is True and r._mixed_d_u_total == 745
    r.extend_range.start = 42240              # the next chunk: the first total stays (no shrinking of a big request)
    mds.stamp_chunked(r)
    assert r._mixed_d_u_total == 745
    big = _req("pdflip-20-226", 3787, seqlen=62047, start=56064)
    mds.stamp_chunked(big)
    assert big._mixed_d_u_total == 5983


def test_review_a1_readmission_on_a_shorter_prefix_restamps_big():
    """S50 review A1: a request first chunked SHORT (745 under halbX 3116), then retracted / parked and re-admitted on a
    lost prefix (seqlen 42473, prefix now 36864: 5609 uncached >= halbX) is BIG -- its chunks must not mix, although each
    chunk alone (512) and the batch sum stay under halbX. The first-admission total may not keep it short."""
    r = _req("pdflip-14-180", 3116, seqlen=42473, start=41728)
    mds.stamp_chunked(r)
    assert r._mixed_d_u_total == 745
    assert _agreed(_fake_sched(chunked_req=r), _batch([r], 512)) is True
    r.extend_range.start = 36864              # re-admission: the prefix shrank
    mds.stamp_chunked(r)
    assert r._mixed_d_u_total == 5609
    assert _agreed(_fake_sched(chunked_req=r), _batch([r], 512)) is False
    r.extend_range.start = 37376              # its next chunk reads the smaller rest: the stamp keeps the maximum
    mds.stamp_chunked(r)
    assert r._mixed_d_u_total == 5609
    assert _agreed(_fake_sched(chunked_req=None), _batch([r], 233)) is False


def test_review_a1_an_unreadable_stamp_stays_big_for_life():
    r = _req("r", 3116)                        # no seqlen: the stamp cannot be read
    mds.stamp_chunked(r)
    assert r._mixed_d_u_total is None
    r.seqlen = 1000
    r.extend_range = types.SimpleNamespace(start=900, end=None)   # a later chunk reads a small rest
    mds.stamp_chunked(r)
    assert r._mixed_d_u_total is None
    assert _agreed(_fake_sched(), _batch([r], 100)) is False


def test_boundary_total_equal_halfx_is_big_one_below_is_short():
    hx = 3116
    for total, want in ((hx, False), (hx - 1, True), (hx + 1, False)):
        r = _req("r", hx)
        r._mixed_d_was_chunked = True
        r._mixed_d_u_total = total
        assert _agreed(_fake_sched(chunked_req=r), _batch([r], 512)) is want, total


def test_chunked_without_a_stamped_total_is_big():
    r = _req("r", 3116)
    r._mixed_d_was_chunked = True             # stamp without total (no invented bound)
    assert _agreed(_fake_sched(), _batch([r], 233)) is False
    s = _fake_sched(chunked_req=_req("c", 3116))   # a chunked_req never stamped
    assert _agreed(s, _batch([_req("a", 3116)], 100)) is False


def test_a_big_chunked_request_behind_the_batch_still_blocks():
    big = _req("big", 3116)
    big._mixed_d_was_chunked = True
    big._mixed_d_u_total = 9000
    assert _agreed(_fake_sched(chunked_req=big), _batch([_req("a", 3116)], 100)) is False


def test_batch_sum_still_bounds_short_chunks_of_two_requests():
    a, b = _req("a", 3116), _req("b", 3116)
    for r in (a, b):
        r._mixed_d_was_chunked = True
        r._mixed_d_u_total = 2000
    assert _agreed(_fake_sched(), _batch([a, b], 4000)) is False   # S of the batch >= halbX: not short (R2a)
    assert _agreed(_fake_sched(), _batch([a, b], 3000)) is True


def test_switch_off_is_the_s32_form(monkeypatch):
    assert envs.FLLIPER_PDFLIP_MIXED_D_SHORT_CHUNKED.default is True
    monkeypatch.setenv("FLLIPER_PDFLIP_MIXED_D_SHORT_CHUNKED", "0")
    r = _req("pdflip-14-180", 3116)
    r._mixed_d_was_chunked = True
    r._mixed_d_u_total = 745
    assert _agreed(_fake_sched(chunked_req=r), _batch([r], 512)) is False
    assert _agreed(_fake_sched(), _batch([_req("n", 3116)], 512)) is True   # never chunked: unchanged


# ------------------------------------------------------------------------------------------------ rank agreement (R11)
def test_group_min_decides_and_is_entered_once_per_decision():
    r = _req("pdflip-14-180", 3116)
    r._mixed_d_was_chunked = True
    r._mixed_d_u_total = 745
    s = _fake_sched(chunked_req=r, others=[1, 0])      # another rank says no (its own term)
    assert _agreed(s, _batch([r], 512)) is False
    assert s._pdflip_group_min_flags.calls == 1
    big = _req("big", 3116)
    big._mixed_d_was_chunked = True
    big._mixed_d_u_total = 9000
    s = _fake_sched(chunked_req=big, others=[1, 1])
    assert _agreed(s, _batch([big], 512)) is False
    assert s._pdflip_group_min_flags.calls == 1         # a big chunk still enters the collective (replicated entry)


def test_e2_handoff_and_missing_halfx_still_never_mix(monkeypatch):
    from flliper.srt.pdflip import tail_adopt

    r = _req("pdflip-18-217", 3116)
    r._mixed_d_was_chunked = True
    r._mixed_d_u_total = 4
    monkeypatch.setattr(tail_adopt, "SKIP_PLANS", {"pdflip-18-217": object()})
    assert _agreed(_fake_sched(), _batch([r], 4)) is False
    monkeypatch.setattr(tail_adopt, "SKIP_PLANS", {})
    h = _req("HEALTH_CHECK_x", None)
    assert _agreed(_fake_sched(), _batch([h], 1)) is False


# --------------------------------------------------------------------------------------------------------- census
def test_census_counts_mixed_share_and_names_the_refusal(monkeypatch, caplog):
    monkeypatch.setenv("FLLIPER_PDFLIP_MIXED_D_CENSUS_N", "4")
    monkeypatch.setattr(mds, "_CENSUS", {"n": 0, "mixed": 0, "refused": {}})
    short = _req("s", 3116)
    short._mixed_d_was_chunked = True
    short._mixed_d_u_total = 745
    big = _req("b", 3116)
    big._mixed_d_was_chunked = True
    big._mixed_d_u_total = 9000
    with caplog.at_level(logging.INFO, logger=mds.logger.name):
        _agreed(_fake_sched(), _batch([short], 233))
        _agreed(_fake_sched(), _batch([big], 815))
        _agreed(_fake_sched(), _batch([_req("h", None)], 1))
        _agreed(_fake_sched(others=[0]), _batch([_req("v", 3116)], 10))
    c = mds._CENSUS
    assert c["n"] == 4 and c["mixed"] == 1
    assert c["refused"] == {"big_chunk": 1, "no_halfx": 1, "group_veto": 1}
    lines = [m.getMessage() for m in caplog.records if "MIXED-D CENSUS" in m.getMessage()]
    assert len(lines) == 1 and "decisions=4 mixed=1 share=0.250" in lines[0]


def test_refusal_reason_order():
    assert mds.refusal_reason(10, 0, 3116, False, False, True) == "not_d"
    assert mds.refusal_reason(10, RIEGEL, 3116, True, True, True) == "e2_handoff"
    assert mds.refusal_reason(10, RIEGEL, 3116, True, False, True) == "big_chunk"
    assert mds.refusal_reason(10, RIEGEL, 0, False, False, True) == "no_halfx"
    assert mds.refusal_reason(3116, RIEGEL, 3116, False, False, True) == "u_ge_halfx"
    assert mds.refusal_reason(10, RIEGEL, 3116, False, False, False) == "local_block"
    assert mds.refusal_reason(10, RIEGEL, 3116, False, False, True) == ""


# ------------------------------------------------------------------------------------------------------- call edge
def test_call_edge_the_stamp_rides_the_counted_chunk_and_runs_before_the_gate():
    t = ast.parse((SRT / "managers/scheduler.py").read_text())
    raw = next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw")
    src = ast.unparse(raw)
    i = src.index("if _pp_seat.counts_middle_chunk(self.chunked_req, can_run_list):")  # 27B-PP-SKIPOUT-1024
    j = src.index("_mds_stamp.stamp_chunked(self.chunked_req)")
    k = src.index("self._mixed_d_agreed(")
    assert i < j < k
    assert "self.chunked_req.inflight_middle_chunks += 1" in src[i:j]
