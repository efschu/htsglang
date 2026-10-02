"""D-Nachlauf nach dem P->D-Wake (y3m 09292136, Flipzeit 5,1-6,9 s gegen x178
2,1 s): E2 (H24/H24c, P's END state, no target forward) stirbt in einer
gemischten Wake-Kohorte.

ep8 21:43:20-24: order_waiting stellte die flip-geparkten Resumes (194/608
frische Tokens) vor die Hand-offs; die Hand-offs trafen einen nicht leeren
Batch (``adopt=skipped:end_only:batch_not_empty``), verloren ihr END und die
ganze Kohorte lief EIN Extend (903 neu / 147904 gecacht, 3,56 s) vor dem
ersten Token. ep6 21:42:25: der Skip lief zuerst (29 ms), aber der naechste
Pass nahm sofort das echte Extend (611 Tokens, 2,5 s); das Ergebnis des Skips
(P's Token) kam erst danach -- erstes Token 3 s nach dem Flip.
"""
import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)


def _req(rid, seq=0, site=None):
    r = types.SimpleNamespace(rid=rid, kv_arrival_seq=seq)
    if site is not None:
        ds.mark_parked(r, site)
    return r


# ------------------------------------------------------------------ order
def test_ep8_the_end_state_hand_offs_lead_the_pass():
    from sglang.srt.weg2 import skip_first as sf

    p16, p18 = _req("weg2-6-16", 1, ds.SITE_FLIP), _req("weg2-6-18", 2, ds.SITE_FLIP)
    h17, h19, h20 = _req("weg2-6-17", 3), _req("weg2-6-19", 4), _req("weg2-6-20", 5)
    skips = {"weg2-6-17", "weg2-6-19", "weg2-6-20"}
    queue, rids = sf.order([p16, p18, h17, h19, h20], lambda r: r.rid in skips)
    assert [r.rid for r in queue] == ["weg2-6-17", "weg2-6-19", "weg2-6-20", "weg2-6-16", "weg2-6-18"]
    assert rids == frozenset(skips)


def test_without_an_end_state_request_the_queue_keeps_its_order():
    from sglang.srt.weg2 import skip_first as sf

    q = [_req("a"), _req("b", site=ds.SITE_FLIP), _req("c")]
    queue, rids = sf.order(q, lambda r: False)
    assert queue == q and rids == frozenset()


# ------------------------------------------------------------ park barrier
def test_ep8_the_barrier_does_not_hold_an_end_state_hand_off():
    """The hand-offs now come BEFORE the parked resumes; without the bypass
    the barrier (parked still waiting) would skip them and the pass would
    admit nothing it can skip. Their seats are the wake's own."""
    p16 = _req("weg2-6-16", 1, ds.SITE_FLIP)
    h17 = _req("weg2-6-17", 3)
    g = ds.admission_gate([h17, p16], running=[], pending_outside=[])
    assert g.barrier
    assert g.skip(h17, admitted=[]) == "weg2_d_park_first"      # a newcomer that needs a forward
    assert g.skip(h17, admitted=[], skip_extend=True) is None    # base: TypeError


def test_the_bypass_never_frees_a_blocked_pressure_park():
    older = _req("older", 0)
    p = _req("p", 1, ds.SITE_PRESSURE)
    g = ds.admission_gate([p], running=[older], pending_outside=[])
    assert g.skip(p, admitted=[], skip_extend=True) == "weg2_d_park_older_live"


# ------------------------------------------------------ decode before extend
def _batch(mode_extend=True, skip=True, reqs=1):
    mode = types.SimpleNamespace(is_extend=lambda: mode_extend)
    return types.SimpleNamespace(forward_mode=mode, weg2_skip_extend=skip, reqs=[object()] * reqs,
                                 is_empty=lambda: reqs == 0)


def test_ep6_the_pass_after_a_skip_batch_decodes_first(caplog):
    from sglang.srt.weg2 import skip_first as sf

    skip_batch = _batch()
    running = _batch(skip=False, mode_extend=False, reqs=1)
    with caplog.at_level(logging.INFO):
        assert sf.hold_prefill_after_skip(skip_batch, running) is True
    assert any(sf.HOLD_MARK in r.getMessage() for r in caplog.records)
    assert sf.hold_prefill_after_skip(skip_batch, running) is False   # once only


def test_no_hold_after_a_real_extend_an_aliased_decode_or_an_empty_batch():
    from sglang.srt.weg2 import skip_first as sf

    running = _batch(skip=False, mode_extend=False, reqs=1)
    assert sf.hold_prefill_after_skip(_batch(skip=False), running) is False
    assert sf.hold_prefill_after_skip(_batch(mode_extend=False), running) is False
    assert sf.hold_prefill_after_skip(_batch(), _batch(skip=False, reqs=0)) is False
    assert sf.hold_prefill_after_skip(None, running) is False


# ----------------------------------------------------------------- wiring
def test_the_scheduler_orders_marks_and_holds():
    from sglang.srt.managers import schedule_batch as sb
    from sglang.srt.managers import scheduler as sch

    src = inspect.getsource(sch.Scheduler)
    assert "_weg2_skip_first.order(" in src and "_weg2_tail_adopt.skip_joinable" in src
    assert "skip_extend=str(req.rid) in _skip_first_rids" in src
    assert "new_batch.weg2_skip_extend = bool(adder.weg2_skip_extend_taken)" in src
    assert "_weg2_skip_first.hold_prefill_after_skip(last_batch, running_batch)" in src
    assert sb.ScheduleBatch.__dataclass_fields__["weg2_skip_extend"].default is False
