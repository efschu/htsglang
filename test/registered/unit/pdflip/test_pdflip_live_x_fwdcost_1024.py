"""LIVE-X-FWDCOST 1024: the live X re-solve prices D's fixed cost per FORWARD.

THE DEFECT, MEASURED. ``phase_policy.solve_x_cost_line`` solved the break-even
``a + (b + c*p)*u = price/k + P(u)``: D's fixed cost ``a`` once per REQUEST.
But ``a`` comes from a PER-FORWARD fit (``fit_cost_line`` over the
``Prefill rank batch`` ring, one row per forward), and D prefills a request in
forwards of at most W tokens (the #794 corridor width; NF after the PCIe switch
1664-2038, D's nominal chunk 4096 never reached). A 7-10k request is 4-5 D
forwards, so the solve under-priced D by 1.2-2.4 s and put X 2-3k too high:
NF 1023_201915 X* 9766 at a live price of 5.67 s, 234711 X* 11670 on the ski
record price 7.06 s; the fitted curve (x_curves.nf-int4-h6-abl.x16-1023.json,
D chunk 1984) gives 7276-7331. An X above the true crossing lets D prefill
requests that are dearer on D than the flip -- the danger direction of SOLL v2
R7 (no flip below X, so everything below X goes to D) and KEIN-D-PREFILL-UEBER-X.

THE FIX (cost term only, the break-even ``D(u) = price/k + P(u)`` unchanged):
``D(u) = ceil(u/W)*a + sum_j (b + c*(p + j*W))*n_j``; X* is the first crossing,
solved exactly per forward (D(u)-P(u) jumps by ``a`` at each new forward, linear
inside one). The front's W is the widest D forward of the boot (the ring rows),
else the line's ``n_hi``.

Inputs below are the logged ``PDFLIP X COST-LINE RE-SOLVE`` lines of the NF
boots (docker-acceptance/nf/evidence/*_1023_{201915,234711}.front.log).
"""

from __future__ import annotations

import collections
import math
import re
import types

import pytest

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.srt.pdflip import prefill_clock as pc  # noqa: E402

# 201915 21:01:51: D line [live:3p:n=53] fitted range n 64..1984, P line [live:3p:n=38]
NF_201915 = dict(line={"a_ms": 593.0, "b_ms": 0.706, "c_ms": 0.0007, "n": 53, "n_lo": 64,
                       "n_hi": 1984, "prefix_med": 32000},
                 line_p={"a_ms": 1063.0, "b_ms": 0.082, "c_ms": 0.0006},
                 price_s=5.67, prefix=32000.0, x_star_logged=9766)
# 234711 00:21:27: D line [live:3p:n=48] fitted range n 64..1788, P line [live:3p:n=28]
NF_234711 = dict(line={"a_ms": 388.0, "b_ms": 0.773, "c_ms": 0.0006, "n": 48, "n_lo": 64,
                       "n_hi": 1788, "prefix_med": 0},
                 line_p={"a_ms": 911.0, "b_ms": 0.098, "c_ms": 0.0001},
                 price_s=5.07, prefix=0.0, x_star_logged=8292)
# the record every NF boot seeds from (dauer10220606, before the switch: D 2880 wide)
NF_RECORD = dict(line={"a_ms": 525.0, "b_ms": 1.040, "c_ms": 0.0003, "n": 40, "n_lo": 157,
                       "n_hi": 2880, "prefix_med": 0},
                 line_p={"a_ms": 1312.0, "b_ms": 0.099, "c_ms": 0.0015},
                 price_s=7.06, prefix=0.0, x_star_logged=8335)


def _d_ref(line, u, prefix, w):
    """D's cost of ``u`` tokens as D runs them: forwards of <= w, each paying a."""
    a, b, c = line["a_ms"], line["b_ms"], line["c_ms"]
    total, done = 0.0, 0
    while done < u:
        piece = min(w, u - done)
        total += a + (b + c * (prefix + done) / 1000.0) * piece
        done += piece
    return total


def _excess(case, u, *, k=1.0, w=None, price_s=None, prefix=None):
    """D's cost minus the flip route's, ms: <= 0 = D is the cheaper route."""
    line, lp = case["line"], case["line_p"]
    p = case["prefix"] if prefix is None else prefix
    w = int(line["n_hi"]) if w is None else int(w)
    price = case["price_s"] if price_s is None else price_s
    p_ms = lp["a_ms"] + (lp["b_ms"] + lp["c_ms"] * p / 1000.0) * u
    return _d_ref(line, u, p, w) - p_ms - 1000.0 * price / k


def _first_crossing(case, **kw):
    u = 1
    while _excess(case, u, **kw) <= 0.0:
        u += 1
    return u - 1                                    # the last size cheaper on D


# ---------------------------------------------------------------- the cost term
def test_u_just_over_k_forwards_pays_one_more_fixed_cost_and_goes_to_p():
    """a=600, b=0.7 vs P a=1000, b=0.1, W=2000, price 4.7 s: three full
    forwards (6000 tok) cost D 6000 ms <= 4700+1000+600 = 6300 on the flip;
    6001 tok need a FOURTH forward (6600.7 ms > 6300.1). X = 6000 exactly.
    The one-fixed-cost solve said (4700+1000-600)/0.6 = 8500: D would have
    prefilled 6001..8500 tokens at up to ~1.2 s more than the flip costs."""
    case = dict(line={"a_ms": 600.0, "b_ms": 0.7, "c_ms": 0.0, "n_lo": 64, "n_hi": 2000},
                line_p={"a_ms": 1000.0, "b_ms": 0.1, "c_ms": 0.0}, price_s=4.7, prefix=0.0)
    x, why = pp.solve_x_cost_line(price_s=4.7, k=1.0, line=case["line"], r_p=None,
                                  prefix_tokens=0.0, line_p=case["line_p"])
    assert why == "ok"
    assert x == pytest.approx(6000.0)
    assert _excess(case, 6000) <= 0.0 < _excess(case, 6001)
    # the same request one forward narrower: 3 x 1999 = 5997 is the edge
    x2, _ = pp.solve_x_cost_line(price_s=4.7, k=1.0, line=case["line"], r_p=None,
                                 prefix_tokens=0.0, line_p=case["line_p"], d_forward_tokens=1999)
    assert x2 == pytest.approx(5997.0)


def test_d_request_ms_pays_a_per_forward_and_reads_each_chunk_at_its_depth():
    line = {"a_ms": 500.0, "b_ms": 0.7, "c_ms": 0.001}
    assert pp.d_request_ms(line, 2000, 10000, 2000) == pytest.approx(500 + (0.7 + 0.01) * 2000)
    # 2001 tokens: a second forward, its one token behind 12000 cached
    assert pp.d_request_ms(line, 2001, 10000, 2000) == pytest.approx(
        2 * 500 + (0.7 + 0.01) * 2000 + (0.7 + 0.012) * 1)
    assert pp.d_request_ms(line, 0, 0, 2000) == 0.0
    assert pp.d_request_ms(line, 4000, 0, None) == pytest.approx(500 + 0.7 * 4000)  # one forward


@pytest.mark.parametrize("case,k", [(NF_201915, 1.0), (NF_234711, 1.0), (NF_RECORD, 1.0),
                                    (NF_201915, 2.0), (NF_234711, 3.0)],
                         ids=["201915", "234711", "record", "201915-k2", "234711-k3"])
def test_x_is_the_first_size_where_d_costs_more_than_the_flip(case, k):
    """Every size up to X is cheaper on D, the next token is not -- checked
    token by token against D's cost as D runs it (W = the line's n_hi)."""
    x, why = pp.solve_x_cost_line(price_s=case["price_s"], k=k, line=case["line"], r_p=None,
                                  prefix_tokens=case["prefix"], line_p=case["line_p"])
    assert why == "ok"
    assert int(math.floor(x)) == _first_crossing(case, k=k)


def test_201915_one_fixed_cost_let_d_prefill_2s_over_its_crossing():
    """R7 / KEIN-D-PREFILL-UEBER-X, the danger direction on the measured
    inputs: the logged X* 9766 is a request D runs in 5 forwards; there D costs
    > 2 s more than the flip. The per-forward X* lies at ~6.9k."""
    c = NF_201915
    lp, ln = c["line_p"], c["line"]
    pk = c["prefix"] / 1000.0
    old = ((1000.0 * c["price_s"] + lp["a_ms"] - ln["a_ms"])
           / (ln["b_ms"] + ln["c_ms"] * pk - lp["b_ms"] - lp["c_ms"] * pk))
    assert old == pytest.approx(c["x_star_logged"], rel=0.005)   # the logged inputs reproduce it
    assert _excess(c, int(old)) > 2000.0
    x, _ = pp.solve_x_cost_line(price_s=c["price_s"], k=1.0, line=ln, r_p=None,
                                prefix_tokens=c["prefix"], line_p=lp)
    assert 6800.0 < x < 7100.0
    assert old - x > 2500.0


def test_a_line_without_any_d_width_is_priced_as_one_forward_and_says_so():
    line = {k: v for k, v in NF_201915["line"].items() if k != "n_hi"}
    x, why = pp.solve_x_cost_line(price_s=5.67, k=1.0, line=line, r_p=None, prefix_tokens=32000,
                                  line_p=NF_201915["line_p"])
    assert why == "ok(one-forward:no-D-width)"
    assert x == pytest.approx(NF_201915["x_star_logged"], rel=0.005)


# ---------------------------------------------------------------- the front
def _front(rows, **attrs):
    from flliper.srt.pdflip import front as F

    ns = types.SimpleNamespace(
        tp_prefill_max_tokens=4096, flip_min_work_tokens=4096, _x_min_work_follows=True,
        x_ceiling_tokens=12288, x_floor_tokens=4096, counters=collections.Counter(),
        _x_samples={"r_d": collections.deque(maxlen=32), "r_p": collections.deque(maxlen=32),
                    "flip_s": collections.deque(maxlen=32)},
        _x_last_missing=[], _x_cost_last_missing=[], _x_r_d_src="none yet",
        _x_seed_note="launcher solve X=4096", X_SAMPLE_WINDOW=32,
        _d_cost_rows=collections.deque(maxlen=64), _d_cost_mark=None,
        _p_cost_rows=collections.deque(maxlen=64), _p_cost_mark=None,
        _d_cost_all=collections.deque(maxlen=4096), _p_cost_all=collections.deque(maxlen=4096),
        _p_phase_k=collections.deque([1], maxlen=32),
        _x_cost_seed=dict(NF_RECORD["line"], k=1.0, boot_tag="dauer", at="2026-10-22 06:45:43,925"),
        _x_cost_seed_p=dict(NF_RECORD["line_p"], n=40, n_lo=7979, n_hi=16384, prefix_med=0, k=1.0,
                            boot_tag="dauer", at="2026-10-22 06:59:17,785"),
        _x_cost_written={"D": 0, "P": 0},
        _park_rt_seed={"round_trip_s": 7.06, "boot_tag": "dauer", "at": "2026-10-22 06:54:30,618"},
        _park_form_key="", measured_record="", p_phase_max_requests=6,
        flip_log=[], _resume_ms_log=[],
    )
    ns._park_warm_legs_ms = lambda: F.Front._park_warm_legs_ms(ns)
    ns._x_cost_inputs = lambda: F.Front._x_cost_inputs(ns)
    ns._resolve_x_cost_line = lambda: F.Front._resolve_x_cost_line(ns)
    ns._note_x_cost_line = lambda *a: F.Front._note_x_cost_line(ns, *a)
    ns._x_cost_line_of = lambda *a: F.Front._x_cost_line_of(ns, *a)
    ns._note_cost_rows = lambda b, side: F.Front._note_cost_rows(ns, b, side)
    ns.x_flip_s_provenance = lambda: F.Front.x_flip_s_provenance(ns)
    for k, v in attrs.items():
        setattr(ns, k, v)
    pc._reset_for_tests()
    for n, cached, ms in rows:
        pc.note_batch_cost(n, cached, ms)
    ns._note_cost_rows(pc.cost_snapshot(), "D")
    return F.Front, ns


def _resolve(F, ns, caplog):
    with envs.FLLIPER_PDFLIP_ENABLE_X_COST_LINE.override(True), \
            envs.FLLIPER_PDFLIP_X_COST_MAX_STEP.override(0.0), caplog.at_level("INFO"):
        x = F.resolve_x_live(ns)
    msg = next(m for m in caplog.messages if "X COST-LINE RE-SOLVE" in m)
    return x, msg, float(re.search(r"X\*=([\d.]+)", msg).group(1))


def test_front_prices_d_in_the_boots_widest_forward_not_the_records(caplog):
    """215911's state: the record line (fitted at D 2880 wide, before the
    switch) prices a boot whose D forwards are at most 1856 wide (#794 corridor).
    Too few rows for a live fit, so the line is the record's -- but D's request
    runs in 1856-token forwards, and X follows them."""
    rows = [(1856, 0, 2500.0), (1600, 0, 2200.0), (300, 0, 800.0)]
    F, ns = _front(rows)
    x, msg, x_star = _resolve(F, ns, caplog)
    assert "[record:" in msg
    assert "a per D forward of <=1856 tok [live:widest-of-3]" in msg
    assert x == _first_crossing(NF_RECORD, w=1856)       # X = int(X*), hysteresis off
    assert abs(x_star - x) <= 1.0
    # the old solve's 8339 and the record's own width (2880) are both refused
    assert x < _first_crossing(NF_RECORD, w=2880) < 8335


def test_front_falls_back_to_the_lines_widest_forward_without_a_wide_d_forward(caplog):
    rows = [(900, 0, 1200.0), (300, 0, 800.0)]
    F, ns = _front(rows)
    x, msg, x_star = _resolve(F, ns, caplog)
    assert "a per D forward of <=2880 tok [line:n_hi(boot widest 900 < 1024)]" in msg
    assert x == _first_crossing(NF_RECORD, w=2880)


def test_x_k_flip_prices_the_riders_in_the_same_forwards(caplog):
    rows = [(1856, 0, 2500.0), (1600, 0, 2200.0)]
    F, ns = _front(rows)
    with envs.FLLIPER_PDFLIP_X_K_FLIP.override(True):
        x1, _msg, _xs = _resolve(F, ns, caplog)
        xk = F._x_for_flip(ns, 2, "rid", "test")
    assert xk == min(_first_crossing(NF_RECORD, k=2.0, w=1856), x1)
    assert xk < x1
