"""X-CURVES 1006 (AUFTRAG-x-kurven-1006, user go 06.10. ~10:15Z): X from
profile curves instead of a fixed or a live-solved X.

The user's decisions this file pins:

* the fixed X stops being a standing profile value; NOT computed live -- two
  curves (P prefill, D prefill over the context) and the flip price per depth,
  measured once per model x form x hardware, one file the profile points at;
* the front sets X PER REQUEST from them, at the request's cached depth, and
  more open requests make the flip pay more;
* CHANGE 06.10. ~11:00Z: ``--x-mode`` has ONLY ``fixed`` and ``curve`` ("es
  soll kein live geben und kein curve-capped"); ``live`` / ``curve-capped``
  are refused by name (W196). No flag = today's front and launcher byte for
  byte (what the unflagged default should become is the user's call). Under
  ``curve`` D's W50 riegel = the curves' envelope, a hard bound no request X
  crosses; a manual ``--x-ceiling-tokens`` beside it is refused (W195) -- an
  open user decision, not decided here;
* refusals by name (W190..W196), no silent fallback, no silent extrapolation.

NOTE on the monotony the Auftrag's test list states ("mehr offene Requests ->
X nicht kleiner"): the break-even is D(d,n) <= price/k + P(d,n); a larger k
LOWERS the price share, so the flip pays MORE and X can only FALL -- which is
the user's own sentence ("mehr gleichzeitig offene Requests lohnen den Flip
ohnehin mehr") and X-K-FLIP's contract (X_k <= X_1). Pinned that way round.

Every case is red on the base (cand4c 369f31e2e2): the module, the flags and
the front's kwargs do not exist there.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
from types import SimpleNamespace

import pytest

from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import x_curves as xc
from sglang.srt.weg2 import x_curves_build as xb
from sglang.srt.weg2.front import Front

MODEL = "Qwen3.8-Flash-Next-int4"
FORM_AXES = "arch=moe,experts=offload,draft=dflash,kv=qsa_forma,flip=family"
FORM_ENV_VALUE = ("arch=moe,experts=offload,draft=dflash,p_draft=cold,kv=qsa_forma,flip=family,"
                  f"vision=off,profile=nextflash,model=/models/{MODEL}")
HW = "RTX5090,RTX3080,RTX3080"
CARDS = [{"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607, "cc": (12, 0)},
         {"name": "NVIDIA GeForce RTX 3080", "total_mib": 20480, "cc": (8, 6)},
         {"name": "NVIDIA GeForce RTX 3080", "total_mib": 20480, "cc": (8, 6)}]
START_X = 4096
CEILING = 12288
DEPTHS = (0, 32768, 131072, 262144)
NS = (512, 1024, 2048, 4096)


def _d_ms(depth, n):          # D: dear per token, dearer with depth
    return 40.0 + (0.9 + 0.004 * depth / 1000.0) * n


def _p_ms(depth, n):          # P: a fixed cost per chunk, cheap per token
    return 120.0 + (0.08 + 0.0008 * depth / 1000.0) * n


def _curve(fn, *, chunk=4096, depths=DEPTHS, ns=NS):
    return xc.PrefillCurve(
        rows=tuple(xc.CurveRow(depth=d, n=tuple(ns), ms=tuple(fn(d, n) for n in ns)) for d in depths),
        chunk_tokens=chunk, instrument="synthetic")


def _curves(*, d=None, p=None, price=None, ident=None):
    return xc.XCurves(
        format=xc.X_CURVES_FORMAT,
        identity=ident or xc.CurveIdentity(model=MODEL, form=FORM_AXES, hardware=HW,
                                           created="2026-10-06T10:30:00Z", source="synthetic"),
        p_curve=p or _curve(_p_ms), d_curve=d or _curve(_d_ms),
        flip_price=price or xc.FlipPrice(depth=(0, 131072, 262144), seconds=(4.0, 5.0, 7.0),
                                         attribution="calib_prefix", pairs=9))


def _write(tmp_path, curves, name="nf.xcurves.json"):
    path = tmp_path / name
    path.write_bytes(xc.encode(curves))
    return str(path)


# ----------------------------------------------------------------- A: schema


def test_file_round_trips_and_every_malformation_is_refused_by_name(tmp_path):
    curves = _curves()
    assert xc.load(_write(tmp_path, curves)) == curves
    with pytest.raises(xc.XCurvesRefused) as e:
        xc.load(str(tmp_path / "absent.json"))
    assert e.value.code == xc.W_X_CURVES_MISSING
    raw = json.loads(xc.encode(curves))
    bad = [
        dict(raw, format="weg2-x-curves/0"),
        dict(raw, extra=1),                                              # unknown field
        dict(raw, d_curve=dict(raw["d_curve"], rows=list(reversed(raw["d_curve"]["rows"])))),
        dict(raw, p_curve=dict(raw["p_curve"], rows=[dict(raw["p_curve"]["rows"][0], ms=[1.0])])),
        dict(raw, flip_price=dict(raw["flip_price"], seconds=[4.0, -1.0, 7.0])),
        dict(raw, identity=dict(raw["identity"], hardware=" ")),
    ]
    for obj in bad:
        with pytest.raises(xc.XCurvesRefused) as e:
            xc.decode(json.dumps(obj).encode())
        assert e.value.code == xc.W_X_CURVES_MALFORMED


def test_foreign_curves_are_refused_naming_every_mismatch():
    curves = _curves()
    assert "checked by the launcher" in xc.check_identity(curves, model=MODEL, form=FORM_AXES, hardware=None)
    with pytest.raises(xc.XCurvesRefused) as e:
        xc.check_identity(curves, model="Qwen3.8-27B", form=FORM_AXES.replace("moe", "dense"),
                          hardware="RTX5090,RTX5090")
    assert e.value.code == xc.W_X_CURVES_FOREIGN
    assert all(w in e.value.detail for w in ("model", "form", "hardware"))
    with pytest.raises(xc.XCurvesRefused) as e:                       # no published form
        xc.check_identity(curves, model="", form="", hardware=None)
    assert e.value.code == xc.W_X_CURVES_FOREIGN


def test_a_chunked_request_sums_its_chunks_each_at_its_own_depth():
    curve = _curve(_d_ms)
    got = xc.request_ms(curve, depth=1000, n=2 * 4096 + 2048)
    want = _d_ms(1000, 4096) + _d_ms(1000 + 4096, 4096) + _d_ms(1000 + 8192, 2048)
    assert got == pytest.approx(want, rel=1e-9)        # rows are linear in depth: exact blend


def test_extrapolation_is_named_never_silent():
    notes = set()
    xc.forward_ms(_curve(_d_ms), depth=300000, n=8000, notes=notes, tag="d")
    assert {"d-depth-edge", "d-n-edge"} <= notes
    notes = set()
    xc.flip_seconds(xc.FlipPrice(depth=(0,), seconds=(4.0,), attribution="none"), depth=50000, notes=notes)
    assert notes == {"flip-depthless"}


# ------------------------------------------------------------ B: x_from_curves


def _linear_one_forward(a_d, b_d, a_p, b_p, price_s):
    d = _curve(lambda _d, n: a_d + b_d * n, chunk=0, depths=(0, 262144), ns=(256, 65536))
    p = _curve(lambda _d, n: a_p + b_p * n, chunk=0, depths=(0, 262144), ns=(256, 65536))
    return _curves(d=d, p=p, price=xc.FlipPrice(depth=(0,), seconds=(price_s,), attribution="none"))


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_x_is_the_closed_form_break_even_on_linear_curves(k):
    """D = 200 + 1.0 n, P = 900 + 0.1 n, price 4 s: X = (4000/k + 700) / 0.9."""
    curves = _linear_one_forward(200.0, 1.0, 900.0, 0.1, 4.0)
    x = xc.x_from_curves(depth=1000, open_requests=k, curves=curves)
    assert x == math.floor((4000.0 / k + 700.0) / 0.9)


def test_more_open_requests_never_raise_x():
    curves = _curves()
    for depth in (0, 5000, 40000, 131072, 200000, 262144):
        xs = [xc.x_from_curves(depth=depth, open_requests=k, curves=curves) for k in range(1, 9)]
        assert xs == sorted(xs, reverse=True), (depth, xs)
        assert xs[0] > xs[-1]


def test_deeper_d_prefill_lowers_x_and_the_lone_request_is_the_envelope():
    curves = _curves()
    xs = [xc.x_from_curves(depth=d, open_requests=1, curves=curves) for d in (0, 65536, 200000)]
    assert xs == sorted(xs, reverse=True) and xs[0] > xs[-1]
    env = xc.x_envelope(curves)
    assert all(xc.x_from_curves(depth=d, open_requests=k, curves=curves) <= env
               for d in range(0, 262145, 8192) for k in (1, 2, 4))


def test_ds_riegel_caps_the_curve_x_and_says_so():
    """The front's cap is D's W50 riegel (x_ceiling): no per-request X above it."""
    v = xc.x_verdict_from_curves(depth=0, open_requests=1, curves=_curves(), cap=3000)
    assert (v.x, v.clamp) == (3000, "cap") and v.x_curve > 3000
    assert xc.x_envelope(_curves(), cap=3000) == 3000


def test_the_measured_n_range_bounds_x_on_both_sides():
    never = _linear_one_forward(200.0, 0.2, 900.0, 0.1, 30.0)       # D never dearer within range
    v = xc.x_verdict_from_curves(depth=0, open_requests=1, curves=never)
    assert (v.x, v.why, v.clamp) == (65536, "d-cheaper-to-n_hi", "n_hi")
    always = _linear_one_forward(5000.0, 1.0, 100.0, 0.1, 0.5)      # P cheaper at the first point
    v = xc.x_verdict_from_curves(depth=0, open_requests=1, curves=always)
    assert (v.x, v.clamp) == (256, "n_lo")


def test_a_request_deeper_than_the_curves_is_refused_or_clamped_by_name():
    curves = _curves(d=_curve(_d_ms, depths=(0, 32768, 131072)))
    assert xc.reach_depth(curves) == 131072
    with pytest.raises(xc.XCurvesRefused) as e:
        xc.x_from_curves(depth=140000, open_requests=1, curves=curves)
    assert e.value.code == xc.W_X_CURVE_DEPTH_BEYOND
    v = xc.x_verdict_from_curves(depth=140000, open_requests=1, curves=curves, beyond=xc.BEYOND_CLAMP)
    assert v.depth_used == 131072 and "depth-clamped(140000->131072)" in v.notes


# --------------------------------------------------------------- A: builder


def _calib_rows():
    rows = []
    t = 1_759_740_000

    def fwd(role, ranks, depth, n, ms, chunks=1):
        nonlocal t
        t += 2
        for i, rk in enumerate(ranks):
            rows.append({"role": role, "t": t, "rank": rk, "new_tokens": n, "cached_depth": depth,
                         "chunks": chunks, "gpu_ms": ms + i, "compute_ms": ms, "wait_ms": 0.0,
                         "tag": "s9wwu9"})
    for d in (0, 40960, 204800):
        for n in (512, 2048, 4096):
            fwd("p_chunk", ("PP0", "PP1", "PP2"), d, n, _p_ms(d, n))
            fwd("d_forward", ("TP0", "TP1", "TP2"), d, n, _d_ms(d, n))
    fwd("p_chunk", ("PP0",), 0, 4096, 99999.0, chunks=3)                # folded: skipped
    rows.append({"role": "d_extend", "cached_depth": 40960, "new_tokens": 8000, "gpu_ms_max": 7777.0,
                 "d_direct": True, "ambiguous": False, "error": None})
    rows.append({"role": "d_extend", "cached_depth": 0, "new_tokens": 512, "gpu_ms_max": 1.0,
                 "d_direct": True, "ambiguous": False, "error": None})     # duplicate shape: dropped
    for depth, (t0, dp, pd) in ((4000, (t + 100, 1800, 2200)), (200000, (t + 200, 2600, 3400))):
        rows.append({"role": "calib_prefix", "depth_target": depth, "prompt_tokens": depth + 120,
                     "t_send": t0 - 1, "t_end": t0 + 6})
        rows.append({"role": "flip", "t": t0 + dp / 1000.0, "epoch": 1, "direction": "D>P",
                     "flip_total_ms": dp})
        rows.append({"role": "flip", "t": t0 + 5, "epoch": 2, "direction": "P>D", "flip_total_ms": pd})
    return rows


def test_builder_turns_a_harvest_into_a_valid_curve_file(tmp_path):
    curves, notes = xb.build(_calib_rows(), model=MODEL, form=FORM_AXES, hardware=HW, source="s9wwu9",
                             p_chunk_tokens=4096, d_chunk_tokens=4096, created="2026-10-06T13:00:00Z")
    assert [r.depth for r in curves.d_curve.rows] == [0, 40960, 204800]
    row0 = curves.p_curve.rows[0]
    assert row0.n == (512, 2048, 4096)
    assert row0.ms[0] == pytest.approx(_p_ms(0, 512) + 2)            # max over the three PP ranks
    assert 8000 in curves.d_curve.rows[1].n and 7777.0 in curves.d_curve.rows[1].ms
    assert curves.d_curve.rows[0].ms[0] == pytest.approx(_d_ms(0, 512) + 2)   # forward kept, d_extend dup dropped
    fp = curves.flip_price
    assert fp.attribution == "calib_prefix" and fp.pairs == 2
    assert fp.depth == (4120, 200120) and fp.seconds == (4.0, 6.0)
    assert any("1 folded" in n for n in notes)
    assert xc.load(_write(tmp_path, curves)) == curves
    table = xc.text_table(curves, cap=CEILING)
    assert "X by depth x k (cap 12288)" in table and "flip price: 4120:4.00s" in table


def test_builder_without_a_prefix_window_names_a_depthless_price():
    rows = [r for r in _calib_rows() if r["role"] != "calib_prefix"]
    curves, _ = xb.build(rows, model=MODEL, form=FORM_AXES, hardware=HW, source="s",
                         p_chunk_tokens=4096, d_chunk_tokens=4096)
    assert curves.flip_price.attribution == "none" and curves.flip_price.depth == (0,)
    with pytest.raises(xc.XCurvesRefused) as e:
        xb.build([r for r in rows if r["role"] != "flip"], model=MODEL, form=FORM_AXES, hardware=HW,
                 source="s", p_chunk_tokens=4096, d_chunk_tokens=4096)
    assert e.value.code == xc.W_X_CURVES_MALFORMED


# ----------------------------------------------------- C/D: launcher half


def _resolve(**kw):
    base = dict(mode=None, curves_path=None, beyond=None, ceiling_flag=0, x_tokens=START_X,
                model=MODEL, form=FORM_AXES, hardware=HW)
    base.update(kw)
    return xc.resolve_launch(**base)


def test_no_flag_is_the_live_mode_with_nothing_added():
    xm = _resolve(ceiling_flag=CEILING)
    assert (xm.mode, xm.given, xm.ceiling_for_d, xm.front_argv) == ("live", False, CEILING, ())
    assert xm.provenance == "" and xm.ceiling_note == ""


@pytest.mark.parametrize("kw,code", [
    (dict(mode="fixed", curves_path="x.json"), xc.W_X_CURVES_WITHOUT_MODE),
    (dict(mode=None, beyond="clamp"), xc.W_X_CURVES_WITHOUT_MODE),
    (dict(mode="curve"), xc.W_X_CURVES_MISSING),
    (dict(mode="curve-capped", curves_path="x.json"), xc.W_X_MODE_UNKNOWN),     # removed 06.10.
    (dict(mode="live"), xc.W_X_MODE_UNKNOWN),                                  # removed 06.10.
    (dict(mode="curve", curves_path="/nonexistent/x.json"), xc.W_X_CURVES_MISSING),
    (dict(mode="curve", curves_path="x.json", ceiling_flag=CEILING), xc.W_X_CEILING_UNCAPPED_MODE),
    (dict(mode="adaptive"), xc.W_X_MODE_UNKNOWN),
])
def test_launch_words_are_refused_by_name(kw, code):
    with pytest.raises(xc.XCurvesRefused) as e:
        _resolve(**kw)
    assert e.value.code == code


def test_curve_modes_size_ds_riegel_and_hand_the_front_its_flags(tmp_path):
    path = _write(tmp_path, _curves())
    env = xc.x_envelope(_curves())
    cur = _resolve(mode="curve", curves_path=path)
    assert cur.ceiling_for_d == env and cur.front_argv == ("--x-mode", "curve", "--x-curves", path)
    assert f"D's W50 riegel = {max(env, START_X)}" in cur.line
    ref = _resolve(mode="curve", curves_path=path, beyond="refuse")
    assert ref.front_argv[-2:] == ("--x-curves-beyond", "refuse")
    with pytest.raises(xc.XCurvesRefused) as e:
        _resolve(mode="curve", curves_path=path, hardware="RTX5090")
    assert e.value.code == xc.W_X_CURVES_FOREIGN


def test_launcher_refuses_foreign_cards_at_launch_and_passes_its_own(tmp_path):
    from sglang.srt.weg2 import form as weg2_form
    path = _write(tmp_path, _curves())
    boot_form = weg2_form.parse_form(FORM_ENV_VALUE)
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--x-mode", "curve",
                                      "--x-curves", path])
    ns.weg2_boot_form, ns.model = boot_form, f"/models/{MODEL}"
    xm = L.resolve_x_mode_launch(ns, CARDS, START_X)
    assert xm.mode == "curve" and xm.ceiling_for_d == xc.x_envelope(_curves())
    d_x, front_c, line = L.resolve_x_ceiling(xm.ceiling_for_d, START_X)
    assert d_x == front_c == max(START_X, xm.ceiling_for_d)      # D's riegel admits every curve X
    for removed in ("live", "curve-capped"):
        with pytest.raises(SystemExit):
            L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--x-mode", removed])
    with pytest.raises(SystemExit, match="W192 Weg2XCurvesForeign"):
        L.resolve_x_mode_launch(ns, CARDS[:2], START_X)


def test_front_argv_without_x_flags_is_byte_identical():
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
    base = L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 1, 1, START_X, START_X, "D")
    assert base == L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 1, 1, START_X, START_X, "D",
                                    x_mode_argv=())
    assert not any(a.startswith("--x-mode") or a.startswith("--x-curves") for a in base)
    fa = L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 1, 1, START_X, START_X, "D",
                          x_mode_argv=("--x-mode", "fixed"))
    assert fa == base + ["--x-mode", "fixed"]


# ------------------------------------------------------------ C: the front


def _front(**kw) -> Front:
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                 carrier_max_tokens=262144, tp_prefill_max_tokens=START_X, **kw)


def _feed(f: Front) -> None:
    f.note_x_sample("r_p", 3649.0)
    f.note_x_sample("flip_s", 2.6)
    f.note_x_sample("r_d", 1036.0)        # completes the triple: the live mode re-solves here


def test_default_front_is_the_live_mode_unchanged(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    assert f.x_mode == "live" and f._x_setup is None
    assert not any("X MODE" in m for m in caplog.messages)
    assert Front._x_route_note(f) == ""                     # the ROUTE-VERDICT stays byte-identical
    _feed(f)
    assert sum(len(v) for v in f._x_samples.values()) == 3  # the live samples still land


def test_fixed_pins_x_no_sample_no_resolve(monkeypatch):
    f = _front(x_mode="fixed", x_ceiling_tokens=CEILING)
    monkeypatch.setattr(Front, "resolve_x_live", lambda self: pytest.fail("fixed X re-solved"))
    _feed(f)
    assert f.tp_prefill_max_tokens == START_X and not any(f._x_samples.values())
    assert Front._x_route_of(f, rid="r", depth=0, k_flip=3) == START_X
    assert Front._x_route_note(f) == "; x_mode=fixed (no live re-solve)"


def test_curve_front_routes_each_request_on_its_own_x(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_WEG2_FORM", FORM_ENV_VALUE)
    monkeypatch.setattr(Front, "resolve_x_live", lambda self: pytest.fail("curve X re-solved"))
    caplog.set_level(logging.INFO, logger="weg2.front")
    path = _write(tmp_path, _curves())
    env = xc.x_envelope(_curves())
    f = _front(x_mode="curve", x_curves=path, x_ceiling_tokens=env)   # as the launcher hands it
    assert f.tp_prefill_max_tokens == env and f.flip_min_work_tokens == env
    assert any(m.startswith("WEG2 X MODE curve:") for m in caplog.messages)
    shallow = Front._x_route_of(f, rid="a", depth=0, k_flip=1)
    deep = Front._x_route_of(f, rid="b", depth=200000, k_flip=1)
    shared = Front._x_route_of(f, rid="c", depth=200000, k_flip=4)
    assert env >= shallow > deep > shared
    assert deep == xc.x_from_curves(depth=200000, open_requests=1, curves=_curves(), cap=env)
    note = Front._x_route_note(f)
    assert note.startswith(f"; x_mode=curve X_req={shared} ") and "k=4" in note
    assert "curves=nf.xcurves.json@synthetic" in note
    assert F.serviceable_route(deep + 1, deep + 1, deep, 373536) == "long"
    _feed(f)
    assert not any(f._x_samples.values())
    st = xc.state_block(mode=f.x_mode, source=f._x_setup.source, envelope=f._x_setup.envelope,
                        last=f._x_curve_last)
    assert st["x_mode"] == "curve" and st["x_curves_last"]["k"] == 4.0
    with pytest.raises(xc.XCurvesRefused) as e:
        _front(x_mode="live")
    assert e.value.code == xc.W_X_MODE_UNKNOWN


def test_curve_front_refuses_without_a_published_form_and_beyond_the_reach(tmp_path, monkeypatch):
    curves = _curves(d=_curve(_d_ms, depths=(0, 32768, 131072)))
    path = _write(tmp_path, curves)
    monkeypatch.delenv("SGLANG_WEG2_FORM", raising=False)
    with pytest.raises(xc.XCurvesRefused) as e:
        _front(x_mode="curve", x_curves=path)
    assert e.value.code == xc.W_X_CURVES_FOREIGN
    monkeypatch.setenv("SGLANG_WEG2_FORM", FORM_ENV_VALUE)
    f = _front(x_mode="curve", x_curves=path, x_curves_beyond="refuse",
               x_ceiling_tokens=xc.x_envelope(curves))
    with pytest.raises(xc.XCurvesRefused) as e:
        Front._x_route_of(f, rid="deep", depth=200000, k_flip=1)
    resp = Front._x_curve_refusal(f, "/v1/chat/completions", "deep", e.value)
    assert resp.status == 503 and b"W193 Weg2XCurveDepthBeyond" in resp.body
    f2 = _front(x_mode="curve", x_curves=path, x_ceiling_tokens=xc.x_envelope(curves))
    Front._x_route_of(f2, rid="deep", depth=200000, k_flip=1)
    assert f2.counters["x_curve_depth_clamped"] == 1 and "depth=200000->131072" in Front._x_route_note(f2)


def test_the_route_site_and_the_verdict_line_are_wired():
    """Bookkeeping: the arrival is routed through _x_route_of (not X-K-FLIP
    alone), a W193 answers before any verdict, and the ROUTE-VERDICT line
    carries the mode note as its last argument."""
    import inspect

    src = inspect.getsource(F)
    assert "_x_arrival = Front._x_route_of(self, rid=rid, depth=store_span," in src
    assert "return Front._x_curve_refusal(self, request.path, rid, _xcr)" in src
    assert '"est_prompt=%d chars=%d (#1290)%s",' in src
    assert "_x_note = Front._x_route_note(self)\n        route = serviceable_route(" in src
    assert "len(text), _x_note," in src
    assert "self._x_for_flip(1 + self._x_riders(), rid, \"route\")" not in src
