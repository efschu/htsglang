"""KALIB-LIVE-X #235 (24.10.): the front takes the curve measured on THIS boot while serving.

User order 24.10. (memory kalibrierung-je-boot-p-kurve-x-1024): "fuer jedes unserer boots vorher nochmal einmal p kurve und
X durchgemessen" -- the measured X goes into the boot, not fixed profile values. SOLL v2 section 2 (X = D-Prefill-Max at the
cache depth from the curves), R7 (no D prefill over X; D's W50 riegel stays the hard bound), R11 (the front decides alone).

Metal (boot_weg2_dkr27bbar1drei10241731_c07b574410_1024_173149): KALIB wrote X@0k 8418 at 17:36:21Z, the front kept
routing on the live re-solve X=12288 (55/62 SET-VERDICTs after it) -- it read X only at start, and KALIB-GUARD had
shelved the previous image's curve at 17:31:33Z, so the boot started in the live mode.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging

import pytest

from flliper.srt.pdflip import front as F
from flliper.srt.pdflip import x_curves as xc
from flliper.srt.pdflip.front import Front

MODEL = "Qwen3.8-27B-INT8-gdncov-vocabembed"
FORM_AXES = "arch=dense,experts=none,draft=dflash,kv=paged_dcp,flip=family"
FORM_ENV_VALUE = ("arch=dense,experts=none,draft=dflash,p_draft=cold,kv=paged_dcp,flip=family,"
                  f"vision=transient,profile=qwen27b,model=/models/{MODEL}")
HW = "RTX5090,RTX3080,RTX3080"
TAG = "dkr27bbar1drei10241731"
COMMIT = "c07b57441093560f17c75899c6ff5cbc01a257ac"
SOURCE = f"kalib:boot_weg2_{TAG}_{COMMIT[:10]}_1024_173149"
START_X = 4096
CEILING = 12288
DEPTHS = (0, 32768, 131072, 262144)
NS = (256, 512, 1024, 2048, 4096)


def _d_ms(depth, n, k=1.0):
    return 40.0 + k * (0.9 + 0.004 * depth / 1000.0) * n


def _p_ms(depth, n):
    return 120.0 + (0.08 + 0.0008 * depth / 1000.0) * n


def _curve(fn):
    return xc.PrefillCurve(
        rows=tuple(xc.CurveRow(depth=d, n=NS, ms=tuple(fn(d, n) for n in NS)) for d in DEPTHS),
        chunk_tokens=256, instrument="synthetic")


def _curves(*, d_k=1.0, source=SOURCE, model=MODEL, form=FORM_AXES):
    return xc.XCurves(
        format=xc.X_CURVES_FORMAT,
        identity=xc.CurveIdentity(model=model, form=form, hardware=HW, created="2026-10-24T17:36:21Z", source=source),
        p_curve=_curve(_p_ms), d_curve=_curve(lambda d, n: _d_ms(d, n, d_k)),
        flip_price=xc.FlipPrice(depth=(0,), seconds=(2.954,), attribution="row+theil-sen-flat", pairs=6))


def _write(tmp_path, curves, name="flliper-27b-int8.json"):
    p = tmp_path / name
    p.write_bytes(xc.encode(curves))
    return str(p)


def _front(tag=TAG, **kw) -> Front:
    kw.setdefault("x_ceiling_tokens", CEILING)
    return Front("http://p", "http://d", "D", tag, "", 0, 0, {}, 45.0, commit=COMMIT,
                 carrier_max_tokens=262144, tp_prefill_max_tokens=START_X, **kw)


@pytest.fixture(autouse=True)
def _form(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FORM", FORM_ENV_VALUE)


def _env(curves):
    return xc.x_envelope(curves, cap=CEILING)


# ------------------------------------------------------------------ the reload switches


def test_live_started_front_takes_the_same_boot_curve_and_x_changes(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="pdflip.front")
    f = _front()
    f.tp_prefill_max_tokens = CEILING            # the live re-solve stood at the ceiling (metal: X=12288)
    assert f.x_mode == xc.X_MODE_LIVE and f._x_setup is None
    curves = _curves()
    res = Front.reload_x_curves(f, _write(tmp_path, curves), who="kalib_boot test")
    assert res["switched"] is True
    env = _env(curves)
    assert env < CEILING                          # the measured X is not the ceiling
    assert f.x_mode == xc.X_MODE_CURVE and f.tp_prefill_max_tokens == env
    assert f.flip_min_work_tokens == env          # follows X as in a curve-started boot
    assert Front._x_route_of(f, rid="a", depth=0, k_flip=1) == xc.x_from_curves(
        depth=0, open_requests=1, curves=curves, cap=env)
    assert Front._x_at_depth(f, 131072) < Front._x_at_depth(f, 0) <= env
    assert res["old"] == {"x_mode": "live", "x_source": "live re-solve (no curve file at start)",
                          "x_at_0": CEILING, "x_boot": CEILING}
    assert res["new"]["envelope"] == env and res["new"]["cap"] == CEILING
    line = [m for m in caplog.messages if m.startswith("PDFLIP X-CURVES RELOAD #1:")]
    assert len(line) == 1 and "X source live (live re-solve" in line[0] and f"X@0k={CEILING}" in line[0]
    assert f"-> curve flliper-27b-int8.json@{SOURCE}" in line[0] and f"envelope X={env}" in line[0]
    assert "[kalib_boot test]" in line[0]
    # curve mode from here on: no live sample lands, nothing re-solves the X back to the ceiling
    monkeypatch.setattr(Front, "resolve_x_live", lambda self: pytest.fail("re-solved after the reload"))
    f.note_x_sample("r_p", 3649.0)
    f.note_x_sample("flip_s", 2.6)
    f.note_x_sample("r_d", 1036.0)
    assert f.tp_prefill_max_tokens == env and f.counters["x_curves_reloads"] == 1
    st = xc.state_block(mode=f.x_mode, source=f._x_setup.source, envelope=f._x_setup.envelope, last=f._x_curve_last)
    assert st["x_mode"] == "curve" and st["x_curves_envelope"] == env


def test_a_rewritten_curve_of_the_same_boot_moves_x_again(tmp_path):
    f = _front()
    path = _write(tmp_path, _curves())
    assert Front.reload_x_curves(f, path)["switched"]
    x_first = Front._x_at_depth(f, 0)
    dearer = _curves(d_k=1.6)                     # a later measurement: D dearer per token -> X lower
    path2 = _write(tmp_path, dearer)
    res = Front.reload_x_curves(f, path2)
    assert res["switched"] and res["old"]["x_mode"] == "curve" and res["old"]["x_at_0"] == x_first
    assert Front._x_at_depth(f, 0) < x_first and f.tp_prefill_max_tokens == _env(dearer)
    assert f.counters["x_curves_reloads"] == 2


def test_the_cap_is_ds_riegel_as_this_boot_runs_it(tmp_path):
    f = _front(x_ceiling_tokens=4600)
    curves = _curves()
    assert xc.x_envelope(curves) > 4600
    assert Front.reload_x_curves(f, _write(tmp_path, curves))["switched"]
    assert f.tp_prefill_max_tokens == 4600 and Front._x_at_depth(f, 0) <= 4600   # never above what D admits (H84)


# ------------------------------------------------------------------ invalid -> nothing changes


@pytest.mark.parametrize("make,code,part", [
    (lambda tp: str(tp / "absent.json"), xc.W_X_CURVES_MISSING, "absent.json"),
    (lambda tp: "", xc.W_X_CURVES_MISSING, "without a file path"),
    (lambda tp: (tp / "bad.json").write_text("{not json") and str(tp / "bad.json"), xc.W_X_CURVES_MALFORMED, ""),
    (lambda tp: _write(tp, _curves(model="Qwen3.8-Flash-Next-int4")), xc.W_X_CURVES_FOREIGN, "model"),
    (lambda tp: _write(tp, _curves(form="arch=moe,experts=offload,draft=dflash,kv=qsa_forma,flip=family")),
     xc.W_X_CURVES_FOREIGN, "form"),
    (lambda tp: _write(tp, _curves(source=f"kalib:boot_weg2_dkr27bbar1drei10241106_{COMMIT[:10]}_1024_110644")),
     xc.W_X_CURVES_FOREIGN, "names another boot"),
    (lambda tp: _write(tp, _curves(source=f"kalib:boot_weg2_{TAG}_1ecda02a58_1024_173149")),
     xc.W_X_CURVES_FOREIGN, "another tree"),
])
def test_an_invalid_curve_is_refused_by_name_and_the_old_x_stays(tmp_path, caplog, make, code, part):
    caplog.set_level(logging.INFO, logger="pdflip.front")
    f = _front()
    f.tp_prefill_max_tokens = CEILING
    res = Front.reload_x_curves(f, make(tmp_path), who="kalib_boot test")
    assert res["switched"] is False and res["code"] == code and part in res["detail"]
    assert f.x_mode == xc.X_MODE_LIVE and f._x_setup is None and f.tp_prefill_max_tokens == CEILING
    assert f.counters["x_curves_reloads"] == 0 and f.counters["x_curves_reload_refused"] == 1
    line = [m for m in caplog.messages if m.startswith(f"PDFLIP X-CURVES RELOAD REFUSED {code}:")]
    assert len(line) == 1 and "X stays live" in line[0] and f"X@0k={CEILING}" in line[0]


def test_a_refused_reload_keeps_an_earlier_curve(tmp_path):
    f = _front()
    good = _curves()
    assert Front.reload_x_curves(f, _write(tmp_path, good))["switched"]
    setup = f._x_setup
    res = Front.reload_x_curves(f, _write(tmp_path, _curves(model="other", d_k=3.0), name="other.json"))
    assert not res["switched"] and f._x_setup is setup and f.tp_prefill_max_tokens == _env(good)


def test_dual_and_an_explicit_fixed_pin_are_refused(tmp_path):
    path = _write(tmp_path, _curves())
    fd = _front()
    fd.dual_layout = True
    res = Front.reload_x_curves(fd, path)
    assert not res["switched"] and res["code"] == xc.W_X_MODE_IN_DUAL
    ff = _front(x_mode="fixed")
    res = Front.reload_x_curves(ff, path)
    assert not res["switched"] and res["code"] == xc.W_X_CURVES_WITHOUT_MODE and ff.x_mode == "fixed"
    assert ff.tp_prefill_max_tokens == START_X


def test_a_front_without_a_boot_tag_takes_no_reload(tmp_path):
    f = _front(tag="")
    res = Front.reload_x_curves(f, _write(tmp_path, _curves()))
    assert not res["switched"] and "no boot tag" in res["detail"]


# ------------------------------------------------------------------ the admin route


class _Req:
    def __init__(self, method="POST", body=None, auth=None, remote="172.17.0.1"):
        self.method = method
        self.headers = {} if auth is None else {"Authorization": auth}
        self.remote = remote
        self._body = body

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _call(f, req):
    return asyncio.run(Front.handle_x_curves(f, req))


def test_route_needs_the_boot_admin_key_and_reloads(tmp_path):
    f = _front()
    f.admin_key = "k" * 43
    path = _write(tmp_path, _curves())
    assert _call(f, _Req(body={"path": path}, auth="Bearer wrong")).status == 401
    assert f.x_mode == xc.X_MODE_LIVE
    r = _call(f, _Req(body={"path": path, "who": "kalib_boot test"}, auth=f"Bearer {f.admin_key}"))
    assert r.status == 200 and json.loads(r.text)["switched"] is True and f.x_mode == xc.X_MODE_CURVE
    r = _call(f, _Req(body={"path": str(tmp_path / "absent.json")}, auth=f"Bearer {f.admin_key}"))
    assert r.status == 409 and json.loads(r.text)["code"] == xc.W_X_CURVES_MISSING
    g = json.loads(_call(f, _Req(method="GET", auth=f"Bearer {f.admin_key}")).text)
    assert g["x_mode"] == "curve" and g["reloads"] == 1 and g["refused"] == 1
    assert _call(f, _Req(body=ValueError("no json"), auth=f"Bearer {f.admin_key}")).status == 400


def test_route_without_a_key_serves_loopback_only(tmp_path):
    f = _front()
    f.admin_key = None
    path = _write(tmp_path, _curves())
    assert _call(f, _Req(body={"path": path})).status == 403
    assert _call(f, _Req(body={"path": path}, remote="127.0.0.1")).status == 200


def test_the_route_is_registered_on_every_boot():
    src = inspect.getsource(F.main)
    assert 'app.router.add_post("/pdflip/x-curves", front.handle_x_curves)' in src
    assert 'app.router.add_get("/pdflip/x-curves", front.handle_x_curves)' in src


def test_same_boot_problems_is_pure():
    assert xc.same_boot_problems(SOURCE, boot_tag=TAG, boot_commit=COMMIT) == []
    assert xc.same_boot_problems(SOURCE, boot_tag=TAG, boot_commit="") == []
    assert len(xc.same_boot_problems("synthetic", boot_tag=TAG, boot_commit=COMMIT)) == 2
