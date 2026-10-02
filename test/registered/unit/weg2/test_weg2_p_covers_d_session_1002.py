"""LAW (user 02.10. ~07:55Z): "kann nie passieren, weil P ja bis 262k prefillen
kann und eine einzelne D session nur maximal 262k haben kann. das selbe gilt
fuer yarn ... deswegen ja die stufen in P".

P's KV stages (262k, 524k YaRN, 786k) always cover D's maximum decode session,
so no request is ever D's to prefill for a capacity reason:
  * the CARRIER-EXCEEDS route ("carrier_single", one prefill on D, no leg 1)
    is deleted -- it fired 0x as a route on the last 12 NF boots of 02.10.
    (carrier 373536 > 262144);
  * the launcher refuses a boot whose P stage / P cap or carrier is below D's
    session (P-COVERS-D-SESSION), never a runtime fallback to a D prefill.
"""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

X = 4096


# ---- the route --------------------------------------------------------------------

def test_above_the_carrier_is_the_p_route_never_a_d_prefill():
    assert F.serviceable_route(500, 30000, X, 27466) == "long"
    assert F.serviceable_route(500, 30000, 0, 27466) == "long"
    assert F.serviceable_route(500, 20000, X, 27466) == "short"
    for unc in (1, X, X + 1, 10 ** 6):
        for est in (1, 27466, 27467, 10 ** 6):
            assert F.serviceable_route(unc, est, X, 27466) != "carrier_single"


def test_the_router_has_no_carrier_branch_and_the_drain_no_skip():
    src = inspect.getsource(F.Front)
    assert 'if route == "carrier_single":' not in src
    assert "CARRIER-EXCEEDS -> D single prefill" not in src
    assert "if p.skip_leg1:  # route CARRIER-EXCEEDS" not in src


# ---- the launcher invariant -------------------------------------------------------

def _argv_p(cap, pool, extra=()):
    return ["--max-kv-per-request", str(cap), "--max-total-tokens", str(pool), *extra]


def test_nf_standard_and_yarn_forms_pass():
    # NF y6y: P --max-kv-per-request 262144, the cut's pool 574337 (extra-p repeats 262144)
    ok, line = L.p_covers_d_session(_argv_p(262144, 574337, ("--max-total-tokens", "262144")), 262144)
    assert ok and "P-COVERS-D-SESSION D session (--max-kv-per-request)=262144" in line
    # -st-yarn2: P and D --max-kv-per-request 524288, P --max-total-tokens 524288
    assert L.p_covers_d_session(_argv_p(524288, 524288), 524288)[0]
    # 27B yarn2: 512000 on both (the cap+chunk floor), the cut's pool >= 514048
    assert L.p_covers_d_session(_argv_p(512000, 516574), 512000)[0]


def test_a_p_stage_below_ds_session_refuses_the_boot():
    ok, line = L.p_covers_d_session(_argv_p(524288, 262144), 524288)
    assert not ok
    assert "REFUSED" in line and "P --max-total-tokens (KV stage)=262144 < D session 524288" in line
    ok, line = L.p_covers_d_session(_argv_p(262144, 600000), 524288)
    assert not ok and "P --max-kv-per-request=262144 < D session 524288" in line


def test_ds_session_reads_its_extra_d_value_last_wins():
    assert L.d_session_tokens("--context-length 524288 --max-kv-per-request 524288", 262144) == 524288
    assert L.d_session_tokens("", 262144) == 262144
    assert L.d_session_tokens("--max-kv-per-request 1 --max-kv-per-request=300000", 262144) == 300000


def test_the_carrier_must_hold_ds_session():
    assert L.carrier_covers_d_session(373536, 262144)[0]
    ok, line = L.carrier_covers_d_session(373536, 524288)
    assert not ok and "P-COVERS-D-SESSION carrier_max=373536 D session=524288 REFUSED" in line


def test_the_launcher_refuses_before_spec_p_and_after_the_carrier_census():
    src = inspect.getsource(L)
    i = src.index("_cov_ok, _cov_line = p_covers_d_session(")
    assert i < src.index('spec_p = GroupSpec("P", PORT_P, transport_argv(shipped_argv_p')
    assert "raise Weg2LaunchRefused(_cov_line)" in src[i:i + 400]
    j = src.index("_cc_ok, _cc_line = carrier_covers_d_session(")
    assert src.index("state.carrier_max_tokens = carrier_max_tokens") < j
    assert "raise Weg2LaunchRefused(_cc_line)" in src[j:j + 500]


# ---- 27B port (02.10.): the two 27B boot forms, numbers from their boot logs -------------

def test_27b_row_authority_cut43_passes():
    # boot_weg2_dkr27browauthoritycut43bar1fs10020837_6dd8d7e68c: 'group P: --max-running-requests 1
    # --max-kv-per-request 262144', 'PP-CUT P-CAP: --max-total-tokens=310242', group D argv
    # '--max-kv-per-request 262144', front 'carrier_max_tokens=648806'
    argv_p = ["--max-kv-per-request", "262144", "--max-total-tokens=310242"]
    ok, line = L.p_covers_d_session(argv_p, L.d_session_tokens("", 262144))
    assert ok, line
    assert "P cap=262144 P stage pool=310242 ok" in line
    assert L.carrier_covers_d_session(648806, 262144)[0]


def test_27b_nvfp4_dual1i_passes():
    # boot_weg2_dkr27bnvfp4dual1ibar1fs10010932_90945aeba9: P and D argv '--max-kv-per-request 131072',
    # the shipped P argv names no --max-total-tokens (unnamed, never assumed), carrier 648806
    argv_p = ["--context-length", "262144", "--max-kv-per-request", "131072"]
    ok, line = L.p_covers_d_session(argv_p, L.d_session_tokens("", 131072))
    assert ok and "P stage pool=unnamed ok" in line, line
    assert L.carrier_covers_d_session(648806, 131072)[0]
