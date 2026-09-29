# SPDX-License-Identifier: Apache-2.0
"""27B early D start, step 1 (29.09.): D's expectation from P's records.

Why: ``--weg2-d-early-start`` plans group D before P's first sleep through the
one selector ``d_expect_dormant_other``. On the 27B that selector returned the
legacy term ``dc_expect_d + P_WINDOWS_MIB - D_WINDOWS_MIB`` -- D's own reserve
standing in for P's residue. Boot dkr27browauthoritybar1w109290020
(bb82fbcb68): dc_expect_d 2042/1614/1614 (nvml1/0/2) -> legacy 2090/1662/1662
MiB, while the launcher measured P after its sleep at 1104/588/814 MiB. The
gate only checks planned <= measured and never re-solves D, so an early 27B D
would have run ~1 GiB per card short of the serial one (NF z30r3: 540-960).

What changes:
* a registry switch of its own, ``d_expect_from_p_records`` (qwen27b True,
  nextflash True) -- ``d_residue_census`` (the RESERVE question) untouched;
* ``d_early_start_proven`` (nextflash True, qwen27b False): ``auto`` = on only
  where the metal proved it; ``auto`` on None / "" / qwen27b stays OFF;
* W185 refuses an explicit ``on`` only where the expectation is still legacy,
  so the 27B proof boot can run with ``on``.

Fixtures: group-P dormant-image records of the 27B sidecar
(/spinning/docker-acceptance/27b/evidence/weg2_measured_record.json), the
newest 10 same-form boots, values verbatim. Hermetic: no NVML, no GPU.
"""
import argparse
import dataclasses
import inspect
import json
import os
import tempfile
import unittest
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as weg2_form
from sglang.srt.weg2 import launcher as L

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=4, suite="base-a-test-cpu")
except Exception:  # pragma: no cover - registration is optional off-CI
    pass

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"  # nvml1 5090
S0 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"   # nvml0 3080
S2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"   # nvml2 3080
CARDS = [
    L.Card(1, BIG, "NVIDIA GeForce RTX 5090", 32607),
    L.Card(0, S0, "NVIDIA GeForce RTX 3080", 20480),
    L.Card(2, S2, "NVIDIA GeForce RTX 3080", 20480),
]
XCHG = L.WEIGHT_SOURCE_EXCHANGE
Q27 = weg2_form.PROFILE_QWEN27B
NF = weg2_form.PROFILE_NEXTFLASH

# dkr27browauthoritybar1w109290020 launcher.log:55 (#1444 DC-RESIDUE group=D)
DC_EXPECT_D_27B = {BIG: 2042, S0: 1614, S2: 1614}
LEGACY_27B = {BIG: 2090, S0: 1662, S2: 1662}
# same boot, launcher.log:238-240 (WEG2-DC group=P, after P's sleep)
DC_P_27B = {BIG: 1104, S0: 588, S2: 814}
# the 27B sidecar's group-P exchange records, newest first (boot, at, commit,
# nvml1, nvml0, nvml2)
SERIES_27B = [
    ("dkr27browauthoritybar1w109290020", "2026-09-29T00:23:18Z", "bb82fbcb68", 1138, 598, 840),
    ("dkr27browauthoritybar1w109281851", "2026-09-28T18:53:56Z", "85386b1df1", 1178, 614, 842),
    ("dkr27browauthoritybar1w109281715", "2026-09-28T17:18:14Z", "dc1d5d3da0", 1178, 614, 842),
    ("dkr27bparkdraftbar1w109281421", "2026-09-28T14:24:56Z", "82fa502795", 1240, 630, 852),
    ("dkr27breleasedraftbar1w109281340", "2026-09-28T13:43:57Z", "82fa502795", 1182, 648, 842),
    ("dkr27browauthoritybar1w109280820", "2026-09-28T08:23:19Z", "6617ab7d71", 1372, 730, 992),
    ("dkr27browauthoritybar1w5l309280802", "2026-09-28T08:04:44Z", "1961f756ad", 1312, 690, 914),
    ("dkr27browauthoritybar1w109280741", "2026-09-28T07:43:38Z", "1961f756ad", 1272, 684, 908),
    ("dkr27breleasedraftbar1w109271525", "2026-09-27T15:28:34Z", "21d7e3b188", 1340, 724, 914),
    ("dkr27breleasedraftbar1w109271344", "2026-09-27T13:47:39Z", "732d629b21", 1354, 704, 926),
]


def _rec(row):
    tag, at, commit, b, s0, s2 = row
    return {"group": "P", "boot_tag": tag, "at": at, "commit": commit,
            "model_digest": "", "sampled_at_flip_epoch": 1, "interleaved": True,
            "rss_shmem_gib": 97.59,  # an image entry (the sidecar reader's shape filter)
            "form_key": "ranks=3;wtags=28.83", "vram_residue_form": XCHG,
            "vram_residue_mib": {S0: s0, BIG: b, S2: s2}}


def _ns(mode, profile):
    return argparse.Namespace(weg2_d_early_start=mode, profile=profile)


@pytest.fixture(autouse=True)
def _no_kill_switch(monkeypatch):
    monkeypatch.delenv(L.P_DORMANT_EXPECT_ENV, raising=False)


# --- the registry: two new fields, the reserve switch untouched --------------


def test_registry_rows():
    q, n = weg2_form.PROFILES[Q27], weg2_form.PROFILES[NF]
    assert (q.d_expect_from_p_records, q.d_early_start_proven) == (True, False)
    assert (n.d_expect_from_p_records, n.d_early_start_proven) == (True, True)
    # the reserve question keeps its own answer
    assert (q.d_residue_census, n.d_residue_census) == (False, True)
    assert not L.xchg_census_is_reserve(Q27) and L.xchg_census_is_reserve(NF)
    # a row without the fields (a future profile) defaults to legacy / unproven
    f = {x.name: x.default for x in dataclasses.fields(weg2_form.ModelProfile)}
    assert f["d_expect_from_p_records"] is False and f["d_early_start_proven"] is False


# --- 27B: the expectation is P's measured residue, not the legacy term -------


def test_27b_expectation_is_the_record_not_legacy():
    newest = [_rec(SERIES_27B[0])]
    for prof in (Q27, None):
        got, why = L.d_expect_dormant_other(CARDS, DC_EXPECT_D_27B, newest, XCHG, prof)
        assert got == {BIG: 1138, S0: 598, S2: 840}, prof
        assert not why.startswith("legacy") and "dkr27browauthoritybar1w109290020" in why
    # what the legacy term booked on this boot, and what P really left (launcher):
    # the record sits 34/10/26 MiB above the launcher's reading -> gate go
    assert {u: LEGACY_27B[u] - got[u] for u in got} == {BIG: 952, S0: 1064, S2: 822}
    assert {u: got[u] - DC_P_27B[u] for u in got} == {BIG: 34, S0: 10, S2: 26}


def test_27b_expectation_is_the_max_over_the_newest_8_boots():
    got, why = L.p_dormant_from_records([_rec(r) for r in SERIES_27B], CARDS, XCHG)
    # newest 8 = 09290020 .. 09280741; 09280820 carries the maximum
    assert got == {BIG: 1372, S0: 730, S2: 992}
    assert "newest 8 of N=8" in why and "dkr27breleasedraftbar1w109271525" not in why
    sel, _ = L.d_expect_dormant_other(CARDS, DC_EXPECT_D_27B,
                                      [_rec(r) for r in SERIES_27B], XCHG, Q27)
    assert sel == got
    # still far below legacy; the planned D budget is lower than the serial
    # one by (max - launcher reading) per card -- the price the gate lets pass
    assert {u: LEGACY_27B[u] - sel[u] for u in sel} == {BIG: 718, S0: 932, S2: 670}
    assert {u: sel[u] - DC_P_27B[u] for u in sel} == {BIG: 268, S0: 142, S2: 178}


def test_27b_without_record_is_named_legacy():
    for recs in (None, []):
        got, why = L.d_expect_dormant_other(CARDS, DC_EXPECT_D_27B, recs, XCHG, Q27)
        assert got == LEGACY_27B and "UNMEASURED" in why


def test_27b_kill_switch(monkeypatch):
    monkeypatch.setenv(L.P_DORMANT_EXPECT_ENV, "0")
    got, why = L.d_expect_dormant_other(CARDS, DC_EXPECT_D_27B, [_rec(SERIES_27B[0])], XCHG, Q27)
    assert got == LEGACY_27B and "=0" in why


def test_27b_reads_its_p_records_through_the_gated_reader():
    rows = [_rec(r) for r in SERIES_27B] + [dict(_rec(SERIES_27B[0]), group="D")]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "weg2_measured_record.json")
        with open(path, "w") as f:
            json.dump({"samples": rows}, f)
        with mock.patch.object(L, "measured_record_path", return_value=path):
            for prof in (Q27, None):
                recs = L.p_dormant_records(prof, lambda e: True)
                # every group-P entry, oldest first (the D row is not P's)
                assert [r["boot_tag"] for r in recs] == [r[0] for r in reversed(SERIES_27B)]
            # no calibration identity: still no read
            assert L.p_dormant_records(Q27, None) is None


def test_a_legacy_profile_still_never_reads_and_prices_legacy():
    row = dataclasses.replace(weg2_form.PROFILES[Q27], d_expect_from_p_records=False)
    with mock.patch.dict(weg2_form.PROFILES, {Q27: row}), \
            mock.patch.object(L.host_ledger, "read_measured_records",
                              side_effect=AssertionError("read the sidecar")):
        assert L.p_dormant_records(Q27, lambda e: True) is None
        got, why = L.d_expect_dormant_other(CARDS, DC_EXPECT_D_27B, [_rec(SERIES_27B[0])],
                                            XCHG, Q27)
        assert got == LEGACY_27B and why.startswith("legacy (profile's")


# --- auto / on / off and W185 -------------------------------------------------


def test_auto_resolution_is_pinned():
    assert L._d_early_start_armed(_ns("auto", NF))
    for prof in (Q27, None, ""):
        assert not L._d_early_start_armed(_ns("auto", prof)), prof
    for prof in (NF, Q27, None):
        assert L._d_early_start_armed(_ns("on", prof))
        assert not L._d_early_start_armed(_ns("off", prof))
    # the parser default: auto on the default profile (27B) = serial
    ns = L.build_parser().parse_known_args(["--tree", "t", "--tag", "x"])[0]
    assert ns.weg2_d_early_start == "auto" and not L._d_early_start_armed(ns)


def test_auto_follows_the_proof_field_not_the_reserve_field():
    proven = dataclasses.replace(weg2_form.PROFILES[Q27], d_early_start_proven=True)
    with mock.patch.dict(weg2_form.PROFILES, {Q27: proven}):
        assert L._d_early_start_armed(_ns("auto", Q27))
        assert not L.xchg_census_is_reserve(Q27)
    # proof without the record expectation does not arm auto
    half = dataclasses.replace(weg2_form.PROFILES[NF], d_expect_from_p_records=False)
    with mock.patch.dict(weg2_form.PROFILES, {NF: half}):
        assert not L._d_early_start_armed(_ns("auto", NF))


def test_w185_lets_the_27b_proof_boot_run_with_on():
    for prof in (Q27, None, NF):
        L.refuse_d_early_start_unreviewed(_ns("on", prof))
        L.refuse_d_early_start_unreviewed(_ns("auto", prof))
        L.refuse_d_early_start_unreviewed(_ns("off", prof))


def test_w185_refuses_on_where_the_expectation_is_still_legacy():
    row = dataclasses.replace(weg2_form.PROFILES[Q27], d_expect_from_p_records=False)
    with mock.patch.dict(weg2_form.PROFILES, {Q27: row}):
        with pytest.raises(L.Weg2DEarlyStartUnreviewed, match="W185.*d_expect_from_p_records"):
            L.refuse_d_early_start_unreviewed(_ns("on", Q27))
        # a default never refuses
        L.refuse_d_early_start_unreviewed(_ns("auto", Q27))


# --- the D-EXPECT CHECK names the early plan on the 27B (no map pass there) ---


def test_early_branch_publishes_the_expectation_for_the_check():
    src = inspect.getsource(L.main)
    early = src[src.index("_early_d = None\n    if _d_early_start_armed(ns):"):
                src.index("def _d_early_verdict(")]
    i_sel = early.index("d_expect_dormant_other(")
    i_pub = early.index("ns._d_expect_other = _early_other")
    assert i_sel < i_pub < early.index("_d_spec_from(")
    # only when the map pass did not already publish it (NF), only a record value
    guard = early[early.rindex("if (", 0, i_pub):i_pub]
    assert 'getattr(ns, "_d_expect_other", None) is None' in guard
    assert '_early_other_why.startswith("legacy")' in guard


if __name__ == "__main__":
    unittest.main()
