# SPDX-License-Identifier: Apache-2.0
"""#584 -- the card-rate measurement pass, and the refusal it makes remediable.

WHAT THESE TESTS PIN, and why each one exists rather than being obvious.

#485's window (WINDOW_VERDICT_485_R12.md section 2, defect 3) found that
``--pp-solve-cut`` refuses every card on every rig, because:

  * no SEED_CARDS entry carries gemm_tflops/membw_gbs, and
  * ``_pp_cut_card_rates`` built a seed-only ``CardLibrary()``, never calling
    the ``CardLibrary.load`` that exists for exactly this, and
  * ``load``/``save`` take an explicit path that NOTHING in the tree computed,
    so the store had no location and could not be filled by anyone.

The refusal was therefore correct and unremediable at the same time. These
tests pin both halves of the fix: the refusal still fires and is now
actionable, and a measured artifact is actually loaded and used.

The one that matters most is T4. A test suite can pass while the wiring is
absent, if every test constructs its own library. T4 goes through the real
``_pp_cut_card_rates`` with only a file on disk, so it fails if the handler
ever goes back to ``CardLibrary()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang.srt.planner import card_rate_pass as crp
from sglang.srt.planner.card_library import CardLibrary
from sglang.srt.rigmon.card_probe import CardProbeMeasurement, CardProbeProfile

# The two cards this rig actually carries, with the rates a previous shift
# measured on them (evidence-631/s50/gate_check.py). Used as fixtures only --
# nothing here reads a device.
R5090 = "NVIDIA GeForce RTX 5090"
R3080 = "NVIDIA GeForce RTX 3080"
U5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
U3080_A = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
U3080_B = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"


def _measurement(uuid, name, gemm, read_gbs, total_mib, throttle=()):
    return CardProbeMeasurement(
        uuid=uuid,
        name=name,
        cuda_index=0,
        total_mib=total_mib,
        gemm_bf16_tflops=gemm,
        membw_read_gbs=read_gbs,
        throttle_reasons=list(throttle),
    )


def _profile(*cards):
    return CardProbeProfile(cards=list(cards))


@pytest.fixture
def rig_profile():
    """The real three-card rig: one 5090 and TWO 3080s that differ slightly."""
    return _profile(
        _measurement(U5090, R5090, 231.97, 1533.8, 32607),
        _measurement(U3080_A, R3080, 65.57, 717.4, 20480),
        # The second 3080 measures a little slower. This is the case a
        # name-keyed store cannot represent and must therefore resolve.
        _measurement(U3080_B, R3080, 61.20, 702.1, 20480),
    )


@pytest.fixture
def lib_path(tmp_path, monkeypatch):
    p = tmp_path / "card_library.json"
    monkeypatch.setenv("SGLANG_CARD_LIBRARY", str(p))
    return p


# ---------------------------------------------------------------------------
# T1 -- the store now HAS a location, which is why load could never be called
# ---------------------------------------------------------------------------

def test_t1_library_path_resolves_and_is_overridable(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_CARD_LIBRARY", raising=False)
    default = crp.card_library_path()
    assert default.endswith(crp.CARD_LIBRARY_BASENAME)
    # It must live beside the #213 probe cache the rates are projected from.
    from sglang.srt.rigmon.card_probe import CACHE_DIR
    assert str(Path(default).parent) == str(Path(CACHE_DIR))

    monkeypatch.setenv("SGLANG_CARD_LIBRARY", str(tmp_path / "elsewhere.json"))
    assert crp.card_library_path() == str(tmp_path / "elsewhere.json")
    # An explicit argument beats the environment.
    assert crp.card_library_path(str(tmp_path / "explicit.json")).endswith(
        "explicit.json"
    )


def test_t1b_no_seed_card_carries_a_measured_rate(self_check=None):
    """The premise of the whole ticket, pinned so it cannot rot silently."""
    from sglang.srt.planner.card_library import SEED_CARDS

    rated = [
        s.name for s in SEED_CARDS.values() if s.gemm_tflops or s.membw_gbs
    ]
    assert rated == [], (
        f"seed cards now carry measured rates ({rated}); if that was "
        f"deliberate, the gate can be fed by fabrication and #584's refusal is "
        f"no longer meaningful"
    )


# ---------------------------------------------------------------------------
# T2 -- identity: UUID is the key, and a duplicate NAME resolves conservatively
# ---------------------------------------------------------------------------

def test_t2_rates_are_keyed_by_uuid_never_by_index(rig_profile):
    rates = crp.rates_by_uuid(rig_profile)
    assert set(rates) == {U5090, U3080_A, U3080_B}
    assert rates[U5090].gemm_tflops == pytest.approx(231.97)
    assert rates[U5090].membw_gbs == pytest.approx(1533.8)
    # Two distinct cards share a name and remain distinct entries.
    assert rates[U3080_A].name == rates[U3080_B].name == R3080
    assert rates[U3080_A].gemm_tflops != rates[U3080_B].gemm_tflops


def test_t3_duplicate_name_takes_the_slowest_instance(rig_profile):
    """The pacer sets the makespan, so the faster twin must not price the name.

    Taking the faster 3080 (65.57) would under-predict the makespan of whichever
    stage lands on the slower one (61.20), and an under-predicted makespan is
    how a cut gets admitted that should have been refused.
    """
    by_name = crp.rates_by_name(crp.rates_by_uuid(rig_profile))
    assert by_name[R3080].gemm_tflops == pytest.approx(61.20)
    assert by_name[R3080].membw_gbs == pytest.approx(702.1)
    assert by_name[R5090].gemm_tflops == pytest.approx(231.97)
    # A combined entry must not claim to be one card's measurement.
    assert by_name[R3080].uuid == ""
    assert by_name[R3080].pci_bus_id is None


def test_t3b_incomplete_measurement_never_wins_a_min(rig_profile):
    """A card that measured nothing must be skipped, not folded in as None."""
    partial = _profile(
        _measurement(U3080_A, R3080, 65.57, 717.4, 20480),
        _measurement(U3080_B, R3080, None, None, 20480),
    )
    by_name = crp.rates_by_name(crp.rates_by_uuid(partial))
    assert by_name[R3080].gemm_tflops == pytest.approx(65.57)


# ---------------------------------------------------------------------------
# T4 -- THE WIRING. The real handler, fed only by a file on disk.
# ---------------------------------------------------------------------------

def test_t4_solver_loads_and_uses_the_measured_library(rig_profile, lib_path):
    """The can-fail proof for the wiring itself.

    Nothing here hands the handler a library. It writes an artifact, then calls
    ``_pp_cut_card_rates``, which must find it. If that method ever reverts to
    ``CardLibrary()`` this test fails -- which is the exact regression #485
    spent a window discovering.
    """
    from sglang.srt.server_args import ServerArgs

    report = crp.run_card_rate_pass(path=str(lib_path), profile=rig_profile)
    assert report.wrote, report.format_text()
    assert lib_path.is_file()

    args = ServerArgs.__new__(ServerArgs)
    args.pp_size = 3
    rates = args._pp_cut_card_rates([R5090, R3080, R3080])

    assert [r[0] for r in rates] == [R5090, R3080, R3080]
    assert rates[0][1] == pytest.approx(231.97)      # 5090 gemm
    assert rates[0][2] == pytest.approx(1533.8)      # 5090 membw
    # Both 3080 stages priced at the SLOWER twin's rate.
    assert rates[1][1] == pytest.approx(61.20)
    assert rates[2][1] == pytest.approx(61.20)


def test_t5_solver_refuses_loudly_when_no_pass_has_run(lib_path):
    """No artifact -> refuse, and the refusal must name the remedy.

    The seed catalog must NOT be used as a fallback: it carries capacity only,
    so a fallback would price every stage from an absent number while looking
    like it had a catalog.
    """
    from sglang.srt.server_args import ServerArgs

    assert not lib_path.exists()
    args = ServerArgs.__new__(ServerArgs)
    args.pp_size = 3

    with pytest.raises(ValueError) as exc:
        args._pp_cut_card_rates([R5090, R3080, R3080])

    msg = str(exc.value)
    assert "no measured card-rate library" in msg
    assert "card_rate_pass" in msg, "the refusal does not name its remedy"
    assert str(lib_path) in msg, "the refusal does not say WHERE it looked"


def test_t6_a_library_without_this_card_still_refuses(rig_profile, lib_path):
    """A pass that covered other cards must not price this one."""
    from sglang.srt.server_args import ServerArgs

    crp.run_card_rate_pass(
        path=str(lib_path),
        profile=_profile(_measurement(U5090, R5090, 231.97, 1533.8, 32607)),
    )
    args = ServerArgs.__new__(ServerArgs)
    args.pp_size = 2
    with pytest.raises(ValueError) as exc:
        args._pp_cut_card_rates([R5090, R3080])
    assert "carries no measured gemm/bandwidth rate" in str(exc.value)


def test_t7_seed_capacity_survives_the_measurement(rig_profile, lib_path):
    """The pass must add rates without discarding curated capacity/nameplate."""
    seeded = CardLibrary()
    before = seeded.get(R5090) if seeded.has(R5090) else None
    crp.run_card_rate_pass(path=str(lib_path), profile=rig_profile)

    lib = crp.load_measured_library(str(lib_path))
    spec = lib.get(R5090)
    assert spec.gemm_tflops == pytest.approx(231.97)
    if before is not None:
        assert spec.total_mib == before.total_mib
        assert spec.peak_membw_gbs == before.peak_membw_gbs


# ---------------------------------------------------------------------------
# T8 -- the artifact keeps the identity the name-keyed library cannot
# ---------------------------------------------------------------------------

def test_t8_sidecar_preserves_per_uuid_rates(rig_profile, lib_path):
    """The library is name-keyed, so saving only it would DROP the spread.

    The two 3080s differ by 4.37 TFLOPS. That difference is the evidence that
    a name is not an identity, and it must survive on disk.
    """
    crp.run_card_rate_pass(path=str(lib_path), profile=rig_profile)
    side = Path(str(lib_path) + ".by-uuid.json")
    assert side.is_file(), "no UUID-keyed sidecar was written"

    data = json.loads(side.read_text())
    by_uuid = data["rates_by_uuid"]
    assert set(by_uuid) == {U5090, U3080_A, U3080_B}
    assert by_uuid[U3080_A]["gemm_tflops"] == pytest.approx(65.57)
    assert by_uuid[U3080_B]["gemm_tflops"] == pytest.approx(61.20)


def test_t9_no_probe_refuses_and_writes_nothing(lib_path):
    report = crp.run_card_rate_pass(path=str(lib_path), profile=None)
    assert not report.wrote
    assert not lib_path.exists()
    assert any("no card probe on disk" in c for c in report.caveats), report.caveats


def test_t10_throttled_measurement_is_recorded_not_dropped(lib_path):
    prof = _profile(
        _measurement(U5090, R5090, 120.0, 900.0, 32607, throttle=["sw_thermal"]),
    )
    report = crp.run_card_rate_pass(path=str(lib_path), profile=prof)
    assert report.wrote, "a throttled measurement must still be usable"
    assert any("THROTTLED" in c for c in report.caveats), report.caveats


def test_t11_round_trips_through_the_real_library_format(rig_profile, lib_path):
    from sglang.srt.planner.card_library import _canonical

    crp.run_card_rate_pass(path=str(lib_path), profile=rig_profile)
    lib = CardLibrary.load(str(lib_path))
    # Looked up by name AND capacity: the driver's "NVIDIA GeForce RTX 3080"
    # names two catalogue entries and only the 20 GB one is this rig's card.
    # (See test_card_library_guards.py -- a name-only `get` here reads the
    # 10 GB seed, which is the collision this rig actually has.)
    assert lib.resolve(R3080, total_mib=20480).gemm_tflops == pytest.approx(61.20)
    # Cards the pass never measured keep their seeded, unrated form rather
    # than acquiring a number from anywhere. Compared canonically: the pass
    # writes under the DRIVER's name, and the catalog keys on the stripped
    # form, so a literal name comparison would miss the entry it just wrote.
    measured_keys = {_canonical(R5090), _canonical("RTX 3080 20GB")}
    for name in lib.names():
        spec = lib.get(name)
        if _canonical(name) not in measured_keys:
            assert not spec.gemm_tflops, f"{name} acquired an unmeasured rate"


def test_t11b_measured_capacity_corrects_a_colliding_catalog_entry(
    rig_profile, lib_path
):
    """The RTX 3080 name collision, pinned because it is silent and real.

    The seed catalog deliberately holds "RTX 3080" (10240 MiB) and
    "RTX 3080 20GB" as distinct entries. The driver calls BOTH
    "NVIDIA GeForce RTX 3080", and `_canonical` strips only the vendor words,
    so this rig's 20 GB cards land on the 10 GB profile. Without the
    correction the library would describe a 20480 MiB card as 10240 MiB while
    carrying that card's measured rates -- a profile internally inconsistent
    with the machine it was measured on.

    SUPERSEDED IN PART. #584 corrected this by OVERWRITING the colliding
    entry's capacity, which fixed the reading of this rig by breaking the
    reading of the catalog: "RTX 3080" then claimed 20480 MiB, so a composed
    rig built from the 10 GB card silently got a 20 GB one -- the same
    substitution, one level up. The measurement now lands on the entry that
    matches BOTH name and capacity (the 20 GB variant the catalog already
    carried), and the 10 GB entry is left exactly as it was. The assertion
    below is therefore about which ENTRY took the rates, not about one entry
    changing size.
    """
    seeded = CardLibrary()
    assert seeded.get(R3080).total_mib == 10240, (
        "the colliding seed entry changed; re-check which profile the driver "
        "name now resolves to"
    )

    report = crp.run_card_rate_pass(path=str(lib_path), profile=rig_profile)
    lib = crp.load_measured_library(str(lib_path))
    resolved = lib.resolve(R3080, total_mib=20480)
    assert resolved.name == "RTX 3080 20GB"
    assert resolved.total_mib == 20480
    assert resolved.gemm_tflops == pytest.approx(61.20)
    # The variant the measurement is NOT about keeps its curated capacity and
    # acquires no rate.
    assert lib.get("RTX 3080").total_mib == 10240
    assert lib.get("RTX 3080").gemm_tflops is None
    assert "RTX 3080 20GB" in report.names_written, report.names_written


def test_t12_projection_will_not_invent_a_capacity(lib_path):
    """An unknown card with no measured total must be skipped, not given one."""
    unknown = _profile(_measurement("GPU-ffff", "Some Unlisted Card", 10.0, 100.0, None))
    lib, written, caveats = crp.project_onto_library(
        crp.rates_by_uuid(unknown), library=CardLibrary(), total_mib_by_name={}
    )
    assert written == []
    assert not lib.has("Some Unlisted Card")
    assert any("fabricated capacity" in c for c in caveats), caveats


def test_t13_census_records_the_uuid_alongside_the_name():
    """A name cannot identify a card on a rig with two of the same model."""
    import inspect

    from sglang.srt.planner import residency_census

    src = inspect.getsource(residency_census)
    assert '"gpu_uuid"' in src, (
        "the residency census records only gpu_name; on this rig two RTX 3080s "
        "share it, so the census cannot say which card a stage ran on"
    )
