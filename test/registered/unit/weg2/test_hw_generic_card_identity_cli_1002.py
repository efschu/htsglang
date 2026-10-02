"""HW-GENERIC 1002: the ``card_identity`` CLI is the release entrypoint's GPU gate.

User order 02.10.: the release runs on any sm_86/sm_120 NVIDIA card set, not
only 1x RTX 5090 + 2x RTX 3080. SM89-DURCHSPIEL-1002: sm_89 passes the arch
gate too -- it has no calibration class, so it lands on HW-UNCALIBRATED
(rc 4) rather than HW-ARCH. The entrypoint's
``preflight_gpus`` (staged: /spinning/gpu-arb/docker/entrypoint.sh.hwgeneric-staged)
runs ``python -m sglang.srt.weg2.card_identity --expect-count N [--inventory L]``
and acts on its exit code and on the FIRST line carrying one of these prefixes:

* rc 3, ``refuse HW-ARCH: ...``  -> REFUSED GPU (always, also under HTSGLANG_EXPECT_GPUS=0)
* rc 3, ``refuse HW-COUNT: ...`` -> REFUSED GPU (warning under HTSGLANG_EXPECT_GPUS=0)
* rc 4, ``HW-UNCALIBRATED: ...`` -> REFUSED UNCALIBRATED (warning under HTSGLANG_EXPECT_GPUS=0)

These tests pin the exit codes, the prefixes and the card order on synthetic
NVML recordings (``SGLANG_NVML_REPLAY_JSON``), in-process via ``_cli(argv)``.
"""

from __future__ import annotations

import json

import pytest

from sglang.srt.registry import nvml
from sglang.srt.weg2 import card_identity as ci
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1024 * 1024
REF_INV = "RTX5090,RTX3080,RTX3080"


def _card(i, name, mib, cc, bus=None, clk=None):
    row = {
        "index": i,
        "uuid": f"GPU-synth-{i:02d}",
        "name": name,
        "total_bytes": mib * MIB,
        "reserved_bytes": 0,
        "pci_bus_id": f"00000000:{0x10 + i:02X}:00.0",
    }
    if cc is not None:
        row["cc_major"], row["cc_minor"] = cc
    if bus:
        row["mem_bus_width_bits"] = bus
    if clk:
        row["mem_clock_max_mhz"] = clk
    return row


def _r3080(i, mib=20480):
    return _card(i, "NVIDIA GeForce RTX 3080", mib, (8, 6), 320, 9501)


def _r5090(i):
    return _card(i, "NVIDIA GeForce RTX 5090", 32607, (12, 0), 512, 14001)


INVENTORIES = {
    # this rig: nvml0 3080, nvml1 5090, nvml2 3080
    "rig": [_r3080(0), _r5090(1), _r3080(2)],
    "3x3090": [_card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6), 384, 9751) for i in range(3)],
    "2x5090_3080": [_r5090(0), _r5090(1), _r3080(2)],
    "pro6000_2xa6000": [
        _card(0, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 97887, (12, 0), 512, 14001),
        _card(1, "NVIDIA RTX A6000", 49140, (8, 6), 384, 8001),
        _card(2, "NVIDIA RTX A6000", 49140, (8, 6), 384, 8001),
    ],
    "rig_plus_4090": [_r3080(0), _r5090(1), _r3080(2),
                      _card(3, "NVIDIA GeForce RTX 4090", 24564, (8, 9), 384, 10501)],
    "3x4090": [_card(i, "NVIDIA GeForce RTX 4090", 24564, (8, 9), 384, 10501) for i in range(3)],
    "4x_sm86": [_r3080(i) for i in range(4)],
    # a stock 10 GB RTX 3080 is NOT the 20 GB calibration class
    "rig_10g_3080": [_r3080(0, mib=10240), _r5090(1), _r3080(2)],
    # an NVML that did not answer the compute capability
    "rig_cc_unreported": [_card(0, "NVIDIA GeForce RTX 3080", 20480, None), _r5090(1), _r3080(2)],
}


@pytest.fixture
def run_cli(tmp_path, monkeypatch, capsys):
    def _run(inventory, *argv):
        path = tmp_path / f"{inventory}.json"
        path.write_text(json.dumps(INVENTORIES[inventory]))
        monkeypatch.setenv(nvml.ENV_NVML_REPLAY, str(path))
        rc = ci._cli(list(argv))
        return rc, capsys.readouterr().out

    return _run


def _named(out, prefix):
    """The first line with ``prefix`` -- what the entrypoint extracts."""
    hits = [ln for ln in out.splitlines() if ln.startswith(prefix)]
    assert hits, f"no line starting with {prefix!r} in:\n{out}"
    return hits[0]


def _ordinal_rows(out):
    return [ln for ln in out.splitlines() if ln.startswith("ordinal ")]


def test_reference_rig_passes_in_5090_first_order(run_cli):
    rc, out = run_cli("rig", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 0, out
    rows = _ordinal_rows(out)
    assert [r.split(":")[0] for r in rows] == ["ordinal 0", "ordinal 1", "ordinal 2"]
    assert rows[0].startswith("ordinal 0: nvml1 NVIDIA GeForce RTX 5090") and rows[0].endswith("class=RTX5090")
    assert rows[1].startswith("ordinal 1: nvml0 NVIDIA GeForce RTX 3080") and rows[1].endswith("class=RTX3080")
    assert rows[2].startswith("ordinal 2: nvml2 NVIDIA GeForce RTX 3080") and rows[2].endswith("class=RTX3080")
    assert "HW-" not in out


def test_reference_rig_json_rows(run_cli):
    rc, out = run_cli("rig", "--expect-count", "3", "--inventory", REF_INV, "--json")
    assert rc == 0, out
    rows = json.loads(out)
    assert [r["nvml_index"] for r in rows] == [1, 0, 2]
    assert [r["class"] for r in rows] == ["RTX5090", "RTX3080", "RTX3080"]
    assert rows[0]["key"] == "RTX5090/32607MiB/sm120"


def test_3x3090_uncalibrated_with_inventory(run_cli):
    rc, out = run_cli("3x3090", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 4, out
    msg = _named(out, "HW-UNCALIBRATED:")
    assert "[RTX5090, RTX3080, RTX3080]" in msg
    assert "ordinal 0: live RTX3090/24576MiB/sm86 vs calibrated RTX5090" in msg
    assert "ordinal 2: live RTX3090/24576MiB/sm86 vs calibrated RTX3080" in msg
    # the card table is printed before the refusal (the entrypoint logs it)
    assert len(_ordinal_rows(out)) == 3


def test_3x3090_passes_without_inventory(run_cli):
    rc, out = run_cli("3x3090", "--expect-count", "3")
    assert rc == 0, out
    assert [r.split()[2] for r in _ordinal_rows(out)] == ["nvml0", "nvml1", "nvml2"]
    assert "HW-" not in out


def test_two_5090_one_3080_uncalibrated_names_only_the_differing_ordinal(run_cli):
    rc, out = run_cli("2x5090_3080", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 4, out
    msg = _named(out, "HW-UNCALIBRATED:")
    assert "ordinal 1: live RTX5090 vs calibrated RTX3080" in msg
    assert msg.count("vs calibrated") == 1


def test_pro6000_two_a6000_uncalibrated_biggest_first(run_cli):
    rc, out = run_cli("pro6000_2xa6000", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 4, out
    rows = _ordinal_rows(out)
    assert "RTX PRO 6000 Blackwell" in rows[0] and "sm120" in rows[0]
    assert "RTX A6000" in rows[1] and "RTX A6000" in rows[2]
    msg = _named(out, "HW-UNCALIBRATED:")
    assert "RTXPRO6000BlackwellWorkstationEdition/97887MiB/sm120 vs calibrated RTX5090" in msg
    assert "RTXA6000/49140MiB/sm86 vs calibrated RTX3080" in msg


def test_sm89_card_passes_the_gate_and_the_count_speaks(run_cli):
    # SM89-DURCHSPIEL-1002: 4 cards with --expect-count 3 -- the arch gate no
    # longer speaks for 8.9, the COUNT refusal does (and names all four cards).
    rc, out = run_cli("rig_plus_4090", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 3, out
    msg = _named(out, "refuse HW-COUNT:")
    assert "4 card(s) visible, this launch needs exactly 3" in msg
    assert "HW-ARCH" not in out


def test_pure_sm89_inventory_is_uncalibrated_not_refused(run_cli):
    # 8.9 is arch-accepted; with no calibration class it names the
    # HW-UNCALIBRATED path and the calibration procedure.
    rc, out = run_cli("3x4090", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 4, out
    msg = _named(out, "HW-UNCALIBRATED:")
    assert "RTX4090/24564MiB/sm89" in msg
    assert "card_rate_pass --run" in msg
    assert "HW-ARCH" not in out


def test_four_sm86_cards_refused_by_count(run_cli):
    rc, out = run_cli("4x_sm86", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 3, out
    msg = _named(out, "refuse HW-COUNT:")
    assert "4 card(s) visible, this launch needs exactly 3" in msg
    assert "--gpus / NVIDIA_VISIBLE_DEVICES" in msg
    assert "HW-ARCH" not in out


def test_10g_rtx3080_is_not_the_20g_class(run_cli):
    rc, out = run_cli("rig_10g_3080", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 4, out
    msg = _named(out, "HW-UNCALIBRATED:")
    assert "RTX3080/10240MiB/sm86" in msg


def test_unreported_compute_capability_refused_by_name(run_cli):
    rc, out = run_cli("rig_cc_unreported", "--expect-count", "3", "--inventory", REF_INV)
    assert rc == 3, out
    msg = _named(out, "refuse HW-ARCH:")
    assert "nvml0 'NVIDIA GeForce RTX 3080': compute capability not reported by NVML" in msg


def test_reference_inventory_constant_matches_the_profiles_string():
    # the staged profiles write PROFILE_INVENTORY="RTX5090,RTX3080,RTX3080"
    assert ci.parse_inventory(REF_INV) == ci.REFERENCE_INVENTORY
    assert {c.label for c in ci.CALIBRATED_CLASSES} == set(ci.REFERENCE_INVENTORY)
