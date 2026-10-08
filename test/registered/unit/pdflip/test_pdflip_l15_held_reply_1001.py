# SPDX-License-Identifier: Apache-2.0
"""L15-13c part 2: the D sleep reply carries the held KV MiB per card.

Part 1 (28ffde11e2) taught ``front.sample_dormant_image`` to SUBTRACT the
L1.5 hold from the residue record, but nothing handed it the hold.  Part 2
closes that gap: the release leg's answer
(``ReleaseMemoryOccupationReqOutput``) grows an optional
``l15_held_mib`` -- ``{card uuid: MiB}`` kv_cache bytes still mapped after
the pause -- gathered over the group's ranks by the group fence's own
``all_gather_object``; the front reads it off the D sleep reply and passes
it at BOTH dormant-image stamp sites (the H78 worker-thread path and the
inline path).

Master off: the field stays None, nothing is read, nothing is gathered --
the reply and the record are byte-identical to before.

Plain pytest functions -- CustomTestCase.retry() hides failures.
Hermetic: no NVML, no boot, no GPU (CUDA_VISIBLE_DEVICES="").
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.io_struct import ReleaseMemoryOccupationReqOutput
from flliper.srt.pdflip import host_ledger, l15_plan
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
WEIGHT_UPDATER_PY = (
    _ROOT / "python" / "flliper" / "srt" / "managers" / "scheduler_components"
    / "weight_updater.py"
)
FRONT_PY = _ROOT / "python" / "flliper" / "srt" / "pdflip" / "front.py"


def _func_source(path: Path, name: str) -> str:
    src = path.read_text()
    start = src.index("def %s(" % name)
    end = src.index("\n    def ", start + 1)
    return src[start:end]


# -- (a) the Output accepts and defaults the field ----------------------------


def test_output_defaults_l15_held_mib_to_none():
    assert ReleaseMemoryOccupationReqOutput().l15_held_mib is None


def test_output_carries_the_field_when_given():
    out = ReleaseMemoryOccupationReqOutput(l15_held_mib={"u": 300})
    assert out.l15_held_mib == {"u": 300}


# -- (b) the wiring is present on both sides ----------------------------------


def test_release_leg_fills_the_field():
    body = _func_source(WEIGHT_UPDATER_PY, "release_memory_occupation")
    assert "_pdflip_tag_mapped_bytes" in body
    assert "l15_held=_l15_held" in body
    assert "l15_held_mib=" in body


def test_both_front_stamp_sites_pass_l15_held_mib():
    src = FRONT_PY.read_text()
    sites = [
        ln
        for ln in src.splitlines()
        if "sample_dormant_image(" in ln and "def " not in ln
    ]
    assert len(sites) == 2, "expected the H78 and the inline stamp site"
    for ln in sites:
        assert "l15_held_mib=" in ln, ln


# -- (c) the stamp subtracts the held MiB from the record ---------------------


def _fake_front():
    grp = SimpleNamespace(sid=None, outstanding=())
    return SimpleNamespace(
        dormant_image={},
        groups={"D": grp, "P": grp},
        tag="t",
        commit="c",
        ledger_arm=None,
        epoch=0,
        weight_form="exchange",
        d_bs=6,
        d_residue_context_tokens=0,
        queue=[],
    )


def test_d_record_subtracts_the_held_mib(monkeypatch, tmp_path):
    from flliper.srt.pdflip import front

    monkeypatch.setenv(l15_plan.L15_MASTER_ENV, "1")
    monkeypatch.setattr(host_ledger, "read_cgroup", lambda: {"current": 1, "reclaimable": 2})
    monkeypatch.setattr(host_ledger, "read_cgroup_shmem_bytes", lambda: 0)
    rec = front.Front.sample_dormant_image(
        _fake_front(),
        "D",
        0,
        vram_residue_mib={"u": 1000},
        l15_held_mib={"u": 300},
        persist=False,
    )
    assert rec is not None
    assert rec["vram_residue_mib"] == {"u": 700}


def test_master_off_leaves_the_record_untouched(monkeypatch):
    from flliper.srt.pdflip import front

    monkeypatch.delenv(l15_plan.L15_MASTER_ENV, raising=False)
    monkeypatch.setattr(host_ledger, "read_cgroup", lambda: {"current": 1, "reclaimable": 2})
    monkeypatch.setattr(host_ledger, "read_cgroup_shmem_bytes", lambda: 0)
    rec = front.Front.sample_dormant_image(
        _fake_front(),
        "D",
        0,
        vram_residue_mib={"u": 1000},
        l15_held_mib={"u": 300},
        persist=False,
    )
    assert rec is not None
    assert rec["vram_residue_mib"] == {"u": 1000}


def test_front_parser_reads_the_field_off_the_reply_body():
    from flliper.srt.pdflip import front

    body = '{"l15_held_mib": {"u": 300}}'
    assert front.l15_held_mib_from_body(body) == {"u": 300}
    assert front.l15_held_mib_from_body("not json") is None
    assert front.l15_held_mib_from_body("{}") is None
