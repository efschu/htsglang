"""RC1 (weg2rc1, 24.09. 21:46Z, tree fe316bdf40 = xsn438 + vision): the P cut
fell from 42,11,11 to 39,13,12 (ranked #57), P ladder -9..-11 %.

NOT VISION. The one input that moved is ``dormant_other`` -- group D's
measured sleeping residue that P's budget subtracts on every card:
xsn437 2416/1832/1832 MiB, RC1 3526/3062/3060. Its #1444 source was the
newest D sample of the line, boot weg2xsn439 (21:26:59Z), which ran D with
--d-bs 32 (--max-running-requests 32 -> a bigger resident CUDA-graph set):
3206/2742/2740 MiB, W19 at its first sleep. RC1 (d_bs 6, like xsn437/438)
was priced with it: P budgets 24896/14512/14240 instead of 26008/15744/15472,
the 42,11,11 pool 212,871 < the 262,656 floor. The vision boot itself
measured D at 2096/1512/1512 (xsn438) and 2072/1490/1490 (RC1) -- unchanged.

The fix keys the #1444 sample on D's CAPTURE SET: a D sample names its
--max-running-requests (new samples a field, older ones via the ``group D
argv:`` line of their boot's front log), and the launcher prices only a
sample of its own. The planner call of both forms (vision transient / off)
on the record frozen at RC1's launch then chooses 42,11,11, pool 269,805 --
xsn437's -- on both (commit message; the call reads the live P logs, so it is
a verification, not this hermetic test). Hermetic, CPU. Pinned:
  * a D sample carries its capture set, a P sample stays as it was;
  * an older sample's capture set is read off its boot's front log;
  * in the RC1 record state a d_bs=6 boot selects xsn438, not the newer
    xsn439, and its dormant_other terms are xsn437's 2416/1832/1832; a
    d_bs=32 boot selects xsn439;
  * the planner's budget and cut code has no vision input at all.
"""

import inspect
import json
import os
from types import SimpleNamespace

import pytest

from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import line_identity

CARDS = [
    L.Card(nvml_index=1, uuid="GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d",
           name="NVIDIA GeForce RTX 5090", total_mib=32607),
    L.Card(nvml_index=0, uuid="GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
           name="NVIDIA GeForce RTX 3080", total_mib=20480),
    L.Card(nvml_index=2, uuid="GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4",
           name="NVIDIA GeForce RTX 3080", total_mib=20480),
]
U5090, U3080A, U3080B = (c.uuid for c in CARDS)
#: (tag, at, commit, D --max-running-requests, residue 5090/3080a/3080b) -- the record
BOOTS = [
    ("weg2xsn437", "2026-09-24T21:08:45Z", "5a3f533f0f", 6, (2096, 1512, 1512)),
    ("weg2xsn438", "2026-09-24T21:19:10Z", "fe316bdf40", 6, (2096, 1512, 1512)),
    ("weg2xsn439", "2026-09-24T21:26:59Z", "5a3f533f0f", 32, (3206, 2742, 2740)),
]


def _sample(tag, at, commit, residue, capture_bs=None):
    s = host_ledger.dormant_image_sample(
        group="D", shmem_before_bytes=None, shmem_after_bytes=None, pids=[],
        weight_tags_gib=27.15, interleaved=True, boot_tag=tag, commit=commit, at=at,
        vram_residue_mib=dict(zip((U5090, U3080A, U3080B), residue)),
        vram_residue_form="exchange", vram_residue_capture_bs=capture_bs,
    )
    return s


def _front_log(evidence, tag, commit, day_time, mrr):
    path = os.path.join(evidence, f"boot_weg2_{tag}_{commit}_{day_time}.front.log")
    with open(path, "w") as f:
        f.write(f"[x] WEG2-LAUNCH === WEG2 BOOT tag={tag} tree=/t @ {commit}\n")
        f.write("[x] WEG2-LAUNCH group D argv: /py -m sglang.launch_server --model-path "
                f"/m/Qwen3.8-27B --max-running-requests {mrr} --port 30032\n")
    return path


@pytest.fixture
def rc1_state(tmp_path):
    """The record and the front logs as they stood when RC1 launched (the
    samples written before the field existed: no capture set in them)."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for i, (tag, at, commit, mrr, _res) in enumerate(BOOTS):
        _front_log(str(evidence), tag, commit, f"0924_21{i}000", mrr)
    record = tmp_path / "weg2_measured_record.json"
    samples = [_sample(tag, at, commit, res) for tag, at, commit, _m, res in BOOTS]
    record.write_text(json.dumps({"samples": samples}))
    ident = line_identity.LineIdentity(model="/m/Qwen3.8-27B", repo="/r", head="fe316bdf40",
                                       evidence_dir=str(evidence))
    # the line/ancestry verdict is not this test's subject (every boot above is
    # of the line); the capture-set reader is the real one
    line_id = SimpleNamespace(
        accepts_sample=lambda s: str(s.get("boot_tag", "")).startswith("weg2xsn"),
        d_max_running_requests=ident.d_max_running_requests,
        describe=ident.describe,
    )
    return SimpleNamespace(record=str(record), line_id=line_id, ident=ident)


def test_a_d_sample_names_its_capture_set_and_a_p_sample_stays():
    d = _sample("weg2xsn440", "t", "c", (1, 2, 3), capture_bs=6)
    assert d["vram_residue_capture_bs"] == 6
    p = host_ledger.dormant_image_sample(
        group="P", shmem_before_bytes=None, shmem_after_bytes=None, pids=[],
        weight_tags_gib=28.83, interleaved=True, boot_tag="t", commit="c")
    assert "vram_residue_capture_bs" not in p


def test_the_front_hands_d_its_d_bs_and_p_nothing():
    from sglang.srt.weg2 import front

    src = inspect.getsource(front.Front.sample_dormant_image)
    assert 'vram_residue_capture_bs=(getattr(self, "d_bs", None) if group == "D" else None)' in src


def test_an_older_sample_s_capture_set_is_read_off_its_front_log(rc1_state):
    assert rc1_state.ident.d_max_running_requests("weg2xsn439") == 32
    assert rc1_state.ident.d_max_running_requests("weg2xsn438") == 6
    assert rc1_state.ident.d_max_running_requests("weg2nope") is None


def _dormant_other(rec):
    dc, _prov = L.dc_residue_from_record(rec, CARDS, "exchange")
    slack = L.reserve_slack_mib("bar1")
    return [dc[c.uuid] + slack for c in CARDS]


def test_rc1_s_d_bs_6_is_priced_with_its_own_capture_set(rc1_state):
    rec6 = host_ledger.read_measured_record(
        rc1_state.record, accept=L.d_residue_record_accept(rc1_state.line_id, 6)).get("D")
    assert rec6["boot_tag"] == "weg2xsn438", "not the newer bs32 sample"
    assert _dormant_other(rec6) == [2416, 1832, 1832], "xsn437's terms -> P 26008/15744/15472"
    rec32 = host_ledger.read_measured_record(
        rc1_state.record, accept=L.d_residue_record_accept(rc1_state.line_id, 32)).get("D")
    assert rec32["boot_tag"] == "weg2xsn439"
    assert _dormant_other(rec32) == [3526, 3062, 3060], "what RC1 was priced with"


def test_a_sample_with_the_field_needs_no_front_log(tmp_path):
    record = tmp_path / "r.json"
    record.write_text(json.dumps({"samples": [
        _sample("weg2xsn450", "2026-09-24T22:00:00Z", "c1", (2096, 1512, 1512), capture_bs=6),
        _sample("weg2xsn451", "2026-09-24T22:10:00Z", "c2", (3206, 2742, 2740), capture_bs=32),
    ]}))
    line_id = SimpleNamespace(accepts_sample=lambda s: True,
                              d_max_running_requests=lambda tag: None)
    rec = host_ledger.read_measured_record(
        str(record), accept=L.d_residue_record_accept(line_id, 6)).get("D")
    assert rec["boot_tag"] == "weg2xsn450"


def test_an_unprovable_capture_set_does_not_price_the_residue(tmp_path):
    record = tmp_path / "r.json"
    record.write_text(json.dumps({"samples": [
        _sample("weg2xsn460", "2026-09-24T22:00:00Z", "c1", (3206, 2742, 2740))]}))
    line_id = SimpleNamespace(accepts_sample=lambda s: True,
                              d_max_running_requests=lambda tag: None)
    assert host_ledger.read_measured_record(
        str(record), accept=L.d_residue_record_accept(line_id, 6)).get("D") is None


def test_the_planner_has_no_vision_input():
    """The user's rule (24.09.): the image stage takes room only while it runs
    -- no permanent post may move the cut or the pool."""
    for fn in (L.solve_p_cut, L.budgets_from_dc, L.dc_residue_from_record,
               L.d_residue_record_accept):
        assert "vision" not in inspect.getsource(fn).lower(), fn.__name__
