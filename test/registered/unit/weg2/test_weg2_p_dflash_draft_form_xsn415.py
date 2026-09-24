"""weg2xsn415 (24.09.): under --spec-form DFLASH group P must load the DFlash2
draft -- the same checkpoint as group D -- not the target's NEXTN/MTP head.

Metal: xsn415 (76d3a369f9) died at the first flip on all six ranks with
``W68 Weg2XchgPlanDisagree: fc.weight_scale: itemsize 4 on P rank 2 against 2
on D``. P's argv carried ``--speculative-algorithm NEXTN --speculative-num-steps
2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
--speculative-draft-kv-only`` (the NF-line ``p_draft_kv_flags`` constants that
survived the merge f295e098eb), so P's weights_draft held the MTP head's fc
(int8 [10240, 5120], f32 per-channel scale [5120, 1]) while D's held DFlash2's
(packed int32, bf16 group scale [200, 5120] sharded 89/56/55). xsn411
(eb5d04453f) shipped P with ``spec_flags(producer=True)`` and flipped.

Hermetic. Pins: P's DFLASH producer flags ARE xsn411's; the NEXTN (NF) form is
untouched; the pre-spawn W10a guard refuses the xsn415 argv pair and passes
the xsn411 and NF pairs; main runs it on the shipped argv.
"""

import inspect
import json
from types import SimpleNamespace

import pytest

import sglang.srt.weg2.launcher as L

DRAFT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-DFlash2-W8-lued"

#: boot_weg2xsn411.json, group P, the speculative tokens (eb5d04453f).
XSN411_P_SPEC = [
    "--speculative-algorithm", "DFLASH",
    "--speculative-draft-model-path", DRAFT,
    "--speculative-num-draft-tokens", "8",
    "--speculative-draft-kv-only",
]
#: boot_weg2xsn415.json, group P, the speculative tokens (76d3a369f9).
XSN415_P_SPEC = [
    "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "2",
    "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "3",
    "--speculative-draft-kv-only",
]


@pytest.fixture
def restore_form():
    saved = dict(L._SPEC_FORM)
    try:
        yield
    finally:
        L._SPEC_FORM.clear()
        L._SPEC_FORM.update(saved)


def _dflash(path=DRAFT):
    # apply_spec_form checks the directory; the form fields are what matter
    L._SPEC_FORM.update({"form": "DFLASH", "draft_path": path, "block": 8, "window": 2048})


def test_p_carries_the_dflash_draft_under_dflash(restore_form):
    _dflash()
    assert list(L.p_draft_kv_flags([])) == XSN411_P_SPEC
    # D's --extra-d cannot turn P back into an MTP producer
    assert list(L.p_draft_kv_flags(["--speculative-num-draft-tokens", "4"])) == XSN411_P_SPEC


def test_nextn_form_still_reads_ds_depth(restore_form):
    L._SPEC_FORM.update({"form": "NEXTN"})
    flags = list(L.p_draft_kv_flags(["--speculative-num-steps", "3",
                                     "--speculative-num-draft-tokens", "4"]))
    assert flags == [
        "--speculative-algorithm", L.SPEC_ALGORITHM, "--speculative-num-steps", "3",
        "--speculative-eagle-topk", str(L.SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens", "4", "--speculative-draft-kv-only",
    ]


STORE_CFG = json.dumps(
    {"max_size": str(30 * 1024 ** 3), "min_free_space": str(32 * 1024 ** 3),
     "max_size_scope": "shared"},
    separators=(",", ":"),
)


def test_shipped_argv_p_carries_the_draft_checkpoint(restore_form):
    from sglang.srt.weg2.ring_table import p_carries_drafter

    _dflash()
    argv = L.argv_p(
        py="/nonexistent/python",
        model="/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed",
        budgets=[27960, 17064, 16552], s_gb=48, m_mib=2400, store_cfg=STORE_CFG,
        extra=[], p_max_total_tokens=463763, draft_kv_on_p=True,
        spec_flags=L.p_draft_kv_flags([]),
    )
    spec = [t for t in argv if t.startswith("--speculative")]
    assert spec == [t for t in XSN411_P_SPEC if t.startswith("--speculative")]
    assert argv[argv.index("--speculative-draft-model-path") + 1] == DRAFT
    assert p_carries_drafter(argv)


def test_w10a_refuses_the_xsn415_pair(restore_form):
    _dflash()
    d_spec = list(L.spec_flags(producer=False))
    with pytest.raises(L.Weg2LaunchRefused, match="W10a Weg2DrafterArgvMismatch"):
        L.check_drafter_argv_agreement(XSN415_P_SPEC, d_spec)


def test_w10a_passes_the_xsn411_pair_and_the_nf_pair(restore_form):
    _dflash()
    v = L.check_drafter_argv_agreement(XSN411_P_SPEC, list(L.spec_flags(producer=False)))
    assert v["match"] and v["P"]["--speculative-algorithm"] == "DFLASH"
    # NF (fnFL2x139): the MTP path reaches P via --extra-p, D via --extra-d
    mtp = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-MTP-INT4-g32-albucino"
    p = ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
         "--speculative-draft-kv-only", "--speculative-draft-model-path", mtp]
    d = ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", "2",
         "--speculative-algorithm", "NEXTN", "--speculative-draft-model-path", mtp]
    assert L.check_drafter_argv_agreement(p, d)["match"]
    # a P without the producer carries no drafter: not graded
    assert L.check_drafter_argv_agreement([], list(L.spec_flags(producer=False)))["match"]


def test_main_grades_the_shipped_argv_before_spawn():
    src = inspect.getsource(L.main)
    i_ship = src.index("shipped_argv_p = argv_p(")
    i_chk = src.index("check_drafter_argv_agreement(")
    i_spawn = src.index('spec_p = GroupSpec("P"')
    assert i_ship < i_chk < i_spawn
    assert "list(spec_flags(producer=False)) + shlex.split(ns.extra_d)" in src
