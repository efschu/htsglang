"""Task #49 (20.09.): the Marlin C-buffer discriminator and the reduce-branch
override.

What this file gates, and why each gate exists:

  * the two switches parse and DEFAULT to today's behaviour -- a probe that
    changed the default would turn every later boot into a different
    experiment.
  * ``resolve_atomic_add`` reports its SOURCE ('hardware' / 'env'), so the
    one-shot log can never claim a branch the run did not take. That is the
    same null-result trap the workspace-provenance log closes: fn8aq's private
    workspace could not separate 'shared lock buffer' from 'atomic protocol'
    because nothing printed which branch the rank was on.
  * ``classify_c_probe`` is the verdict logic. It is pure, so the U/W split is
    proven here rather than argued from a boot log.
  * the sentinel VALUE must survive the fp16/bf16 round trip bit-for-bit --
    the whole probe rests on an exact equality test against it, and a value
    that rounds (or flushes to zero) would report every row as 'written'.
  * the sentinel is wired into the arena BEFORE the views are taken and read
    back through the SAME storage the second GEMM writes; an emulation of
    'some rows never stored to' shows the probe separating the two classes on
    real tensors, on the CPU.

Run:  CUDA_VISIBLE_DEVICES="" python -m pytest tests/moe_offload/test_marlin_c_probe_0920.py -q
"""

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton import fused_marlin_moe as fm


@pytest.fixture(autouse=True)
def _reset_memos():
    fm._C_SENTINEL["on"] = None
    fm._ATOMIC_ADD["mode"] = None
    fm._ATOMIC_LOGGED["done"] = False
    yield
    fm._C_SENTINEL["on"] = None
    fm._ATOMIC_ADD["mode"] = None
    fm._ATOMIC_LOGGED["done"] = False


# --- the switches ---------------------------------------------------------


def test_sentinel_switch_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_MARLIN_C_SENTINEL", raising=False)
    assert fm.marlin_c_sentinel_on() is False
    for raw, want in (("0", False), ("1", True), ("on", True), ("true", True)):
        fm._C_SENTINEL["on"] = None
        monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", raw)
        assert fm.marlin_c_sentinel_on() is want, raw


def test_atomic_add_override_defaults_to_the_hardware_rule(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_MARLIN_ATOMIC_ADD", raising=False)
    assert fm.marlin_atomic_add_override() is None
    assert fm.resolve_atomic_add(True) == (True, "hardware")
    fm._ATOMIC_ADD["mode"] = None
    assert fm.resolve_atomic_add(False) == (False, "hardware")


@pytest.mark.parametrize(
    "raw, forced",
    [("0", False), ("false", False), ("off", False), ("1", True), ("on", True)],
)
def test_atomic_add_override_forces_both_ways(monkeypatch, raw, forced):
    monkeypatch.setenv("SGLANG_MOE_MARLIN_ATOMIC_ADD", raw)
    # forced in BOTH directions: the 5090's hardware rule says True, the
    # 3080's says False, and the boot must be able to put either rank on
    # either branch to A/B the fault.
    for hardware in (True, False):
        fm._ATOMIC_ADD["mode"] = None
        assert fm.resolve_atomic_add(hardware) == (forced, "env")


def test_unknown_value_is_not_silently_an_override(monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_MARLIN_ATOMIC_ADD", "yes-please")
    assert fm.marlin_atomic_add_override() is None
    assert fm.resolve_atomic_add(True) == (True, "hardware")


# --- the verdict ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad, untouched, want",
    [
        (0, 0, "CLEAN"),
        (144, 144, "U-UNWRITTEN"),
        (144, 0, "W-WRITTEN-NONFINITE"),
        (2208, 7, "MIXED"),
        (1, 1, "U-UNWRITTEN"),
    ],
)
def test_classify_c_probe(bad, untouched, want):
    assert fm.classify_c_probe(bad, untouched) == want


def test_classify_refuses_an_impossible_count():
    # untouched > bad would mean the scan and the finiteness test disagreed
    # about the same rows; a verdict computed from that is worse than none.
    with pytest.raises(ValueError):
        fm.classify_c_probe(10, 11)
    with pytest.raises(ValueError):
        fm.classify_c_probe(-1, 0)


# --- the sentinel value ---------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_sentinel_survives_the_dtype_round_trip_exactly(dtype):
    t = torch.empty(8, dtype=dtype).fill_(fm.C_SENTINEL_VALUE)
    # exact equality in the storage dtype AND back in python float: a value
    # that rounded would make every row compare 'written'.
    assert bool((t == fm.C_SENTINEL_VALUE).all())
    assert float(t[0].to(torch.float64)) == fm.C_SENTINEL_VALUE
    assert torch.isfinite(t).all()


def test_sentinel_is_not_zero_and_not_a_common_activation():
    # zero would collide with the legitimate output of a padded slot, which is
    # exactly the value the expert-major path relies on being zero.
    assert fm.C_SENTINEL_VALUE != 0.0
    assert fm.C_SENTINEL_VALUE < 0.0


# --- the probe on real tensors (CPU) --------------------------------------


def _emulate_gemm(M, topk, K, dtype, unwritten_pairs, nonfinite_pairs):
    """The arena as fused_marlin_moe builds it, with a kernel that skips
    ``unwritten_pairs`` and writes a non-finite value into ``nonfinite_pairs``.

    Returns (out, cache3_rows) in the shapes ``_c_probe_report`` reads."""
    arena = torch.empty(M * topk * K, dtype=dtype)
    arena.fill_(fm.C_SENTINEL_VALUE)
    cache3 = arena[: M * topk * K].view(-1, K)
    for p in range(M * topk):
        if p in unwritten_pairs:
            continue
        cache3[p].fill_(float("nan") if p in nonfinite_pairs else 1.0)
    # moe_sum_reduce over the k axis
    out = cache3.view(M, topk, K).to(torch.float32).sum(dim=1).to(dtype)
    return out, cache3


def _counts(out, cache3, M, topk):
    bad = ~torch.isfinite(out).reshape(out.shape[0], -1).all(dim=1)
    untouched = (cache3 == fm.C_SENTINEL_VALUE).all(dim=1).reshape(M, topk).any(dim=1)
    return int(bad.sum()), int((bad & untouched).sum())


def test_probe_separates_unwritten_from_written_nonfinite():
    M, topk, K = 6, 2, 4
    # (a) rows 1 and 4 have a pair no block ever stored to
    out, c3 = _emulate_gemm(
        M, topk, K, torch.bfloat16, unwritten_pairs={1 * topk, 4 * topk + 1},
        nonfinite_pairs=set(),
    )
    n_bad, n_untouched = _counts(out, c3, M, topk)
    # the sentinel itself is FINITE, so an unwritten row is only 'bad' once a
    # real kernel leaves garbage there; the probe's job is the attribution, so
    # assert the attribution directly on the counts the report computes.
    assert n_bad == 0 and n_untouched == 0
    assert fm.classify_c_probe(n_bad, n_untouched) == "CLEAN"

    # (b) the same skipped pairs, but now the arena held garbage NaN, which is
    # what torch.empty gives without the probe -- attributed as U.
    out2 = out.clone()
    out2[1] = float("nan")
    out2[4] = float("inf")
    n_bad2, n_untouched2 = _counts(out2, c3, M, topk)
    assert (n_bad2, n_untouched2) == (2, 2)
    assert fm.classify_c_probe(n_bad2, n_untouched2) == "U-UNWRITTEN"

    # (c) every pair written, two of them written NON-FINITE -> W
    out3, c33 = _emulate_gemm(
        M, topk, K, torch.bfloat16, unwritten_pairs=set(),
        nonfinite_pairs={2 * topk, 5 * topk + 1},
    )
    n_bad3, n_untouched3 = _counts(out3, c33, M, topk)
    assert (n_bad3, n_untouched3) == (2, 0)
    assert fm.classify_c_probe(n_bad3, n_untouched3) == "W-WRITTEN-NONFINITE"


def test_probe_report_logs_the_verdict(caplog):
    M, topk, K = 4, 2, 4
    out, c3 = _emulate_gemm(
        M, topk, K, torch.bfloat16, unwritten_pairs={0, 3 * topk},
        nonfinite_pairs=set(),
    )
    out[0] = float("nan")
    out[3] = float("nan")
    with caplog.at_level("ERROR", logger=fm.__name__):
        fm._c_probe_report(out, c3, M, topk, 16, True)
    assert "[nan-probe-c] VERDICT U-UNWRITTEN" in caplog.text
    assert "bad_rows=2" in caplog.text and "untouched_bad_rows=2" in caplog.text


def test_probe_report_is_silent_on_a_clean_output(caplog):
    M, topk, K = 4, 2, 4
    out, c3 = _emulate_gemm(
        M, topk, K, torch.bfloat16, unwritten_pairs=set(), nonfinite_pairs=set()
    )
    with caplog.at_level("ERROR", logger=fm.__name__):
        fm._c_probe_report(out, c3, M, topk, 16, True)
    assert "[nan-probe-c] VERDICT" not in caplog.text


# --- the wiring -----------------------------------------------------------


def _module_source() -> str:
    # ``fused_marlin_moe`` is wrapped by register_custom_op, so inspect cannot
    # reach the python body; read the module file instead.
    import pathlib

    return pathlib.Path(fm.__file__).read_text()


def test_the_fill_sits_before_the_views_and_the_report_after_the_reduce():
    src = _module_source()
    i_fill = src.index("intermediate_cache13.fill_(C_SENTINEL_VALUE)")
    i_view = src.index("intermediate_cache1 = intermediate_cache13[")
    i_reduce = src.index("        moe_sum_reduce(")
    i_report = src.index("_c_probe_report(", i_reduce)
    # fill first (the views alias the arena), report after the reduce (the
    # output has to exist before its rows can be called bad).
    assert i_fill < i_view < i_reduce < i_report


def test_the_override_is_applied_after_the_hardware_rule():
    src = _module_source()
    i_rule = src.index("use_atomic_add = (")
    i_override = src.index("use_atomic_add, _aa_source = resolve_atomic_add(")
    i_gemm = src.index("intermediate_cache1 = moe_wna16_marlin_gemm(")
    assert i_rule < i_override < i_gemm
