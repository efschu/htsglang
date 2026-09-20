"""Task #49 (2026-09-20): the Marlin switch logic, proven without a GPU.

Three kinds of claim are tested here, and only these:

  (a) the Python parsers agree BYTE FOR BYTE with the ``std::getenv`` parsers in
      ``csrc/gemm/marlin_moe/moe_wna16_marlin.cuh``. This is the whole reason a
      Python copy exists: it only feeds the per-rank log line, and a log line
      whose parser disagrees with the kernel's would report a switch the boot
      did not run.
  (b) the defaults are the pre-#49 launch, except for the epilogue barrier,
      which is the fix and is therefore ON by default.
  (c) the arch override produces a DISJOINT JIT cache key and embeds PTX, so an
      overridden module can never be served from, or confused with, the native
      sm_120 build.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/spinning/wt-nan49c-0920/python \\
    /spinning/htsglang-gpu/.venv/bin/python -m pytest -q \\
    tests/moe_offload/test_marlin_switches_0920.py
"""

import pathlib
import re

import pytest

from sglang.jit_kernel.marlin_switches import (
    ENV_ARCH_OVERRIDE,
    ENV_EPILOGUE_SYNC,
    ENV_NO_K_SPLIT,
    ENV_SMS_OVERRIDE,
    arch_override_cuda_cflags,
    arch_override_from_env,
    effective_sms,
    epilogue_sync_on,
    format_switch_log,
    no_k_split_on,
    parse_arch_override,
    smem_optin_verdict,
    sms_override,
    switch_census,
)

CUH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh"
)
TEMPLATE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h"
)


# --- (b) defaults: an unset environment is the pre-#49 launch ---------------


def test_defaults_are_the_old_launch_except_the_barrier():
    env = {}
    assert no_k_split_on(env) is False
    assert sms_override(env) == 0
    assert effective_sms(170, env) == 170
    assert arch_override_from_env(env) is None
    # The one deliberate exception: the barrier is the fix, so it is ON.
    assert epilogue_sync_on(env) is True


def test_empty_string_is_the_same_as_unset():
    env = {
        ENV_NO_K_SPLIT: "",
        ENV_SMS_OVERRIDE: "",
        ENV_EPILOGUE_SYNC: "",
        ENV_ARCH_OVERRIDE: "",
    }
    assert no_k_split_on(env) is False
    assert sms_override(env) == 0
    assert epilogue_sync_on(env) is True
    assert arch_override_from_env(env) is None


# --- (a) the Python parsers mirror the C++ ones -----------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "Y", "t"])
def test_no_k_split_truthy_first_chars(raw):
    assert no_k_split_on({ENV_NO_K_SPLIT: raw}) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "N", "F", "junk"])
def test_no_k_split_everything_else_is_off(raw):
    # The C++ tests the FIRST CHARACTER only; 'off' starts with 'o', which is
    # not in the truthy set, so it is off -- as is any junk. Pinned so a later
    # "let's be more liberal" edit on one side breaks this test, not a boot.
    assert no_k_split_on({ENV_NO_K_SPLIT: raw}) is False


@pytest.mark.parametrize("raw", ["0", "false", "no", "F", "N", "n"])
def test_epilogue_sync_falsy_first_chars_disable_the_fix(raw):
    assert epilogue_sync_on({ENV_EPILOGUE_SYNC: raw}) is False


@pytest.mark.parametrize("raw", ["1", "true", "on", "yes", "junk"])
def test_epilogue_sync_anything_not_falsy_keeps_the_fix(raw):
    assert epilogue_sync_on({ENV_EPILOGUE_SYNC: raw}) is True


@pytest.mark.parametrize(
    "raw,want",
    [("68", 68), ("  68  ", 68), ("68abc", 68), ("abc", 0), ("-4", 0), ("0", 0)],
)
def test_sms_override_is_c_atoi(raw, want):
    assert sms_override({ENV_SMS_OVERRIDE: raw}) == want


def test_sms_override_applies_downward_only():
    # The clamp exists because the shared lock workspace was sized
    # hardware_sms * 4 at weight-load time: a WIDER grid would fail the
    # workspace check in the .cuh instead of running.
    assert effective_sms(170, {ENV_SMS_OVERRIDE: "68"}) == 68
    assert effective_sms(170, {ENV_SMS_OVERRIDE: "340"}) == 170
    assert effective_sms(170, {ENV_SMS_OVERRIDE: "170"}) == 170
    assert effective_sms(68, {ENV_SMS_OVERRIDE: "68"}) == 68


# --- the same claim, checked against the CUDA source itself -----------------


def test_cuda_source_reads_exactly_these_env_names():
    src = CUH.read_text()
    for name in (ENV_EPILOGUE_SYNC, ENV_SMS_OVERRIDE):
        assert f'std::getenv("{name}")' in src, name
    # NO_K_SPLIT goes through the shared gate parser, not a bare getenv.
    assert f'marlin_parse_gate("{ENV_NO_K_SPLIT}")' in src


def test_cuda_truthiness_sets_match_the_python_ones():
    src = CUH.read_text()
    # no_k_split truthiness now lives in marlin_parse_gate, same character set
    block = src.split("inline MarlinGate marlin_parse_gate")[1].split("\n}\n")[0]
    assert set(re.findall(r"raw\[0\] == '(.)'", block)) == {"1", "t", "T", "y", "Y"}
    # epilogue_sync: NOT ('0','f','F','n','N'), default true
    block = src.split("marlin_moe_epilogue_sync()")[1].split("}")[0]
    assert set(re.findall(r"raw\[0\] == '(.)'", block)) == {"0", "f", "F", "n", "N"}
    assert "return true;" in block


def test_cuda_clamps_the_sms_override_downward():
    src = CUH.read_text()
    assert "sms_override > 0 && sms_override < sms" in src


def test_the_barrier_exists_and_is_gated_by_the_kernel_argument():
    src = TEMPLATE.read_text()
    # The barrier must sit between the epilogue's cp_async_wait<0>() and the
    # first sh_red write, and must be uniform (a kernel argument, not a
    # per-thread predicate) so it is never a divergent __syncthreads.
    epi = src.split("// Process results and, if necessary,")[1]
    head = epi.split("bool last = slice_idx")[0]
    assert "cp_async_wait<0>();" in head
    assert "if (epilogue_sync) {" in head
    assert "__syncthreads();" in head
    assert head.index("cp_async_wait<0>();") < head.index("if (epilogue_sync) {")


def test_no_k_split_rounds_iters_to_whole_k_columns():
    src = TEMPLATE.read_text()
    assert "if (no_k_split) {" in src
    assert "iters = k_tiles * div_ceil(iters, k_tiles);" in src


# --- (c) the arch override --------------------------------------------------


@pytest.mark.parametrize(
    "raw,want",
    [
        ("9.0", (9, 0, "", True)),
        ("90", (9, 0, "", True)),
        ("sm_90", (9, 0, "", True)),
        ("compute_90", (9, 0, "", True)),
        ("10.0", (10, 0, "", True)),
        ("12.0a", (12, 0, "a", True)),
        ("9.0+ptx", (9, 0, "", True)),
        ("9.0+PTX", (9, 0, "", True)),
        ("9.0+noptx", (9, 0, "", False)),
    ],
)
def test_parse_arch_override_accepts_the_usual_spellings(raw, want):
    ov = parse_arch_override(raw)
    assert (ov.major, ov.minor, ov.suffix, ov.ptx) == want


@pytest.mark.parametrize("raw", ["ninety", "9", "9.0.0", "sm90a+", "7.5x", "-9.0"])
def test_parse_arch_override_rejects_junk_loudly(raw):
    # A typo in a boot line must fail at the first Marlin call, not run the
    # default silently and be reported as "the override did nothing".
    with pytest.raises(ValueError):
        parse_arch_override(raw)


def test_pre_ampere_is_refused():
    with pytest.raises(ValueError):
        parse_arch_override("6.1")


def test_ptx_is_embedded_by_default_because_a_cubin_would_not_load():
    ov = parse_arch_override("9.0")
    flags = arch_override_cuda_cflags(ov)
    assert flags == ["-gencode=arch=compute_90,code=compute_90"]


def test_noptx_emits_no_extra_flag():
    assert arch_override_cuda_cflags(parse_arch_override("9.0+noptx")) == []


def test_no_override_emits_no_extra_flag():
    assert arch_override_cuda_cflags(None) == []


def test_arch_override_lands_in_the_jit_cache_key():
    """The build directory name carries the arch, so the overridden module and
    the native sm_120 module cannot share a cache entry."""
    from sglang.jit_kernel.utils import _jit_build_dir_name, override_jit_cuda_arch

    with override_jit_cuda_arch(12, 0):
        native = _jit_build_dir_name("m", "b0")
    with override_jit_cuda_arch(9, 0):
        overridden = _jit_build_dir_name("m", "b0")
    assert native != overridden
    assert "arch_12.0" in native and "arch_9.0" in overridden


# --- the log line -----------------------------------------------------------


@pytest.mark.parametrize(
    "value,want",
    [
        (101376, "ok"),
        (None, "unknown"),
        (0, "DRIVER-POISONED"),
        (-1, "DRIVER-POISONED"),
        # The value RTX 5090 owners report from early Blackwell drivers.
        (4294967297, "DRIVER-POISONED"),
        (49152, "unexpected"),
    ],
)
def test_smem_optin_verdict_names_the_driver_report(value, want):
    assert smem_optin_verdict(value) == want


def test_census_and_log_line_name_every_switch():
    env = {
        ENV_NO_K_SPLIT: "1",
        ENV_SMS_OVERRIDE: "68",
        ENV_EPILOGUE_SYNC: "0",
        ENV_ARCH_OVERRIDE: "9.0",
    }
    census = switch_census(170, env, smem_optin=101376, device_cap=(12, 0))
    assert census == {
        "arch_override_applied": True,
        "arch_override_reason": "applies",
        "no_k_split_requested": True,
        "epilogue_sync": False,
        "no_k_split": True,
        "sms_requested": 68,
        "sms_hw": 170,
        "sms_effective": 68,
        "arch_override": "9.0",
        "ptx_jit": True,
        "smem_optin": 101376,
        "smem_optin_verdict": "ok",
    }
    line = format_switch_log("12.0", census)
    assert line.startswith("[nan-49c] marlin switches:")
    for fragment in (
        "device_arch=12.0",
        "arch_override=9.0",
        "ptx_jit=True",
        "epilogue_sync=False",
        "no_k_split=True",
        "sms_hw=170",
        "sms_effective=68",
    ):
        assert fragment in line, fragment


def test_log_line_on_a_default_boot_says_so():
    line = format_switch_log(
        "8.6", switch_census(68, {}, smem_optin=101376, device_cap=(8, 6))
    )
    assert "arch_override=None" in line
    assert "epilogue_sync=True" in line
    assert "no_k_split=False" in line
    assert "sms_hw=68 sms_effective=68" in line
    assert "smem_optin=101376(ok)" in line


# --- fn8c4: the override must be a per-RANK decision, not a per-BOOT env ----


def test_override_is_refused_on_a_device_older_than_its_ptx():
    """fn8c4 died exactly here: both sm_86 ranks built compute_90 and hit
    'no kernel image is available for execution on the device' at the first
    Marlin launch. PTX is forward-compatible only."""
    from sglang.jit_kernel.marlin_switches import arch_override_applies

    ov = parse_arch_override("9.0")
    assert arch_override_applies(ov, 12, 0) == (True, "applies")
    ok, reason = arch_override_applies(ov, 8, 6)
    assert ok is False and reason == "device-8.6-older-than-ptx-9.0"


def test_override_is_refused_when_the_capability_is_unknown():
    from sglang.jit_kernel.marlin_switches import arch_override_applies

    assert arch_override_applies(parse_arch_override("9.0"), None, None) == (
        False,
        "device-capability-unknown",
    )


def test_no_override_is_never_applied():
    from sglang.jit_kernel.marlin_switches import arch_override_applies

    assert arch_override_applies(None, 12, 0) == (False, "unset")


def test_explicit_gate_restricts_to_one_capability():
    from sglang.jit_kernel.marlin_switches import arch_override_applies

    ov = parse_arch_override("9.0@12.0")
    assert ov.only_on == (12, 0)
    assert arch_override_applies(ov, 12, 0) == (True, "applies")
    assert arch_override_applies(ov, 8, 6) == (False, "gated-to-12.0")
    # The gate wins even where the physical rule would allow it.
    assert arch_override_applies(ov, 10, 0) == (False, "gated-to-12.0")


def test_gate_junk_raises():
    with pytest.raises(ValueError):
        parse_arch_override("9.0@twelve")


def test_gate_combines_with_the_ptx_suffix():
    ov = parse_arch_override("9.0@12.0+noptx")
    assert (ov.major, ov.minor, ov.ptx, ov.only_on) == (9, 0, False, (12, 0))


def test_census_reports_applied_separately_from_requested():
    """After fn8c4 'requested' and 'applied' are not allowed to be one field:
    the boot log showed arch_override=9.0 ptx_jit=True on BOTH 3080 ranks,
    which read like a working arm right up to the crash."""
    env = {ENV_ARCH_OVERRIDE: "9.0"}
    on5090 = switch_census(170, env, smem_optin=101376, device_cap=(12, 0))
    on3080 = switch_census(68, env, smem_optin=101376, device_cap=(8, 6))
    assert on5090["arch_override"] == on3080["arch_override"] == "9.0"
    assert on5090["arch_override_applied"] is True
    assert on3080["arch_override_applied"] is False
    assert on3080["ptx_jit"] is False
    assert "older-than-ptx" in on3080["arch_override_reason"]
    line = format_switch_log("8.6", on3080)
    assert "arch_override_applied=False(device-8.6-older-than-ptx-9.0)" in line


def test_ptx_flag_is_what_the_fn8c4_build_actually_used():
    """The flag pair verified by hand against the real cache entry
    /root/.cache/tvm-ffi/...__cuda_arch_9.0__.../build.ninja after fn8c4:
    -gencode=arch=compute_90,code=sm_90 (from TVM_FFI_CUDA_ARCH_LIST) plus the
    one this module adds. cuobjdump -lptx on that .so listed a sm_90 PTX
    section, so the fatbin was NOT the failure -- the 3080 ranks were."""
    assert arch_override_cuda_cflags(parse_arch_override("9.0")) == [
        "-gencode=arch=compute_90,code=compute_90"
    ]


# --- fn8c6/fn8c7: does NO_K_SPLIT actually reach slice_count == 1? ----------

from sglang.jit_kernel.marlin_switches import marlin_slice_census

#: The fn8c6 shapes. Qwen3.8-Flash-Next: hidden 2560, moe_intermediate 640,
#: block_size_m 64 -> thread_m_blocks 4 -> large_batch_thread_configs[0] =
#: {thread_k 64, thread_n 256}. k_tiles = K/16/4, n_tiles = N/16/16.
GEMM1_TILES = (40, 5)   # gate_up: K=2560, N=2*640=1280
GEMM2_TILES = (10, 10)  # down:    K=640,  N=2560


@pytest.mark.parametrize("tiles", [GEMM1_TILES, GEMM2_TILES])
@pytest.mark.parametrize("blocks", [170, 68])
@pytest.mark.parametrize("M", [32850, 28909, 20161])
def test_no_k_split_reaches_slice_count_one(tiles, blocks, M):
    """The claim fn8c7 rests on, as an assertion instead of a hope."""
    k_tiles, n_tiles = tiles
    parallel = -(-M // 64) + 8
    off = marlin_slice_census(k_tiles, n_tiles, parallel, blocks)
    on = marlin_slice_census(k_tiles, n_tiles, parallel, blocks, no_k_split=True)
    assert off["max_slice_count"] == 2, "without the switch, columns DO split"
    assert on["max_slice_count"] == 1
    assert on["slices_with_split"] == 0
    assert on["iters"] % k_tiles == 0, "whole k-columns is the mechanism"


def test_the_k_split_is_not_what_makes_gemm2_special():
    """Both GEMMs split, in the same order of magnitude -- so a mechanism that
    is equally present in the GEMM that stays CLEAN cannot on its own explain
    the one that does not. Pinned so the next reader does not re-derive it."""
    parallel = -(-32850 // 64) + 8
    g1 = marlin_slice_census(*GEMM1_TILES, parallel, 170)
    g2 = marlin_slice_census(*GEMM2_TILES, parallel, 170)
    assert g1["slices_with_split"] > 0 and g2["slices_with_split"] > 0
    ratio = g1["slices_with_split"] / g2["slices_with_split"]
    assert 0.5 < ratio < 2.0, (g1, g2)


def test_the_5090_is_exposed_to_more_split_slices_than_the_3080():
    """The real asymmetry the arm is worth a boot for: 170 SMs against 68."""
    parallel = -(-32850 // 64) + 8
    on5090 = marlin_slice_census(*GEMM2_TILES, parallel, 170)
    on3080 = marlin_slice_census(*GEMM2_TILES, parallel, 68)
    assert on5090["slices_with_split"] > 2 * on3080["slices_with_split"]


def test_slice_census_rejects_nonsense():
    with pytest.raises(ValueError):
        marlin_slice_census(0, 10, 100, 170)


# --- fn8c7: NO_K_SPLIT must be a per-RANK decision too ----------------------


def test_bare_no_k_split_still_applies_everywhere():
    """Backward compatible: fn8c7's spelling keeps fn8c7's meaning."""
    from sglang.jit_kernel.marlin_switches import no_k_split_on

    assert no_k_split_on({ENV_NO_K_SPLIT: "1"}, (12, 0)) is True
    assert no_k_split_on({ENV_NO_K_SPLIT: "1"}, (8, 6)) is True


def test_gated_no_k_split_touches_only_the_named_capability():
    """fn8c7 died because the sm_86 ranks took the coarser grid too. They run
    this form with ~1 % of card headroom (card free 0.17-0.37 GiB of 19.58 in
    BOTH fn8c6 and fn8c7), so moving their kernel timing is enough to tip
    self_attention into CUDA OOM."""
    from sglang.jit_kernel.marlin_switches import no_k_split_on

    env = {ENV_NO_K_SPLIT: "1@12.0"}
    assert no_k_split_on(env, (12, 0)) is True
    assert no_k_split_on(env, (8, 6)) is False
    assert no_k_split_on(env, None) is False, "unknown card -> do not touch it"


@pytest.mark.parametrize("gate", ["1@120", "1@12.0"])
def test_both_gate_spellings_parse(gate):
    from sglang.jit_kernel.marlin_switches import parse_gate

    assert parse_gate(gate) == (True, (12, 0))


@pytest.mark.parametrize("raw", ["1@twelve", "1@", "1@x.y"])
def test_a_malformed_gate_disables_rather_than_widens(raw):
    """The dangerous failure is 'gate unparsable -> apply everywhere'."""
    from sglang.jit_kernel.marlin_switches import no_k_split_on

    assert no_k_split_on({ENV_NO_K_SPLIT: raw}, (12, 0)) is False


def test_census_separates_requested_from_applied_for_no_k_split():
    env = {ENV_NO_K_SPLIT: "1@12.0"}
    on3080 = switch_census(68, env, smem_optin=101376, device_cap=(8, 6))
    assert on3080["no_k_split"] is False
    assert on3080["no_k_split_requested"] is True
    assert "no_k_split=False(req=True)" in format_switch_log("8.6", on3080)


def test_cuda_source_parses_the_same_gate():
    src = CUH.read_text()
    assert "marlin_parse_gate" in src
    assert "cudaDevAttrComputeCapabilityMajor" in src
    assert "marlin_moe_no_k_split(dev)" in src
    # A gate it cannot parse must disable, not widen.
    block = src.split("inline MarlinGate marlin_parse_gate")[1].split("\n}\n")[0]
    assert "g.enabled = false;" in block
