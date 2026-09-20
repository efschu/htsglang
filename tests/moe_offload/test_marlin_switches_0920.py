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
    for name in (ENV_EPILOGUE_SYNC, ENV_NO_K_SPLIT, ENV_SMS_OVERRIDE):
        assert f'std::getenv("{name}")' in src, name


def test_cuda_truthiness_sets_match_the_python_ones():
    src = CUH.read_text()
    # no_k_split: raw[0] == '1' || 't' || 'T' || 'y' || 'Y'
    block = src.split("marlin_moe_no_k_split()")[1].split("}")[0]
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


def test_census_and_log_line_name_every_switch():
    env = {
        ENV_NO_K_SPLIT: "1",
        ENV_SMS_OVERRIDE: "68",
        ENV_EPILOGUE_SYNC: "0",
        ENV_ARCH_OVERRIDE: "9.0",
    }
    census = switch_census(170, env)
    assert census == {
        "epilogue_sync": False,
        "no_k_split": True,
        "sms_requested": 68,
        "sms_hw": 170,
        "sms_effective": 68,
        "arch_override": "9.0",
        "ptx_jit": True,
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
    line = format_switch_log("8.6", switch_census(68, {}))
    assert "arch_override=None" in line
    assert "epilogue_sync=True" in line
    assert "no_k_split=False" in line
    assert "sms_hw=68 sms_effective=68" in line
