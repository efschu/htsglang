# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for the dashboard server-manager control-plane.

NO GPU boot and NO real sglang server: the supervisor lifecycle is exercised
with a FAKE child process (a trivial ``python3 -c "... time.sleep"``), and every
NVML / huggingface_hub touch is injected/mocked so the suite is hermetic and
network-free.
"""

import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from sglang.srt.planner.server_manager import (
    DEFAULT_MODEL_ROOTS,
    DownloadInfo,
    LaunchSettings,
    SglangSupervisor,
    SupervisorBusyError,
    _tail_lines,
    available_downloads,
    discover_models,
    download_model,
    model_root_writable,
)


# ---------------------------------------------------------------------------
# Helpers to build a fake model tree + a minimal real GGUF file.
# ---------------------------------------------------------------------------
def _write_config(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f)


# GGUF value-type enums (subset) mirrored from the header spec.
_T_UINT32, _T_STRING = 4, 8


def _write_min_gguf(path, arch="qwen35", file_type=15):
    """Write a minimal but VALID GGUF header with general.architecture (string)
    and general.file_type (uint32) so the config-authoritative reader resolves
    the quant from the header, not the file name."""
    def kv_string(key, val):
        b = b""
        b += struct.pack("<Q", len(key)) + key.encode()
        b += struct.pack("<I", _T_STRING)
        b += struct.pack("<Q", len(val)) + val.encode()
        return b

    def kv_u32(key, val):
        b = b""
        b += struct.pack("<Q", len(key)) + key.encode()
        b += struct.pack("<I", _T_UINT32)
        b += struct.pack("<I", val)
        return b

    kvs = [
        kv_string("general.architecture", arch),
        kv_u32("general.file_type", file_type),
    ]
    body = b"".join(kvs)
    header = b"GGUF"
    header += struct.pack("<I", 3)          # version
    header += struct.pack("<Q", 0)          # tensor count
    header += struct.pack("<Q", len(kvs))   # kv count
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header + body)


# ===========================================================================
# _tail_lines: the O(tail size) log-tail read behind status()/log_tail(),
# on the /api/live_snapshot poll hot path (every LAND_POLL_MS while a server
# is monitored). Must match plain readlines()[-n:] semantics exactly, without
# reading the whole file.
# ===========================================================================
class TestTailLines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tail_")

    def _write(self, name, lines):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.writelines(lines)
        return path

    def test_matches_readlines_baseline(self):
        lines = [f"line {i}\n" for i in range(500)]
        path = self._write("big.log", lines)
        # NOTE: n=0 is included deliberately -- Python's lines[-0:] equals
        # lines[0:] (there is no negative zero), i.e. the WHOLE file, which
        # is the plain-readlines() baseline this helper must match exactly
        # even at that edge case.
        for n in (0, 1, 20, 60, 499, 500, 501, 10000):
            expected = "".join(lines[-n:])
            self.assertEqual(_tail_lines(path, n), expected, msg=f"n={n}")

    def test_crosses_chunk_boundary(self):
        # Force multiple 8192-byte backward-seek chunks so the loop's
        # chunk-stitching path (not just the single-chunk fast case) is hit.
        lines = [f"{i:08d} padding padding padding\n" for i in range(5000)]
        path = self._write("chunky.log", lines)
        self.assertEqual(_tail_lines(path, 20), "".join(lines[-20:]))

    def test_no_trailing_newline_on_last_line(self):
        path = self._write("noeol.log", ["a\n", "b\n", "c (no eol)"])
        self.assertEqual(_tail_lines(path, 2), "b\nc (no eol)")

    def test_empty_file(self):
        path = self._write("empty.log", [])
        self.assertEqual(_tail_lines(path, 20), "")

    def test_missing_file_returns_placeholder(self):
        self.assertEqual(
            _tail_lines(os.path.join(self.tmp, "nope.log"), 20), "(no log)"
        )

    def test_log_tail_uses_tail_lines(self):
        """SglangSupervisor.log_tail() delegates to _tail_lines (the poll-hot
        O(file size) regression this guards is exactly the earlier
        f.readlines()[-n:] implementation of log_tail itself)."""
        path = self._write("sup.log", [f"line {i}\n" for i in range(200)])
        sup = SglangSupervisor.__new__(SglangSupervisor)
        sup._log_path = path
        self.assertEqual(sup.log_tail(10), "".join([f"line {i}\n" for i in range(190, 200)]))


# ===========================================================================
# Discovery.
# ===========================================================================
class TestDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="disco_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hf_hub_snapshot_config_authoritative(self):
        # Dir name LIES ("bf16-fp16"); config.json says fp8 -> config wins.
        snap = os.path.join(
            self.tmp, "models--org--Model-bf16-fp16", "snapshots", "deadbeef")
        _write_config(os.path.join(snap, "config.json"), {
            "architectures": ["Qwen3ForCausalLM"],
            "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"},
        })
        ms = discover_models(roots=[self.tmp])
        self.assertEqual(len(ms), 1)
        m = ms[0]
        self.assertEqual(m.format, "hf")
        self.assertEqual(m.quant_method, "fp8")  # NOT inferred from the name
        self.assertEqual(m.name, "org/Model-bf16-fp16")
        self.assertFalse(m.vision)

    def test_hf_vision_and_bf16(self):
        d = os.path.join(self.tmp, "vlm-model")
        _write_config(os.path.join(d, "config.json"), {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "vision_config": {"depth": 24},
        })
        ms = discover_models(roots=[self.tmp])
        self.assertEqual(len(ms), 1)
        self.assertEqual(ms[0].quant_method, "bf16")  # no quant config
        self.assertTrue(ms[0].vision)

    def test_gguf_variants_header_authoritative(self):
        d = os.path.join(self.tmp, "MyModel-Q2-lie-GGUF")
        # File names claim Q2; headers say Q4_K_M (15) and Q6_K (18).
        _write_min_gguf(os.path.join(d, "MyModel-Q2-a.gguf"), file_type=15)
        _write_min_gguf(os.path.join(d, "MyModel-Q2-b.gguf"), file_type=18)
        # sidecars must NOT become selectable variants.
        _write_min_gguf(os.path.join(d, "mmproj-BF16.gguf"), file_type=32)
        _write_min_gguf(os.path.join(d, "mtp-MyModel.gguf"), file_type=15)
        ms = discover_models(roots=[self.tmp])
        self.assertEqual(len(ms), 1)
        m = ms[0]
        self.assertEqual(m.format, "gguf")
        quants = sorted(v.quant for v in m.gguf_variants)
        self.assertEqual(quants, ["Q4_K_M", "Q6_K"])  # header, not name; no sidecars
        self.assertTrue(m.vision)  # mmproj sidecar -> multimodal
        self.assertEqual(m.quant_method, "gguf")  # multi-variant label

    def test_single_gguf_quant_label(self):
        d = os.path.join(self.tmp, "Solo-GGUF")
        _write_min_gguf(os.path.join(d, "solo.gguf"), file_type=12)  # Q3_K_M
        ms = discover_models(roots=[self.tmp])
        self.assertEqual(ms[0].quant_method, "Q3_K_M")

    def test_bad_model_tagged_not_thrown(self):
        # One dir with unreadable JSON, one good dir -> good still found, bad
        # tagged with .error (never raised).
        bad = os.path.join(self.tmp, "broken")
        os.makedirs(bad)
        with open(os.path.join(bad, "config.json"), "w") as f:
            f.write("{ this is not json ")
        good = os.path.join(self.tmp, "good")
        _write_config(os.path.join(good, "config.json"), {
            "architectures": ["Qwen3ForCausalLM"]})
        ms = discover_models(roots=[self.tmp])
        by_name = {m.name: m for m in ms}
        self.assertIn("good", by_name)
        self.assertIn("broken", by_name)
        self.assertIsNotNone(by_name["broken"].error)
        self.assertIsNone(by_name["good"].error)

    def test_missing_root_is_robust(self):
        # Non-existent roots must not throw.
        ms = discover_models(roots=["/no/such/dir/at/all"])
        self.assertEqual(ms, [])

    def test_org_subdir_recursion(self):
        # An org dir containing a model one level down is still found.
        d = os.path.join(self.tmp, "unsloth", "Some-Model")
        _write_config(os.path.join(d, "config.json"), {
            "architectures": ["Qwen3ForCausalLM"]})
        ms = discover_models(roots=[self.tmp])
        self.assertEqual(len(ms), 1)
        # The sub-dir stays in the display name -- two org dirs may hold the
        # same model basename.
        self.assertEqual(ms[0].name, "unsloth/Some-Model")

    def test_default_roots_constant(self):
        # Without either env var the defaults are the generic locations.
        from sglang.srt.planner.server_manager import _model_roots_from_env

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_MODEL_ROOTS", None)
            os.environ.pop("SGLANG_PLANNER_MODEL_ROOTS", None)
            self.assertEqual(
                _model_roots_from_env(),
                ("~/.cache/huggingface/hub", "./models"))

    def test_model_roots_env_override(self):
        # SGLANG_MODEL_ROOTS (colon-separated) replaces the generic defaults.
        from sglang.srt.planner.server_manager import _model_roots_from_env

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_PLANNER_MODEL_ROOTS", None)
            os.environ["SGLANG_MODEL_ROOTS"] = "/a/models:/b/cache"
            self.assertEqual(_model_roots_from_env(), ("/a/models", "/b/cache"))

    def test_planner_env_var_wins_over_legacy(self):
        # SGLANG_PLANNER_MODEL_ROOTS is the documented name; the older
        # SGLANG_MODEL_ROOTS stays accepted but loses when both are set.
        from sglang.srt.planner.server_manager import _model_roots_from_env

        with mock.patch.dict(
            os.environ,
            {"SGLANG_PLANNER_MODEL_ROOTS": "/new", "SGLANG_MODEL_ROOTS": "/old"},
        ):
            self.assertEqual(_model_roots_from_env(), ("/new",))

    def test_set_model_roots_override_and_clear(self):
        # --model-root wins over the env vars; clearing restores the env layer.
        from sglang.srt.planner.server_manager import (
            model_roots,
            set_model_roots,
        )

        with mock.patch.dict(
            os.environ, {"SGLANG_PLANNER_MODEL_ROOTS": "/from/env"}
        ):
            try:
                set_model_roots(["/from/flag", "/second"])
                self.assertEqual(model_roots(), ("/from/flag", "/second"))
                # The legacy alias reads through to the live value.
                self.assertEqual(
                    list(DEFAULT_MODEL_ROOTS), ["/from/flag", "/second"])
                set_model_roots(None)
                self.assertEqual(model_roots(), ("/from/env",))
            finally:
                set_model_roots(None)

    def test_nested_dirs_keep_their_subpath_in_the_name(self):
        # Two checkpoints with the same basename in different sub-dirs must
        # stay tellable apart instead of colliding under one display name.
        for sub in ("unsloth", "legacy"):
            _write_config(
                os.path.join(self.tmp, sub, "Same-Name", "config.json"),
                {"architectures": ["Qwen3ForCausalLM"]})
        ms = discover_models(roots=[self.tmp])
        names = sorted(m.name for m in ms)
        self.assertEqual(names, ["legacy/Same-Name", "unsloth/Same-Name"])

    def test_symlink_loop_does_not_duplicate_models(self):
        # A root containing a symlink back to itself must not yield the same
        # model twice (and must not recurse forever).
        _write_config(
            os.path.join(self.tmp, "Only-Model", "config.json"),
            {"architectures": ["Qwen3ForCausalLM"]})
        try:
            os.symlink(self.tmp, os.path.join(self.tmp, "self-link"))
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        ms = discover_models(roots=[self.tmp])
        self.assertEqual([m.name for m in ms], ["Only-Model"])

    def test_duplicate_root_scanned_once(self):
        _write_config(
            os.path.join(self.tmp, "Only-Model", "config.json"),
            {"architectures": ["Qwen3ForCausalLM"]})
        ms = discover_models(roots=[self.tmp, self.tmp])
        self.assertEqual(len(ms), 1)


# ===========================================================================
# LaunchSettings validation + argv.
# ===========================================================================
class TestLaunchSettings(unittest.TestCase):
    def test_rank_gpu_id_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            LaunchSettings(model_path="/m", tp_size=4,
                           rank_gpu_id=[0, 1]).validate()

    def test_rank_tp_ratio_length_checked(self):
        with self.assertRaises(ValueError):
            LaunchSettings(model_path="/m", tp_size=2,
                           rank_tp_ratio=[1, 1, 1]).validate()

    def test_valid_matches(self):
        LaunchSettings(model_path="/m", tp_size=4,
                       rank_gpu_id=[0, 1, 1, 2]).validate()  # no raise

    def test_bad_spec_mode(self):
        with self.assertRaises(ValueError):
            LaunchSettings(model_path="/m", spec_mode="turbo").validate()

    def test_argv_contains_expected_flags(self):
        s = LaunchSettings(
            model_path="/m", tp_size=2, rank_gpu_id=[0, 1],
            rank_gpu_memory_mib=[16000, 16000], kv_cache_dtype="fp8",
            spec_mode="mtp", speculative_num_steps=3, port=8100,
            served_model_name="Qwen", mem_fraction_static=0.82,
            chat_template="/tpl.jinja", tool_call_parser="qwen")
        cmd = s.launch_command()
        j = " ".join(cmd)
        self.assertIn("--model-path /m", j)
        self.assertIn("--tp-size 2", j)
        self.assertIn("--rank-gpu-id 0,1", j)
        self.assertIn("--rank-gpu-memory-mib 16000,16000", j)
        self.assertIn("--kv-cache-dtype fp8", j)
        self.assertIn("--speculative-algorithm NEXTN", j)
        self.assertIn("--speculative-num-steps 3", j)
        self.assertIn("--port 8100", j)
        self.assertIn("--served-model-name Qwen", j)
        self.assertIn("--mem-fraction-static 0.82", j)
        self.assertIn("--chat-template /tpl.jinja", j)
        self.assertIn("--tool-call-parser qwen", j)

    def test_gguf_variant_path_resolution(self):
        s = LaunchSettings(
            model_path="/models/Repo-GGUF", format="gguf",
            gguf_variant="model-Q4_K_M.gguf")
        self.assertEqual(
            s.resolved_model_path(), "/models/Repo-GGUF/model-Q4_K_M.gguf")

    def test_extra_env_wins_over_supervisor_defaults(self):
        # A profile's launch env (flags.profile_env) must override the
        # supervisor defaults so a launched profile matches its reference
        # command exactly (PYTHONPATH / LD_LIBRARY_PATH / SGLANG_UNEVEN_*).
        from sglang.srt.planner.server_manager import SglangSupervisor

        sup = SglangSupervisor(nvml=object())
        s = LaunchSettings(
            model_path="/m",
            extra_env={"SGLANG_UNEVEN_DCP": "1", "PYTHONPATH": "/custom"},
        )
        env = sup._build_env(s)
        self.assertEqual(env["SGLANG_UNEVEN_DCP"], "1")
        self.assertEqual(env["PYTHONPATH"], "/custom")

    def test_no_extra_env_keeps_default_behavior(self):
        from sglang.srt.planner.server_manager import SglangSupervisor

        sup = SglangSupervisor(nvml=object())
        env = sup._build_env(LaunchSettings(model_path="/m"))
        self.assertIn("PYTHONPATH", env)


# ===========================================================================
# Supervisor lifecycle with a FAKE child (never a real sglang boot).
# ===========================================================================
def _fake_child_settings(port):
    # A harmless real LaunchSettings; the actual argv is overridden in start().
    return LaunchSettings(model_path="/fake/model", port=port)


def _sleep_argv(seconds=300):
    # A child that itself spawns a grandchild in the SAME process group, so we
    # can prove killpg reaps the whole group (not just the direct child).
    code = (
        "import subprocess, time, sys;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(%d)']);"
        "time.sleep(%d)" % (seconds, seconds)
    )
    return [sys.executable, "-c", code]


def _pgid_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class TestSupervisorLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sup_")
        self.sup = SglangSupervisor(log_dir=self.tmp)
        self._siblings = []

    def tearDown(self):
        try:
            if self.sup.is_running():
                self.sup.stop(wait_vram=False)
        except Exception:
            pass
        for p in self._siblings:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _spawn_sibling(self):
        # An UNRELATED process in its OWN group -> must survive supervisor stop.
        p = subprocess.Popen(_sleep_argv(), start_new_session=True)
        self._siblings.append(p)
        return p

    def test_start_running_then_stop_kills_group(self):
        st = self.sup.start(
            _fake_child_settings(39001), argv=_sleep_argv(), wait_ready=False)
        self.assertEqual(st["state"], "booting")
        self.assertTrue(self.sup.is_running())
        pgid = self.sup._pgid
        self.assertTrue(_pgid_alive(pgid))
        rep = self.sup.stop(wait_vram=False)
        self.assertFalse(self.sup.is_running())
        self.assertTrue(rep["group_gone"])
        # The whole process group (child + grandchild) is gone.
        self.assertFalse(_pgid_alive(pgid))
        self.assertEqual(self.sup.state, "stopped")

    def test_status_flips_booting_to_ready_when_server_answers(self):
        # start(wait_ready=False) returns in BOOTING so the dashboard stays
        # responsive; the poll-driven status() must flip to READY once the
        # server answers. The real-boot bug was status() sitting on BOOTING
        # forever while the server was up and serving.
        self.sup.start(
            _fake_child_settings(39010), argv=_sleep_argv(), wait_ready=False)
        self.assertEqual(self.sup.state, "booting")

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with mock.patch("urllib.request.urlopen", return_value=_Resp()):
            st = self.sup.status()
        self.assertEqual(st["state"], "ready")

    def test_status_flips_booting_to_error_past_deadline(self):
        # A boot that never comes up must not stay BOOTING forever.
        self.sup.start(
            _fake_child_settings(39011), argv=_sleep_argv(), wait_ready=False)
        self.sup._boot_deadline = time.time() - 1  # already elapsed

        def _boom(*a, **k):
            raise OSError("connection refused")

        with mock.patch("urllib.request.urlopen", side_effect=_boom):
            st = self.sup.status()
        self.assertEqual(st["state"], "error")
        self.assertIn("boot deadline", st["error"])

    def test_stop_does_not_kill_unrelated_sibling(self):
        sibling = self._spawn_sibling()
        sibling_pgid = os.getpgid(sibling.pid)
        self.sup.start(
            _fake_child_settings(39002), argv=_sleep_argv(), wait_ready=False)
        managed_pgid = self.sup._pgid
        self.assertNotEqual(sibling_pgid, managed_pgid)
        self.sup.stop(wait_vram=False)
        # NO broad kill: the sibling's group is untouched.
        self.assertFalse(_pgid_alive(managed_pgid))
        self.assertTrue(_pgid_alive(sibling_pgid))
        self.assertIsNone(sibling.poll())  # still alive

    def test_restart_gives_new_pid(self):
        self.sup.start(
            _fake_child_settings(39003), argv=_sleep_argv(), wait_ready=False)
        pid1 = self.sup.proc.pid
        pgid1 = self.sup._pgid
        self.sup.restart(
            _fake_child_settings(39003), argv=_sleep_argv(), wait_ready=False)
        pid2 = self.sup.proc.pid
        self.assertNotEqual(pid1, pid2)
        self.assertTrue(self.sup.is_running())
        # Old group is gone; new group is alive.
        self.assertFalse(_pgid_alive(pgid1))
        self.assertTrue(_pgid_alive(self.sup._pgid))

    def test_is_busy_guard_refuses_restart(self):
        self.sup.start(
            _fake_child_settings(39004), argv=_sleep_argv(), wait_ready=False)
        pid1 = self.sup.proc.pid
        self.sup.set_busy(True)
        with self.assertRaises(SupervisorBusyError):
            self.sup.restart(_fake_child_settings(39004), argv=_sleep_argv(),
                             wait_ready=False)
        # The running child is untouched by the refused restart.
        self.assertTrue(self.sup.is_running())
        self.assertEqual(self.sup.proc.pid, pid1)
        self.sup.set_busy(False)

    def test_start_rejects_double_start(self):
        self.sup.start(
            _fake_child_settings(39005), argv=_sleep_argv(), wait_ready=False)
        with self.assertRaises(RuntimeError):
            self.sup.start(_fake_child_settings(39005), argv=_sleep_argv(),
                           wait_ready=False)

    def test_child_death_surfaces_as_error(self):
        # A child that exits immediately -> status() reports error, no wedge.
        self.sup.start(
            _fake_child_settings(39006),
            argv=[sys.executable, "-c", "import sys; sys.exit(3)"],
            wait_ready=False)
        # Give it a moment to exit.
        for _ in range(50):
            if self.sup.proc.poll() is not None:
                break
            time.sleep(0.05)
        st = self.sup.status()
        self.assertEqual(st["state"], "error")
        self.assertIsNotNone(st["error"])

    def test_stop_when_never_started_is_safe(self):
        rep = self.sup.stop(wait_vram=False)
        self.assertEqual(self.sup.state, "stopped")
        self.assertIsInstance(rep, dict)


# ===========================================================================
# VRAM-free guard with a MOCK nvml (no GPU).
# ===========================================================================
class _FakeMem:
    def __init__(self, free_mib):
        self.free = free_mib * 2 ** 20


class _FakeNvml:
    """Deterministic pynvml-like: free memory recovers after N polls."""

    def __init__(self, baseline_free_mib, recover_after=2):
        self.baseline = baseline_free_mib
        self.recover_after = recover_after
        self._calls = 0

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetMemoryInfo(self, h):
        # First call = baseline (captured at start). After boot the child
        # "used" 4 GiB; it recovers to baseline after recover_after polls.
        self._calls += 1
        if self._calls == 1:
            return _FakeMem(self.baseline)
        if self._calls <= 1 + self.recover_after:
            return _FakeMem(self.baseline - 4096)
        return _FakeMem(self.baseline)


class TestVramGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vram_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_vram_recovers_true(self):
        nvml = _FakeNvml(baseline_free_mib=20000, recover_after=1)
        sup = SglangSupervisor(log_dir=self.tmp, nvml=nvml)
        s = LaunchSettings(model_path="/fake", tp_size=1, rank_gpu_id=[0],
                           port=39100)
        sup.start(s, argv=_sleep_argv(), wait_ready=False)
        # baseline captured at start (call 1).
        rep = sup.stop(wait_vram=True, vram_timeout_s=5)
        self.assertTrue(rep["vram_recovered"])

    def test_vram_no_indices_returns_none(self):
        # No rank_gpu_id -> nothing to poll -> None (can't judge).
        nvml = _FakeNvml(baseline_free_mib=20000)
        sup = SglangSupervisor(log_dir=self.tmp, nvml=nvml)
        s = LaunchSettings(model_path="/fake", tp_size=1, port=39101)
        sup.start(s, argv=_sleep_argv(), wait_ready=False)
        rep = sup.stop(wait_vram=True, vram_timeout_s=2)
        self.assertIsNone(rep["vram_recovered"])


# ===========================================================================
# Download layer (writability + gated download + quant->file mapping).
# ===========================================================================
class TestWritability(unittest.TestCase):
    def test_rw_dir_true(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(model_root_writable(d))

    def test_readonly_dir_false(self):
        d = tempfile.mkdtemp()
        try:
            os.chmod(d, 0o500)  # r-x, no write
            # (running as non-root this is a hard denial; guard for root.)
            if os.geteuid() != 0:
                self.assertFalse(model_root_writable(d))
            else:
                # As root, os.access/W_OK is bypassed; the temp-file probe is
                # the real signal but root can write anywhere. Assert the probe
                # at least does not throw.
                self.assertIsInstance(model_root_writable(d), bool)
        finally:
            os.chmod(d, 0o700)
            os.rmdir(d)

    def test_missing_dir_false(self):
        self.assertFalse(model_root_writable("/no/such/dir"))


class _FakeHfApi:
    def __init__(self, files):
        self._files = files
        self.calls = []

    def list_repo_files(self, repo_id):
        self.calls.append(repo_id)
        return self._files


class TestDownloads(unittest.TestCase):
    def test_available_downloads_parses_gguf_variants(self):
        api = _FakeHfApi([
            "README.md",
            "Model-Q3_K_M.gguf",
            "Model-Q4_K_M.gguf",
            "Model-Q6_K.gguf",
            "mmproj-BF16.gguf",  # sidecar, excluded
        ])
        info = available_downloads("org/Model-GGUF", hf_api=api)
        self.assertIsInstance(info, DownloadInfo)
        self.assertTrue(info.is_gguf)
        quants = sorted(v.quant for v in info.gguf_variants)
        self.assertEqual(quants, ["Q3_K_M", "Q4_K_M", "Q6_K"])

    def test_available_downloads_non_gguf(self):
        api = _FakeHfApi(["config.json", "model.safetensors"])
        info = available_downloads("org/HF-Model", hf_api=api)
        self.assertFalse(info.is_gguf)
        self.assertEqual(info.gguf_variants, [])

    def test_download_refuses_on_readonly_root(self):
        if os.geteuid() == 0:
            self.skipTest("running as root: W_OK/chmod probe is bypassed")
        d = tempfile.mkdtemp()
        try:
            os.chmod(d, 0o500)
            called = {"n": 0}

            def fake_dl(**kw):
                called["n"] += 1
                return "/x"

            with self.assertRaises(PermissionError):
                download_model("org/Repo", quant="Q4_K_M", root=d,
                               hf_hub_download=fake_dl)
            self.assertEqual(called["n"], 0)  # never reached the network
        finally:
            os.chmod(d, 0o700)
            os.rmdir(d)

    def test_download_refuses_when_probe_says_readonly(self):
        # uid-independent: patch the writability probe to False and assert the
        # gate refuses BEFORE any network call (covers the refusal path even
        # when the suite runs as root, where chmod cannot restrict writes).
        called = {"n": 0}

        def fake_dl(**kw):
            called["n"] += 1
            return "/x"

        with mock.patch(
            "sglang.srt.planner.server_manager.model_root_writable",
            return_value=False,
        ):
            with self.assertRaises(PermissionError):
                download_model("org/Repo", quant="Q4_K_M", root="/some/mount",
                               hf_hub_download=fake_dl,
                               hf_api=_FakeHfApi(["Model-Q4_K_M.gguf"]))
        self.assertEqual(called["n"], 0)

    def test_gguf_quant_maps_to_single_file(self):
        api = _FakeHfApi([
            "Model-Q3_K_M.gguf",
            "Model-Q4_K_M.gguf",
            "Model-Q6_K.gguf",
        ])
        with tempfile.TemporaryDirectory() as d:
            recorded = {}

            def fake_hub_download(**kw):
                recorded.update(kw)
                out = os.path.join(kw["local_dir"], kw["filename"])
                return out

            path = download_model(
                "org/Model-GGUF", quant="Q4_K_M", root=d,
                hf_api=api, hf_hub_download=fake_hub_download)
            # Only the chosen quant file is fetched (no whole-repo pull).
            self.assertEqual(recorded["filename"], "Model-Q4_K_M.gguf")
            self.assertEqual(recorded["repo_id"], "org/Model-GGUF")
            self.assertEqual(recorded["local_dir"], d)
            self.assertTrue(path.endswith("Model-Q4_K_M.gguf"))

    def test_full_hf_snapshot_download(self):
        with tempfile.TemporaryDirectory() as d:
            recorded = {}

            def fake_snapshot(**kw):
                recorded.update(kw)
                return os.path.join(kw["local_dir"], "snap")

            path = download_model(
                "org/HF-Model", quant=None, root=d,
                snapshot_download=fake_snapshot)
            self.assertEqual(recorded["repo_id"], "org/HF-Model")
            self.assertEqual(recorded["local_dir"], d)
            self.assertTrue(path.endswith("snap"))

    def test_progress_callback_invoked(self):
        api = _FakeHfApi(["Model-Q4_K_M.gguf"])
        with tempfile.TemporaryDirectory() as d:
            events = []
            download_model(
                "org/Model-GGUF", quant="Q4_K_M", root=d, hf_api=api,
                hf_hub_download=lambda **kw: "/x",
                progress_cb=events.append)
            stages = [e["stage"] for e in events]
            self.assertIn("start", stages)
            self.assertIn("done", stages)


class TestHfHubImportable(unittest.TestCase):
    def test_huggingface_hub_present(self):
        import huggingface_hub  # sglang dependency; download layer needs it

        self.assertTrue(hasattr(huggingface_hub, "hf_hub_download"))


if __name__ == "__main__":
    unittest.main()


class TestGgufAndSpecArgv(unittest.TestCase):
    """A GGUF boot is a loader identity, not a rewritten --model-path, and the
    speculative depth is part of the configuration."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ggufargv_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _argv(self, **kw):
        return LaunchSettings(model_path=self.tmp, **kw).launch_command()

    def test_gguf_derives_loader_and_tokenizer(self):
        argv = self._argv(format="gguf", gguf_variant="m-Q3_K_M.gguf")
        self.assertIn("--load-format", argv)
        self.assertEqual(argv[argv.index("--load-format") + 1], "gguf")
        # the .gguf file has no HF tokenizer beside it -> the model DIR is it
        self.assertIn("--tokenizer-path", argv)
        self.assertEqual(argv[argv.index("--tokenizer-path") + 1], self.tmp)
        self.assertTrue(
            argv[argv.index("--model-path") + 1].endswith("m-Q3_K_M.gguf"))

    def test_explicit_loader_wins_over_the_derived_one(self):
        argv = self._argv(format="gguf", gguf_variant="m.gguf",
                          load_format="dummy", tokenizer_path="/tok")
        self.assertEqual(argv[argv.index("--load-format") + 1], "dummy")
        self.assertEqual(argv[argv.index("--tokenizer-path") + 1], "/tok")

    def test_hf_format_adds_no_loader_flags(self):
        argv = self._argv()
        self.assertNotIn("--load-format", argv)
        self.assertNotIn("--tokenizer-path", argv)

    def test_spec_depth_and_reserve_reach_the_command(self):
        argv = self._argv(spec_mode="mtp", speculative_num_steps=3,
                          speculative_eagle_topk=1,
                          speculative_num_draft_tokens=4,
                          speculative_draft_model_path="/draft",
                          rank_auto_reserve_mib=2700,
                          quantization="gguf")
        for flag, val in (("--speculative-algorithm", "NEXTN"),
                          ("--speculative-num-steps", "3"),
                          ("--speculative-eagle-topk", "1"),
                          ("--speculative-num-draft-tokens", "4"),
                          ("--speculative-draft-model-path", "/draft"),
                          ("--rank-auto-reserve-mib", "2700"),
                          ("--quantization", "gguf")):
            self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index(flag) + 1], val)


class TestModelPathRequired(unittest.TestCase):
    def test_empty_model_path_is_rejected_before_boot(self):
        with self.assertRaises(ValueError) as cm:
            LaunchSettings(model_path="").validate()
        self.assertIn("model_path is required", str(cm.exception))

    def test_whitespace_only_is_rejected_too(self):
        with self.assertRaises(ValueError):
            LaunchSettings(model_path="   ").validate()
