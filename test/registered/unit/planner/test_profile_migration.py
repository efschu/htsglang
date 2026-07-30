"""What a PROFILE_VERSION bump costs the next boot (task #303).

Both bumps of the stage-0 hardware profile so far only ADDED per-GPU fields
(v2: the three membw rates; v3: the quantized GEMM lanes). The cache key
carries the version, so a bump moved the cache file and every
auto-performance boot afterwards ran a fresh stage-0 probe -- including the
pairwise NCCL link matrix, the one phase that joins a process group and can
therefore wait on something other than this rig's own hardware. On the
reference rig that phase hung and cost 600 s of card time per boot.

What is asserted here, all on CPU with the probe stubbed out:

* a v2 cache is MIGRATED, not discarded -- every measured value survives,
  including the link matrix;
* only the fields the newer version added are reported missing;
* the lazy top-up runs exactly the probe groups those fields belong to, and
  never the link matrix;
* a failed top-up still yields the carried-over profile, with a stored reason
  for each field it could not measure;
* the link phase is bounded by a wall clock, and on expiry the profile is
  cached with the per-card measurements plus a named reason;
* nothing in the probe path waits without a deadline.
"""

import json
import os
import time
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DRIVER = "595.58.03"
_UUIDS = ["GPU-aaa", "GPU-bbb", "GPU-ccc"]
_GPUS = [
    {"cuda_index": 0, "uuid": "GPU-aaa", "name": "RTX 5090", "total_mib": 32607},
    {"cuda_index": 1, "uuid": "GPU-bbb", "name": "RTX 3080", "total_mib": 20480},
    {"cuda_index": 2, "uuid": "GPU-ccc", "name": "RTX 3080", "total_mib": 20480},
]
_LINKS = {
    "GPU-aaa|GPU-bbb": {"p2p_gbs": 5.1},
    "GPU-aaa|GPU-ccc": {"p2p_gbs": 9.06},
    "GPU-bbb|GPU-ccc": {"p2p_gbs": 5.83},
    "__group__": {"ar_10kb_us": 32.4, "ar_1mb_us": 361.3},
}


def _v2_profile():
    """A real-shaped v2 cache file: every v2 field, no lanes."""
    gpus = {}
    for i, g in enumerate(_GPUS):
        gpus[g["uuid"]] = {
            "name": g["name"],
            "cuda_index": i,
            "total_mib": g["total_mib"],
            "gemm_tflops": 232.97 if i == 0 else 62.72,
            "membw_gbs": 1664.2 if i == 0 else 717.8,
            "membw_read_gbs": 1664.2 if i == 0 else 717.8,
            "membw_copy_gbs": 1520.0 if i == 0 else 700.0,
            "membw_gemv_gbs": 1529.7 if i == 0 else 717.8,
        }
    return {
        "version": 2,
        "driver": _DRIVER,
        "uuids": sorted(_UUIDS),
        "gpus": gpus,
        "links": dict(_LINKS),
        "probe_seconds": 11.6,
        "created": "2026-07-27 15:34:30",
    }


class TestMigrateProfile(CustomTestCase):
    """The pure migration: values carried over, added fields named."""

    def test_v2_keeps_every_measured_value(self):
        old = _v2_profile()
        migrated, missing = uneven_perf.migrate_profile(old)
        self.assertEqual(migrated["version"], uneven_perf.PROFILE_VERSION)
        self.assertEqual(migrated["migrated_from"], 2)
        self.assertEqual(migrated["links"], _LINKS)
        self.assertEqual(migrated["driver"], _DRIVER)
        for uuid, entry in old["gpus"].items():
            for key, value in entry.items():
                self.assertEqual(migrated["gpus"][uuid][key], value)
        self.assertEqual(sorted(missing), sorted(_UUIDS))
        for gaps in missing.values():
            self.assertEqual(sorted(gaps), ["gemm_lane_notes", "gemm_lanes"])

    def test_migration_does_not_mutate_the_input(self):
        old = _v2_profile()
        frozen = json.dumps(old, sort_keys=True)
        uneven_perf.migrate_profile(old)
        self.assertEqual(json.dumps(old, sort_keys=True), frozen)

    def test_v1_reports_both_version_steps(self):
        old = _v2_profile()
        old["version"] = 1
        for entry in old["gpus"].values():
            for key in ("membw_read_gbs", "membw_copy_gbs", "membw_gemv_gbs"):
                entry.pop(key)
        _, missing = uneven_perf.migrate_profile(old)
        self.assertEqual(
            sorted(missing["GPU-aaa"]),
            [
                "gemm_lane_notes",
                "gemm_lanes",
                "membw_copy_gbs",
                "membw_gemv_gbs",
                "membw_read_gbs",
            ],
        )

    def test_probe_groups_for_fields_maps_only_what_is_missing(self):
        self.assertEqual(
            uneven_perf.probe_groups_for_fields(["gemm_lanes", "gemm_lane_notes"]),
            [uneven_perf.PROBE_GROUP_LANES],
        )
        self.assertEqual(
            uneven_perf.probe_groups_for_fields(["membw_gemv_gbs", "gemm_lanes"]),
            [uneven_perf.PROBE_GROUP_LANES, uneven_perf.PROBE_GROUP_MEMBW],
        )

    def test_every_declared_field_has_a_probe_group(self):
        """A future bump that forgets to register the group would migrate a
        field nothing can measure."""
        for version, fields in uneven_perf._PROFILE_VERSION_FIELDS.items():
            for field in fields:
                self.assertIn(
                    field,
                    uneven_perf._FIELD_PROBE_GROUP,
                    f"v{version} field {field} belongs to no probe group",
                )


class _CacheFixture:
    """A cache directory holding a v2 profile for the fake rig."""

    def __init__(self, tmpdir, profile=None):
        self.dir = tmpdir
        self.profile = _v2_profile() if profile is None else profile
        version = int(self.profile["version"])
        self.old_path = os.path.join(
            tmpdir, os.path.basename(self._path(version=version))
        )
        self.new_path = os.path.join(tmpdir, os.path.basename(self._path()))
        with open(self.old_path, "w") as f:
            json.dump(self.profile, f)

    @staticmethod
    def _path(version=None):
        return uneven_perf.profile_cache_path(_UUIDS, _DRIVER, version=version)


class TestLazyTopUp(CustomTestCase):
    """The boot path: migrate from cache, top up only the added fields."""

    def setUp(self):
        self._tmp = __import__("tempfile").TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(uneven_perf, "PROFILE_CACHE_DIR", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        inv = mock.patch.object(
            uneven_perf, "_nvml_gpu_inventory", return_value=(_GPUS, _DRIVER)
        )
        inv.start()
        self.addCleanup(inv.stop)
        self.cache = _CacheFixture(self._tmp.name)
        self.calls = []

    def _fake_probe(self, lanes=True):
        """Stand in for the probe subprocess: record the groups it was asked
        for, and merge plausible values for exactly those groups."""

        def run(path, groups):
            self.calls.append(list(groups) if groups else None)
            if groups is None:
                # A full probe writes a profile from scratch.
                profile = _v2_profile()
                profile["links"] = {"reprobed": True}
            else:
                with open(path) as f:
                    profile = json.load(f)
            for entry in profile["gpus"].values():
                if groups is None or uneven_perf.PROBE_GROUP_LANES in groups:
                    entry["gemm_lanes"] = (
                        {uneven_perf.LANE_FP8_NATIVE: 566.88} if lanes else {}
                    )
                    entry["gemm_lane_notes"] = {}
                if groups is None or uneven_perf.PROBE_GROUP_MEMBW in groups:
                    entry["membw_gemv_gbs"] = 1529.7
            profile["version"] = uneven_perf.PROFILE_VERSION
            profile["probe_seconds"] = 4.2
            with open(path, "w") as f:
                json.dump(profile, f)
            return None

        return run

    def test_v2_cache_is_migrated_and_only_the_lanes_are_reprobed(self):
        with mock.patch.object(
            uneven_perf, "_run_probe_subprocess", side_effect=self._fake_probe()
        ):
            profile, source, gpus = uneven_perf.get_hardware_profile()
        self.assertEqual(self.calls, [[uneven_perf.PROBE_GROUP_LANES]])
        self.assertIn("migrated v2", source)
        self.assertEqual(profile["version"], uneven_perf.PROFILE_VERSION)
        # The link matrix was NOT re-measured: it is the v2 one, verbatim.
        self.assertEqual(profile["links"], _LINKS)
        self.assertEqual(profile["gpus"]["GPU-aaa"]["gemm_tflops"], 232.97)
        self.assertEqual(
            profile["gpus"]["GPU-aaa"]["gemm_lanes"],
            {uneven_perf.LANE_FP8_NATIVE: 566.88},
        )
        self.assertEqual([g["uuid"] for g in gpus], _UUIDS)

    def test_the_migrated_profile_is_written_to_the_current_version_path(self):
        with mock.patch.object(
            uneven_perf, "_run_probe_subprocess", side_effect=self._fake_probe()
        ):
            uneven_perf.get_hardware_profile()
        self.assertTrue(os.path.exists(self.cache.new_path))
        with open(self.cache.new_path) as f:
            written = json.load(f)
        self.assertEqual(written["version"], uneven_perf.PROFILE_VERSION)
        # A second boot is a plain cache hit: no probe of any kind.
        self.calls.clear()
        with mock.patch.object(
            uneven_perf, "_run_probe_subprocess", side_effect=self._fake_probe()
        ):
            _, source, _ = uneven_perf.get_hardware_profile()
        self.assertEqual(self.calls, [])
        self.assertIn("cache (", source)

    def test_a_failed_topup_keeps_the_carried_over_values_with_a_reason(self):
        with mock.patch.object(
            uneven_perf,
            "_run_probe_subprocess",
            return_value="probe subprocess exited with rc=1",
        ):
            profile, source, _ = uneven_perf.get_hardware_profile()
        self.assertIn("top-up failed", source)
        self.assertEqual(profile["version"], uneven_perf.PROFILE_VERSION)
        self.assertEqual(profile["links"], _LINKS)
        self.assertEqual(profile["gpus"]["GPU-aaa"]["membw_gemv_gbs"], 1529.7)
        self.assertNotIn("gemm_lanes", profile["gpus"]["GPU-aaa"])
        note = profile[uneven_perf.PROFILE_NOTES_KEY]["gemm_lanes"]
        self.assertIn("rc=1", note)
        self.assertIn("SGLANG_PERF_REPROBE", note)

    def test_reprobe_forced_ignores_the_migration_entirely(self):
        with mock.patch.object(
            uneven_perf, "_run_probe_subprocess", side_effect=self._fake_probe()
        ), mock.patch.dict(os.environ, {"SGLANG_PERF_REPROBE": "1"}):
            _, source, _ = uneven_perf.get_hardware_profile()
        self.assertEqual(self.calls, [None])
        self.assertIn("fresh probe", source)

    def test_no_cache_at_all_runs_the_full_probe(self):
        os.remove(self.cache.old_path)
        with mock.patch.object(
            uneven_perf, "_run_probe_subprocess", side_effect=self._fake_probe()
        ):
            _, source, _ = uneven_perf.get_hardware_profile()
        self.assertEqual(self.calls, [None])
        self.assertIn("fresh probe", source)

    def test_a_foreign_rig_key_is_not_migrated(self):
        """A profile of DIFFERENT cards must never be carried forward."""
        stale = _v2_profile()
        stale["uuids"] = ["GPU-zzz"]
        with open(self.cache.old_path, "w") as f:
            json.dump(stale, f)
        with mock.patch.object(
            uneven_perf, "_run_probe_subprocess", side_effect=self._fake_probe()
        ):
            _, source, _ = uneven_perf.get_hardware_profile()
        self.assertEqual(self.calls, [None])
        self.assertIn("fresh probe", source)

    def test_cache_only_reader_migrates_without_probing(self):
        profile, gpus = uneven_perf.get_cached_hardware_profile()
        self.assertIsNotNone(profile)
        self.assertEqual(profile["version"], uneven_perf.PROFILE_VERSION)
        self.assertEqual(profile["gpus"]["GPU-aaa"]["membw_gemv_gbs"], 1529.7)
        self.assertNotIn("gemm_lanes", profile["gpus"]["GPU-aaa"])
        # Cache-only means cache-only: nothing was written back.
        self.assertFalse(os.path.exists(self.cache.new_path))
        self.assertEqual([g["uuid"] for g in gpus], _UUIDS)


class _FakeProc:
    """A worker that ignores SIGTERM -- the case the teardown has to bound."""

    def __init__(self):
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return not self.killed

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        return None


class _FakeCtx:
    """An mp.spawn context whose workers never finish."""

    def __init__(self, nprocs):
        self.processes = [_FakeProc() for _ in range(nprocs)]

    def join(self, timeout=None):
        if timeout:
            time.sleep(min(timeout, 0.05))
        return False


class TestLinkPhaseTimeCap(CustomTestCase):
    """The network phase of the probe is bounded, and says why it gave up."""

    def _run_with_hung_link(self, timeout_s, reached=(0,)):
        results = {f"reached_rendezvous:{r}": True for r in reached}
        ctx = _FakeCtx(3)
        fake_mp = mock.MagicMock()
        fake_mp.Manager.return_value.dict.return_value = results
        fake_mp.spawn.return_value = ctx
        fake_torch = mock.MagicMock()
        fake_torch.cuda.device_count.return_value = 3
        # `import torch.multiprocessing as mp` resolves the submodule as an
        # ATTRIBUTE of the torch module object, so patching sys.modules alone
        # would hand the probe an auto-created mock instead of this one.
        fake_torch.multiprocessing = fake_mp
        with mock.patch.dict(
            "sys.modules",
            {"torch": fake_torch, "torch.multiprocessing": fake_mp},
        ):
            links, reason = uneven_perf.probe_link_matrix(_GPUS, timeout_s=timeout_s)
        return links, reason, ctx

    def test_a_hung_link_phase_returns_within_its_budget(self):
        t0 = time.time()
        links, reason, ctx = self._run_with_hung_link(0.5)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 10.0, "the link phase did not honour its cap")
        self.assertEqual(links, {})
        self.assertIn("timed out after", reason)
        # And the workers are gone: a probe that gave up must not leave CUDA
        # contexts on the cards the server is about to load.
        self.assertTrue(all(p.terminated for p in ctx.processes))

    def test_the_reason_names_the_ranks_that_never_arrived(self):
        _, reason, _ = self._run_with_hung_link(0.3, reached=(0, 2))
        self.assertIn("[0, 2]", reason)
        self.assertIn("never arrived: [1]", reason)
        self.assertIn("SGLANG_PERF_PROBE_LINK_TIMEOUT_S", reason)

    def test_a_timed_out_link_phase_still_caches_the_card_measurements(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "profile.json")
            with mock.patch.object(
                uneven_perf, "_nvml_gpu_inventory", return_value=(_GPUS, _DRIVER)
            ), mock.patch.object(
                uneven_perf,
                "_probe_one_gpu",
                side_effect=lambda g, groups: {"gemm_tflops": 1.0},
            ), mock.patch.object(
                uneven_perf,
                "probe_link_matrix",
                return_value=({}, "link matrix timed out after 45 s"),
            ):
                profile = uneven_perf.run_probe(out)
            with open(out) as f:
                written = json.load(f)
        self.assertEqual(profile["links"], {})
        self.assertEqual(written["gpus"]["GPU-aaa"]["gemm_tflops"], 1.0)
        self.assertIn("timed out", written[uneven_perf.PROFILE_NOTES_KEY]["links"])

    def test_skip_links_is_recorded_as_a_reason_not_a_silence(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "profile.json")
            with mock.patch.object(
                uneven_perf, "_nvml_gpu_inventory", return_value=(_GPUS, _DRIVER)
            ), mock.patch.object(
                uneven_perf, "_probe_one_gpu", side_effect=lambda g, groups: {}
            ), mock.patch.dict(
                os.environ, {"SGLANG_PERF_PROBE_SKIP_LINKS": "1"}
            ):
                profile = uneven_perf.run_probe(out)
        self.assertEqual(profile["links"], {})
        self.assertIn("skipped", profile[uneven_perf.PROFILE_NOTES_KEY]["links"])

    def test_the_topup_path_never_touches_the_link_matrix(self):
        import tempfile

        base = _v2_profile()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "profile.json")
            with mock.patch.object(
                uneven_perf, "_nvml_gpu_inventory", return_value=(_GPUS, _DRIVER)
            ), mock.patch.object(
                uneven_perf,
                "_probe_one_gpu",
                side_effect=lambda g, groups: {"gemm_lanes": {}, "gemm_lane_notes": {}},
            ), mock.patch.object(
                uneven_perf, "probe_link_matrix"
            ) as link:
                profile = uneven_perf.run_probe(
                    out, groups=[uneven_perf.PROBE_GROUP_LANES], base=base
                )
        link.assert_not_called()
        self.assertEqual(profile["links"], _LINKS)
        self.assertEqual(profile["topup_groups"], [uneven_perf.PROBE_GROUP_LANES])
        # The v2 rates the top-up did not measure are still there.
        self.assertEqual(profile["gpus"]["GPU-aaa"]["membw_gemv_gbs"], 1529.7)


class TestRendezvousIsNotInherited(CustomTestCase):
    """The root of the 600 s hang: the probe's own rendezvous endpoint.

    torch's ``env://`` rendezvous is steered by a handful of environment
    variables. The link workers used ``setdefault``, so an inherited
    ``MASTER_ADDR`` won: rank 0 bound its store on the wildcard address and
    waited for workers that were dialling somewhere else, and every rank sat
    in the ``TCPStore`` constructor for the full process-group timeout.
    """

    def test_every_steering_variable_is_removed_and_overwritten(self):
        env = {
            "MASTER_ADDR": "192.0.2.7",
            "MASTER_PORT": "29517",
            "WORLD_SIZE": "8",
            "RANK": "5",
            "LOCAL_RANK": "5",
            "TORCHELASTIC_USE_AGENT_STORE": "True",
            "TORCHELASTIC_RUN_ID": "job-42",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            uneven_perf._link_rendezvous_env(45123)
            self.assertEqual(os.environ["MASTER_ADDR"], "127.0.0.1")
            self.assertEqual(os.environ["MASTER_PORT"], "45123")
            for var in (
                "WORLD_SIZE",
                "RANK",
                "LOCAL_RANK",
                "TORCHELASTIC_USE_AGENT_STORE",
                "TORCHELASTIC_RUN_ID",
            ):
                self.assertNotIn(var, os.environ, var)

    def test_the_port_is_free_and_not_a_constant(self):
        a = uneven_perf._free_tcp_port()
        self.assertGreater(a, 1024)
        self.assertLess(a, 65536)

    def test_master_addr_and_port_are_in_the_cleared_set(self):
        """Both must be CLEARED before being set, not defaulted onto."""
        self.assertIn("MASTER_ADDR", uneven_perf._LINK_ENV_TO_CLEAR)
        self.assertIn("MASTER_PORT", uneven_perf._LINK_ENV_TO_CLEAR)
        self.assertIn("TORCHELASTIC_USE_AGENT_STORE", uneven_perf._LINK_ENV_TO_CLEAR)


class TestNoUnboundedWaitInTheProbePath(CustomTestCase):
    """Rank-local condition before the collective, deadline on every wait.

    The audit that keeps finding this bug family: a rank enters a group
    operation before proving its own local precondition, and the group wait
    has no deadline, so the failure surfaces as a hang with no rank named.
    """

    def test_the_probe_source_has_no_deadline_free_join(self):
        import inspect

        src = inspect.getsource(uneven_perf.probe_link_matrix)
        self.assertNotIn("join(timeout=None)", src)
        self.assertIn("deadline", src)
        # join() is only ever called with a timeout derived from the deadline.
        self.assertNotIn("ctx.join()", src)

    def test_the_worker_publishes_that_it_reached_the_rendezvous(self):
        import inspect

        src = inspect.getsource(uneven_perf._link_worker)
        # local proof first ...
        local = src.index("torch.zeros(1, device=dev)")
        marker = src.index("reached_rendezvous")
        collective = src.index("init_process_group")
        self.assertLess(local, marker)
        self.assertLess(marker, collective)
        # ... and the collective itself carries an explicit timeout.
        self.assertIn("timeout=timedelta(", src)

    def test_terminate_is_bounded_at_every_step(self):
        procs = [_FakeProc() for _ in range(3)]
        ctx = _FakeCtx(0)
        ctx.processes = procs
        t0 = time.time()
        uneven_perf._terminate_spawn_context(ctx, grace_s=0.2)
        self.assertLess(time.time() - t0, 5.0)
        self.assertTrue(all(p.terminated for p in procs))
        self.assertTrue(all(p.killed for p in procs))


if __name__ == "__main__":
    unittest.main()
