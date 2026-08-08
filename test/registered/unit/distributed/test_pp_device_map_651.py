"""#651 W1 + W2b: a mixed-device pipeline (CPU stage feeding a GPU stage).

Three layers, all runnable without an accelerator:

* ``--pp-device-map`` argument validation and the derived rank-uniform
  backend override (``TestPpDeviceMapValidation``,
  ``TestDerivedBackendOverride``);
* the backend-aware p2p route decision and the send-side host staging it
  drives (``TestP2PRoute``, ``TestStageTensorDictForWire``) -- the pure
  function is the seam, because a CUDA payload cannot be produced here;
* a real 2-process gloo world with the per-rank device override installed:
  groups form, the pipeline group's device is ``cpu`` on a CUDA build, no
  PyNccl communicator is constructed, and the p2p tensor-dict path completes
  in BOTH directions (``TestMixedDeviceWorld``).

Run: CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<repo>/python python -m pytest -q \
    test/registered/unit/distributed/test_pp_device_map_651.py
"""

import contextlib
import importlib.util
import multiprocessing as mp
import os
import socket
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_sgl_kernel_stub():
    if importlib.util.find_spec("sgl_kernel") is not None:
        return

    def _make(name, pkg=False):
        mod = types.ModuleType(name)
        if pkg:
            mod.__path__ = []

        def _getattr(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return lambda *a, **k: None

        mod.__getattr__ = _getattr
        sys.modules.setdefault(name, mod)

    _make("sgl_kernel", pkg=True)
    _make("sgl_kernel.quantization")
    _make("sgl_kernel.kvcacheio")


_install_sgl_kernel_stub()

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
)

import torch  # noqa: E402

import sglang.srt.distributed.parallel_state as ps  # noqa: E402
from sglang.srt.distributed.parallel_state import (  # noqa: E402
    _p2p_route,
    _p2p_wire_backend,
    _split_tensor_dict,
    _stage_tensor_dict_for_wire,
    should_build_pynccl,
    should_pre_warm_nccl,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.utils.common import hide_all_devices  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def make_args(**kwargs):
    """model_path='dummy' short-circuits __post_init__, so the handler under
    test is driven directly (same pattern as test_uneven_tp_memory.py)."""
    return ServerArgs(model_path="dummy", **kwargs)


class TestPpDeviceMapValidation(CustomTestCase):
    def _reject(self, **kwargs):
        args = make_args(**kwargs)
        with self.assertRaises(ValueError) as ctx:
            args._handle_pp_device_map()
        return str(ctx.exception)

    def test_default_path_is_untouched(self):
        args = make_args(tp_size=2)
        args._handle_pp_device_map()
        self.assertIsNone(args.pp_device_map)
        self.assertFalse(args.mixed_device_world())
        self.assertIsNone(args.distributed_backend_override())
        self.assertIsNone(args.device_for_pp_rank(0))

    def test_accepts_cpu_cuda_pipeline(self):
        args = make_args(pp_size=2, tp_size=1, pp_device_map=["cpu", "cuda"])
        args._handle_pp_device_map()
        self.assertTrue(args.mixed_device_world())
        self.assertEqual(args.device_for_pp_rank(0), "cpu")
        self.assertEqual(args.device_for_pp_rank(1), "cuda")

    def test_uniform_cuda_map_is_not_a_mixed_world(self):
        args = make_args(pp_size=2, tp_size=1, pp_device_map=["cuda", "cuda"])
        args._handle_pp_device_map()
        self.assertFalse(args.mixed_device_world())
        self.assertIsNone(args.distributed_backend_override())

    def test_rejects_without_pipeline(self):
        self.assertIn("--pp-size > 1", self._reject(pp_size=1, pp_device_map=["cpu"]))

    def test_rejects_length_mismatch(self):
        msg = self._reject(pp_size=2, tp_size=1, pp_device_map=["cpu", "cuda", "cuda"])
        self.assertIn("must equal --pp-size", msg)

    def test_rejects_unknown_device(self):
        msg = self._reject(pp_size=2, tp_size=1, pp_device_map=["cpu", "npu"])
        self.assertIn("unsupported device", msg)

    def test_rejects_all_cpu_map(self):
        msg = self._reject(pp_size=2, tp_size=1, pp_device_map=["cpu", "cpu"])
        self.assertIn("at least one stage on 'cuda'", msg)

    def test_rejects_tensor_parallelism(self):
        msg = self._reject(pp_size=2, tp_size=2, pp_device_map=["cpu", "cuda"])
        self.assertIn("--tp-size 1", msg)

    def test_rejects_data_parallelism(self):
        msg = self._reject(
            pp_size=2, tp_size=1, dp_size=2, pp_device_map=["cpu", "cuda"]
        )
        self.assertIn("--dp-size", msg)

    def test_rejects_expert_parallelism(self):
        msg = self._reject(
            pp_size=2, tp_size=1, ep_size=2, pp_device_map=["cpu", "cuda"]
        )
        self.assertIn("--ep-size", msg)

    def test_rejects_multi_node(self):
        msg = self._reject(
            pp_size=2, tp_size=1, nnodes=2, pp_device_map=["cpu", "cuda"]
        )
        self.assertIn("single-node only", msg)

    def test_rejects_rank_gpu_id_combination(self):
        msg = self._reject(
            pp_size=2,
            tp_size=1,
            pp_device_map=["cpu", "cuda"],
            rank_gpu_id=[0, 1],
            rank_gpu_memory_mib=16000,
        )
        self.assertIn("--rank-gpu-id", msg)


class TestDerivedBackendOverride(CustomTestCase):
    """The one place the mixed-device rule becomes a backend string.

    ``ModelRunner.init_torch_distributed`` consumes exactly this, so the
    rank-uniformity claim ("gloo on EVERY rank") is checkable here: the value
    is a function of the CLI alone, with no per-process input.
    """

    def test_mixed_world_forces_gloo(self):
        args = make_args(pp_size=2, tp_size=1, pp_device_map=["cpu", "cuda"])
        args._handle_pp_device_map()
        self.assertEqual(args.distributed_backend_override(), "gloo")

    def test_no_override_without_the_flag(self):
        self.assertIsNone(make_args(pp_size=2).distributed_backend_override())


class TestPynccGate(CustomTestCase):
    def test_mixed_world_blocks_pynccl_construction(self):
        # Rank-uniform by construction: the extra input is derived from the
        # same CLI on every rank.
        self.assertTrue(should_build_pynccl(True, 2, False))
        self.assertFalse(should_build_pynccl(True, 2, False, True))

    def test_default_reduction_unchanged(self):
        self.assertFalse(should_build_pynccl(True, 1, False))
        self.assertFalse(should_build_pynccl(False, 2, False))
        self.assertFalse(should_build_pynccl(True, 2, True))


class TestPreWarmNcclGate(CustomTestCase):
    def test_cpu_rank_never_warms_up(self):
        # A CPU pipeline stage has no torch.cuda.current_device() and no
        # device communicator; the parent-side guard cannot see this.
        self.assertFalse(should_pre_warm_nccl(True, "cpu", 1, 2, 1))
        self.assertFalse(should_pre_warm_nccl(True, "cpu", 4, 1, 1))

    def test_default_reduction_unchanged(self):
        self.assertTrue(should_pre_warm_nccl(True, "cuda", 2, 1, 1))
        self.assertTrue(should_pre_warm_nccl(True, "cuda", 1, 2, 1))
        self.assertTrue(should_pre_warm_nccl(True, "cuda", 1, 1, 2))
        self.assertFalse(should_pre_warm_nccl(True, "cuda", 1, 1, 1))
        self.assertFalse(should_pre_warm_nccl(False, "cuda", 2, 2, 2))


class TestHideAllDevices(CustomTestCase):
    def test_empties_and_restores_every_vendor_variable(self):
        names = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=False):
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            with hide_all_devices(7) as gpu_id:
                self.assertEqual(gpu_id, 7)
                for name in names:
                    self.assertEqual(os.environ[name], "")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
            self.assertNotIn("HIP_VISIBLE_DEVICES", os.environ)
            self.assertNotIn("ROCR_VISIBLE_DEVICES", os.environ)


class TestSpawnPathInstalls(CustomTestCase):
    """(b) of the slice: the two per-rank installs around ``proc.start()``.

    Parent side picks the child's device VISIBILITY (it cannot be narrowed
    once the process exists); child side installs the device STRING (the
    visibility alone does not move the platform probe).
    """

    def setUp(self):
        self.addCleanup(ps.set_local_device_override, None)
        self.addCleanup(ps.set_mixed_device_world, False)

    def _mapped_args(self):
        args = make_args(pp_size=2, tp_size=1, pp_device_map=["cpu", "cuda"])
        args._handle_pp_device_map()
        return args

    def test_cpu_stage_child_sees_no_accelerator(self):
        from sglang.srt.entrypoints.engine import _rank_device_visibility

        args = self._mapped_args()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            with _rank_device_visibility(args, 0, 0) as gpu_id:
                self.assertEqual(gpu_id, 0)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "")
                self.assertEqual(os.environ["ROCR_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1")

    def _delegation_target(self, args, pp_rank):
        """Which helper the branch hands the child to, and with what.

        Asserted by delegation rather than by the resulting environment: what
        ``maybe_reindex_device_id`` does to CUDA_VISIBLE_DEVICES depends on
        env that a real boot sets (SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS) and
        on a cuda-alike host -- neither is this test's subject, and pinning
        them here would test the pre-#651 helper, not the new branch.
        """
        import sglang.srt.entrypoints.engine as engine_module

        with patch.object(engine_module, "maybe_reindex_device_id") as reindex:
            reindex.return_value = contextlib.nullcontext(0)
            with engine_module._rank_device_visibility(args, pp_rank, 3, [4]):
                pass
        return reindex.call_args

    def test_cuda_stage_keeps_the_historical_reindex(self):
        self.assertEqual(self._delegation_target(self._mapped_args(), 1), ((3, [4]),))

    def test_default_path_takes_the_reindex_branch(self):
        self.assertEqual(self._delegation_target(make_args(tp_size=2), 0), ((3, [4]),))

    def test_cpu_stage_does_not_reach_the_reindex_helper(self):
        self.assertIsNone(self._delegation_target(self._mapped_args(), 0))

    def test_child_install_sets_device_and_world_flag(self):
        from sglang.srt.managers.scheduler import install_pp_stage_device

        args = self._mapped_args()
        self.assertEqual(install_pp_stage_device(args, 0), "cpu")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(ps.get_local_device_override(), "cpu")
        self.assertTrue(ps.in_mixed_device_world())

        args = self._mapped_args()
        self.assertEqual(install_pp_stage_device(args, 1), "cuda")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(ps.get_local_device_override(), "cuda")
        # Rank-uniform: the GPU rank installs the SAME world flag.
        self.assertTrue(ps.in_mixed_device_world())

    def test_child_install_is_a_no_op_by_default(self):
        from sglang.srt.managers.scheduler import install_pp_stage_device

        args = make_args(tp_size=2)
        args.device = "cuda"
        self.assertIsNone(install_pp_stage_device(args, 0))
        self.assertEqual(args.device, "cuda")
        self.assertIsNone(ps.get_local_device_override())
        self.assertFalse(ps.in_mixed_device_world())


class TestCoordinatorDevice(CustomTestCase):
    """The build-probe trap this slice exists to close.

    ``is_cuda_alike()`` answers for the WHEEL, not for the process: on a
    CUDA/ROCm build a card-less process still says True. It is forced True
    here because this box answers False once CUDA_VISIBLE_DEVICES points
    nowhere -- which is precisely why a full-world test cannot falsify this
    on the rig, only on the ROCm laptop.
    """

    def setUp(self):
        self.addCleanup(ps.set_local_device_override, None)

    def test_override_outranks_the_build_probe(self):
        with patch.object(ps, "is_cuda_alike", return_value=True):
            ps.set_local_device_override("cpu")
            self.assertEqual(ps._coordinator_device(0).type, "cpu")
            self.assertEqual(ps.get_local_device_override(), "cpu")

    def test_probe_still_decides_without_an_override(self):
        ps.set_local_device_override(None)
        with patch.object(ps, "is_cuda_alike", return_value=True):
            self.assertEqual(ps._coordinator_device(0), torch.device("cuda:0"))

    def test_cpu_platform_default_is_unchanged(self):
        ps.set_local_device_override(None)
        with patch.object(ps, "is_cuda_alike", return_value=False):
            self.assertEqual(ps._coordinator_device(0), torch.device("cpu"))


class TestP2PRoute(CustomTestCase):
    """The routing decision that replaced ``metadata_group if tensor.is_cpu``."""

    def test_nccl_world_is_unchanged(self):
        self.assertEqual(_p2p_route("cpu", "nccl"), "wire_cpu")
        self.assertEqual(_p2p_route("cuda", "nccl"), "device_group")

    def test_gloo_device_group_cannot_carry_a_device_tensor(self):
        # The W2b gap: pre-fix this returned the device group, and gloo
        # cannot send a CUDA tensor.
        self.assertEqual(_p2p_route("cuda", "gloo"), "wire_cpu")
        self.assertEqual(_p2p_route("cpu", "gloo"), "wire_cpu")

    def test_backend_string_case_and_variants(self):
        self.assertEqual(_p2p_route("cuda", "GLOO"), "wire_cpu")
        self.assertEqual(_p2p_route("cuda", "mpi"), "wire_cpu")
        self.assertEqual(_p2p_route("cuda", ""), "device_group")


def _fake_cpu(self):
    """Stand-in for ``Tensor.cpu()`` on a device torch cannot materialize
    here (``meta``). Returns a real host tensor of the same shape/dtype."""
    return torch.zeros(self.shape, dtype=self.dtype)


class TestStageTensorDictForWire(CustomTestCase):
    def test_nccl_world_returns_the_same_object(self):
        payload = {"hidden": torch.zeros(2, 3), "note": "x"}
        self.assertIs(_stage_tensor_dict_for_wire(payload, "nccl"), payload)

    def test_gloo_world_with_cpu_payload_returns_the_same_object(self):
        payload = {"hidden": torch.zeros(2, 3), "note": "x"}
        self.assertIs(_stage_tensor_dict_for_wire(payload, "gloo"), payload)

    def test_gloo_world_stages_a_device_tensor_to_host(self):
        payload = {
            "hidden": torch.zeros(2, 3, dtype=torch.float16, device="meta"),
            "note": "passthrough",
        }
        with patch.object(torch.Tensor, "cpu", _fake_cpu):
            staged = _stage_tensor_dict_for_wire(payload, "gloo")
        self.assertIsNot(staged, payload)
        self.assertEqual(staged["hidden"].device.type, "cpu")
        self.assertEqual(staged["note"], "passthrough")
        # The original dict is not mutated (the caller may still hold it).
        self.assertEqual(payload["hidden"].device.type, "meta")

    def test_staged_metadata_declares_cpu_so_the_receiver_matches(self):
        """The design claim of W2b: the wire buffer's declared device is the
        STAGED one, not the sender's original. Otherwise the receiver
        allocates a CUDA buffer and picks the device group -- reopening the
        mismatch. The receiver's own ``_move_received_tensor`` hop (W2) is
        what restores its device."""
        payload = {"hidden": torch.zeros(2, 3, dtype=torch.float16, device="meta")}
        with patch.object(torch.Tensor, "cpu", _fake_cpu):
            staged = _stage_tensor_dict_for_wire(payload, "gloo")
        metadata_list, _ = _split_tensor_dict(staged)
        self.assertEqual(metadata_list[0][1].device, "cpu")


class TestP2PWireBackend(CustomTestCase):
    """The backend lookup must tolerate the stub coordinators the p2p tests
    drive the codec through (no real device group), and must cache."""

    def test_caches_and_survives_a_missing_group(self):
        stub = SimpleNamespace()
        self.assertEqual(_p2p_wire_backend(stub), "")
        stub._cached_device_group_backend = "nccl"
        self.assertEqual(_p2p_wire_backend(stub), "nccl")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PAYLOAD_FORWARD = {
    "hidden_states": torch.arange(12, dtype=torch.float16).reshape(2, 6),
    "residual": torch.full((2, 6), 2.0, dtype=torch.float16),
    "stage": "forward",
}
PAYLOAD_BACKWARD = {
    "hidden_states": torch.arange(6, dtype=torch.float32).reshape(1, 6),
    "stage": "backward",
}


def _world_worker(rank, port, out_queue):
    try:
        _world_worker_body(rank, port, out_queue)
    except Exception:
        import traceback

        out_queue.put({"_worker_error": f"rank {rank}: {traceback.format_exc()}"})


def _describe(tensor_dict):
    return {
        k: (
            (v.device.type, tuple(v.shape), str(v.dtype))
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in tensor_dict.items()
    }


def _world_worker_body(rank, port, out_queue):
    import torch.distributed as dist

    from sglang.srt.distributed import parallel_state as ps

    # What `configure_scheduler_process` installs in the child, in the same
    # order: per-rank device string first, rank-uniform world flag second.
    ps.set_local_device_override("cpu")
    ps.set_mixed_device_world(True)

    ps.init_distributed_environment(
        backend="gloo",
        world_size=2,
        rank=rank,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        backend="gloo",
    )
    try:
        pp = ps.get_pp_group()
        report = {
            "world_backend": str(dist.get_backend(ps.get_world_group().device_group)),
            "pp_backend": str(dist.get_backend(pp.device_group)),
            "pp_device": pp.device.type,
            "world_device": ps.get_world_group().device.type,
            "pynccl_is_none": pp.pynccl_comm is None,
        }
        if rank == 0:
            pp.send_tensor_dict(dict(PAYLOAD_FORWARD))
            got = pp.recv_tensor_dict()
            report["backward"] = _describe(got)
            report["backward_values_ok"] = bool(
                torch.equal(got["hidden_states"], PAYLOAD_BACKWARD["hidden_states"])
            )
        else:
            got = pp.recv_tensor_dict()
            report["forward"] = _describe(got)
            report["forward_values_ok"] = bool(
                torch.equal(got["hidden_states"], PAYLOAD_FORWARD["hidden_states"])
                and torch.equal(got["residual"], PAYLOAD_FORWARD["residual"])
            )
            pp.send_tensor_dict(dict(PAYLOAD_BACKWARD))
        out_queue.put({f"rank{rank}": report})
    finally:
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()


class TestMixedDeviceWorld(CustomTestCase):
    """(c) of the slice: a pp_size=2/tp_size=1 world whose ranks run the
    mixed-device install path completes group setup and a p2p roundtrip.

    Both ranks are CPU processes here (that is what a CI box offers), so the
    payloads exercise the wire path; the CUDA-side ``.cpu()`` staging is
    covered by ``TestStageTensorDictForWire`` through the same
    ``_p2p_route``.
    """

    def test_groups_form_and_p2p_roundtrips(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        port = _free_port()
        procs = [
            ctx.Process(target=_world_worker, args=(r, port, queue)) for r in (0, 1)
        ]
        for p in procs:
            p.start()
        results = {}
        for _ in range(2):
            item = queue.get(timeout=180)
            if "_worker_error" in item:
                for p in procs:
                    p.kill()
                self.fail(item["_worker_error"])
            results.update(item)
        for p in procs:
            p.join(timeout=60)

        for rank in ("rank0", "rank1"):
            report = results[rank]
            # Rank-uniform gloo world -- the CPU stage could not join an
            # NCCL one, and a per-process derivation would deadlock init.
            self.assertEqual(report["world_backend"], "gloo", rank)
            self.assertEqual(report["pp_backend"], "gloo", rank)
            # RED without set_local_device_override: is_cuda_alike() is a
            # BUILD probe, so a card-less process on a CUDA build would
            # claim cuda:0 here and die allocating the group's marker.
            self.assertEqual(report["pp_device"], "cpu", rank)
            self.assertEqual(report["world_device"], "cpu", rank)
            self.assertTrue(report["pynccl_is_none"], rank)

        self.assertTrue(results["rank1"]["forward_values_ok"])
        self.assertEqual(results["rank1"]["forward"]["stage"], "forward")
        self.assertEqual(
            results["rank1"]["forward"]["hidden_states"],
            ("cpu", (2, 6), "torch.float16"),
        )
        self.assertTrue(results["rank0"]["backward_values_ok"])
        self.assertEqual(results["rank0"]["backward"]["stage"], "backward")


if __name__ == "__main__":
    unittest.main()
