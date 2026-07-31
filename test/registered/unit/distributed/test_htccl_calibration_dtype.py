"""HTCCL pipe-chunk calibration must not measure a dtype the rank emulates.

The sweep is a real all_reduce over a synthetic payload, and the chunk size it
picks is used for every subsequent transfer. The payload was bfloat16. On the
cards this transport exists for -- a Turing (sm75) rank and a gfx900 rank --
bfloat16 has no hardware path: the kernel's `to_f` / `from_f` converters are
emulated, while the traffic those ranks really carry is float16, converted by a
single instruction. So the calibration paid a per-element cost the production
path never pays and picked a chunk size for arithmetic that never runs.

float16 is the honest probe everywhere: it is the same 2 bytes per element, so
the bandwidth being measured is identical, and it is natively converted on
every card this transport runs on, including the ones with native bfloat16.

CPU only: torch.distributed, the device and the transport are all stubbed.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import sglang
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_COMM_DIR = (
    Path(sglang.__file__).parent / "srt" / "distributed" / "device_communicators"
)


def _load_standalone(name):
    """Import the module from its file, bypassing the package __init__
    (which initializes CUDA)."""
    spec = importlib.util.spec_from_file_location(
        f"_htccl_calib_{name}", _COMM_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestCalibrationPayloadDtype(unittest.TestCase):
    def _run_calibration(self):
        """Drive _resolve_pipe_chunk with everything collective stubbed and
        report the dtypes the payload was allocated with."""
        module = _load_standalone("htccl_device")
        seen = []

        real_randn = torch.randn

        def fake_randn(*args, **kwargs):
            seen.append(kwargs.get("dtype"))
            # Pin to CPU explicitly: a sibling test may leave a CUDA default
            # device installed, and this test must not need a GPU.
            kwargs["device"] = "cpu"
            return real_randn(*args, **kwargs)

        import torch.distributed as dist

        fake_self = types.SimpleNamespace(
            device="cpu",
            rank=0,
            world_size=1,
            cpu_group=None,
            _pipe_chunk_bytes=0,
            all_reduce=lambda t: None,
        )

        patches = [
            mock.patch.dict("os.environ", {}, clear=False),
            mock.patch.object(module, "_tune_get", lambda key: None),
            mock.patch.object(module, "_tune_report", lambda key, value: None),
            mock.patch.object(module.torch, "randn", fake_randn),
            mock.patch.object(
                module.torch.cuda, "synchronize", lambda device=None: None
            ),
            # `async_op` is part of the real signature, and the sweep's
            # barriers are bounded (task #312) so they pass it. A stub
            # narrower than the API it stands in for fails on the call
            # rather than on the behaviour under test. Returning None is
            # what a completed collective looks like to `bounded_barrier`.
            mock.patch.object(
                dist, "barrier", lambda group=None, async_op=False: None
            ),
            mock.patch.object(
                dist,
                "all_gather_object",
                lambda out, obj, group=None: out.__setitem__(0, obj),
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        import os

        os.environ.pop("SGLANG_HTCCL_PIPE_CHUNK_MIB", None)

        module.HTCCLDeviceTransport._resolve_pipe_chunk(fake_self)
        return seen, fake_self

    def test_payload_is_float16(self):
        seen, _ = self._run_calibration()
        self.assertTrue(seen, "the calibration never allocated a payload")
        for dtype in seen:
            self.assertNotEqual(
                dtype,
                torch.bfloat16,
                "bfloat16 is emulated on every card this transport targets",
            )
            self.assertEqual(dtype, torch.float16)

    def test_a_chunk_size_is_still_chosen(self):
        """The behaviour around the dtype must not move."""
        _, fake_self = self._run_calibration()
        self.assertIn(
            fake_self._pipe_chunk_bytes,
            [mib * 1024 * 1024 for mib in (1, 2, 4, 8)],
        )


if __name__ == "__main__":
    unittest.main()
