"""#651 W3: the CPU stage of a mixed-device pipeline must not capture graphs.

``CPUGraphRunner.__init__`` asserts ``pp_size == 1``. That assertion is correct
-- its capture machinery really is single-stage -- so the fix is NOT to lift it
but to stop reaching it: ``init_decode_cuda_graph`` returns early on a CPU rank
whenever ``pp_size > 1``, leaving that stage eager while the GPU stage keeps
capturing its decode graphs.

The test drives the real ``ModelRunner.init_decode_cuda_graph`` as an unbound
function over a light stub, because constructing a ModelRunner needs a loaded
model and an accelerator. What matters here is the branch structure, and the
branch structure is exactly what the stub exercises.

RED-FIRST. ``test_single_stage_cpu_still_captures`` is the can-fail half: with
``pp_size == 1`` the function must NOT take the new early return, and must walk
on into capture. If someone widens the new condition to "any CPU rank", that
test fails. Without it, the guard could be trivially satisfied by returning
early always -- which would silently disable CPU graph capture for every
single-stage CPU deployment.

Run: PYTHONPATH=<repo>/python python -m pytest -q \
    test/registered/unit/distributed/test_cpu_stage_eager_651.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _Sentinel(Exception):
    """Raised by the stub to mark 'execution continued past the early return'."""


def _make_runner_stub(device: str, pp_size: int):
    """Minimal stand-in carrying only what init_decode_cuda_graph touches."""
    return SimpleNamespace(
        decode_cuda_graph_runner="untouched",
        graph_mem_usage=-1,
        is_generation=True,
        device=device,
        pp_size=pp_size,
        gpu_id=0,
        is_draft_worker=False,
        server_args=SimpleNamespace(model_impl="auto"),
    )


def _call(runner):
    from sglang.srt.model_executor.model_runner import ModelRunner

    # Past the CPU early-returns the function immediately asks the clock and
    # then the device for free memory. Raising there is how the test observes
    # "did not return early" without needing a real accelerator.
    with patch(
        "sglang.srt.model_executor.model_runner.get_available_gpu_memory",
        side_effect=_Sentinel("reached capture path"),
    ), patch(
        "sglang.srt.model_executor.model_runner.get_flags",
        return_value=SimpleNamespace(
            capture=SimpleNamespace(enable_torch_compile=True)
        ),
    ):
        return ModelRunner.init_decode_cuda_graph(runner)


class TestCpuStageEager(unittest.TestCase):
    def test_cpu_stage_of_a_pipeline_skips_capture(self):
        """pp_size > 1 on a CPU rank: return early, leave the runner unset."""
        runner = _make_runner_stub(device="cpu", pp_size=2)
        result = _call(runner)
        self.assertIsNone(result)
        self.assertIsNone(
            runner.decode_cuda_graph_runner,
            "the CPU stage must end with no graph runner, i.e. eager",
        )

    def test_single_stage_cpu_still_captures(self):
        """pp_size == 1 on CPU: unchanged behaviour, capture is attempted.

        This is the can-fail proof for the guard above. If the new condition
        were widened to every CPU rank, no _Sentinel would be raised and this
        test would fail.
        """
        runner = _make_runner_stub(device="cpu", pp_size=1)
        with self.assertRaises(_Sentinel):
            _call(runner)

    def test_gpu_stage_of_a_pipeline_still_captures(self):
        """The GPU stage of the SAME pipeline must keep capturing.

        Decode after the #651 phase flip runs GPU-only, so the whole feature
        would be pointless if pp_size > 1 disabled capture on the GPU side too.
        """
        runner = _make_runner_stub(device="cuda", pp_size=2)
        with patch(
            "sglang.srt.model_executor.model_runner.check_cuda_graph_backend",
            return_value=False,
        ):
            with self.assertRaises(_Sentinel):
                _call(runner)


if __name__ == "__main__":
    unittest.main()
