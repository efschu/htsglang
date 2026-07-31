# SPDX-License-Identifier: Apache-2.0
"""#355: the masked KV writer's bound check fires LOUDLY, and without it the
same write corrupts a live row in silence.

``masked_set_kv_buffer_kernel`` is the path every target-side DCP write takes.
Before #355 it had no index bound: an escaped compact row -- a #345-family
regression, a bad reshard, a scheduler bug -- wrote at ``loc * H * D`` into
whatever that address happened to be. Inside the allocation that is not a
crash, it is a wrong answer, and the wrong answer is attributable to nothing.

Each case runs in its OWN subprocess. A device assert poisons the CUDA context
for the whole process, so a passing in-process assertion would prove nothing
about a later case. Structure:

* ``test_out_of_range_loc_is_refused_by_name`` -- the guard fires and the
  message names the writer.
* ``test_without_the_guard_the_same_write_corrupts_silently`` -- the SAME
  launch with ``SGLANG_DISABLE_KV_MASKED_BOUND_CHECK=1`` returns exit 0 and a
  row that should not have been touched now holds the written value. This is
  the defect being closed; the test asserts the defect is real, so a future
  regression that quietly drops the guard fails here too.
* ``test_in_range_write_is_unchanged`` -- the guard costs the legal path
  nothing, byte-compared against the pre-#355 semantics.
* ``test_production_pool_path_is_guarded`` -- the check is reached through
  ``MHATokenToKVPool.set_kv_buffer(dcp_kv_mask=...)``, not only through a
  hand-rolled launch.

    python -m pytest test/registered/mem_cache/test_masked_kv_bound_falsifier.py -v
"""

import os
import subprocess
import sys
import textwrap
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu")

# Decode-shaped: one token on the Qwen3.6-27B row geometry (4 replicated KV
# heads x head_dim 256). Few rows on purpose -- the point is the index.
H, D = 4, 256
ROWS = 64  # rows the kernel is allowed to address
ALLOC_ROWS = 256  # rows actually allocated behind it
OOB_LOC = 100  # inside the allocation, outside the bound: the silent case

_PRELUDE = textwrap.dedent(
    f"""
    import torch
    from sglang.srt.mem_cache.memory_pool import (
        masked_set_kv_buffer_kernel, kv_bound_check_enabled,
    )

    H, D, ROWS, ALLOC_ROWS = {H}, {D}, {ROWS}, {ALLOC_ROWS}
    dev = torch.device("cuda")
    kbuf = torch.zeros(ALLOC_ROWS, H, D, dtype=torch.bfloat16, device=dev)
    vbuf = torch.zeros(ALLOC_ROWS, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.full((1, H, D), 7.0, dtype=torch.bfloat16, device=dev)
    v = torch.full((1, H, D), 9.0, dtype=torch.bfloat16, device=dev)
    mask = torch.ones(1, dtype=torch.bool, device=dev)

    def write(loc_value, bound=ROWS):
        loc = torch.full((1,), loc_value, dtype=torch.int64, device=dev)
        masked_set_kv_buffer_kernel[(1,)](
            k, v, kbuf, vbuf, loc, mask, bound,
            1, H, D, 128,
            k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            debug=kv_bound_check_enabled(),
        )
        torch.cuda.synchronize()
    """
)


def _run(body: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestMaskedKvBoundFalsifier(CustomTestCase):
    def test_in_range_write_is_unchanged(self):
        r = _run(
            """
            write(5)
            assert float(kbuf[5, 0, 0]) == 7.0, kbuf[5, 0, 0]
            assert float(vbuf[5, 0, 0]) == 9.0, vbuf[5, 0, 0]
            # Nothing else moved.
            touched = int((kbuf.reshape(ALLOC_ROWS, -1) != 0).any(dim=1).sum())
            assert touched == 1, touched
            print("IN-RANGE OK")
            """
        )
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        self.assertIn("IN-RANGE OK", r.stdout)

    def test_out_of_range_loc_is_refused_by_name(self):
        r = _run(
            f"""
            write({OOB_LOC})
            print("NO ASSERT -- the guard did not fire")
            """
        )
        self.assertNotEqual(r.returncode, 0, f"guard did not fire: {r.stdout}")
        combined = r.stdout + r.stderr
        self.assertIn(
            "masked_set_kv_buffer: loc out of range",
            combined,
            f"the failure does not name the writer:\n{combined[-4000:]}",
        )

    def test_without_the_guard_the_same_write_corrupts_silently(self):
        """THE defect #355 closes. Off-switch on: exit 0, wrong bytes."""
        r = _run(
            f"""
            assert not kv_bound_check_enabled()
            write({OOB_LOC})
            corrupted = float(kbuf[{OOB_LOC}, 0, 0])
            print("SILENT", corrupted)
            """,
            env_extra={"SGLANG_DISABLE_KV_MASKED_BOUND_CHECK": "1"},
        )
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        self.assertIn(
            "SILENT 7.0",
            r.stdout,
            f"expected a silently corrupted row, got:\n{r.stdout}\n{r.stderr[-2000:]}",
        )

    def test_production_pool_path_is_guarded(self):
        """Through MHATokenToKVPool.set_kv_buffer(dcp_kv_mask=...), i.e. the
        launch the DCP decode step actually performs."""
        r = _run(
            """
            from types import SimpleNamespace
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            pool = MHATokenToKVPool(
                size=ROWS - 1, page_size=1, dtype=torch.bfloat16,
                head_num=H, head_dim=D, layer_num=1, device="cuda",
                enable_memory_saver=False,
            )
            layer = SimpleNamespace(layer_id=0)
            loc = torch.full((1,), ROWS + 32, dtype=torch.int64, device=dev)
            pool.set_kv_buffer(layer, loc, k, v, dcp_kv_mask=mask)
            torch.cuda.synchronize()
            print("NO ASSERT -- the pool path is unguarded")
            """
        )
        self.assertNotEqual(r.returncode, 0, f"pool path unguarded: {r.stdout}")
        combined = r.stdout + r.stderr
        self.assertIn("masked_set_kv_buffer: loc out of range", combined)


if __name__ == "__main__":
    unittest.main()
