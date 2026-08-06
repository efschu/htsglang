"""The chunked GDN extend kernel must skip the -1 padding sentinel.

Ported from upstream sglang #33810 ("fix(gdn): skip the -1 padding sentinel in
the chunked extend kernel"). Upstream's own regression signal is an 8-GPU
``--dp=8 --enable-dp-attention`` model test, which cannot run at the desk, so
this file drives the SAME production kernel
(``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``) through Triton's
interpreter on CPU tensors. It is the real kernel, not a mirror of it.

The bug
-------
A batch padded up to a captured/synced shape stamps ``PAD_SLOT_ID`` (-1) over
the tail of ``cache_indices`` (``hybrid_linear_attn_backend.py:96-99``), and
``gdn_triton.py:183`` forwards that very tensor as ``initial_state_indices``.
The decode kernel guards on the sentinel (``fused_recurrent.py:908`` /
``:997``); the chunked extend path did not. ``index * stride_h`` with
``index == -1`` therefore addresses one whole slot pitch BELOW the state pool
base -- the padded lane both READ its initial state from and WROTE its final
state to out-of-bounds memory.

What is pinned here
-------------------
* **No out-of-bounds write.** The state pool is carved out of a larger buffer
  whose first slot is a poisoned guard, so slot ``-1`` lands inside the guard
  rather than in unrelated memory. The guard must come back bit-identical.
* **No out-of-bounds read.** Re-running with a different poison must not change
  the kernel's output; if the padded lane loads the guard, it does.
* **The valid lane is untouched.** The real sequence's per-chunk ``h`` and its
  in-place-updated state slot must be bit-identical to a run without the padded
  lane.
* **Control.** A VALID index must actually write its slot, otherwise "the guard
  is untouched" would also hold for a kernel that stopped writing state at all.

Falsifier: dropping either ``valid_state`` conjunct in ``chunk_delta_h.py``
turns the two out-of-bounds assertions red (verified both directions).

Why a subprocess
----------------
``@triton.jit`` reads ``TRITON_INTERPRET`` at DECORATION time, i.e. when the
module body first runs, and Python caches that module object process-wide. In a
shared pytest process the kernel module (or one of its ``@triton.jit`` helpers
in ``fla.op``) is usually already imported COMPILED by an earlier suite, and
reloading only part of that graph yields "Cannot call @triton.jit'd outside of
the scope of a kernel". Setting the env var in-process would equally poison
every later suite. A child interpreter is isolated in both directions and makes
this file's result independent of collection order.

CPU only, no GPU: the interpreter executes the index arithmetic the bug lives
in.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAD_SLOT_ID = -1
GUARD = 3.5

# Deliberately tiny: the interpreter is slow, and the defect is in the index
# arithmetic, not in the tile sizes.
_WORKER = textwrap.dedent("""
    import json, os, sys
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

    import torch
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE,
        chunk_gated_delta_rule_fwd_h,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64 as KERNEL,
    )

    H, K, V, NUM_SLOTS = 1, 64, 32, 2
    MAX_SEQS = 2
    GUARD = 3.5
    PAD = -1

    def inputs(num_seqs, seed=0):
        # Always drawn at the SAME size and then sliced: sampling directly at
        # num_seqs*CHUNK_SIZE would advance the generator differently per
        # tensor, so the 1-lane and 2-lane runs would not share lane 0's
        # inputs. Sampled on CPU (torch.randn is not arch-identical).
        g = torch.Generator().manual_seed(seed)
        full = MAX_SEQS * CHUNK_SIZE
        k = torch.randn(1, full, H, K, generator=g, dtype=torch.float32)
        w = torch.randn(1, full, H, K, generator=g, dtype=torch.float32)
        u = torch.randn(1, full, H, V, generator=g, dtype=torch.float32)
        t = num_seqs * CHUNK_SIZE
        cu = torch.arange(0, (num_seqs + 1) * CHUNK_SIZE, CHUNK_SIZE,
                          dtype=torch.int32)
        return (k[:, :t].contiguous(), w[:, :t].contiguous(),
                u[:, :t].contiguous(), cu)

    def pool(guard_value, seed=7):
        # backing[0] is the guard: `state` slot -1 IS backing[0].
        g = torch.Generator().manual_seed(seed)
        backing = torch.empty(NUM_SLOTS + 1, H, V, K, dtype=torch.float32)
        backing[0].fill_(guard_value)
        backing[1:] = torch.randn(NUM_SLOTS, H, V, K, generator=g,
                                  dtype=torch.float32)
        return backing, backing[1:]

    def run(indices, guard_value=GUARD):
        k, w, u, cu = inputs(len(indices))
        backing, state = pool(guard_value)
        idx = torch.tensor(indices, dtype=torch.int32)
        h, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, initial_state=state,
            initial_state_indices=idx, cu_seqlens=cu,
        )
        return h, backing

    out = {}
    out["interpreted"] = type(getattr(KERNEL, "fn", KERNEL)).__name__

    h_pad, backing_pad = run([0, PAD])
    guard = backing_pad[0]
    out["guard_max_dev"] = float((guard - GUARD).abs().max())

    h_alt, _ = run([0, PAD], guard_value=-GUARD)
    out["h_stable_under_guard_poison"] = bool(torch.equal(h_pad, h_alt))

    h_solo, backing_solo = run([0])
    nt = h_solo.shape[1]
    out["valid_lane_h_equal"] = bool(torch.equal(h_pad[:, :nt], h_solo))
    out["valid_lane_state_equal"] = bool(
        torch.equal(backing_pad[1], backing_solo[1])
    )

    _, backing_valid = run([0, 1])
    _, fresh = pool(GUARD)
    out["control_slot_written"] = not bool(
        torch.equal(backing_valid[1], fresh[1])
    )

    print("__RESULT__" + json.dumps(out))
    """)


def _probe():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "99"
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    marker = [
        line for line in proc.stdout.splitlines() if line.startswith("__RESULT__")
    ]
    if not marker:
        raise AssertionError(
            "interpreter probe produced no result\n"
            f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    return json.loads(marker[-1][len("__RESULT__") :])


class TestGdnChunkHPadSentinel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # One child run for the whole class: the interpreter is slow.
        cls.res = _probe()

    def test_the_kernel_really_runs_under_the_interpreter(self):
        """Without this, every other assertion here could be vacuous."""
        self.assertEqual(
            self.res["interpreted"],
            "InterpretedFunction",
            "the kernel was compiled, not interpreted: the assertions below "
            "would not be executing the kernel at all",
        )

    def test_padded_lane_does_not_write_below_the_pool_base(self):
        """slot -1 is one full stride_h BELOW `initial_state`."""
        self.assertEqual(
            self.res["guard_max_dev"],
            0.0,
            "the -1 padding lane wrote its final state one slot below the "
            "state pool base: the chunked extend kernel is missing the "
            "sentinel guard the decode kernel already has",
        )

    def test_padded_lane_does_not_read_below_the_pool_base(self):
        """The guard's CONTENT must not reach the padded lane's accumulator."""
        self.assertTrue(
            self.res["h_stable_under_guard_poison"],
            "changing the poison in the slot below the pool base changed the "
            "kernel output: the -1 padding lane is loading its initial state "
            "out of bounds",
        )

    def test_the_valid_lane_is_unaffected_by_a_padded_neighbour(self):
        self.assertTrue(
            self.res["valid_lane_h_equal"],
            "the real sequence's per-chunk states changed once a padded lane "
            "was appended to the batch",
        )
        self.assertTrue(
            self.res["valid_lane_state_equal"],
            "the real sequence's in-place state slot changed once a padded "
            "lane was appended to the batch",
        )

    def test_the_guard_is_actually_addressable(self):
        """Control: with a VALID index the kernel does write that slot."""
        self.assertTrue(
            self.res["control_slot_written"],
            "the kernel did not update its state slot in place, so 'the guard "
            "is untouched' proves nothing about the sentinel",
        )


if __name__ == "__main__":
    unittest.main()
