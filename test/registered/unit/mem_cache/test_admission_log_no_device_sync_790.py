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
"""#790: an admission-path log line must never read a device tensor's value.

THE METAL FACT. A boot wedged with 120 ADMISSION-WEDGE markers. Two py-spy
dumps 25 minutes apart were byte-identical: PP0's MainThread sat inside
`logging.emit`; PP1/PP2 starved in `pp_chain_receiver.recv`, waiting on a
chain send PP0 never reached. The trigger was the first RADIX-CARRYING
request admitted (6443 tokens) reaching `HybridReqToTokenPool.alloc` via
`get_new_batch_prefill -> prepare_for_extend -> alloc_for_extend ->
alloc_req_slots`, landing on the `#767 carry-without-copy` instrument:

    logger.warning("... slot=%s ...", req.mamba_pool_idx)

`mamba_pool_idx` is a 1-element CUDA tensor (`Req.mamba_pool_idx`, shape
(1); see `schedule_batch.py`). `%s` formatting a tensor calls
`Tensor.__repr__`, which forces a device-to-host copy and a stream
synchronize -- INSIDE `logging.emit`, on the admission hot path, holding
whatever lock/position the scheduler thread held to get there. On that boot
the device was occupied by a spinning kernel, so the sync never returned.
Zero "#767 carry-without-copy" lines ever reached the log, because the
record died mid-format.

THE FIX (`sync_free_tensor_repr` in `memory_pool.py`, used at this call
site and at every other member of the same family found in the #790
sweep) never reads the tensor's VALUE. It prints shape/dtype/device and
`id()` -- all resident on the host already, none of it requires a sync --
and falls through to a plain `str()` for anything that is not a tensor
(covering the general case: this call site, and its siblings, sometimes
carry an int instead once a value has already been pulled off-device
upstream).

WHAT THIS IS NOT. It is not a claim that `.item()`/`.tolist()`/`.cpu()`
themselves are always wrong -- plenty of other call sites in this codebase
need the actual value and pay for the sync deliberately, in places where a
sync is going to happen soon regardless of the log line (see the #790
sweep's phase_flip_resident_carry.py finding, explicitly commented "one D2H
per cutover, not per round"). The defect this test guards is narrower and
sharper: a sync whose ONLY reason to exist is to print a value, sitting on
a path that must not block, invoked whether or not anything ever reads the
log output.

COLOUR CONVENTION. Every test below either exercises the FIXED call site
(must stay green: the tripwire it carries must never fire) or deliberately
reproduces the PRE-#790 pattern (a raw tensor reaching a `%s` slot) to
prove the tripwire is not a dead switch -- those are named `..._trips_...`
and are expected to raise `_TripwireError`, caught by `assertRaises`. A
green run of THIS FILE, on this CPU-only box, is not a claim that nothing
would have synced on real CUDA -- it is a claim that the call that WOULD
have synced never happens at all, which is the only thing #790's fix
changes and the only thing a hermetic test can prove.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

_LOGGER_NAME = "sglang.srt.mem_cache.memory_pool"


class _TripwireError(AssertionError):
    """Raised the instant anything reads a torch.Tensor's VALUE through one
    of the methods #790 showed can force a device sync: __repr__, __str__,
    .item(), .tolist(), .cpu(), __float__, __int__. On real CUDA every one
    of these is a D2H copy + stream synchronize; on this CPU-only box none
    of them would actually block, so the trap substitutes for the hardware
    -- it proves the CALL never happens, which is the only thing #790's fix
    changes (the sync itself is unavoidable IF the call happens; the fix is
    to not make the call)."""


class _TensorTripwire:
    """Installs/restores, around torch.Tensor's value-reading methods, for
    the lifetime of a `with` block. Records every trip in `.calls`."""

    _NAMES = ("__repr__", "__str__", "item", "tolist", "cpu", "__float__", "__int__")

    def __enter__(self):
        self.calls = []
        self._orig = {name: getattr(torch.Tensor, name) for name in self._NAMES}

        def _make(name):
            def _tripped(_self, *a, **kw):
                self.calls.append(name)
                raise _TripwireError(
                    f"torch.Tensor.{name}() was called -- on real CUDA this "
                    "is a D2H copy + stream synchronize, and #790 hit "
                    "exactly this shape inside logging.emit on the "
                    "admission hot path."
                )

            return _tripped

        for name in self._NAMES:
            setattr(torch.Tensor, name, _make(name))
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(torch.Tensor, name, fn)
        return False


def _carry_req(mamba_pool_idx):
    """A request in exactly the #767/#790 branch: it already owns a mamba
    slot (`mamba_pool_idx is not None`) and has no pending COW copy, which
    is the `carry-without-copy` case the instrument logs on every one of
    its first three occurrences (`n <= 3`)."""
    return SimpleNamespace(
        rid="tripwire-req",
        req_pool_idx=None,
        mamba_pool_idx=mamba_pool_idx,
        mamba_cow_src_index=None,
        mamba_needs_clear=False,
        mamba_ping_pong_track_buffer=None,
    )


def _drive_alloc(m, req):
    """Construct a bare HybridReqToTokenPool -- no CUDA, no real pool
    memory -- with just enough state for `.alloc()` to reach and run the
    real #767/#790 log line, then return once `.alloc()` returns.

    `ReqToTokenPool.alloc` (the parent-class half of the method) is
    replaced with a stub returning a one-element row list: it needs a live
    token-slot table this test has no reason to build, and nothing
    downstream of the branch under test reads its return value except by
    length.
    """
    pool = m.HybridReqToTokenPool.__new__(m.HybridReqToTokenPool)
    pool.enable_mamba_extra_buffer = False

    class _AcceptsAnyIndex:
        def __setitem__(self, key, value):
            pass

    pool.req_index_to_mamba_index_mapping = _AcceptsAnyIndex()

    with mock.patch.object(m.ReqToTokenPool, "alloc", lambda self, reqs: [0]):
        return pool.alloc([req])


class TheHelperNeverReadsTheTensorsValue(unittest.TestCase):
    """`sync_free_tensor_repr` in isolation: the general-case guard (tensor
    OR plain value) must not trip the wire for either branch, and the trap
    itself must be proven live (can-fail case) so a passing test here
    means something."""

    def setUp(self):
        from sglang.srt.mem_cache import memory_pool as m

        self.m = m

    def test_a_device_shaped_tensor_never_trips_the_wire(self):
        t = torch.tensor([42], dtype=torch.int64)
        with _TensorTripwire() as trap:
            result = self.m.sync_free_tensor_repr(t)
        self.assertEqual([], trap.calls)
        self.assertRegex(
            result,
            r"^<tensor shape=\(1,\) dtype=torch\.int64 device=cpu id=\d+>$",
            "the identity must be built from shape/dtype/device/id alone "
            "-- none of them require reading the tensor's VALUE.",
        )

    def test_a_plain_int_or_none_passes_through_untouched(self):
        with _TensorTripwire() as trap:
            self.assertEqual("3", self.m.sync_free_tensor_repr(3))
            self.assertEqual("None", self.m.sync_free_tensor_repr(None))
        self.assertEqual(
            [],
            trap.calls,
            "plain values must never even reach torch.Tensor's methods.",
        )

    def test_the_trap_itself_is_not_a_dead_switch(self):
        """Can-fail case: point the exact pre-#790 pattern -- a raw tensor
        as a %s log argument -- at the trap and confirm it fires. If this
        test cannot fail, none of the tests above prove anything."""
        t = torch.tensor([42], dtype=torch.int64)
        with self.assertRaises(_TripwireError):
            with _TensorTripwire():
                "%s" % (t,)


class TheAdmissionLogSiteIsSyncFree(unittest.TestCase):
    """Drives the REAL `HybridReqToTokenPool.alloc()` #767/#790 branch, not
    a re-implementation of its log line -- a copy of the line could stay
    green while the real one regresses."""

    def setUp(self):
        from sglang.srt.mem_cache import memory_pool as m

        self.m = m

    def test_alloc_logs_the_carry_without_copy_line_without_touching_the_tensor(
        self,
    ):
        req = _carry_req(torch.tensor([42], dtype=torch.int64))
        with _TensorTripwire() as trap:
            with self.assertLogs(_LOGGER_NAME, level="WARNING") as cm:
                select_index = _drive_alloc(self.m, req)
        self.assertEqual(
            [],
            trap.calls,
            "the admission log line touched the tensor's value -- this is "
            "the #790 wedge, reproduced.",
        )
        self.assertEqual([0], select_index)
        body = "\n".join(cm.output)
        self.assertIn("carry-without-copy", body)
        self.assertIn("slot=<tensor", body)

    def test_pre_fix_pattern_trips_the_wire_from_inside_the_real_branch(self):
        """Can-fail case at the integration level: patch the module's
        sync-free helper back to identity -- exactly what the call site
        passed before #790's fix, a bare tensor reaching the %s slot --
        and prove the SAME branch, exercised the SAME way, now trips the
        wire. The raise below is `logging.emit`'s stream synchronize,
        played back on a box with no GPU to actually hang on.
        """
        req = _carry_req(torch.tensor([42], dtype=torch.int64))
        with mock.patch.object(self.m, "sync_free_tensor_repr", lambda v: v):
            with self.assertRaises(_TripwireError):
                with _TensorTripwire():
                    with self.assertLogs(_LOGGER_NAME, level="WARNING"):
                        _drive_alloc(self.m, req)


if __name__ == "__main__":
    unittest.main()
