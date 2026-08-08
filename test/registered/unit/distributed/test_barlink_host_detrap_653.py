"""#653: a tripped barlink HOST spin kernel must not destroy the CUDA context.

The device transport was cured of this on 2026-08-05 (#583): its three spin
kernels called ``__trap()`` on deadline expiry, a device trap destroys the
CUDA context, and from that moment every later CUDA call in the process
returns a sticky ``cudaErrorLaunchFailure`` ("unspecified launch failure") at
whatever unrelated call site happens to be next. In the production boot that
opened #583 the crash surfaced inside Triton's ``load_binary`` and named a
kernel that was only the victim.

``barlink_host.py`` kept the same mechanism -- a ``BARLINK_HOST_TRAP()`` macro
called from ``wait_ge`` -- for three more days. This file pins the port of the
device transport's fix: the spin writes an abort code into ``seq_dev[1]``,
returns, and :meth:`BarlinkHostTransport.check_aborted` raises a structured
``HostCollectiveAborted`` on the host instead.

The thread scoping is the part that is NOT a copy of the device fix. The
device transport's spin kernels are single-threaded; the host transport's
three spin sites live inside kernels whose whole block must pass a
``__syncthreads()`` afterwards, so thread 0 cannot return alone -- it publishes
the outcome in ``__shared__ int abortS`` and every thread leaves together.
That is the bar1 mesh kernel's K_1BLK pattern, and it is what the source
invariants below check.

Layout of the file:

* ``TestTheOldSourceReallyHadTheTrap`` -- the RED RECORD. The pre-fix text is
  read out of git and asserted to contain exactly what the rest of this file
  forbids. Without it, every assertion below could be vacuously true.
* ``TestNoTrapSurvives`` / ``TestTheSharedAbortPattern`` -- the fixed source.
* ``TestCheckAbortedRaises`` -- the host side, on a ``__new__`` stand-in.
* ``TestTheKernelsStillCompile`` -- nvcc, source only, no card.
* ``TestOnCardAbortLeavesTheContextAlive`` -- the on-card falsifier. SKIPS
  without CUDA, which is the expected desk result.
* ``TestHostTransportCannotAppearMidRun`` -- the reachability claim the whole
  fix rests on.
"""

import os
import re
import subprocess
import sysconfig
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.distributed.device_communicators.barlink_host import (
    _ABORT_WAITS,
    BarlinkHostTransport,
    HostCollectiveAborted,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


_MODULE = "python/sglang/srt/distributed/device_communicators/barlink_host.py"

#: The three spin sites, and the abort code each one must write. The codes are
#: distinct because the three point at different culprits -- a reuse-guard
#: expiry means the PEERS are behind, a publish wait means a peer never got
#: here at all.
_SITES = {
    "barlink_host_put_kernel": 1,
    "barlink_host_reduce_kernel": 2,
    "barlink_host_copyout_kernel": 3,
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MODULE).is_file():
            return parent
    raise AssertionError(f"could not locate {_MODULE} above {here}")


def _cuda_src(text: str) -> str:
    match = re.search(r'_CUDA_SRC = r"""(.*?)"""', text, re.S)
    assert match is not None, "the inline CUDA source moved"
    return match.group(1)


def _strip_line_comments(text: str) -> str:
    """Drop ``//`` comments.

    The bans below are about generated INSTRUCTIONS, not prose: the fixed
    kernels name ``__trap()`` repeatedly, precisely to explain why it must not
    be called, and matching those would make the pin unmaintainable.
    """
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def _fixed_source() -> str:
    return (_repo_root() / _MODULE).read_text()


def _head_source() -> str:
    """The module as it stands at HEAD -- the pre-fix text, for the red record.

    Read through ``git show`` rather than kept as a copied excerpt: an excerpt
    is a claim about the old file, this is the old file.
    """
    out = subprocess.run(
        ["git", "-C", str(_repo_root()), "show", f"HEAD:{_MODULE}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def _kernel_bodies(src: str) -> dict:
    """The three spin kernels, by name, comments stripped."""
    bodies = {}
    for name in _SITES:
        start = src.index(f"__global__ void {name}(")
        rest = src[start + 1 :]
        stops = [
            rest.index(marker)
            for marker in ("\n__global__ ", "\ntemplate ", "\n// ---")
            if marker in rest
        ]
        bodies[name] = _strip_line_comments(rest[: min(stops)] if stops else rest)
    return bodies


# ---------------------------------------------------------------------------
# (a) THE RED RECORD
# ---------------------------------------------------------------------------


class TestTheOldSourceReallyHadTheTrap(CustomTestCase):
    """The discriminating power of every assertion below, stated once.

    A source-invariant test that has never been shown to fail against the code
    it was written for is a hope, not a pin. So the pre-fix module is read out
    of git here and asserted to contain the very things the fixed one is
    forbidden to contain. If this class ever goes green-by-absence -- because
    HEAD moved past the fix -- it says so instead of quietly passing.
    """

    def setUp(self):
        self.head = _head_source()
        if "BARLINK_HOST_TRAP" not in self.head:
            self.skipTest(
                "HEAD no longer contains the pre-#653 source (the fix was "
                "committed); the red record has nothing left to record."
            )

    def test_the_old_wait_ge_called_the_trap_macro(self):
        body = _cuda_src(self.head)
        start = body.index("__device__ __forceinline__ void wait_ge(")
        old_wait_ge = body[start : body.index("\n}", start)]
        self.assertIn(
            "BARLINK_HOST_TRAP()",
            old_wait_ge,
            "the red record is empty: the pre-fix wait_ge did not trap after "
            "all, so the fix below pins nothing",
        )

    def test_the_old_wait_ge_returned_void(self):
        """No return value means no call site could have consumed one."""
        self.assertIn(
            "__device__ __forceinline__ void wait_ge(", _cuda_src(self.head)
        )
        self.assertNotIn(
            "__device__ __forceinline__ bool wait_ge(", _cuda_src(self.head)
        )

    def test_the_old_macro_expanded_to_a_device_trap(self):
        self.assertRegex(
            self.head, r"#define BARLINK_HOST_TRAP\(\)\s+__trap\(\)"
        )

    def test_the_old_seq_tensor_had_no_room_for_an_abort_word(self):
        """One element: ``seq_dev[1]`` would have been an out-of-bounds store,
        which is why the fix had to grow the tensor and not only the kernel."""
        self.assertIn(
            "self._seq = torch.ones(1, dtype=torch.int64, device=device)",
            self.head,
        )


# ---------------------------------------------------------------------------
# (a) THE FIXED SOURCE
# ---------------------------------------------------------------------------


class TestNoTrapSurvives(CustomTestCase):
    """Zero device traps outside historical-reference prose."""

    def test_the_macro_is_gone(self):
        text = _fixed_source()
        self.assertNotIn(
            "#define BARLINK_HOST_TRAP",
            text,
            "the macro must be deleted, not merely unused -- an unused trap "
            "macro is an invitation to call it again",
        )

    def test_no_kernel_invokes_a_trap_of_any_spelling(self):
        src = _strip_line_comments(_cuda_src(_fixed_source()))
        for token in ("BARLINK_HOST_TRAP()", "__trap()", "assert(", "abort()"):
            self.assertNotIn(
                token,
                src,
                f"{token} destroys or aborts the CUDA context; #653 replaced "
                f"it with an abort word the host reads",
            )

    def test_every_spin_site_still_has_its_deadline(self):
        """Removing the trap must not have removed the bound with it."""
        src = _strip_line_comments(_cuda_src(_fixed_source()))
        self.assertIn("(u64)(clock64() - start) > timeout", src)


class TestTheSharedAbortPattern(CustomTestCase):
    """wait_ge reports, and the whole BLOCK acts on the report."""

    def setUp(self):
        self.src = _cuda_src(_fixed_source())
        self.bodies = _kernel_bodies(self.src)

    def test_wait_ge_returns_bool_and_takes_the_word_and_a_code(self):
        signature = re.search(
            r"__device__ __forceinline__ bool wait_ge\((.*?)\)\s*\{",
            self.src,
            re.S,
        )
        self.assertIsNotNone(signature, "wait_ge must return bool")
        args = " ".join(signature.group(1).split())
        self.assertIn("u64* seq_dev", args)
        self.assertIn("u64 code", args)

    def test_wait_ge_writes_the_code_and_returns_true_on_expiry(self):
        body = _strip_line_comments(
            self.src[
                self.src.index("bool wait_ge(") : self.src.index(
                    "// \"Am I the block that finished last?\""
                )
            ]
        )
        self.assertRegex(
            body,
            r"seq_dev\[1\] = code; return true;",
            "on expiry the word is written and the spin RETURNS -- falling "
            "through would keep spinning past the deadline",
        )
        self.assertIn("return false;", body, "the arrival case must report false")

    def test_every_call_site_consumes_the_return_value(self):
        for name, body in self.bodies.items():
            with self.subTest(kernel=name):
                for call in re.findall(r"[^\n]*wait_ge\(", body):
                    self.assertIn(
                        "if (wait_ge(",
                        call + "wait_ge(",
                        f"{name} calls wait_ge without testing it; a dropped "
                        f"return value is a spin that gave up and then ran "
                        f"the payload anyway",
                    )

    def test_every_site_writes_its_own_distinct_code(self):
        seen = {}
        for name, body in self.bodies.items():
            codes = set(re.findall(r"seq_dev,\s*(\d+)ull\)", body))
            self.assertEqual(
                len(codes),
                1,
                f"{name} must write exactly one abort code, found {codes}",
            )
            seen[name] = int(codes.pop())
        self.assertEqual(
            seen,
            _SITES,
            "the three sites must be distinguishable, and _ABORT_WAITS must "
            "describe exactly the codes the kernels write",
        )
        self.assertEqual(set(seen.values()), set(_ABORT_WAITS))

    def test_every_site_uses_a_per_block_shared_abort_flag(self):
        for name, body in self.bodies.items():
            with self.subTest(kernel=name):
                self.assertIn("__shared__ int abortS;", body)
                self.assertIn("abortS = 0;", body, "initialize before use")
                self.assertIn("abortS = 1;", body)

    def test_thread_zero_never_returns_alone(self):
        """The scoping rule.

        Only thread 0 spins, so only thread 0 learns of the expiry -- but the
        block's remaining threads are about to hit a ``__syncthreads()`` (the
        one right after the spin region, and ``last_block``'s own). A bare
        ``return`` inside the ``threadIdx.x == 0`` region would strand them
        there forever, turning a reported abort into the hang the deadline
        exists to prevent. Hence: publish, barrier, everyone leaves.
        """
        for name, body in self.bodies.items():
            with self.subTest(kernel=name):
                region = body[
                    body.index("if (threadIdx.x == 0)") : body.index(
                        "if (abortS) return;"
                    )
                ]
                self.assertNotIn(
                    "return;",
                    region,
                    f"{name} returns from inside the thread-0 spin region",
                )
                self.assertIn("__syncthreads();", region)

    def test_the_block_leaves_before_any_payload_work(self):
        """``if (abortS) return;`` must precede the copy/reduce loops and
        ``last_block`` -- an aborted block that reached either would be
        writing into a slot whose ownership it just failed to establish, or
        waiting at a barrier its siblings already left."""
        for name, body in self.bodies.items():
            with self.subTest(kernel=name):
                leave = body.index("if (abortS) return;")
                self.assertLess(leave, body.index("last_block("))
                for marker in ("for (size_t i = ", "__threadfence_system();"):
                    if marker in body:
                        self.assertLess(
                            leave,
                            body.index(marker),
                            f"{name} does payload work before honouring abortS",
                        )

    def test_the_terminal_abort_contract_is_written_down(self):
        """Cross-block skew and a stale ``blk_ctr`` are accepted residue, and
        the reason has to be in the file -- otherwise the next reader adds the
        cross-block coordination this deliberately does not have."""
        self.assertIn("TERMINAL-ABORT CONTRACT", self.src)

    def test_the_kernel_comments_carry_the_issue_numbers(self):
        self.assertIn("#583", self.src)
        self.assertIn("#653", self.src)
        self.assertIn("unspecified launch failure", self.src)


class TestTheSeqTensorHasRoomForTheWord(CustomTestCase):
    """Every size assumption on the counter tensor, in one place.

    ``seq_dev[1]`` is a store past the end of a one-element tensor, and the
    host transport has THREE counter families (collective, per-destination
    send, per-source recv), all of which reach the same kernels.
    """

    def test_one_backing_tensor_two_words_per_counter(self):
        text = _fixed_source()
        self.assertIn("1 + 2 * _MAX_RANKS, 2, dtype=torch.int64", text)
        self.assertNotIn("self._seq = torch.ones(1,", text)

    def test_the_p2p_counters_are_handed_over_as_rows_not_slices(self):
        """``_send_seq[dst : dst + 1]`` was correct for a flat one-word
        counter and is a corruption for a two-word one: the kernel's abort
        store would land on the NEXT destination's sequence number."""
        text = _fixed_source()
        self.assertIn("seq = self._send_seq[dst]", text)
        self.assertIn("seq = self._recv_seq[src]", text)
        self.assertNotIn("self._send_seq[dst : dst + 1]", text)
        self.assertNotIn("self._recv_seq[src : src + 1]", text)

    def test_the_rows_the_kernels_get_are_contiguous_pairs(self):
        """Behavioural, not textual: the extension takes ``data_ptr()`` and
        indexes [0] and [1], so a row must be two contiguous elements."""
        backing = torch.zeros(1 + 2 * 8, 2, dtype=torch.int64)
        for row in (0, 1, 8, 16):
            with self.subTest(row=row):
                view = backing[row]
                self.assertEqual(view.numel(), 2)
                self.assertTrue(view.is_contiguous())
                self.assertEqual(
                    view.data_ptr(), backing.data_ptr() + row * 2 * 8
                )

    def test_the_sequence_column_still_starts_at_one(self):
        """The put kernel's reuse guard is skipped while ``seq <= lag``; that
        is a statement about column 0 only, and the abort column must start
        clean."""
        backing = torch.zeros(1 + 2 * 8, 2, dtype=torch.int64)
        backing[:, 0] = 1
        self.assertTrue(bool((backing[:, 0] == 1).all()))
        self.assertEqual(int(backing[:, 1].max()), 0)


# ---------------------------------------------------------------------------
# (b) THE HOST SIDE
# ---------------------------------------------------------------------------


def _transport(*, code: int = 0, row: int = 0, rank: int = 1, world: int = 3):
    """A host transport carrying the real guard, without a card or a segment.

    ``__new__`` on purpose: ``check_aborted`` and ``_raise_aborted`` are the
    code under test and must not be re-implemented here, but constructing the
    transport would create a shm segment, page-lock it and build the
    extension. Only what the guard READS is set -- the counter tensor, the
    rank and the world -- so a field this test forgot shows up as an
    AttributeError rather than as a silent pass.
    """
    t = BarlinkHostTransport.__new__(BarlinkHostTransport)
    t._seq_all = torch.zeros(1 + 2 * 8, 2, dtype=torch.int64)
    t._seq_all[:, 0] = 1
    t._seq_all[row, 1] = code
    t.rank = rank
    t.world_size = world
    return t


class TestCheckAbortedRaises(CustomTestCase):
    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_a_clean_word_does_not_raise(self):
        _transport().check_aborted("unit")

    def test_a_transport_without_counters_does_not_raise(self):
        """The ``__new__`` stand-in of some other test, and every path that
        can reach the guard before bring-up finished."""
        t = BarlinkHostTransport.__new__(BarlinkHostTransport)
        t.check_aborted("unit")

    def test_a_tripped_word_raises_with_structured_context(self):
        for code, description in _ABORT_WAITS.items():
            with self.subTest(code=code):
                t = _transport(code=code)
                with self.assertRaises(HostCollectiveAborted) as caught:
                    t.check_aborted("unit-where")
                err = caught.exception
                self.assertEqual(err.rank, 1)
                self.assertEqual(err.world, 3)
                self.assertEqual(err.code, code)
                self.assertEqual(err.where, "unit-where")
                message = str(err)
                # The decoded phase, or the reader is back to guessing which
                # of the three waits expired -- the whole point of the codes.
                self.assertIn(description.split(" (")[0], message)
                self.assertIn("unit-where", message)
                self.assertIn("rank 1/3", message)
                # The garbage-results contract, so nobody reads this as a
                # warning that can be caught and continued past.
                self.assertIn("garbage", message)
                # #650, always appended, even disarmed.
                self.assertIn("peer statement", message.lower())

    def test_a_p2p_counter_trip_is_seen_too(self):
        """The send/recv counters reach the same kernels, so their abort words
        must reach the same guard -- that is why they share one tensor."""
        t = _transport(code=3, row=12)
        with self.assertRaises(HostCollectiveAborted):
            t.check_aborted("unit")

    def test_the_peer_statement_is_appended_under_the_650_heading(self):
        t = _transport(code=1)
        with self.assertRaises(HostCollectiveAborted) as caught:
            t.check_aborted("unit")
        self.assertIn("PEER POSITIONS (#650)", str(caught.exception))

    def test_the_gate_switch_restores_the_old_silence(self):
        t = _transport(code=2)
        old = os.environ.get(barlink_abort_gate.ENV_ENABLE)
        os.environ[barlink_abort_gate.ENV_ENABLE] = "0"
        try:
            t.check_aborted("unit")
        finally:
            if old is None:
                os.environ.pop(barlink_abort_gate.ENV_ENABLE, None)
            else:
                os.environ[barlink_abort_gate.ENV_ENABLE] = old

    def test_capture_suppresses_the_read(self):
        """Reading a device word inside a stream capture is illegal, not just
        slow. The replay boundary picks the kernels up instead, which is what
        the abort-gate registration at bring-up is for."""
        import sglang.srt.distributed.device_communicators.barlink as barlink

        t = _transport(code=2)
        original = barlink.graph_capture_running
        barlink.graph_capture_running = lambda: True
        try:
            t.check_aborted("unit")
        finally:
            barlink.graph_capture_running = original

    def test_the_dispatch_seam_reaches_this_guard(self):
        """The wiring, driven rather than described.

        ``BarlinkCommunicator._after_transport`` is duck-typed (#431): it
        looks up ``check_aborted`` on whatever transport just ran. Growing the
        method IS the wiring, so this asserts the seam actually calls it.
        """
        from sglang.srt.distributed.device_communicators.barlink import (
            BarlinkCommunicator,
        )

        comm = BarlinkCommunicator.__new__(BarlinkCommunicator)
        t = _transport(code=1)
        with self.assertRaises(HostCollectiveAborted) as caught:
            comm._after_transport(t, "all_reduce")
        self.assertEqual(caught.exception.where, "all_reduce")

    def test_the_replay_boundary_reaches_this_guard(self):
        """The other half of the wiring: a collective that only ever runs
        inside a replayed CUDA graph is never followed by ``_after_transport``
        at all. ``barlink_abort_gate`` is the next host point, and the
        transport registers with it at bring-up."""
        t = _transport(code=2)
        barlink_abort_gate.register(t)
        with self.assertRaises(HostCollectiveAborted):
            barlink_abort_gate.check_aborts("cuda-graph replay")


# ---------------------------------------------------------------------------
# (c) THE COMPILE SMOKE
# ---------------------------------------------------------------------------


def _nvcc() -> str | None:
    for candidate in (
        os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


class TestTheKernelsStillCompile(CustomTestCase):
    """nvcc on the modified source. No card, no process group, no load_inline.

    ``_load_ext`` cannot be called here: it resolves the group's architectures
    through ``dist.all_gather_object`` and hands the sources to
    ``load_inline``, which links a pybind module -- a process group and a
    build directory this test has no business creating. What it does is
    compile the SAME ``_CUDA_SRC`` string with the same flag SHAPE
    (``-gencode=arch=compute_86,code=sm_86``, torch's include paths, C++17),
    which is what a syntax or scoping error in the new shared-abort pattern
    would trip over. Linking is not part of the question.
    """

    def test_nvcc_accepts_the_cuda_source(self):
        nvcc = _nvcc()
        if nvcc is None:
            self.skipTest("no nvcc on this machine")
        from torch.utils.cpp_extension import include_paths

        src = _cuda_src(_fixed_source())
        with tempfile.TemporaryDirectory() as tmp:
            cu = Path(tmp) / "barlink_host.cu"
            cu.write_text(src)
            includes: list[str] = []
            for path in include_paths("cuda") + [sysconfig.get_paths()["include"]]:
                includes += ["-I", path]
            proc = subprocess.run(
                [
                    nvcc,
                    "-c",
                    str(cu),
                    "-o",
                    str(Path(tmp) / "barlink_host.o"),
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    "-Xcompiler",
                    "-fPIC",
                ]
                + includes,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"nvcc rejected the modified kernels:\n{proc.stderr[-4000:]}",
            )


# ---------------------------------------------------------------------------
# (d) THE ON-CARD FALSIFIER
# ---------------------------------------------------------------------------


class TestOnCardAbortLeavesTheContextAlive(CustomTestCase):
    """THE falsifier for #653, and the only test here that can see the bug.

    Every assertion above is about text. What #583 actually cost was a live
    CUDA context, and no amount of source analysis can show that a context
    SURVIVED. This one does: it drives the real extension's put kernel into
    its deadline with a consume flag that can never be satisfied, then

      1. asserts the abort word was written (the kernel reported instead of
         trapping),
      2. runs a trivial, unrelated CUDA op and asserts it succeeds -- this is
         precisely what a ``__trap()`` destroyed, and what would fail with
         "unspecified launch failure" on the pre-#653 kernels,
      3. asserts ``check_aborted`` turns the word into
         ``HostCollectiveAborted``.

    SCOPE. It drives the extension DIRECTLY, single-rank, with ordinary device
    tensors standing in for the mapped host slots and the flag block: the full
    transport would need a shm segment, ``cudaHostRegister`` and a real
    process group, none of which the kernels' abort path depends on. It
    therefore covers the put kernel's reuse-guard wait (code 1) -- the site
    whose flag can be starved single-rank. Codes 2 and 3 wait on PEERS, which
    do not exist at world=1; their pattern is identical and pinned by source
    invariant above.

    SKIPS without CUDA, which is the expected result at a desk.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_the_put_kernel_reports_instead_of_killing_the_context(self):
        import torch.distributed as dist

        from sglang.srt.distributed.device_communicators import barlink_host

        started_group = False
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29653")
            dist.init_process_group(backend="gloo", world_size=1, rank=0)
            started_group = True
        try:
            ext = barlink_host._load_ext(dist.group.WORLD)
            dev = torch.device("cuda", 0)

            # The counter, in the shape the transport builds: [seq, abort].
            # seq = 3 so the reuse guard is ARMED (it is skipped for seq <=
            # lag = 2), which is what makes the wait reachable at all.
            seq = torch.zeros(2, dtype=torch.int64, device=dev)
            seq[0] = 3
            blk = torch.zeros(1, dtype=torch.int32, device=dev)
            # Every flag zero: the consume flag the reuse guard waits for can
            # never reach seq - lag. That is the starved peer, in one line.
            flags = torch.zeros(8 * 32, dtype=torch.int64, device=dev)
            slot = torch.zeros(4096, dtype=torch.float32, device=dev)
            slot_addrs = torch.tensor(
                [slot.data_ptr(), slot.data_ptr()], dtype=torch.int64
            )
            inp = torch.ones(256, dtype=torch.float32, device=dev)
            out = torch.empty_like(inp)

            ext.barlink_host_all_reduce(
                inp, out, slot_addrs, flags.data_ptr(), seq, blk,
                0, 1, 100_000, 1,
            )
            torch.cuda.synchronize()

            self.assertEqual(
                int(seq[1]),
                1,
                "the put kernel's reuse guard must write abort code 1",
            )
            # THE assertion. A trap would have killed the context and this
            # would raise "unspecified launch failure".
            probe = (torch.ones(8, device=dev) * 2).sum().item()
            self.assertEqual(probe, 16.0)

            t = BarlinkHostTransport.__new__(BarlinkHostTransport)
            t._seq_all = seq.view(1, 2)
            t.rank = 0
            t.world_size = 1
            with self.assertRaises(HostCollectiveAborted):
                t.check_aborted("on-card falsifier")
        finally:
            if started_group:
                dist.destroy_process_group()


# ---------------------------------------------------------------------------
# (e) THE REACHABILITY CLAIM
# ---------------------------------------------------------------------------


class TestHostTransportCannotAppearMidRun(CustomTestCase):
    """What this pins: an explicit ``SGLANG_BARLINK_TRANSPORT=host`` boot is
    the ONLY way host-transport code runs.

    That claim is load-bearing twice over. It is why the trap could sit in
    this file for three days after the device transport was cured -- nothing
    on a default boot ever reached it. And it is why the fix does not need a
    mid-run guard: there is no path on which a run that started on ``device``
    or ``bar1`` finds itself executing these kernels later.

    Two halves:

    (i) ``host`` is in the no-fallback set, so a failed bring-up RAISES rather
        than quietly swapping in the gloo plane. Asserted through the real
        lookup path (``parallel_state.capturable_transports`` via
        ``barlink._no_fallback``), never against a copied literal -- the
        module comment is explicit that the set is the capturable set and that
        writing it out twice is how the two drift apart.

    (ii) The transport object is built ONCE and never rebuilt or reassigned:
         the per-op path only reads ``self.transport``.
    """

    def test_host_is_in_the_capturable_set_through_the_real_lookup(self):
        from sglang.srt.distributed.parallel_state import capturable_transports

        self.assertIn("host", capturable_transports())

    def test_host_must_not_fall_back_to_the_gloo_plane(self):
        from sglang.srt.distributed.device_communicators.barlink import (
            _no_fallback,
        )

        self.assertTrue(_no_fallback("host"))
        # The contrast case, so this is a statement about "host" and not about
        # every name: "shm" IS allowed to fall back.
        self.assertFalse(_no_fallback("shm"))

    def test_host_is_never_the_default(self):
        """Selecting it is an explicit act. The module reads the environment
        once, at import, and the default is ``device``."""
        text = (
            _repo_root()
            / "python/sglang/srt/distributed/device_communicators/barlink.py"
        ).read_text()
        self.assertIn(
            '_TRANSPORT = os.environ.get("SGLANG_BARLINK_TRANSPORT", "device")',
            text,
        )

    def test_the_transport_is_built_once_and_never_reassigned(self):
        """Source-invariant on the dispatch module: exactly one assignment to
        ``self.transport``, and it is in ``__init__``. A behavioural assert
        would have to bring up a real transport to be worth anything."""
        text = (
            _repo_root()
            / "python/sglang/srt/distributed/device_communicators/barlink.py"
        ).read_text()
        assignments = re.findall(r"^\s*self\.transport\s*=\s*", text, re.M)
        self.assertEqual(
            len(assignments),
            1,
            "more than one site assigns self.transport; a mid-run swap would "
            "make the reachability claim above false",
        )
        init_at = text.index("    def __init__(\n        self,\n        cpu_group")
        build_at = text.index("self.transport = _build_transport(")
        next_def = text.index("\n    def ", build_at)
        self.assertLess(init_at, build_at)
        self.assertLess(build_at, next_def)

    def test_the_per_op_path_only_reads_the_built_transport(self):
        """``_select`` is the one function every collective goes through."""
        text = (
            _repo_root()
            / "python/sglang/srt/distributed/device_communicators/barlink.py"
        ).read_text()
        start = text.index("    def _select(self, op: str, nbytes: int):")
        body = text[start : text.index("\n    def ", start + 1)]
        self.assertIn("t = self.transport", body)
        self.assertNotIn("self.transport =", body)
        self.assertNotIn("_build_transport(", body)
        self.assertNotIn("TRANSPORT_REGISTRY", body)


if __name__ == "__main__":
    unittest.main()
