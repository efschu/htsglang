"""The #622 consumption-acknowledgment layout and its kernel protocol.

THE DEFECT. The BAR1 mesh and a2a kernels wait on a CONJUNCTION over all
peers, on 256-byte flag lines addressed by (topology, step, sender), for
EQUALITY with the current round number. The round counter is global across
topologies -- it advances once per collective, whichever topology ran -- so
a rank that gets ahead enters its next same-topology collective and
overwrites a line a slow peer is still waiting on. Flags only ever grow, so
the awaited value never reappears and the group deadlocks. That is the
#622/#632 specimen.

THE FIX under test here: two acknowledgment banks appended to the END of the
flag region (one line per rank each, mesh and a2a separately), a watermark
word per topology in the round tensor, and an entry wait in both kernels
that blocks until every peer has confirmed consuming this rank's PREVIOUS
same-topology collective.

Pure source and arithmetic analysis. No card, no nvcc, no build.
"""

import ast
import re
import unittest
from pathlib import Path

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    ackbase_a2a,
    ackbase_mesh,
    fbase_a2a,
    flags_requirement,
)
from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (
    pipe_flags_extra,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_EXT = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/distributed/device_communicators/barlink_bar1_ext.py"
)

_WORLDS = tuple(range(2, 9))


def _prefix_622_flag_bytes(world: int, with_a2a: bool, with_pipe: bool) -> int:
    """The flag-region size as it was BEFORE the acknowledgment banks.

    Written out here rather than imported: the point of the test is that the
    banks were APPENDED, and a reference that shared its arithmetic with the
    implementation could not tell an append from a rewrite.
    """
    base = (2 + 2 * (world - 1) + (1 if with_a2a else 0)) * world * 256
    return base + (pipe_flags_extra(world) if with_pipe else 0)


def _cuda_src() -> str:
    """The ``_CUDA_SRC`` literal, read as text rather than imported."""
    tree = ast.parse(_EXT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if getattr(node.targets[0], "id", "") == "_CUDA_SRC":
                return node.value.value
    raise AssertionError("_CUDA_SRC not found")


def _without_comments(text: str) -> str:
    """Comments blanked, line count preserved -- the comments quote the code."""
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S
    )


def _kernel_body(src: str, name: str) -> str:
    """The braces-balanced body of ``__global__ void <name>(...)``."""
    m = re.search(rf"__global__\s+void\s+{name}\s*\([^)]*\)\s*\{{", src)
    assert m, f"kernel {name} not found"
    start = m.end() - 1
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


class TestTheRegionGrewByExactlyTwoBanks(CustomTestCase):
    def test_the_growth_is_two_lines_per_rank_in_every_configuration(self):
        for world in _WORLDS:
            for with_a2a in (False, True):
                for with_pipe in (False, True):
                    self.assertEqual(
                        flags_requirement(world, with_a2a, with_pipe)
                        - _prefix_622_flag_bytes(world, with_a2a, with_pipe),
                        2 * world * 256,
                        msg=f"world={world} a2a={with_a2a} pipe={with_pipe}",
                    )

    def test_the_banks_are_allocated_even_without_a2a(self):
        """The entry barrier belongs to the transport, not to a topology.

        A region sized only when ``with_a2a`` happens to be on would put the
        mesh bank inside the allocation on one configuration and past its end
        on another -- and past the end is a write into whatever the allocator
        put there, not a crash.
        """
        for world in _WORLDS:
            budget = flags_requirement(world, False, False)
            last = ackbase_a2a(world, False, False) + (world - 1) * 256
            self.assertLessEqual(last + 256, budget)


class TestTheBanksSitBehindEverythingThatExisted(CustomTestCase):
    def test_the_mesh_bank_starts_exactly_at_the_old_end(self):
        for world in _WORLDS:
            for with_a2a in (False, True):
                for with_pipe in (False, True):
                    self.assertEqual(
                        ackbase_mesh(world, with_a2a, with_pipe),
                        _prefix_622_flag_bytes(world, with_a2a, with_pipe),
                        msg=f"world={world} a2a={with_a2a} pipe={with_pipe}",
                    )

    def test_no_pre_existing_line_moved(self):
        """``fbase_a2a`` is the offset a PEER writes to. It must not shift."""
        for world in _WORLDS:
            self.assertEqual(fbase_a2a(world), (2 + 2 * (world - 1)) * world * 256)

    def test_both_banks_lie_behind_the_pipe_rows_when_the_pipe_is_on(self):
        for world in _WORLDS:
            last_pre = _prefix_622_flag_bytes(world, True, True) - 256
            self.assertGreater(ackbase_mesh(world, True, True), last_pre)
            self.assertGreater(ackbase_a2a(world, True, True), last_pre)

    def test_the_two_banks_do_not_overlap_and_stay_inside_the_budget(self):
        for world in _WORLDS:
            for with_a2a in (False, True):
                for with_pipe in (False, True):
                    mesh = [
                        ackbase_mesh(world, with_a2a, with_pipe) + r * 256
                        for r in range(world)
                    ]
                    a2a = [
                        ackbase_a2a(world, with_a2a, with_pipe) + r * 256
                        for r in range(world)
                    ]
                    self.assertEqual(set(mesh) & set(a2a), set())
                    self.assertGreater(min(a2a), max(mesh))
                    for line in mesh + a2a:
                        self.assertEqual(line % 256, 0)
                    self.assertLessEqual(
                        max(a2a) + 256,
                        flags_requirement(world, with_a2a, with_pipe),
                    )

    def test_the_a2a_bank_follows_the_mesh_bank_immediately(self):
        for world in _WORLDS:
            for with_a2a in (False, True):
                for with_pipe in (False, True):
                    self.assertEqual(
                        ackbase_a2a(world, with_a2a, with_pipe)
                        - ackbase_mesh(world, with_a2a, with_pipe),
                        world * 256,
                    )


class TestTheKernelProtocol(CustomTestCase):
    def setUp(self):
        self.src = _cuda_src()
        self.code = _without_comments(self.src)

    def test_both_argument_structs_carry_the_acknowledgment_entries(self):
        for struct in ("Bar1Args", "A2aArgs"):
            m = re.search(rf"struct\s+{struct}\s*\{{(.*?)\n\}};", self.code, re.S)
            self.assertIsNotNone(m, msg=f"{struct} not found")
            body = m.group(1)
            self.assertIn("ackTo", body, msg=f"{struct} has no ackTo")
            self.assertIn("ackFrom", body, msg=f"{struct} has no ackFrom")
            self.assertIn("lastRoundDev", body, msg=f"{struct} has no watermark")

    def test_the_entry_wait_is_monotonic_in_both_kernels(self):
        """``>=``, not ``==``.

        This is the whole reason the fix terminates. An ack line carries a
        watermark: a peer that has already run further has by construction
        consumed the older round, and an equality wait would sit there
        forever waiting for a value that has been passed -- reintroducing the
        very deadlock shape the banks exist to remove.
        """
        for kernel, expr in (
            ("bar1_mesh_kernel", r"readFlag<LA>\(sAckFrom\[s\]\)\s*>=\s*prev"),
            ("bar1_a2a_kernel", r"readFlag<LA>\(A\.ackFrom\[s\]\)\s*>=\s*prev"),
        ):
            body = _without_comments(_kernel_body(self.src, kernel))
            self.assertRegex(body, expr, msg=f"{kernel}: no monotonic entry wait")
            self.assertNotRegex(
                body,
                r"ackFrom\[s\]\)\s*[!=]=\s*prev",
                msg=f"{kernel}: the entry wait compares for equality",
            )

    def test_the_entry_wait_precedes_the_send_phase(self):
        """A wait placed after the send has already overwritten the slots."""
        for kernel, ack in (
            ("bar1_mesh_kernel", "sAckFrom"),
            ("bar1_a2a_kernel", "A.ackFrom"),
        ):
            body = _without_comments(_kernel_body(self.src, kernel))
            self.assertLess(
                body.index(ack),
                body.index("__threadfence_system();"),
                msg=f"{kernel}: the entry wait does not come first",
            )

    def test_the_first_collective_passes_without_waiting(self):
        """``prev == 0`` against a zero-initialized bank must not spin."""
        for kernel in ("bar1_mesh_kernel", "bar1_a2a_kernel"):
            body = _without_comments(_kernel_body(self.src, kernel))
            self.assertIn("while (prev != 0ull)", body, msg=kernel)

    def test_the_entry_abort_has_its_own_status_code(self):
        """Status 2 names a different culprit than status 1.

        1 = a payload barrier gave up, i.e. this rank had sent and was waiting
        for the peers' contribution. 2 = this rank had sent nothing and was
        waiting for the peers to consume the previous collective. Sharing a
        code would erase that distinction in every future report.
        """
        for kernel in ("bar1_mesh_kernel", "bar1_a2a_kernel"):
            body = _without_comments(_kernel_body(self.src, kernel))
            self.assertIn("*A.ctlStatus = 2u;", body, msg=kernel)
            self.assertEqual(body.count("*A.ctlStatus = 2u;"), 1, msg=kernel)
        self.assertEqual(self.code.count("*A.ctlStatus = 2u;"), 2)

    def test_the_acknowledgment_is_published_at_kernel_end(self):
        for kernel, ack in (
            ("bar1_mesh_kernel", "writeU64(sAckTo[z], round);"),
            ("bar1_a2a_kernel", "writeU64(A.ackTo[z], round);"),
        ):
            body = _without_comments(_kernel_body(self.src, kernel))
            self.assertIn(ack, body, msg=f"{kernel}: no ack is ever published")
            # Peers first, fence, then the local watermark: a watermark that
            # moved first would let a peer's entry wait pass against acks that
            # had not crossed the aperture yet.
            pos = body.index(ack)
            tail = body[pos:]
            self.assertLess(
                tail.index("__threadfence_system();"),
                tail.index("*(volatile u64 *)A.lastRoundDev = round;"),
                msg=f"{kernel}: no fence between the acks and the watermark",
            )

    def test_the_ring_kernel_is_untouched(self):
        """2(R-1) single-peer waits wrap the whole ring.

        A ring collective cannot complete on any rank without every rank
        having entered it, so no rank can be a whole collective ahead and no
        line can be rewritten under a waiting peer. An ack barrier here would
        be a spin per step for a race the step structure already excludes.
        """
        ring = _without_comments(_kernel_body(self.src, "bar1_ring_kernel"))
        for token in ("ackTo", "ackFrom", "lastRoundDev", "prev"):
            self.assertNotIn(token, ring, msg=f"the ring kernel references {token}")

    def test_the_watermarks_are_separate_words_of_the_round_tensor(self):
        """Word 1 = mesh, word 2 = a2a. One shared word would let an a2a
        acknowledgment satisfy a mesh entry wait."""
        self.assertIn("((u64 *)round_dev.data_ptr()) + 1", self.code)
        self.assertIn("((u64 *)round_dev.data_ptr()) + 2", self.code)

    def test_the_offsets_are_passed_in_and_not_recomputed_in_cxx(self):
        """The same rule ``fbase_a2a`` follows: one arithmetic source.

        The offset depends on whether a2a and the pipe families are present.
        A second version of that formula in C++ is exactly where sender and
        receiver end up addressing different lines.
        """
        for param in ("int64_t ackbase_mesh", "int64_t ackbase_a2a"):
            self.assertIn(param, self.code)
        for forbidden in ("2 + 2 * (R - 1)", "2 + 2 * (world - 1)"):
            self.assertNotIn(forbidden, self.code)


if __name__ == "__main__":
    unittest.main()
