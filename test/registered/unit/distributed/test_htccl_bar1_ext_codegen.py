"""The mesh kernel must not index the parameter block dynamically.

Kernel parameters live in constant bank 0, which has no dynamic indexing.
One ``A.nzSendRS[z]`` with a running ``z`` therefore makes nvcc copy the
WHOLE parameter block into local memory, per thread. Measured on this
kernel with ``nvcc -cubin -arch=sm_86`` plus ``cuobjdump -res-usage``:

    before   bar1_netz_kernel   REG:39-40  STACK:64  SHARED:0-4
    after    bar1_netz_kernel   REG:37-40  STACK:0    SHARED:512-520

``bar1_ring_kernel`` (fixed indices) had STACK:0 all along, which is what
made the mesh number stand out. The a2a and pipelined kernels avoid it the
same way this fix does: one thread per block stages the pointer tables into
shared memory, then everyone indexes there.

That measurement needs nvcc, takes minutes, and belongs in the validation
notes -- not in a unit test. What belongs here is the INVARIANT, so the
next edit cannot quietly put the spill back: inside ``bar1_netz_kernel``,
the parameter arrays are read only in the staging loop.

Pure source analysis. No nvcc, no card.
"""

import re
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_COMM = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/distributed/device_communicators"
)
_EXT = _COMM / "htccl_bar1_ext.py"
_PIPE = _COMM / "htccl_bar1_pipe_ext.py"

#: The arrays inside ``Bar1Args`` that used to be indexed with a running
#: index from every thread.
_PARAM_ARRAYS = (
    "nzSendRS",
    "nzSendAG",
    "nzRecvRS",
    "nzRecvAG",
    "nzFlagAn",
    "nzFlagVon",
)


def _cuda_src(path: Path = _EXT) -> str:
    """The ``_CUDA_SRC`` literal, read as source rather than imported.

    Importing the module is fine on CPU but pointless here: the question is
    about the text that nvcc will see.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if getattr(target, "id", "") == "_CUDA_SRC":
                return node.value.value
    raise AssertionError(f"_CUDA_SRC not found in {path.name}")


def _without_comments(text: str) -> str:
    """Line and block comments blanked out, line count preserved.

    The comment above the staging loop quotes ``A.nzSendRS[z]`` -- it is the
    explanation, not an access. A scan that cannot tell the two apart would
    fail on its own documentation.
    """
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


class TestMeshKernelHasNoParamSpill(CustomTestCase):
    def setUp(self):
        self.src = _cuda_src()
        self.body = _kernel_body(self.src, "bar1_netz_kernel")

    def test_the_staging_block_exists(self):
        """Otherwise the test below would pass on an empty kernel."""
        self.assertIn("__shared__ uint4       *sSendRS", self.body)
        self.assertIn("if (threadIdx.x == 0) {", self.body)

    def test_staging_uses_syncthreads_not_the_grid_barrier(self):
        """Shared memory is block-local, so the barrier has to be too.

        ``barriere<GRID>()`` is a grid sync in the cooperative variant. Using
        it here would leave every block except block 0 reading a shared array
        nobody in that block filled -- null pointers, not slow code.
        """
        pos = self.body.index("if (threadIdx.x == 0) {")
        after = self.body[pos : pos + 1400]
        self.assertIn("__syncthreads();", after)
        before_sync = after[: after.index("__syncthreads();")]
        self.assertNotIn("barriere<GRID>", before_sync)

    def test_param_arrays_are_read_only_while_staging(self):
        """Every ``A.<array>`` sits inside the thread-0 staging loop."""
        body = _without_comments(self.body)
        pos = body.index("if (threadIdx.x == 0) {")
        end = body.index("__syncthreads();", pos)
        staging = body[pos:end]
        rest = body[:pos] + body[end:]
        offenders = []
        for array in _PARAM_ARRAYS:
            self.assertIn(
                f"A.{array}",
                staging,
                msg=f"{array} is not staged into shared memory",
            )
            for m in re.finditer(rf"A\.{array}\b", rest):
                lineno = rest[: m.start()].count("\n") + 1
                offenders.append(f"A.{array} (kernel-relative line {lineno})")
        self.assertFalse(
            offenders,
            msg=(
                "bar1_netz_kernel reads a Bar1Args pointer array outside the "
                "staging loop. Kernel parameters live in constant bank 0, "
                "which cannot be indexed dynamically -- nvcc answers by "
                "copying the whole parameter block into local memory per "
                "thread (measured: STACK 64 B, vs 0 for bar1_ring_kernel). "
                "Stage it into __shared__ with the others instead:\n  "
                + "\n  ".join(offenders)
            ),
        )

    def test_ring_kernel_is_left_alone(self):
        """It already had STACK 0 -- do not 'fix' what is measured at zero."""
        ring = _kernel_body(self.src, "bar1_ring_kernel")
        self.assertNotIn("__shared__ uint4", ring)
        self.assertIn("A.rgSend[s]", ring)


#: Every spin loop of the BAR1 family ends on the same cycle deadline, so the
#: deadline check is the enumeration of the spin loops -- there is no second
#: list to keep in sync.
_DEADLINE = "> A.deckelZyklen"
#: Expected number of spin loops per source. A change here is a change to the
#: kernel's wait structure and should be a deliberate edit, not a surprise.
_SPIN_LOOPS = {"htccl_bar1_ext.py": 5, "htccl_bar1_pipe_ext.py": 3}


class TestHostAbortProbeInEverySpinLoop(CustomTestCase):
    """A spinning kernel must be reachable from the host.

    The cycle deadline is rank-local: it cannot see that a peer PROCESS is
    gone, it is multiplied by up to 40x inside the JIT cold-build window --
    which is exactly the capture window where an OOM kill lands -- and its
    expiry is silent. ``abbruchWirt`` is the one host-set word that closes
    that gap, and it only closes it in the loops that actually read it. The
    invariant is therefore per loop, not per file.
    """

    def _probe_follows_every_deadline(self, path: Path) -> None:
        src = _without_comments(_cuda_src(path))
        self.assertIn(
            "#define HTCCL_BAR1_WIRT_MASKE",
            _cuda_src(path),
            msg=f"{path.name}: the probe's rate limit is gone",
        )
        self.assertIn(
            "const unsigned int *abbruchWirt;",
            src,
            msg=f"{path.name}: no host abort word in the argument struct",
        )
        stellen = [
            src.count("\n", 0, m.start()) + 1
            for m in re.finditer(re.escape(_DEADLINE), src)
        ]
        self.assertEqual(
            len(stellen),
            _SPIN_LOOPS[path.name],
            msg=(
                f"{path.name} has {len(stellen)} spin loops, expected "
                f"{_SPIN_LOOPS[path.name]}. Adding or removing a wait is "
                f"fine -- update the count here and give the new loop its "
                f"probe."
            ),
        )
        zeilen = src.splitlines()
        ohne = []
        for lineno in stellen:
            # The probe sits immediately after the deadline check, within the
            # macro continuation or the next few statements of the loop body.
            fenster = "\n".join(zeilen[lineno : lineno + 5])
            if "A.abbruchWirt" not in fenster:
                ohne.append(lineno)
        self.assertFalse(
            ohne,
            msg=(
                f"{path.name}: the spin loop(s) whose deadline check is on "
                f"line(s) {ohne} do not probe the host abort word. Such a "
                f"loop keeps spinning for the full "
                f"SGLANG_HTCCL_BAR1_CAP_CYCLES budget after a peer has "
                f"died, and the watchdog has no way to end it."
            ),
        )

    def test_the_collective_kernels_probe(self):
        self._probe_follows_every_deadline(_EXT)

    def test_the_pipelined_kernel_probes(self):
        self._probe_follows_every_deadline(_PIPE)

    def test_the_probe_never_precedes_the_success_break(self):
        """The completing collective must not pay for the probe.

        In every loop the "all peers arrived" break comes first, then the
        deadline check, then the probe. Pinning the probe to the lines right
        after the deadline check keeps it there: moved up in front of the
        break, it would put a host-memory read on the success path of every
        spin iteration of every collective that completes.
        """
        for path in (_EXT, _PIPE):
            zeilen = _without_comments(_cuda_src(path)).splitlines()
            falsch = []
            for i, zeile in enumerate(zeilen):
                if "A.abbruchWirt != nullptr" not in zeile:
                    continue
                if not any(_DEADLINE in z for z in zeilen[max(0, i - 3) : i]):
                    falsch.append(i + 1)
            self.assertFalse(
                falsch,
                msg=(
                    f"{path.name}: the probe(s) on line(s) {falsch} do not "
                    f"sit directly after a deadline check. The order in the "
                    f"loop is success break, deadline, probe -- anything "
                    f"else costs the completing collective."
                ),
            )

    def test_the_probe_is_rate_limited(self):
        """Unmasked, it would be one PCIe read per spin iteration."""
        for path in (_EXT, _PIPE):
            src = _without_comments(_cuda_src(path))
            lese = src.count("A.abbruchWirt != nullptr")
            maske = src.count("& HTCCL_BAR1_WIRT_MASKE")
            self.assertEqual(
                lese,
                maske,
                msg=(
                    f"{path.name}: {lese} null checks of the abort word but "
                    f"{maske} rate limits. Every probe must be masked."
                ),
            )


class TestExtensionSurface(CustomTestCase):
    def test_the_exported_functions_are_unchanged(self):
        """all_gather rides the a2a entry point -- no new export.

        If this list ever grows, the claim "no new kernel was built for
        all_gather" in the commit and in the validation notes stops being
        true, and that should be a deliberate edit.
        """
        src = _EXT.read_text(encoding="utf-8")
        m = re.search(r"functions=\[([^\]]*)\]", src)
        self.assertIsNotNone(m)
        names = re.findall(r'"([^"]+)"', m.group(1))
        self.assertEqual(names, ["bar1_all_reduce", "bar1_all_to_all"])


if __name__ == "__main__":
    unittest.main()
