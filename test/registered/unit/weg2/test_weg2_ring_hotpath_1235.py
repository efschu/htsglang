"""T4 + T5 -- the MATCHED hot-path check for C3/C4/C5/C6.

Why a source scan and not a runtime test: the two properties that can kill this
slice silently are an ORDERING and a STREAM FLAG, and neither is observable
without a CUDA context and a real weights image.

* ORDERING (C3/C4).  ``cudaMemcpyAsync`` returns immediately, so an unmap that
  runs before the single ``cudaStreamSynchronize`` reads unmapped pages, and a
  ``ring_->release`` that runs before it hands the peer a granule whose H2D is
  still in flight.  Both produce corruption, not an error.
* STREAM FLAG (R12).  ``cudaStreamCreateWithFlags(cudaStreamNonBlocking)``
  compiles, links, runs, and silently drops the implicit ordering against
  PyTorch's default stream that today's ``cudaMemcpy`` relies on -- copying
  pages a pending kernel is still writing.  Same silent-corruption class as
  MAP_saver risk 5.

``py_compile``, an import smoke and every python test in this tree are
structurally blind to both.  This file is the check that CAN fail on them, per
the speed-mode rule "name the likeliest failure class of THIS edit and pick the
check that can actually fail on it".

RED-FIRST, verified by mutation before this file was committed -- each of these
edits turns exactly the named test red:

* move ``cudaStreamSynchronize`` in ``pause`` below the unmap loop
* delete ``cudaStreamSynchronize`` from ``resume``
* move ``ring_->release`` in ``resume`` above the synchronize
* swap ``cudaStreamCreate`` for ``cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)``
* drop the ``assert_host_backup_eligible`` call from either leg
* let ``free`` drop the ``cpu_backup_from_ring`` branch
"""

from __future__ import annotations

import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CSRC = os.path.join(TREE, "python", "sglang", "srt", "weg2", "tms_csrc")


def _read(name: str) -> str:
    with open(os.path.join(CSRC, name)) as f:
        return f.read()


def _code(src: str) -> str:
    """``src`` with comments blanked out, same length so offsets still line up.

    A comment that MENTIONS ``cudaStreamSynchronize`` is not a call, and this
    file's whole point is to check where the CALLS are.  Measured while writing
    it: the raw text counted 2 syncs in ``pause`` and 3 in ``resume``, all but
    one of them prose -- the exact prose-marker trap (#995) in source form.
    """
    out = []
    i, n = 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(" " * (j - i))
            i = j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _body(src: str, signature: str) -> str:
    """The text of one function, brace-balanced from its signature."""
    start = src.index(signature)
    depth = 0
    i = src.index("{", start)
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


class PauseResumePassStructureTest(unittest.TestCase):
    """C3/C4: one synchronize per leg, and it lies between the right passes."""

    @classmethod
    def setUpClass(cls):
        cls.raw = _read("core.cpp")
        cls.core = _code(cls.raw)
        cls.pause = _body(cls.core, "void TorchMemorySaver::pause(")
        cls.resume = _body(cls.core, "void TorchMemorySaver::resume(")
        cls.resume_raw = _body(cls.raw, "void TorchMemorySaver::resume(")

    def test_pause_has_exactly_one_synchronize(self):
        self.assertEqual(
            self.pause.count("cudaStreamSynchronize"), 1,
            "C3: ONE synchronize for the whole tag -- one per allocation is the "
            "old shape this slice removes",
        )

    def test_resume_has_exactly_one_synchronize(self):
        self.assertEqual(self.resume.count("cudaStreamSynchronize"), 1)

    def test_pause_synchronizes_between_the_copies_and_the_unmap(self):
        copy = self.pause.index("cudaMemcpyDeviceToHost")
        sync = self.pause.index("cudaStreamSynchronize")
        unmap = self.pause.index("cuMemUnmap")
        self.assertLess(copy, sync, "the D2H copies are issued before the sync")
        self.assertLess(
            sync, unmap,
            "C3/T4: cuMemUnmap must not run before the single "
            "cudaStreamSynchronize -- an async D2H still in flight would read "
            "unmapped pages, silently",
        )

    def test_resume_maps_then_copies_then_syncs_then_releases(self):
        mapping = self.resume.index("cuMemMap")
        copy = self.resume.index("cudaMemcpyHostToDevice")
        sync = self.resume.index("cudaStreamSynchronize")
        release = self.resume.index("cpu_backup_granules.clear()")
        ring_release = self.resume.index("ring_->release(")
        self.assertLess(mapping, copy, "pass 1 maps every allocation before any H2D")
        self.assertLess(copy, sync)
        self.assertLess(
            sync, ring_release,
            "C4/R11: the granules go back to the ring strictly AFTER the "
            "synchronize; releasing earlier hands the peer a granule whose H2D "
            "is still in flight",
        )
        self.assertLess(sync, release)

    def test_the_1233_one_backup_invariant_is_preserved_verbatim(self):
        for phrase in (
            "WEG2 ONE-BACKUP PATCH (#1233, DR-1)",
            'The design statement is "only ONE layout lies in host RAM"',
            "the image is released the moment its bytes are back on the device",
        ):
            self.assertIn(phrase, self.resume_raw, f"#1233 invariant text lost: {phrase!r}")

    def test_a_paused_allocation_is_never_left_without_its_bytes(self):
        # Every granule walk asserts it covered the whole allocation; a partial
        # walk would restore a truncated image with no error anywhere.
        self.assertIn("D2H granule walk did not cover the allocation", self.pause)
        self.assertIn("H2D granule walk did not cover the allocation", self.resume)
        self.assertIn("SIMPLE_CHECK(offset == metadata.size", self.pause)


class BackupStreamFlagsTest(unittest.TestCase):
    """R12: the backup stream is created BLOCKING, in the whole vendored csrc."""

    def test_the_backup_stream_is_created_with_default_flags(self):
        core = _code(_read("core.cpp"))
        self.assertIn("cudaStreamCreate(&backup_stream_)", core)

    def test_no_file_in_the_vendored_csrc_creates_a_non_blocking_stream(self):
        for name in sorted(os.listdir(CSRC)):
            if not name.endswith((".cpp", ".h")):
                continue
            src = _code(_read(name))
            for token in ("cudaStreamNonBlocking", "cudaStreamCreateWithFlags"):
                self.assertNotIn(
                    token, src,
                    f"R12: {name} must not use {token} -- it silently drops the "
                    "ordering against PyTorch's default stream that the copies "
                    "rely on (same class as MAP_saver risk 5)",
                )

    def test_the_hazard_is_documented_where_the_stream_is_declared(self):
        header = _read("core.h")
        self.assertIn("backup_stream_", header)
        self.assertIn("R12", header)
        self.assertIn("cudaStreamNonBlocking", header,
                      "the comment must NAME the hazard, not the gap")


class RingEligibilityAssertionTest(unittest.TestCase):
    """T5/C6/R20: kv_cache never reaches a host backup allocator."""

    @classmethod
    def setUpClass(cls):
        cls.raw = _read("core.cpp")
        cls.core = _code(cls.raw)
        cls.pause = _body(cls.core, "void TorchMemorySaver::pause(")
        cls.resume = _body(cls.core, "void TorchMemorySaver::resume(")

    def test_the_assertion_exists_and_names_r20(self):
        guard = _body(self.raw, "void TorchMemorySaver::assert_host_backup_eligible(")
        self.assertIn("SIMPLE_CHECK(metadata.enable_cpu_backup", guard)
        self.assertIn("R20", guard)
        self.assertIn("kv stays RELEASED", guard)

    def test_both_legs_pass_through_the_assertion(self):
        for leg, body in (("pause", self.pause), ("resume", self.resume)):
            self.assertIn("assert_host_backup_eligible(metadata)", body,
                          f"the {leg} leg must go through the C6 gate")

    def test_no_host_allocation_happens_outside_an_enable_cpu_backup_guard(self):
        # Every ring acquire and every cudaMallocHost in the hot path must sit
        # after a `if (!metadata.enable_cpu_backup) { continue; }` and after the
        # assertion.  A call before either is a kv allocation waiting to happen.
        for leg, body in (("pause", self.pause), ("resume", self.resume)):
            guard = body.find("if (!metadata.enable_cpu_backup)")
            if "ring_->acquire(" not in body and "cudaMallocHost" not in body:
                continue
            self.assertNotEqual(guard, -1, f"{leg}: no enable_cpu_backup guard at all")
            assertion = body.index("assert_host_backup_eligible(metadata)")
            for call in ("ring_->acquire(", "cudaMallocHost"):
                pos = body.find(call)
                if pos == -1:
                    continue
                self.assertLess(guard, pos, f"{leg}: {call} precedes the guard")
                self.assertLess(assertion, pos, f"{leg}: {call} precedes the assertion")

    def test_free_returns_ring_granules_to_the_ring_and_pinned_ones_to_cuda(self):
        free_body = _body(_code(_read("core.cpp")), "cudaError_t TorchMemorySaver::free(")
        self.assertIn("metadata.cpu_backup_from_ring", free_body)
        self.assertIn("ring_->release(metadata.cpu_backup_granules)", free_body)
        self.assertIn("cudaFreeHost", free_body)
        # And the old single-pointer field is gone, so nothing can leak through it.
        self.assertIsNone(
            re.search(r"\bcpu_backup\s*=(?!=)", _code(_read("core.cpp"))),
            "the single `cpu_backup` pointer must be gone -- a survivor would be a "
            "second host-byte bookkeeping beside the granule list",
        )

    def test_the_metadata_carries_a_scatter_list_and_its_owner(self):
        header = _read("core.h")
        self.assertIn("std::vector<void*> cpu_backup_granules;", header)
        self.assertIn("bool cpu_backup_from_ring;", header)
        self.assertNotIn("void* cpu_backup;", header)


class RingApiSurfaceTest(unittest.TestCase):
    """C1: the surface the CARRIER's second consumer will use (A2-1..A2-4)."""

    def test_the_region_is_keyed_by_tag_family_not_by_consumer(self):
        header = _read("host_ring.h")
        for token in ("TMS_RING_FAMILY_FREE", "TMS_RING_FAMILY_FLIP_BACKUP",
                      "TMS_RING_FAMILY_CARRIER"):
            self.assertIn(token, header)
        self.assertIn("uint8_t family", header)
        m = re.search(r"acquire\(size_t bytes, uint8_t family, const std::string& tag\)", header)
        self.assertIsNotNone(m, "acquire must take the family, or a second consumer "
                                "needs a second region")

    def test_the_card_is_resolved_by_uuid_and_never_by_a_cvd_ordinal(self):
        src = _code(_read("host_ring.cpp"))
        self.assertIn("cuDeviceGetUuid", src)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", src)

    def test_the_bounded_wait_is_justified_not_tuned(self):
        header = _read("host_ring.h")
        flat = " ".join(header.replace("//:", " ").split())
        self.assertIn("TMS_RING_ACQUIRE_BUDGET_S", header)
        self.assertIn("NOT a tuning knob", flat)
        self.assertIn("120 s", flat, "the bound must name the budget it inherits")
        # No env knob may reach the budget (spec section 10.3 / R19).
        src = _read("host_ring.cpp")
        for knob in ("TMS_HOST_RING_WAIT", "TMS_HOST_RING_BUDGET", "TMS_HOST_RING_MIB",
                     "TMS_HOST_RING_GRANULE"):
            self.assertNotIn(knob, src)

    def test_only_four_env_names_are_read_and_all_are_launcher_output(self):
        src = _read("host_ring.cpp")
        names = set(re.findall(r'env_or_null\("([A-Z_0-9]+)"\)', src))
        self.assertEqual(
            names,
            {"TMS_HOST_RING_DIR", "TMS_HOST_RING_MAP", "TMS_HOST_RING_EPOCH",
             "TMS_HOST_RING_FORM"},
            "R19: the ring reads exactly the four variables the launcher "
            "publishes -- any fifth would be an operator-facing knob",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
