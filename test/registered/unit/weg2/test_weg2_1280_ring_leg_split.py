"""#1280 -- split the ring restore leg's resume() into its three passes so a
per-tag ms no longer over-attributes to the copy.

Coordinator order: before #1369 (ring teardown) and #1354 (remap flip) both
decide policy over "this leg's time", the leg must say WHICH pass it spent
that time in -- remap_ms (pass 1, VMM cu_mem_create/cuMemMap/cu_mem_set_access,
cost scales with allocs=, never logged before this ticket), copy_ms (pass 2's
async H2D issue alone), sync_ms (pass 3's single blocking synchronise -- the
actual PCIe/duplex-bound wait). On the x4 card this leg is measured at 2.35 s
= 100% of the leg wall, and nobody could say whether that was wire-bound or
map-bound.

HARD BOUNDARY, per the order: NO BEHAVIOUR CHANGE. The ring path is what
every boot runs today (#1369: launcher.py:2712's `--enable-weights-cpu-backup`
is unconditional in `common_flags` for both groups, even under `exchange`),
so an instrument that adds so much as one extra device synchronisation costs
a boot. This file's whole job is to prove the added code is TIMING-ONLY:
pure `std::chrono::steady_clock::now()` reads (never a device call) and a
host-only byte sum, in positions that cannot move an existing
`cudaStreamSynchronize` or add a new one.

Why a source scan and not a runtime test: identical rationale to
test_weg2_ring_hotpath_1235.py -- the properties that can kill this slice
silently (a stray device sync, the new print polluting S7's own `copy_ms`
by running before `note_resume`) are invisible to `py_compile` and an import
smoke, and no GPU is available to this test suite (CUDA_VISIBLE_DEVICES="").

RED-FIRST, verified by hand (see the commit message for the exact
before/after `git diff`): removing `weg2_copy_issue_t1`/`weg2_sync_t1` (or
reordering the print before `note_resume`) turns
`test_the_print_runs_strictly_after_note_resume` /
`test_exactly_two_new_timestamps_are_pure_chrono_reads` red.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CSRC = os.path.join(TREE, "python", "sglang", "srt", "weg2", "tms_csrc")


def _read(name: str) -> str:
    with open(os.path.join(CSRC, name)) as f:
        return f.read()


def _code(src: str) -> str:
    """``src`` with ``//``/``/* */`` comments blanked (same offsets) -- the
    #995 prose-marker trap in source form. Copied verbatim from
    test_weg2_ring_hotpath_1235.py so both files agree on what counts as
    code."""
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


class RingLegSplitStructureTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw = _read("core.cpp")
        cls.core = _code(cls.raw)
        cls.resume = _body(cls.core, "int TorchMemorySaver::resume(")  # xsn289: resume returns the CUresult

    def test_no_new_device_synchronisation_was_added(self):
        """THE HARD BOUNDARY. Exactly one cudaStreamSynchronize must remain
        in resume() -- the pre-existing pass-3 one. A second one, anywhere,
        is exactly the defect class the order named ("wenn du dafuer einen
        [Sync] brauchst, ist das ein Befund")."""
        self.assertEqual(
            self.resume.count("cudaStreamSynchronize"), 1,
            "#1280 must not add a device synchronisation -- if timing this "
            "split needed one, that is a finding to report, not a feature "
            "to build",
        )

    def test_exactly_two_new_timestamps_are_pure_chrono_reads(self):
        """weg2_copy_issue_t1 and weg2_sync_t1 must both be
        std::chrono::steady_clock::now() reads -- the same mechanism
        weg2_map_t0/t1 already use -- never anything that touches the
        device."""
        for name in ("weg2_copy_issue_t1", "weg2_sync_t1"):
            decl = f"const auto {name} = std::chrono::steady_clock::now();"
            self.assertIn(decl, self.resume,
                         f"{name} must be declared as a plain chrono read")

    def test_weg2_copy_issue_t1_sits_between_pass_2_and_pass_3(self):
        """The split point: after pass 2's issue loop (any_copy = true;),
        before pass 3's synchronise call."""
        any_copy_set = self.resume.rindex("any_copy = true;")
        copy_issue_t1 = self.resume.index("weg2_copy_issue_t1 = std::chrono")
        sync_call = self.resume.index("cudaStreamSynchronize(backup_stream_)")
        self.assertLess(any_copy_set, copy_issue_t1,
                        "weg2_copy_issue_t1 must be taken AFTER pass 2's "
                        "issue loop, not before or during it")
        self.assertLess(copy_issue_t1, sync_call,
                        "weg2_copy_issue_t1 must be taken BEFORE pass 3's "
                        "synchronise, or it no longer marks the boundary "
                        "between issue and sync")

    def test_weg2_sync_t1_sits_strictly_after_the_synchronize_call(self):
        sync_call = self.resume.index("cudaStreamSynchronize(backup_stream_)")
        sync_t1 = self.resume.index("weg2_sync_t1 = std::chrono")
        self.assertLess(sync_call, sync_t1,
                        "weg2_sync_t1 must be read AFTER cudaStreamSynchronize "
                        "returns, or sync_ms would not include the wait it "
                        "claims to measure")

    def test_the_print_runs_strictly_after_note_resume(self):
        """S7's own copy_ms (#1273) is read inside note_resume() at the
        moment it is called -- the (slower, unbounded-latency) stderr write
        for WEG2-RING LEG must never run before that call, or it would leak
        into S7's own timing."""
        note_resume_call = self.resume.index("note_resume(tag, matched_ptrs.size()")
        ring_leg_print = self.resume.index('"[core.cpp] WEG2-RING LEG')
        self.assertLess(
            note_resume_call, ring_leg_print,
            "the WEG2-RING LEG print must run AFTER note_resume(), never "
            "before it, or the print's own I/O latency pollutes S7's "
            "existing copy_ms reading",
        )

    def test_note_resumes_own_call_site_and_arguments_are_unchanged(self):
        """#1280 must not alter the S7 (#1273) instrument it sits beside --
        same call, same arguments, same two timestamps."""
        self.assertIn(
            "note_resume(tag, matched_ptrs.size(), weg2_map_t0, weg2_map_t1);",
            self.resume,
        )

    def test_bytes_is_summed_before_pass_1_with_no_device_call_between(self):
        """weg2_leg_bytes must be a pure host-side sum, computed before pass
        1 even begins -- so it cannot itself be timed into remap_ms."""
        bytes_sum = self.resume.index("weg2_leg_bytes +=")
        map_t0 = self.resume.index("weg2_map_t0 = std::chrono")
        self.assertLess(bytes_sum, map_t0,
                        "weg2_leg_bytes must be fully computed before "
                        "weg2_map_t0 is taken, so pass 1's own timing window "
                        "never includes it")
        between = self.resume[bytes_sum:map_t0]
        for device_call in ("cuMem", "cuda", "CUDA_ERROR_CHECK", "CURESULT_CHECK"):
            self.assertNotIn(device_call, between,
                             f"a device-shaped call ({device_call}) sits "
                             "between the byte sum and pass 1's start")

    def test_the_ring_leg_line_carries_every_required_field_in_order(self):
        print_stmt = self.resume[self.resume.index('"[core.cpp] WEG2-RING LEG'):]
        print_stmt = print_stmt[: print_stmt.index("std::endl") + len("std::endl")]
        for token in ("tag=", "remap_ms=", "copy_ms=", "sync_ms=", "allocs=", "bytes="):
            self.assertIn(token, print_stmt, f"{token!r} missing from the WEG2-RING LEG line")
        # ORDER matters: it is the exact shape the coordinator specified.
        positions = [print_stmt.index(t) for t in
                    ("tag=", "remap_ms=", "copy_ms=", "sync_ms=", "allocs=", "bytes=")]
        self.assertEqual(positions, sorted(positions),
                         "WEG2-RING LEG fields are out of the specified order")

    def test_the_three_durations_use_the_correct_endpoint_pairs(self):
        print_stmt = self.resume[self.resume.index('"[core.cpp] WEG2-RING LEG'):]
        print_stmt = print_stmt[: print_stmt.index("std::endl") + len("std::endl")]
        self.assertIn("weg2_map_t1 - weg2_map_t0", print_stmt,
                     "remap_ms must be pass 1's own window (map_t1 - map_t0)")
        self.assertIn("weg2_copy_issue_t1 - weg2_map_t1", print_stmt,
                     "copy_ms must be the issue window (copy_issue_t1 - map_t1)")
        self.assertIn("weg2_sync_t1 - weg2_copy_issue_t1", print_stmt,
                     "sync_ms must be the synchronise window "
                     "(sync_t1 - copy_issue_t1)")

    def test_the_print_is_unconditional_not_behind_a_debug_flag(self):
        """Matching host_ring.cpp's WEG2-RING-OPEN/-ATTACH precedent: an
        informational WEG2-RING line is NOT gated behind TMS_DEBUG_LOG (that
        flag is off by default and the line would never reach a real boot
        log)."""
        print_pos = self.resume.index('"[core.cpp] WEG2-RING LEG')
        # No #ifdef TMS_DEBUG_LOG between note_resume's call and the print.
        note_resume_call = self.resume.index("note_resume(tag, matched_ptrs.size()")
        between = self.raw[
            self.raw.index("note_resume(tag, matched_ptrs.size()"):
            self.raw.index('"[core.cpp] WEG2-RING LEG')
        ]
        self.assertNotIn("#ifdef TMS_DEBUG_LOG", between)

    def test_only_one_call_site_of_the_ring_leg_marker_exists(self):
        self.assertEqual(self.core.count("WEG2-RING LEG"), 1)


class BuildIsContentAddressedTest(unittest.TestCase):
    """(d): the preload .so's filename is sha256(sources)-keyed, so a source
    edit MUST produce a new filename automatically -- no manual cache
    invalidation, and no risk of a boot silently reusing a stale .so."""

    def test_build_script_hashes_exactly_the_nine_vendored_sources(self):
        script = _read(os.path.join("..", "..", "scripts", "weg2", "tms",
                                    "build_tms_preload.sh")) if False else None
        # The script lives outside tms_csrc/; read it directly.
        path = os.path.join(TREE, "scripts", "weg2", "tms", "build_tms_preload.sh")
        with open(path) as f:
            script = f.read()
        for src in ("core.cpp", "core.h", "entrypoint.cpp", "api_forwarder.cpp",
                    "api_forwarder.h", "host_ring.cpp", "host_ring.h", "utils.h",
                    "macro.h"):
            self.assertIn(f'"$SRC"/{src}', script,
                         f"{src} is not part of the SHA the build script keys on")
        self.assertIn("sha256sum", script)
        self.assertIn('OUT="$OUT_DIR/torch_memory_saver_hook_mode_preload_cu13_ring_$SHA.so"',
                     script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
