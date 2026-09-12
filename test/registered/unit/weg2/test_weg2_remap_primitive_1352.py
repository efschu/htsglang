# SPDX-License-Identifier: Apache-2.0
"""#1352 REMAP -- the TMS page primitive's surface and its refusals.

WHAT THIS CAN AND CANNOT PROVE, said first so nobody reads more into a green
run than it carries.  The CUDA path (``cuMemUnmap`` in one VA, ``cuMemMap`` of
the same handle in another) is a METAL fact and is proven on metal.  What is
hermetic here is everything that decides whether that metal run is even
possible, and every item below was a real defect risk rather than a formality:

* the page-granular arm exists in ``malloc``/``free``/``pause``/``resume``, and
  the LEGACY arm is untouched -- an unset ``TMS_REMAP_TAGS`` must be the old
  path byte for byte;
* ``resume`` fills ONLY the holes, which is the entire funding mechanism: a
  page that arrived by remap must not be re-created;
* handles are created EXPORTABLE when asked, because that is a load-time
  decision no flip can repair;
* the adapter turns a missing symbol into a WIRING error and a refusal into a
  REASON, and never the two the wrong way round.
"""

import ast
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import CustomTestCase

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CSRC = os.path.join(ROOT, "python", "sglang", "srt", "weg2", "tms_csrc")
CORE_CPP = os.path.join(CSRC, "core.cpp")
CORE_H = os.path.join(CSRC, "core.h")
UTILS_H = os.path.join(CSRC, "utils.h")
ENTRY = os.path.join(CSRC, "entrypoint.cpp")
ADAPTER = os.path.join(ROOT, "python", "sglang", "srt", "utils",
                       "torch_memory_saver_adapter.py")


def _src(path):
    with open(path) as fh:
        return fh.read()


def _method_src(path, class_name, method_name):
    """The source of ONE method of ONE class, via the AST.

    Slicing by the first ``def <name>`` found the ABSTRACT BASE's stub and
    asserted against ``raise NotImplementedError`` -- the #995 prose trap
    inside a unit test, and the same shape B4n's seat hit twice on this lane.
    """
    src = _src(path)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(src, item) or ""
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


class PrimitiveSurfaceTest(CustomTestCase):
    def test_the_three_entrypoints_are_exported(self):
        src = _src(ENTRY)
        for sym in ("tms_remap_pages", "tms_tag_pages", "tms_page_stats"):
            self.assertIn(sym, src, f"{sym} is not an entrypoint")

    def test_remap_validates_the_whole_batch_before_moving_a_page(self):
        """All-or-nothing: a half-done remap is not recoverable from Python."""
        src = _src(CORE_CPP)
        body = src[src.index("int TorchMemorySaver::remap_pages"):]
        body = body[:body.index("\nvoid TorchMemorySaver::pause")]
        validate = body.index("VALIDATE THE WHOLE BATCH")
        perform = body.index("---- PERFORM ---")
        first_unmap = body.index("cuMemUnmap")
        self.assertLess(validate, perform)
        self.assertLess(perform, first_unmap,
                        "a cuMemUnmap runs before the validation loop closed")

    def test_remap_never_creates_or_releases_a_page(self):
        """THE claim of the design, checked where it can be checked."""
        src = _src(CORE_CPP)
        body = src[src.index("int TorchMemorySaver::remap_pages"):]
        body = body[:body.index("\nvoid TorchMemorySaver::pause")]
        self.assertNotIn("cu_mem_create", body,
                         "remap_pages allocates -- that is the defect it replaces")
        self.assertNotIn("cuMemRelease", body,
                         "remap_pages releases -- the peer would lose the page")

    def test_remap_refuses_a_cross_device_move(self):
        src = _src(CORE_CPP)
        self.assertIn("cannot change card, only VA", src)

    def test_remap_refuses_an_occupied_destination_page(self):
        src = _src(CORE_CPP)
        self.assertIn("is already backed", src)

    def test_remap_refuses_a_source_page_with_no_physical_page(self):
        src = _src(CORE_CPP)
        self.assertIn("refusing rather than allocating one", src)

    def test_every_refusal_carries_the_planner_s_own_w_code(self):
        """One name in a boot log for a planner refusal and a driver refusal."""
        self.assertIn("W95 Weg2RemapPageRefused", _src(CORE_CPP))
        from sglang.srt.weg2 import xchg_pageplan as pp
        self.assertEqual(pp.Weg2RemapPageRefused.__name__, "Weg2RemapPageRefused")


class PageGranularArmTest(CustomTestCase):
    def test_resume_fills_only_the_holes(self):
        """A page that arrived by remap must NOT be re-created.

        This is the funding mechanism itself: if resume re-created every page,
        the remap would move pages and the wake would allocate them anyway, and
        the boot would read as a success while doing exactly what the user's
        order forbids.
        """
        src = _src(CORE_CPP)
        body = src[src.index("void TorchMemorySaver::resume"):]
        body = body[:body.index("--- pass 2")]
        self.assertIn("if (metadata.page_handles[i] != 0) {", body)
        self.assertIn("continue;", body)
        self.assertIn("++created_this_wake", body)

    def test_pause_zeroes_the_slot_it_released(self):
        src = _src(CORE_CPP)
        body = src[src.index("void TorchMemorySaver::pause"):]
        self.assertIn("metadata.page_handles[i] = 0;", body)

    def test_free_skips_a_page_it_no_longer_owns(self):
        """Releasing a handle the peer now owns frees a page under its feet."""
        src = _src(CORE_CPP)
        body = src[src.index("cudaError_t TorchMemorySaver::free"):]
        body = body[:body.index("void TorchMemorySaver::ensure_backup_stream")]
        self.assertIn("if (metadata.page_handles[i] == 0)", body)

    def test_the_legacy_arm_is_still_one_handle_for_the_whole_allocation(self):
        """An unset TMS_REMAP_TAGS must be the old path, unchanged."""
        src = _src(CORE_CPP)
        self.assertIn("CUDAUtils::cu_mem_create(&allocHandle, size, device);", src)
        self.assertIn("cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0)", src)

    def test_the_page_size_is_read_from_the_driver_not_assumed(self):
        """A wrong page constant mis-slices every tensor boundary silently."""
        self.assertIn("cu_mem_min_granularity", _src(CORE_CPP))
        self.assertIn("CU_MEM_ALLOC_GRANULARITY_MINIMUM", _src(UTILS_H))

    def test_the_remap_gate_is_off_by_default(self):
        """No env, no behaviour change -- the boot that does not ask is the old one."""
        src = _src(CORE_CPP)
        self.assertIn('std::getenv("TMS_REMAP_TAGS")', src)
        body = src[src.index("bool TorchMemorySaver::remap_tag"):]
        body = body[:body.index("std::vector<void*> TorchMemorySaver::ordered_ptrs")]
        self.assertIn("return false;", body)


class ExportabilityTest(CustomTestCase):
    def test_exportability_is_requested_at_creation_or_never(self):
        src = _src(UTILS_H)
        self.assertIn("prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR",
                      src)
        self.assertIn("CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED",
                      src, "a device that cannot export must refuse, not create "
                           "handles that can never cross a process")

    def test_page_order_is_address_order_on_both_sides(self):
        """A page index only names the same page if both ends sort alike."""
        src = _src(CORE_CPP)
        body = src[src.index("std::vector<void*> TorchMemorySaver::ordered_ptrs"):]
        self.assertIn("std::sort(out.begin(), out.end());", body[:1200])


class AdapterBindingTest(CustomTestCase):
    def test_all_three_adapter_classes_carry_the_new_methods(self):
        """A missing override on one class is an AttributeError at the flip."""
        tree = ast.parse(_src(ADAPTER))
        wanted = {"tag_pages", "page_stats", "remap_pages"}
        classes = {
            node.name: {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        for cls in ("TorchMemorySaverAdapter", "_TorchMemorySaverAdapterReal",
                    "_TorchMemorySaverAdapterNoop"):
            self.assertIn(cls, classes)
            self.assertTrue(wanted <= classes[cls],
                            f"{cls} is missing {wanted - classes[cls]}")

    def test_a_missing_symbol_is_a_wiring_error_not_a_refusal(self):
        """The two must never be confused: one means 'rebuild the .so', the
        other means 'the move cannot happen'. A boot that read the first as the
        second would keep flipping with a stale hook and blame the plan."""
        body = _method_src(ADAPTER, "_TorchMemorySaverAdapterReal", "remap_pages")
        self.assertIn("raise RuntimeError(", body)
        self.assertIn("wiring defect", body)

    def test_the_noop_adapter_refuses_by_name_rather_than_reporting_success(self):
        from sglang.srt.utils.torch_memory_saver_adapter import (
            _TorchMemorySaverAdapterNoop,
        )

        got = _TorchMemorySaverAdapterNoop().remap_pages("a", 0, "b", 0, 1)
        self.assertIsNotNone(got, "None means SUCCESS -- a wake would be believed funded")
        self.assertIn("W95", got)

    def test_page_stats_separates_absence_from_a_measured_zero(self):
        """``created == 0`` is the acceptance number; None is 'nobody measured'."""
        body = _method_src(ADAPTER, "_TorchMemorySaverAdapterReal", "page_stats")
        self.assertIn("if rc != 1:", body)
        self.assertIn("return None", body)


if __name__ == "__main__":
    unittest.main()
