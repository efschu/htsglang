"""SM89-DURCHSPIEL-1002: FP8 dispatch on sm_89 must not trust a wheel that
carries no sm_89 SASS.

The release wheel is built ``SGL_KERNEL_LIMIT_CUDA_ARCHS=86;120a`` (gencode
without PTX). ``sgl_kernel.fp8_scaled_mm`` dispatches on ``sm_version == 89``
into the CUTLASS ``Sm89`` template -- inside the sm_86 pass that template is
guarded off (``__CUDA_ARCH__ >= 890``), so the cubin only holds
``CUTLASS_NOT_IMPLEMENTED()`` (printf + brkpt): a device trap, not an error.

The dispatch therefore consults the WHEEL (the cubins' ``.note.nv.cuinfo``
records, scanned from the ELF bytes -- never a card-name substring):
* sm_89 + wheel without sm_89 SASS  -> named fallback to FP8-Marlin
  (``can_auto_enable_marlin_fp8`` True, ``cutlass_fp8_supported`` False);
* sm_89 + wheel WITH sm_89 SASS     -> native CUTLASS stays, no fallback;
* detection impossible              -> conservative: fallback (a slow Marlin
  beats a device trap);
* sm_86 / sm_120                    -> untouched, the probe is not consulted.

Red on the base tree: base answers sm_89 with cutlass=True / marlin=False
(the trap configuration) and has no ``wheel_sass`` module at all.
"""

import logging
import struct
import unittest


def _clear_gates():
    from sglang.srt.utils.common import clear_per_device_gate_caches

    clear_per_device_gate_caches()


def cuinfo_note(sm: int) -> bytes:
    """Byte-encoding of the real cubin ``.note.nv.cuinfo`` note
    (namesz 12, descsz 8, type 1000, name "NVIDIA Corp\\0", descriptor u16
    at +2 = SM). Verified against the real installed wheel
    ``sgl_kernel/sm100/common_ops.abi3.so``: 52 notes sm_86, 52 notes
    sm_120, none sm_89."""
    return (struct.pack("<III", 12, 8, 1000) + b"NVIDIA Corp\x00"
            + struct.pack("<HHI", 2, sm, 130) + b"\x00\x00")


class TestWheelSassScan(unittest.TestCase):
    def test_scan_reads_note_archs_from_file_bytes(self):
        from sglang.srt.utils import wheel_sass

        blob = b"\x00JUNK\x00" + cuinfo_note(86) + cuinfo_note(120)
        self.assertEqual({86, 120}, set(wheel_sass.scan_sass_archs(blob)))
        blob89 = b"\x00" + cuinfo_note(86) + cuinfo_note(89) + cuinfo_note(120)
        self.assertEqual({86, 89, 120}, set(wheel_sass.scan_sass_archs(blob89)))

    def test_scan_of_bytes_without_notes_is_empty_not_a_crash(self):
        from sglang.srt.utils import wheel_sass

        self.assertEqual(set(), set(wheel_sass.scan_sass_archs(b"no cubins here")))

    def test_scan_inflates_zstd_framed_fatbins(self):
        """The real ``sm100/common_ops.abi3.so`` stores ALL 104 cubins in
        ZSTD frames (magic 28 b5 2f fd) -- the scan must see through them,
        not only the raw ones (flashmla layout)."""
        import importlib.util
        import shutil
        import subprocess

        inner = cuinfo_note(86) + cuinfo_note(120)
        have_decoder = importlib.util.find_spec("zstandard") is not None
        zstd = shutil.which("zstd")
        if not have_decoder and not zstd:
            self.skipTest("no zstd decoder here (the probe answers UNKNOWN then)")
        if have_decoder:
            import zstandard

            frame = zstandard.ZstdCompressor().compress(inner)
        else:
            frame = subprocess.run([zstd, "-c"], input=inner, capture_output=True,
                                   check=True).stdout
        from sglang.srt.utils import wheel_sass

        blob = b"\x7fELF\x02\x01\x01" + b"\x00" * 100 + frame + b"tail"
        self.assertEqual({86, 120}, set(wheel_sass.scan_sass_archs(blob)))


class TestSm89FallbackDecision(unittest.TestCase):
    """The gate pair on sm_89, driven by a stubbed wheel probe (never by a
    real wheel, never by a card name)."""

    def setUp(self):
        _clear_gates()
        self.addCleanup(_clear_gates)

    def _arm(self, sm: int, carries):
        import sglang.srt.layers.quantization.fp8_utils as fu

        self.saved = {}
        for name, value in (
            ("get_cuda_sm", lambda device_id=None: sm),
            ("get_device_capability", lambda device_id=0: (sm // 10, sm % 10)),
            ("get_cuda_version", lambda: (12, 8)),
            ("wheel_carries_sass", carries if callable(carries) else (lambda cc: carries)),
            ("_is_cuda", True),
        ):
            self.saved[name] = getattr(fu, name)
            setattr(fu, name, value)
        return fu

    def tearDown(self):
        import sglang.srt.layers.quantization.fp8_utils as fu

        for name, value in self.saved.items():
            setattr(fu, name, value)

    def test_sm89_without_sm89_sass_routes_the_named_marlin_fallback(self):
        fu = self._arm(89, False)
        self.assertTrue(fu.can_auto_enable_marlin_fp8(0))
        self.assertFalse(fu.cutlass_fp8_supported(0))

    def test_sm89_with_sm89_sass_keeps_the_native_cutlass_path(self):
        fu = self._arm(89, True)
        self.assertFalse(fu.can_auto_enable_marlin_fp8(0))
        self.assertTrue(fu.cutlass_fp8_supported(0))

    def test_sm89_with_unknown_wheel_routes_the_fallback_not_the_trap(self):
        fu = self._arm(89, None)
        self.assertTrue(fu.can_auto_enable_marlin_fp8(0))
        self.assertFalse(fu.cutlass_fp8_supported(0))

    def test_sm86_and_sm120_never_consult_the_probe(self):
        def _boom(cc):
            raise AssertionError("the wheel probe must not gate this sm")

        for sm, marlin, cutlass in ((86, True, False), (120, False, True)):
            with self.subTest(sm=sm):
                _clear_gates()  # per-device answers must not leak between arms
                fu = self._arm(sm, _boom)
                try:
                    self.assertIs(marlin, fu.can_auto_enable_marlin_fp8(0))
                    self.assertIs(cutlass, fu.cutlass_fp8_supported(0))
                finally:
                    self.tearDown()
                    self.saved = {}

    def test_fallback_log_names_the_fallback(self):
        records = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        fu = self._arm(89, False)
        handler = _Cap()
        fu.logger.addHandler(handler)
        try:
            fu.can_auto_enable_marlin_fp8(0)
        finally:
            fu.logger.removeHandler(handler)
        joined = "\n".join(records)
        self.assertIn("FP8-Marlin", joined)
        self.assertIn("sm_89", joined)
        self.assertIn("SASS", joined)


if __name__ == "__main__":
    unittest.main()
