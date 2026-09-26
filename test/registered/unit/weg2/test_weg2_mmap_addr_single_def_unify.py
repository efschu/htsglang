# SPDX-License-Identifier: Apache-2.0
"""Unify step 1: ``_mmap_addr`` is defined ONCE in weight_exchange_bounce.

ALTLAST SEIT DER BASIS 76f8debf2c, auf BEIDEN Linien: zwei Top-Level-Defs im
selben Modul -- 4a67b30db5 (#1378 xsn46, 14.09. 19:17Z, "for ops.host_register")
und 40 Minuten spaeter efaef5768e (#1378 xsn44 (d), "for ops.memcpy_async and
ops.host_register", mit der xsn46/47-Begruendung: eine GANZZAHL, kein
memoryview). Python bindet beim Import die SPAETERE; die fruehere war nie
erreichbar: alle drei Lesestellen (``_persistent_host_buffer``,
``run_sequential_units`` x2) stehen in Funktionsrumpfen und loesen den Namen erst
zur Aufrufzeit auf, kein Default-Argument, kein Dekorator, kein Top-Level-Aufruf
zwischen den beiden Defs, kein Import des Namens aus einem anderen Modul. Die
fruehere Def ist entfernt, die geltende bleibt byte-gleich.

Was dieser Test festhaelt:
* genau EINE Def (der dup_defs_gate-Fund ``_mmap_addr Z.3270,3367`` ist weg);
* die verbliebene ist die geltende (ihr Docstring nennt ``memcpy_async``);
* ihr Vertrag: Rueckgabe ist ein ``int`` gleich der Adresse des Mappings, und
  sie laesst KEINEN exportierten Puffer stehen -- ``mm.close()`` direkt danach
  gelingt (ein haengender ``c_char.from_buffer``-Export liesse close() mit
  BufferError scheitern und den persistenten Puffer nie wachsen).
"""

from __future__ import annotations

import ast
import ctypes
import mmap
import os
import pathlib
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5)


def _top_level_defs(name: str):
    tree = ast.parse(pathlib.Path(bx.__file__).read_text())
    return [
        n.lineno
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]


class TheMmapAddrIsDefinedOnce(unittest.TestCase):
    def test_exactly_one_definition(self):
        self.assertEqual(len(_top_level_defs("_mmap_addr")), 1, _top_level_defs("_mmap_addr"))

    def test_the_survivor_is_the_one_that_was_in_force(self):
        doc = bx._mmap_addr.__doc__ or ""
        self.assertIn("memcpy_async", doc)
        self.assertIn("INTEGER, not a", doc)

    def test_returns_the_int_address_and_leaves_no_export(self):
        with tempfile.TemporaryFile() as fh:
            fh.truncate(4096)
            mm = mmap.mmap(fh.fileno(), 4096)
            try:
                addr = bx._mmap_addr(mm)
                self.assertIsInstance(addr, int)
                mm[0:4] = b"\x01\x02\x03\x04"
                self.assertEqual(ctypes.string_at(addr, 4), b"\x01\x02\x03\x04")
            finally:
                # Would raise BufferError if the helper kept an export alive.
                mm.close()
            self.assertTrue(mm.closed)


if __name__ == "__main__":
    unittest.main()
