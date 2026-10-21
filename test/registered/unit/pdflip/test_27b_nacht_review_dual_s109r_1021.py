# SPDX-License-Identifier: Apache-2.0
"""27B NACHT REVIEW 21.10. -- Auflage A1: the S109R device read is off on the dual layout at EVERY entry.

69000fbab8 gated only the admission re-read (``scheduler._s109r_admission_anchor_read``). The e1 intake read
(``pdflip_anchor_only_read``) and S43 (``pdflip_anchor_only_read_at``) reach ``_s109r_device_read`` through the
AUX-HOST-LAW branch of ``_pdflip_aor_read_attach`` as well; on dual P (local-PP floor -> ``_s109r_rank_alone``) that
took a rank-local device mamba slot per PP stage. User 20.10.: the dual layout stays as it was (before the merge the
branch answered 0). The flip line keeps the read.
"""
import os
import types
import unittest
from unittest import mock

from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U


class DualS109rDeviceReadOff(unittest.TestCase):
    def _fake(self, calls):
        fake = types.SimpleNamespace(
            _s109r_note=lambda req, depth, how, got=0, extra="": calls.append(("note", how)) or 0,
            _s90b_device_attach=lambda req, node, depth, site, note=None: calls.append(("attach", depth)) or depth,
            _pdflip_mamba_pool=lambda: None,
            components={},
        )
        fake.__dict__["_s109r_written"] = set()
        return fake

    def test_dual_answers_zero_without_attach_or_note(self):
        calls = []
        node = types.SimpleNamespace(get_last_hash_value=lambda: "h")
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1"}), \
                mock.patch.object(U, "_s109r_rank_alone", staticmethod(lambda self: True)):
            self.assertEqual(U._s109r_device_read(self._fake(calls), object(), node, 4096, "match"), 0)
        self.assertEqual(calls, [])

    def test_flip_line_still_reads(self):
        calls = []
        node = types.SimpleNamespace(get_last_hash_value=lambda: "h")
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DUAL_LAYOUT": ""}), \
                mock.patch.object(U, "_s109r_rank_alone", staticmethod(lambda self: True)):
            self.assertEqual(U._s109r_device_read(self._fake(calls), object(), node, 4096, "match"), 4096)
        self.assertEqual(calls, [("attach", 4096)])


if __name__ == "__main__":
    unittest.main()
