"""#105 (23.09.): the 27B rule for the NF flip as a check -- every address a
captured D graph reads must still be mapped after the wake. The CUDA walk
itself is proven on metal; these cases pin the parts that decide what counts
as a pointer and what counts as a finding, so a "looks equivalent" rewrite
cannot quietly turn the check into one that finds nothing.
"""
import unittest
from unittest import mock

from sglang.srt.weg2 import graph_pointer_census as gpc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-weg2-unit")


def _rec(addr, start, size, name="k", graph="full/1"):
    return gpc.PtrRecord(graph=graph, node=3, kind="kernel", name=name, param=1,
                         word=8, addr=addr, range_start=start, range_size=size)


class PointerWords(unittest.TestCase):
    def test_a_struct_by_value_yields_its_embedded_pointers(self):
        # the BAR1 argument block: pointers interleaved with ints and zeros
        words = [0x7F0012340000, 7, 0, 0x7F00ABCD0000, 1 << 50]
        raw = b"".join(w.to_bytes(8, "little") for w in words)
        self.assertEqual(gpc.pointer_words(raw),
                         [(0, 0x7F0012340000), (24, 0x7F00ABCD0000)])

    def test_a_tail_shorter_than_a_word_is_not_read(self):
        raw = (0x7F0012340000).to_bytes(8, "little") + b"\x01\x02\x03"
        self.assertEqual(len(gpc.pointer_words(raw)), 1)


class Verify(unittest.TestCase):
    def test_unmapped_and_moved_ranges_are_findings_an_intact_one_is_not(self):
        live = {0x1000_0000: (0x1000_0000, 4096),     # intact
                0x3000_0000: (0x2FFF_F000, 8192)}     # remapped differently
        recs = [_rec(0x1000_0000, 0x1000_0000, 4096),
                _rec(0x2000_0000, 0x2000_0000, 4096, name="pool_copy"),
                _rec(0x3000_0000, 0x3000_0000, 4096, name="marlin")]
        with mock.patch.object(gpc, "resolve_range", side_effect=live.get):
            found = gpc.verify(recs)
        self.assertEqual([(f.record.name, f.verdict) for f in found],
                         [("pool_copy", "unmapped"), ("marlin", "range-changed")])

    def test_the_summary_names_kernel_verdict_and_address(self):
        f = gpc.Finding(_rec(0x2000_0000, 0x2000_0000, 4096, name="pool_copy"), "unmapped")
        line = gpc.summarize([f, f])[0]
        self.assertIn("unmapped n=2 kernel=pool_copy", line)
        self.assertIn("addr=0x20000000", line)

    def test_off_by_default_records_and_checks_nothing(self):
        with mock.patch.dict(gpc.os.environ, {}, clear=False):
            gpc.os.environ.pop(gpc.ENV_ENABLE, None)
            with mock.patch.object(gpc, "census_graph") as walk:
                gpc.record_capture(123, "full/1")
            walk.assert_not_called()
            self.assertEqual(gpc.verify_registry("wake"), 0)


if __name__ == "__main__":
    unittest.main()
