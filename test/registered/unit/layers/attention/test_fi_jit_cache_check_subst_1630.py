"""#1630 substitute_gencode: flashinfer 0.7.0 build.ninja carries the gencode list in TWO
places (cuda_cflags continuation list + ``cuda_arch_flags =``). The b9j/b9l window test
(22:49Z) reported 'WOULD COMPILE [11/11]' for a current cached module because the old
substitution kept the first flag of the FILE and deleted every later one."""

import unittest

from sglang.srt.layers.attention import fi_jit_cache_check as J

G120A = "-gencode=arch=compute_120a,code=sm_120a"
G120F = "-gencode=arch=compute_120f,code=sm_120f"

NINJA_TWO_PLACES = (
    "cuda_cflags = -O3 $\n"
    "    %s $\n"
    "    -DNDEBUG\n"
    "ldflags = -shared\n"
    "cuda_arch_flags = %s\n"
    "build x.o: cuda x.cu\n"
) % (G120A, G120A)


class SubstituteGencode(unittest.TestCase):
    def test_same_flags_text_unchanged(self):
        self.assertEqual(J.substitute_gencode(NINJA_TWO_PLACES, [G120A]), NINJA_TWO_PLACES)

    def test_other_flags_replaced_in_both_places_once(self):
        text = NINJA_TWO_PLACES.replace(G120A, G120F)
        out = J.substitute_gencode(text, [G120A])
        self.assertEqual(out, NINJA_TWO_PLACES)
        self.assertEqual(out.count(G120A), 2)
        self.assertIn("cuda_arch_flags = %s\n" % G120A, out)

    def test_multi_arch_list_per_place(self):
        want = [G120A, "-gencode=arch=compute_86,code=sm_86"]
        text = (
            "cuda_cflags = -O3 $\n    %s $\n    %s $\n    -DNDEBUG\n"
            "cuda_arch_flags = %s %s\n" % (G120F, "-gencode=arch=compute_80,code=sm_80", G120F,
                                           "-gencode=arch=compute_80,code=sm_80")
        )
        out = J.substitute_gencode(text, want)
        self.assertEqual(out.count(want[0]), 2)
        self.assertEqual(out.count(want[1]), 2)
        self.assertNotIn(G120F, out)
        self.assertNotIn("compute_80", out)
        self.assertIn("cuda_arch_flags = %s %s\n" % tuple(want), out)
        self.assertNotIn(" \n", out.replace(" $\n", "$\n"))  # no stray trailing blanks
        self.assertNotIn("\n     $\n", out)  # no empty continuation line left behind

    def test_no_gencode_untouched(self):
        self.assertEqual(J.substitute_gencode("a = b\n", [G120A]), "a = b\n")


if __name__ == "__main__":
    unittest.main()
