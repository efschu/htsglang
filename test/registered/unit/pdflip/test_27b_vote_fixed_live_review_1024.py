# SPDX-License-Identifier: Apache-2.0
"""Review VOTE-FIXED-LIVE 24.10. (desk/27b-vote-fixed-live-1024 @ 1f193be2dd): review pins.

RED on 1f193be2dd (findings, the danger direction named per test):

* M1 (OOM): a relaunch under the same ``--tag`` (or a rank restart in the boot) writes the same store file
  ``<tag>.tp<r>.json`` and REPLACES the first launch's readings at its first counted pass -- rule 2 ("every reading
  counts upward") breaks, the store F falls.
* S1 (OOM): ONE malformed doc of the right schema makes ``aggregate`` raise; the launcher then drops the WHOLE store
  ("store unreadable") and a record the store had raised falls back to the static 238.
* S2 (OOM, latent while the torch cache cap is off on 27B D): under the P0 cap the vote's ``post`` is the CAPPED
  post (cap - reserved), the reading's ``end`` the driver free -- a reading taken under a binding cap is smaller than
  the driver drop by the gap and under-prices every later pass whose cap does not bind. The meter must read the
  driver post.

GREEN on 1f193be2dd (wiring the author's file only fakes; each kills a mutant that survived it, OOM direction):

* the width vote opens the meter's pass (``live_note_vote`` in ``width_vote``);
* the forward trim marks it trimmed (``live_note_trim`` in ``_before_extend_trim``);
* the draft forward lowers the open reading's end through the census hook (``vram_family_census``).
"""

import json
import math
import os
import sys
import types
import unittest
import unittest.mock as mock

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.srt.pdflip import vote_fixed_live as V  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_27b_vote_fixed_live_1024 as VFL  # noqa: E402  (the author's rig: _rank, _pass, _store, _doc)

START = VFL.START
MIB = float(1 << 20)


class _Cuda:
    """Driver free ``free`` MiB, torch reserved ``reserved`` MiB, never capturing."""

    def __init__(self, free, reserved=0.0):
        self.free = float(free)
        self.reserved = float(reserved)

    def mem_get_info(self, *a):
        return self.free * MIB, 20480 * MIB

    def memory_reserved(self, *a):
        return self.reserved * MIB

    def memory_allocated(self, *a):
        return 0.0

    def is_current_stream_capturing(self):
        return False

    def synchronize(self):
        return None

    def empty_cache(self):
        return None


class TheStoreNeverLosesAReading(CustomTestCase):
    def test_a_relaunch_under_the_same_tag_keeps_the_first_launchs_readings(self):
        """M1 (OOM): launch 1 reads cost 300 at 61 rows (F 282); launch 2 under the SAME tag (or the rank restarted)
        reads cost 50 -- the store's F must stay >= 282 (rule 2)."""
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            extra = {V.ENV_FORM: "f1", V.ENV_DIR: root, V.ENV_BOOT: "dkr27bbar1drei10241106",
                     V.ENV_CARDS: "RTX5090,RTX3080,RTX3080"}
            with VFL._rank(extra=extra):
                VFL._pass(1, 1000, 10, 700)            # first pass of the process: not counted
                VFL._pass(1, 1000, 61, 700)            # cost 300 -> F 281.1 -> 282, persisted
                first = V.aggregate(V.load_docs(root, "f1"), START).fixed_mib
                ET.reset_for_tests()                   # a new rank process, same launcher tag
                VFL._pass(1, 1000, 10, 700)
                VFL._pass(1, 1000, 10, 950)            # cost 50, counted 1 -> persisted
                after = V.aggregate(V.load_docs(root, "f1"), START).fixed_mib
        self.assertEqual(first, math.ceil(300 - START * 61))
        self.assertGreaterEqual(after, first)


class AMalformedDocIsSkippedNotTheStore(CustomTestCase):
    def test_one_malformed_doc_does_not_drop_the_raised_record(self):
        """S1 (OOM): the store raised INT8 drei to 262 (author's test); a second doc of the right schema with a
        malformed reading must be skipped on its own -- the record stays 262, not 238."""
        bad_docs = {
            "three-column reading": '{"schema": "%s", "boot": "b2", "rank": 0, "counted": 5, '
                                    '"readings": [[10, 20, 30]]}' % V.SCHEMA,
            "infinite cost": '{"schema": "%s", "boot": "b2", "rank": 0, "counted": 5, '
                             '"readings": [[10, Infinity]]}' % V.SCHEMA,
            "text count": '{"schema": "%s", "boot": "b2", "rank": 0, "counted": "many", '
                          '"readings": [[10, 20]]}' % V.SCHEMA,
        }
        for what, text in bad_docs.items():
            with self.subTest(what), VFL._store() as root:
                d = VFL._digest(VFL.DREI_MODEL, VFL.PRE._cards())
                VFL._doc(root, d, "b1", 1, {61: 280.0})           # 280 - 0.3091 x 61 = 261.1 -> 262
                with open(os.path.join(V.store_dir(root, d), "b2.tp0.json"), "w") as fh:
                    fh.write(text)
                vals, src = L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 3, VFL.PRE._cards(),
                                                         "/m/" + VFL.DREI_MODEL)
                self.assertEqual(vals, [262.0] * 3, src)


class TheMeterReadsTheDriverPost(CustomTestCase):
    def test_under_the_torch_cache_cap_the_reading_opens_at_the_driver_post(self):
        """S2 (OOM, latent): cap 1500, reserved 900 -> cap_post 600 under a driver free of 1000. The vote prices at
        600 (right), the meter's pass must open at the DRIVER post 1000 (its end is the driver free)."""
        with VFL._rank(), mock.patch.object(ET, "_torch_cap_mib", lambda rank: 1500.0):
            ET.width_vote(_Cuda(1000, reserved=900), 1, 4096, 64, True)
            self.assertIsNotNone(ET._CACHE["live"])
            self.assertEqual(ET._CACHE["live"]._pending[0], 1000.0)


class TheHooksFeedTheMeter(CustomTestCase):
    """Green on 1f193be2dd: the call sites the author's tests replace by direct ET.live_note_* calls."""

    def test_the_width_vote_opens_the_pass(self):
        with VFL._rank():
            ET.width_vote(_Cuda(1000), 1, 4096, 64, True)
            m = ET._CACHE["live"]
            self.assertIsNotNone(m)
            self.assertEqual(m._pending[:2], (1000.0, START))

    def test_the_forward_trim_marks_the_pass_trimmed(self):
        with VFL._rank():
            ET.width_vote(_Cuda(1000), 1, 4096, 64, True)
            worker = types.SimpleNamespace(tp_rank=1, is_draft_worker=False)
            batch = types.SimpleNamespace(
                forward_mode=types.SimpleNamespace(is_extend=lambda: True, is_target_verify=lambda: False),
                extend_num_tokens=61)
            with mock.patch.object(ET, "maybe_trim", lambda *a, **k: "PDFLIP-EXTEND-CACHE-TRIM fake"):
                ET._before_extend_trim(worker, batch)
            self.assertTrue(ET._CACHE["live"]._pending[2])

    def test_the_draft_forward_lowers_the_open_reading_through_the_census(self):
        from flliper.srt.model_executor import vram_family_census as VFC
        from flliper.srt.model_executor import vram_peak_window as VPW

        with VFL._rank(), \
                mock.patch.object(VFC, "_maybe_log_vram_peak_since_pools", lambda *a, **k: None), \
                mock.patch.object(VFC, "_capturing", lambda cuda: False), \
                mock.patch.object(VPW, "on_forward_end", lambda *a, **k: None):
            VFL._pass(1, 1000, 10, 700)                      # first pass: not counted
            ET.live_note_vote(1, 1000, START)
            ET.live_note_trim()
            ET.live_note_window("chunk", 10, 700.0)          # reading open, end 700
            draft = types.SimpleNamespace(is_draft_model_runner=True, tp_rank=1)
            VFC.maybe_log_vram_peak(draft, None, cuda=_Cuda(500))
            self.assertEqual(ET._CACHE["live"]._open[2], 500.0)
            ET.live_note_target_decode()
            self.assertEqual(ET.corridor_line_mib(1), 400.0 + math.ceil(500 - START * 10))


if __name__ == "__main__":
    unittest.main()
