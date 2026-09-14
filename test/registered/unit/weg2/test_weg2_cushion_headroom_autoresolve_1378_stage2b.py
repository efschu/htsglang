# SPDX-License-Identifier: Apache-2.0
"""#1378 Stage 2b -- ``choose_host_ledger`` self-reads the PRIOR boot's cushion
from the measured-record sidecar; ``--prior-cushion-min-gib``/
``--prior-bounce-gib`` stay an OVERRIDE, not the only path.

Stage 2 (test_weg2_cushion_headroom_gate_1378_stage2.py) wired the arithmetic
and the ARM-line print, but the two inputs were operator-typed only -- an
"INERT" gate, in the coordinator's own word, because nobody types two numbers
by hand before a real boot. This file closes THAT gap, with the two orders
that came with it:

1. FORM-KEY, never newest-by-mtime. #1308 named this exact shortcut ("the
   newest record regardless of shape") as the cause of two lost boots -- a
   ring-table default that picked an xchg-shadow boot's row. A candidate of
   a different ``form_key`` never wins here, no matter how recent or how low
   its cushion.
2. MINIMUM over the candidates of THIS form, never the newest. The direction
   is asymmetric: a prior cushion read too LOW tightens the next boot's gate
   (safe), read too HIGH loosens it (unsafe -- the #782 ratchet-from-the-
   free-column class, in the threshold direction). Taking the minimum is the
   one reduction a lucky boot cannot exploit.

Both live in :func:`host_ledger.resolve_prior_cushion` /
:func:`host_ledger.flip_ratchet_candidates`, called from
:func:`launcher.choose_host_ledger` -- ``choose()`` itself is UNCHANGED from
Stage 2 (it still just takes ``prior_cushion_min_gib``/``prior_bounce_gib``
and does not know where they came from).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher
from sglang.test.test_utils import CustomTestCase

GIB = hl.GIB
FORM = "wtags=3"


def _rec(boot_tag, cushion_min, bounce, at, form_key=FORM):
    return hl.flip_ratchet_record(
        pre_gib=1.0, post_gib=2.0, boot_tag=boot_tag, commit="a", at=at,
        form_key=form_key, cushion_min_gib=cushion_min, xchg_bounce_gib=bounce,
    )


def _sidecar(tmp, records):
    path = os.path.join(tmp, "rec.json")
    with open(path, "w") as f:
        json.dump({"samples": records}, f)
    return path


class ThePureResolverIsFormAndMinimumCorrect(CustomTestCase):
    """The two hard constraints, tested directly on
    :func:`host_ledger.resolve_prior_cushion` -- no fake host required."""

    def test_the_minimum_cushion_wins_not_the_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _sidecar(tmp, [
                _rec("newest_high", 20.00, 1.0, "2026-09-13T03:00:00Z"),
                _rec("oldest_lowest", 5.00, 9.0, "2026-09-13T00:00:00Z"),
                _rec("middle", 6.99, 7.79, "2026-09-13T01:00:00Z"),
            ])
            cushion, bounce, prov = hl.resolve_prior_cushion(path, FORM)
            self.assertAlmostEqual(cushion, 5.00)
            self.assertAlmostEqual(bounce, 9.0)
            self.assertIn("oldest_lowest", prov)
            self.assertIn("MINIMUM", prov)
            self.assertIn("never the newest", prov)

    def test_a_foreign_form_never_wins_even_with_a_lower_cushion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _sidecar(tmp, [
                _rec("foreign", -50.00, 1.0, "2026-09-13T02:00:00Z",
                     form_key="wtags=99"),
                _rec("this_form", 6.99, 7.79, "2026-09-13T01:00:00Z"),
            ])
            cushion, bounce, prov = hl.resolve_prior_cushion(path, FORM)
            self.assertAlmostEqual(cushion, 6.99)
            self.assertAlmostEqual(bounce, 7.79)
            self.assertNotIn("foreign", prov)

    def test_no_candidate_of_this_form_is_none_with_a_named_reason_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _sidecar(tmp, [_rec("other_form", 1.0, 1.0,
                                        "2026-09-13T01:00:00Z", form_key="wtags=7")])
            cushion, bounce, prov = hl.resolve_prior_cushion(path, FORM)
            self.assertIsNone(cushion)
            self.assertIsNone(bounce)
            self.assertIn(FORM, prov)
            self.assertIn("no measured FLIP record", prov)

    def test_a_missing_sidecar_file_is_none_with_a_named_reason(self):
        cushion, bounce, prov = hl.resolve_prior_cushion(
            "/nonexistent/rec.json", FORM)
        self.assertIsNone(cushion)
        self.assertIsNone(bounce)
        self.assertIn("/nonexistent/rec.json", prov)

    def test_a_candidate_missing_bounce_is_dropped_not_paired_with_a_stranger(self):
        """A record with cushion_min but no xchg_bounce_gib (an older write,
        or a ring arm whose ledger_arm never carried the term) must not be
        used with a DIFFERENT candidate's bounce -- both fields have to come
        from the same boot's own record, or neither is usable."""
        with tempfile.TemporaryDirectory() as tmp:
            incomplete = _rec("incomplete", 3.00, 7.79, "2026-09-13T00:30:00Z")
            incomplete["xchg_bounce_gib"] = None
            path = _sidecar(tmp, [incomplete,
                                   _rec("complete", 6.99, 7.79,
                                        "2026-09-13T01:00:00Z")])
            cushion, bounce, prov = hl.resolve_prior_cushion(path, FORM)
            self.assertAlmostEqual(cushion, 6.99, msg="the incomplete, lower "
                                    "candidate must be dropped, not chosen")
            self.assertIn("complete", prov)


class TheFrontWritesBothFieldsFromOneProducer(CustomTestCase):
    """Structural: `front.py` is the ONE writer of the FLIP record, and it
    must source `xchg_bounce_gib` from the same `ledger_arm` dict
    `dormant_image_sample` already reads it from -- not re-derive it."""

    def test_write_flip_ratchet_passes_ledger_arms_bounce(self):
        import inspect

        from sglang.srt.weg2 import front

        src = inspect.getsource(front.Front._write_flip_ratchet)
        self.assertIn("xchg_bounce_gib=self.ledger_arm.get(\"xchg_bounce_gib\")",
                      src)


class ChooseHostLedgerAutoResolvesEndToEnd(CustomTestCase):
    """Hermetic: fake /proc/meminfo + fake cgroup2 tree + a real sidecar file,
    driven through the real seam ``main`` calls -- not a hand-built kwarg."""

    MEMTOTAL_GIB = 123.78
    MEMAVAIL_GIB = 110.0

    def _fake_host(self, tmp, cg_current_gib=9.0, reclaim_extra_gib=3.0):
        meminfo = os.path.join(tmp, "meminfo")
        with open(meminfo, "w") as f:
            f.write(
                f"MemTotal:       {int(self.MEMTOTAL_GIB * GIB // 1024)} kB\n"
                f"MemFree:        1000000 kB\n"
                f"MemAvailable:   {int(self.MEMAVAIL_GIB * GIB // 1024)} kB\n"
                f"Shmem:          1048576 kB\n"
                f"SwapTotal:      0 kB\n"
            )
        cg = os.path.join(tmp, "cgroup")
        os.makedirs(cg, exist_ok=True)
        cur_b = int(cg_current_gib * GIB)
        shmem_b = int(1.0 * GIB)
        file_b = shmem_b + int(reclaim_extra_gib * GIB)
        with open(os.path.join(cg, "memory.current"), "w") as f:
            f.write(f"{cur_b}\n")
        with open(os.path.join(cg, "memory.peak"), "w") as f:
            f.write(f"{cur_b}\n")
        with open(os.path.join(cg, "memory.max"), "w") as f:
            f.write(f"{int(self.MEMTOTAL_GIB * GIB)}\n")
        with open(os.path.join(cg, "memory.events"), "w") as f:
            f.write("oom_kill 0\n")
        with open(os.path.join(cg, "memory.stat"), "w") as f:
            f.write(f"anon 1000000\nfile {file_b}\nshmem {shmem_b}\n"
                     "unevictable 0\nslab_reclaimable 0\n")
        return meminfo, cg

    def _choose(self, tmp, record_path, **kw):
        meminfo, cg = self._fake_host(tmp)
        return launcher.choose_host_ledger(
            int(12.0 * GIB), int(4.0 * GIB), meminfo_path=meminfo, cgroup_root=cg,
            record_path=record_path, pin_m_mib=150, s_gb_d=4, **kw,
        )

    def test_no_flags_no_form_key_is_the_pre_stage_2b_default(self):
        """A caller that never learned about this file at all (no
        `flip_ratchet_form_key`, no flags) must still get exactly Stage 2's
        behaviour: unmeasured, printed as such, never a refusal."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _sidecar(tmp, [_rec("prior", 6.99, 7.79,
                                        "2026-09-13T01:00:00Z")])
            arm, headroom, lines, _cg = self._choose(tmp, path)
            self.assertIsNotNone(arm)
            src_line = [l for l in lines
                        if l.startswith("WEG2-CUSHION-HEADROOM-SOURCE")][0]
            self.assertIn("no flip_ratchet_form_key", src_line)
            arm_line = [l for l in lines
                        if l.startswith("WEG2-HOST-LEDGER ARM")][0]
            self.assertIn("cushion_headroom=not measured", arm_line)

    def test_the_form_key_alone_auto_resolves_from_the_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _sidecar(tmp, [_rec("prior", 6.99, 7.79,
                                        "2026-09-13T01:00:00Z")])
            arm, headroom, lines, _cg = self._choose(
                tmp, path, flip_ratchet_form_key=FORM)
            self.assertIsNotNone(arm)
            src_line = [l for l in lines
                        if l.startswith("WEG2-CUSHION-HEADROOM-SOURCE")][0]
            self.assertIn("auto-resolved", src_line)
            self.assertIn("prior", src_line)
            arm_line = [l for l in lines
                        if l.startswith("WEG2-HOST-LEDGER ARM")][0]
            self.assertIn("cushion_headroom=", arm_line)
            self.assertNotIn("cushion_headroom=not measured", arm_line)

    def test_an_explicit_flag_overrides_the_sidecar_not_the_reverse(self):
        """The order's own wording: flags are an OVERRIDE. A sidecar minimum
        of 6.99 must NOT be what the arm sees when the operator cited a
        different number by hand."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _sidecar(tmp, [_rec("prior", 6.99, 7.79,
                                        "2026-09-13T01:00:00Z")])
            arm, headroom, lines, _cg = self._choose(
                tmp, path, flip_ratchet_form_key=FORM,
                prior_cushion_min_gib=1.23, prior_bounce_gib=4.56)
            self.assertIsNotNone(arm)
            src_line = [l for l in lines
                        if l.startswith("WEG2-CUSHION-HEADROOM-SOURCE")][0]
            self.assertIn("override", src_line)
            arm_line = [l for l in lines
                        if l.startswith("WEG2-HOST-LEDGER ARM")][0]
            self.assertIn("prior cushion_min=1.23", arm_line)
            self.assertNotIn("prior cushion_min=6.99", arm_line)

    def test_a_foreign_form_candidate_never_wins_through_the_real_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _sidecar(tmp, [
                _rec("foreign", -50.0, 1.0, "2026-09-13T02:00:00Z",
                     form_key="wtags=99"),
                _rec("this_form", 6.99, 7.79, "2026-09-13T01:00:00Z"),
            ])
            arm, headroom, lines, _cg = self._choose(
                tmp, path, flip_ratchet_form_key=FORM)
            self.assertIsNotNone(arm)
            arm_line = [l for l in lines
                        if l.startswith("WEG2-HOST-LEDGER ARM")][0]
            self.assertIn("prior cushion_min=6.99", arm_line,
                          "the foreign-form -50.0 candidate must never be read")

    def test_the_minimum_of_several_same_form_candidates_wins_through_the_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _sidecar(tmp, [
                _rec("newest_high", 20.00, 1.0, "2026-09-13T03:00:00Z"),
                _rec("oldest_lowest", 5.00, 9.0, "2026-09-13T00:00:00Z"),
                _rec("middle", 6.99, 7.79, "2026-09-13T01:00:00Z"),
            ])
            arm, headroom, lines, _cg = self._choose(
                tmp, path, flip_ratchet_form_key=FORM)
            self.assertIsNotNone(arm)
            arm_line = [l for l in lines
                        if l.startswith("WEG2-HOST-LEDGER ARM")][0]
            self.assertIn("prior cushion_min=5.00", arm_line,
                          "the MINIMUM must win, not the newest (20.00)")


class MainReadsItsOwnFormKeyForTheSeam(CustomTestCase):
    """Structural: `main` must build `flip_ratchet_form_key` in the EXACT
    literal shape `front.py` writes it in -- a different spelling would
    auto-resolve nothing, silently (form_key is a plain equality filter)."""

    def test_main_passes_the_same_wtags_shape_the_front_writes(self):
        import inspect

        from sglang.srt.weg2 import front

        main_src = inspect.getsource(launcher.main)
        self.assertIn('flip_ratchet_form_key=f"wtags={len(weights_tags)}"',
                      main_src)
        front_src = inspect.getsource(front.Front._write_flip_ratchet)
        self.assertIn('form_key=f"wtags={len(self.weights_tags)}"', front_src)


if __name__ == "__main__":
    unittest.main()
