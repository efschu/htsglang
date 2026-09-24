# SPDX-License-Identifier: Apache-2.0
"""27B line, xsn420 (24.09.): the sleep-leg gate's sidecar read, memoised on the file's stat.

`Front._sleep_leg_need_gib` runs synchronously in the flip path between
'WEG2-FLIP-ORDER' and 'gathered-legs'. It parsed the whole append-only sidecar
(1854 samples, 2.9 MB) on EVERY flip and, on the 27B line, judged every sample
by the record identity: the ORDER -> gathered-legs gap was 35-62 ms on every
flip of xsn420 against 11-20 ms on xsn407/xsn411.

The answer is a pure function of the file's bytes (and of the front's fixed
identity), so it is memoised on (dev, ino, size, mtime_ns). What must hold:
the memoised answer is the uncached answer, a changed file is read again, a
missing or unreadable file keeps its old answer and is never memoised.
"""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as fr
from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase


def _sample(group, tag, delta, at="2026-09-24T12:00:00Z"):
    return {"group": group, "boot_tag": tag, "at": at,
            "shmem_delta_gib": delta, "rss_shmem_gib": 40.0}


def _write(path, samples):
    with open(path, "w") as f:
        json.dump({"samples": samples}, f)


def _front(path, record_line=None):
    f_ = fr.Front.__new__(fr.Front)
    f_.measured_record = path
    if record_line is not None:
        f_.record_line = record_line
    return f_


def _uncached(path, group, accept=None):
    """The pre-memo computation, spelled out: read, then the same three verdicts."""
    rec = hl.read_measured_record(path, accept=accept)
    e = (rec or {}).get(str(group))
    if not isinstance(e, dict):
        return None, "absent"
    d = e.get("shmem_delta_gib")
    if d is None or float(d) <= 0:
        return None, "unmeasured"
    return float(d), f"record:{e.get('boot_tag', '?')}@{e.get('at', '?')}"


class _Identity:
    """A record identity double: accepts only the named boot tags."""

    def __init__(self, tags):
        self.tags = set(tags)
        self.calls = 0

    def accepts_sample(self, s):
        self.calls += 1
        return s.get("boot_tag") in self.tags


class TheMemoIsTheUncachedAnswer(CustomTestCase):
    def test_every_verdict_matches_the_uncached_read(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            cases = [
                ([], "D"),                                        # absent
                ([_sample("D", "a", None)], "D"),                 # unmeasured
                ([_sample("D", "a", 0.0)], "D"),                  # unmeasured (<= 0)
                ([_sample("D", "a", 4.32)], "D"),                 # measured
                ([_sample("P", "a", 2.5), _sample("D", "b", 3.1)], "P"),
            ]
            for samples, group in cases:
                _write(p, samples)
                f_ = _front(p)
                want = _uncached(p, group)
                self.assertEqual(f_._sleep_leg_need_gib(group), want)
                self.assertEqual(f_._sleep_leg_need_gib(group), want, "memo hit differs")

    def test_the_identity_is_applied_exactly_as_before(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            _write(p, [_sample("D", "nf", 9.9, at="2026-09-24T12:00:00Z"),
                       _sample("D", "x411", 4.2, at="2026-09-20T21:00:00Z")])
            ident = _Identity({"x411"})
            f_ = _front(p, record_line=ident)
            want = _uncached(p, "D", accept=_Identity({"x411"}).accepts_sample)
            self.assertEqual(want, (4.2, "record:x411@2026-09-20T21:00:00Z"))
            self.assertEqual(f_._sleep_leg_need_gib("D"), want)
            n = ident.calls
            self.assertEqual(f_._sleep_leg_need_gib("D"), want)
            self.assertEqual(ident.calls, n, "a memo hit judged the samples again")


class TheFlipPathReadsTheFileOncePerVersion(CustomTestCase):
    def test_unchanged_file_is_read_once(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            _write(p, [_sample("D", "a", 4.32)])
            f_ = _front(p)
            with mock.patch.object(hl, "read_measured_record",
                                   wraps=hl.read_measured_record) as rd:
                for _ in range(5):
                    f_._sleep_leg_need_gib("D")
                self.assertEqual(rd.call_count, 1)

    def test_groups_are_memoised_apart(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            _write(p, [_sample("D", "a", 4.32), _sample("P", "a", 1.5)])
            f_ = _front(p)
            self.assertEqual(f_._sleep_leg_need_gib("D")[0], 4.32)
            self.assertEqual(f_._sleep_leg_need_gib("P")[0], 1.5)
            self.assertEqual(f_._sleep_leg_need_gib("D")[0], 4.32)

    def test_an_append_is_read_again(self):
        """This front's own samples (first sleeps, epoch 2) and another boot's
        append change the file: the next gate must see the new newest entry."""
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            _write(p, [_sample("D", "a", 4.32, at="2026-09-24T12:00:00Z")])
            f_ = _front(p)
            self.assertEqual(f_._sleep_leg_need_gib("D")[0], 4.32)
            _write(p, [_sample("D", "a", 4.32, at="2026-09-24T12:00:00Z"),
                       _sample("D", "b", 5.5, at="2026-09-24T12:01:00Z")])
            self.assertEqual(f_._sleep_leg_need_gib("D"),
                             (5.5, "record:b@2026-09-24T12:01:00Z"))

    def test_a_same_size_rewrite_with_a_new_mtime_is_read_again(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            _write(p, [_sample("D", "a", 4.32)])
            f_ = _front(p)
            self.assertEqual(f_._sleep_leg_need_gib("D")[0], 4.32)
            _write(p, [_sample("D", "a", 4.33)])  # same length, new bytes
            st = os.stat(p)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            self.assertEqual(f_._sleep_leg_need_gib("D")[0], 4.33)


class NothingIsMemoisedThatWasNotRead(CustomTestCase):
    def test_a_missing_file_keeps_its_old_answer_and_is_not_memoised(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            f_ = _front(p)
            self.assertEqual(f_._sleep_leg_need_gib("D"), (None, "absent"))
            self.assertEqual(getattr(f_, "_sleep_leg_need_memo", {}), {})
            _write(p, [_sample("D", "a", 4.32)])
            self.assertEqual(f_._sleep_leg_need_gib("D")[0], 4.32)

    def test_an_unreadable_answer_is_not_memoised(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            _write(p, [_sample("D", "a", 4.32)])
            f_ = _front(p)
            with mock.patch.object(hl, "read_measured_record",
                                   side_effect=RuntimeError("boom")):
                self.assertEqual(f_._sleep_leg_need_gib("D"), (None, "unreadable"))
            self.assertEqual(f_._sleep_leg_need_gib("D")[0], 4.32)

    def test_a_file_that_changes_during_the_read_is_not_memoised(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            _write(p, [_sample("D", "a", 4.32)])
            f_ = _front(p)
            real = hl.read_measured_record

            def read_then_append(path, *a, **k):
                out = real(path, *a, **k)
                _write(path, [_sample("D", "a", 4.32), _sample("D", "b", 7.0)])
                return out

            with mock.patch.object(hl, "read_measured_record", side_effect=read_then_append):
                self.assertEqual(f_._sleep_leg_need_gib("D")[0], 4.32)
            self.assertEqual(getattr(f_, "_sleep_leg_need_memo", {}), {})
            self.assertEqual(f_._sleep_leg_need_gib("D")[0], 7.0)

    def test_no_sidecar_is_unchanged(self):
        f_ = _front("")
        self.assertEqual(f_._sleep_leg_need_gib("D"), (None, "no-sidecar"))


class TheSavingIsReal(CustomTestCase):
    def test_a_2000_sample_sidecar_is_parsed_once(self):
        """The xsn420 size class: the uncached read costs tens of ms, the memo
        hit a stat. Printed, and asserted only as a ratio (desk machines vary)."""
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rec.json")
            pad = "x" * 1400  # ~1.5 KB per sample, as the live sidecar
            _write(p, [dict(_sample("D" if i % 2 else "P", f"b{i}", 1.0 + i / 1e4),
                            note=pad) for i in range(2000)])
            f_ = _front(p)
            t0 = time.perf_counter()
            first = f_._sleep_leg_need_gib("D")
            t1 = time.perf_counter()
            for _ in range(20):
                self.assertEqual(f_._sleep_leg_need_gib("D"), first)
            t2 = time.perf_counter()
            cold_ms, hot_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3 / 20
            print(f"sidecar {os.path.getsize(p) / 1e6:.1f} MB: cold {cold_ms:.1f} ms, "
                  f"memo hit {hot_ms:.3f} ms")
            self.assertLess(hot_ms * 10, cold_ms)


if __name__ == "__main__":
    unittest.main()
