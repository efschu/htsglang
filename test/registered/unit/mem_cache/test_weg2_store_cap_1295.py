"""#1295: the HiCacheFile byte cap must bound the DIRECTORY, not one index.

MEASURED DEFECT (boot weg2sb5h, 57fef0ce6e, store
``/spinning/hicache-weg2/weg2sb5h``): the directory reached 40,429,487,490 B
apparent / 36,622,785,536 B allocated (37.65 / 34.11 GiB) against a configured
``max_size`` of 30,150,131,852 B (28.08 GiB) -- 1.34x the cap in the very unit
the cap is enforced in -- across 655,982 files, with **zero refusals of any
kind** on either group.

The unit was never the bug: ``_allocated_size`` has charged
``max(st_blocks * 512, st_size)`` since #410, and the boot's own log line says
``allocated bytes``. The POPULATION was. Weg 2 runs TWO eviction owners over
ONE directory -- group P's rank 0 and group D's rank 0 -- and:

* ``writer_count = 1 if shared_keys else max(1, tp_size)``
  (``hicache_storage.py``) divides the cap over the ranks INSIDE a group, an
  axis that is 1 for both Weg-2 groups, and never over the groups sharing the
  directory. Both owners therefore took the WHOLE 28.08 GiB
  (``P.log:507`` and ``D.log:933``, byte-identical cap on both);
* each owner's scan filter is its own group's suffix product
  (``_group_scan_suffixes``: P is tp1 x pp3, D is tp3 x pp1), and the two
  products are DISJOINT, so neither ``_total_bytes`` could ever see the
  overshoot;
* the combined enforceable ceiling was 2 x 28.08 = 56.16 GiB and the directory
  reached 67 % of it, which is why 0 refusals is the PREDICTED outcome rather
  than an anomaly.

This file is the desk-scale reproduction of that shape and the grading of the
fix: the cap is enforced against the bytes MEASURED in the directory
(``seen_bytes``, every ``.bin`` under it whatever its suffix, charged at
``_allocated_size``), the wake re-scan that refreshes that measurement is
WIRED rather than merely defined, an eviction run leaves a proof line at INFO,
and an evictor that cannot hold the cap refuses BY NAME instead of being
indistinguishable from "nothing needed evicting".

Hermetic: a fake store directory, no torch, no CUDA, no server.
"""

import ast
import logging
import os
import pathlib
import tempfile
import unittest

from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor
from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreIndexBlind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# Group P's and group D's suffix products are disjoint on the store of record
# (``_0_1``/``_3_x`` against ``_x_3``/``_1_0``). Two one-element sets reproduce
# that property without reproducing the geometry.
P_SFX = "_A"
D_SFX = "_B"


def _dir_allocated_bytes(path: str) -> int:
    """Every ``.bin`` under ``path``, in the evictor's own accounting unit.

    ``max(st_blocks * 512, st_size)`` -- the #410 instrument, so the assertion
    and the thing it grades are measured the same way. Summing apparent size
    here would grade a cap enforced in allocated bytes against a different
    number, which is the instrument confusion the record had to unpick.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            if not name.endswith(".bin"):
                continue
            st = os.stat(os.path.join(dirpath, name))
            blocks = getattr(st, "st_blocks", None)
            total += st.st_size if blocks is None else max(int(blocks) * 512, st.st_size)
    return total


def _owner(path: str, suffix: str, cap: int, **kw) -> LRUFileEvictor:
    """An elected eviction owner of ``path`` that indexes only ``suffix``."""
    return LRUFileEvictor(
        path,
        suffix,
        tp_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        writes_shared_keys=True,
        scan_suffixes=(suffix,),
        extra_config={"max_size": str(cap), "max_size_scope": "shared"},
        writer_count=1,
        **kw,
    )


def _write(path: str, stem: str, nbytes: int) -> None:
    with open(os.path.join(path, f"{stem}.bin"), "wb") as f:
        f.write(b"\x07" * nbytes)


class TestStoreCapBoundsTheDirectory(CustomTestCase):
    """The falsifier: two owners, one directory, one cap."""

    def test_two_owners_over_one_directory_hold_one_cap(self):
        """THE #1295 SHAPE AT DESK SCALE.

        Two elected owners over the same directory, each with the whole
        configured cap and a scan filter the other's files cannot match --
        exactly ``P.log:507`` and ``D.log:933``, which print the same
        ``cap=30150131852 B ... 1 writer ranks`` on both groups. Driven in
        ALTERNATING awake phases with a wake re-scan at each, because that is
        Weg 2's actual form: a sleeping group performs no writes, so the wake
        is the only moment either owner's reading of the directory can be
        corrected. Before the fix each drives its own index to the cap and the
        directory lands at ~2x it with no refusal from either.
        """
        page = 32768
        cap = 40 * page
        with tempfile.TemporaryDirectory() as d:
            owners = ((_owner(d, P_SFX, cap), P_SFX), (_owner(d, D_SFX, cap), D_SFX))
            written = 0
            for phase in range(6):
                owner, sfx = owners[phase % 2]
                # THE WAKE. On the pre-fix tree this same call refuses a warm
                # two-owner store outright (W8b grades the index, and a
                # sibling's legitimate files read as blindness) -- that is a
                # separate finding, pinned by
                # ``test_a_warm_two_owner_store_is_refused_today``. Swallowed so
                # this test grades the BYTES rather than that refusal; the
                # index is rebuilt under the lock before the gate is called,
                # so the rebuild survives either way.
                try:
                    owner.rescan()
                except Weg2StoreIndexBlind:
                    pass
                for _ in range(15):
                    written += 1
                    stem = f"{written:05d}{sfx}"
                    if owner.reserve(stem, page, key=stem):
                        _write(d, stem, page)
                        owner.commit(stem)
            on_disk = _dir_allocated_bytes(d)
            self.assertLessEqual(
                on_disk,
                cap,
                f"two eviction owners over one directory each spent the whole "
                f"cap: {on_disk} B on disk against max_size={cap} B "
                f"({on_disk / cap:.2f}x); "
                f"P index {owners[0][0].stats()['used_bytes']} B, "
                f"D index {owners[1][0].stats()['used_bytes']} B -- neither can "
                f"see the overshoot because their scan filters are disjoint",
            )

    def test_a_warm_two_owner_store_is_refused_today(self):
        """PINNED, NOT FIXED -- the owed half, so it is a decision not a mystery.

        An owner attaching to a directory whose files are all the SIBLING
        group's grades W8b at 0.0 % and raises on a store nothing is wrong
        with. It never fired on the boot of record only because that store was
        COLD (``existing=0 B (0 entries)`` on both groups), i.e. the gate had a
        zero denominator, and the store tag is per-boot today. Re-pointing W8b
        at the #1295 bounded term removes this refusal and was written, run and
        REVERTED: it turns three deliberate launch refusals into runtime
        degradation. The day the retention tier is reused, this is the line to
        read.
        """
        page = 4096
        with tempfile.TemporaryDirectory() as d:
            for i in range(20):
                _write(d, f"{i:04d}{D_SFX}", page)
            with self.assertRaises(Weg2StoreIndexBlind):
                _owner(d, P_SFX, 100 * page)

    def test_the_cap_counts_bytes_this_index_does_not_own(self):
        """The mechanism under the falsifier, isolated.

        A directory over the cap, part of it in a SIBLING owner's suffix. This
        owner's index holds only its own share, so before the fix
        ``_total_bytes`` is under the cap, nothing is evicted, and the
        directory stays over it -- the cap enforced against a subset of one
        filesystem. The own share is the majority here only so W8b (which
        still grades the INDEX, deliberately) does not refuse the fixture.
        """
        page = 32768
        cap = 20 * page
        with tempfile.TemporaryDirectory() as d:
            for i in range(14):
                _write(d, f"{i:04d}{P_SFX}", page)
            for i in range(10):
                _write(d, f"{i:04d}{D_SFX}", page)
            self.assertEqual(_dir_allocated_bytes(d), 24 * page)
            p_owner = _owner(d, P_SFX, cap)
            self.assertEqual(
                p_owner.stats()["foreign_bytes"],
                10 * page,
                "the sibling's bytes were not measured",
            )
            on_disk = _dir_allocated_bytes(d)
            self.assertLessEqual(
                on_disk,
                cap,
                f"attaching to a directory {24 * page} B over a {cap} B cap "
                f"evicted nothing: {on_disk} B still on disk, because this "
                f"index holds only its own "
                f"{p_owner.stats()['used_bytes']} B",
            )

    def test_a_single_owner_over_its_own_files_is_unchanged(self):
        """CAN-FAIL GUARD, and the backward-compatibility grade.

        Where every rank legitimately owns its own suffixed files
        (``writes_shared_keys=False``) the cap is already divided by
        ``writer_count`` and the other ranks' files are NOT this index's
        budget. Counting them would refuse every write on the default,
        non-Weg-2 path -- so the directory term must be inert there.
        """
        page = 4096
        cap = 40 * page
        with tempfile.TemporaryDirectory() as d:
            for i in range(40):
                _write(d, f"{i:04d}{D_SFX}", page)
            evictor = LRUFileEvictor(
                d,
                P_SFX,
                tp_rank=0,
                writes_shared_keys=False,
                extra_config={"max_size": str(cap), "max_size_scope": "shared"},
                writer_count=3,
            )
            self.assertEqual(
                evictor.stats()["directory_bytes"],
                evictor.stats()["used_bytes"],
                "a per-rank index must not be charged for its siblings' files",
            )
            self.assertTrue(
                evictor.reserve(f"0001{P_SFX}", page, key="own"),
                "the per-rank path refused a write it has always admitted",
            )


class TestStoreCapRefusalIsNamedAndCounted(CustomTestCase):
    """0 refusals must mean 'none happened', never 'none were emitted'."""

    def test_an_evictor_that_cannot_hold_the_cap_refuses_by_name(self):
        """A directory a sibling owner alone fills to the cap.

        Attaching evicts this owner's whole share (the honest answer to a
        directory at 2.1x its cap), after which nothing it can reclaim would
        fund another page. Before the fix the sibling's 10 pages are invisible,
        this owner sits at 9 pages under a 10-page cap, and the write is simply
        admitted.
        """
        page = 32768
        cap = 10 * page
        with tempfile.TemporaryDirectory() as d:
            for i in range(11):
                _write(d, f"{i:04d}{P_SFX}", page)
            for i in range(10):
                _write(d, f"{i:04d}{D_SFX}", page)
            p_owner = _owner(d, P_SFX, cap)
            with self.assertLogs(
                "sglang.srt.mem_cache.storage.file.lru_file_evictor", level="WARNING"
            ) as cm:
                self.assertFalse(p_owner.reserve(f"0001{P_SFX}", page, key="k"))
            self.assertTrue(
                any("CAP-UNHOLDABLE" in line for line in cm.output),
                f"the refusal was not named: {cm.output}",
            )
            self.assertEqual(
                p_owner.stats()["cap_refusals"],
                1,
                "a refusal that is not counted cannot be read off a boot log",
            )

    def test_an_eviction_run_leaves_a_proof_line_at_info(self):
        """``0 reclaimed lines`` was a log-level artefact, not a zero.

        The only eviction-success line was ``logger.debug`` while the boot ran
        at ``log_level='info'`` (``grep -c DEBUG P.log`` = 0), so the record
        could not say whether eviction had ever run at all.
        """
        page = 4096
        cap = 10 * page
        with tempfile.TemporaryDirectory() as d:
            evictor = _owner(d, P_SFX, cap)
            for i in range(10):
                stem = f"{i:04d}{P_SFX}"
                self.assertTrue(evictor.reserve(stem, page, key=stem))
                _write(d, stem, page)
                evictor.commit(stem)
            with self.assertLogs(
                "sglang.srt.mem_cache.storage.file.lru_file_evictor", level="INFO"
            ) as cm:
                stem = f"9999{P_SFX}"
                evictor.reserve(stem, page, key=stem)
            self.assertTrue(
                any("EVICTION-RUN" in line for line in cm.output),
                f"an eviction run left no proof at INFO: {cm.output}",
            )


class TestWakeRescanIsWired(CustomTestCase):
    """The designed mitigation existed and had zero callers."""

    def test_the_wake_path_reaches_rescan_eviction_index(self):
        """MATCHED TO THE ERROR CLASS, which an import smoke cannot see.

        ``HiCacheStorage.rescan_eviction_index`` is documented as the whole
        answer to two owners over one directory ("an index built once at boot
        is wrong after hours of the sibling's writes ... Each owner therefore
        re-scans when it wakes") and, at 57fef0ce6e,
        ``grep -rn rescan_eviction_index python/`` returned exactly ONE line:
        its own ``def``. Measured on metal: ``re-scanned at wake`` = 0 in both
        the P and the D log.

        GRADES REACHABILITY, NOT PRESENCE, and the difference is not academic:
        the first version of this test asserted only that SOME caller exists,
        and a mutant that deleted the call from ``resume_memory_occupation``
        SURVIVED it -- the helper still called the method, so a caller still
        existed while the wake no longer reached it. That is the same
        present-but-unwired trap as the defect, one level up. So the check is a
        call-graph walk over ``weight_updater.py`` from the wake entry point.

        LIMITATION, stated: the graph is keyed by bare function name within one
        module, so two same-named methods in that module would be conflated,
        and it follows ``self.x()`` / ``obj.x()`` by attribute name rather than
        by resolved type. It is a wiring check, not a type-accurate call graph.
        """
        root = pathlib.Path(__file__).resolve()
        while root.name != "test" and root.parent != root:
            root = root.parent
        module = (
            root.parent
            / "python/sglang/srt/managers/scheduler_components/weight_updater.py"
        )
        self.assertTrue(module.is_file(), f"could not locate {module}")
        tree = ast.parse(module.read_text(encoding="utf-8"))
        calls: dict = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            targets = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Attribute):
                        targets.add(fn.attr)
                    elif isinstance(fn, ast.Name):
                        targets.add(fn.id)
            calls.setdefault(node.name, set()).update(targets)

        entry = "resume_memory_occupation"
        self.assertIn(entry, calls, "the Weg-2 wake entry point moved or was renamed")
        seen, queue, path_found = {entry}, [entry], False
        while queue:
            cur = queue.pop()
            if "rescan_eviction_index" in calls.get(cur, ()):
                path_found = True
                break
            for nxt in calls.get(cur, ()):
                if nxt in calls and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        self.assertTrue(
            path_found,
            f"no call path from {entry}() reaches rescan_eviction_index() in "
            f"{module.name}: the wake re-scan is defined, documented as the "
            f"mitigation for two eviction owners over one directory, and the "
            f"wake does not run it. Walked {len(seen)} function(s).",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
