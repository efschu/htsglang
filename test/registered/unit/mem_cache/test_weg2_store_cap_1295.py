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
* each owner indexed only what IT had written since ITS OWN attach, and both
  attached COLD (``existing=0 B (0 entries)`` on both group logs), so neither
  ``_total_bytes`` could ever see the other's writes;
* the combined enforceable ceiling was 2 x 28.08 = 56.16 GiB and the directory
  reached 67 % of it, which is why 0 refusals is the PREDICTED outcome rather
  than an anomaly.

ROUND 2 CORRECTS THE THIRD BULLET OF THE FIRST CUT, which said the two groups'
scan suffixes are "DISJOINT". They are not. ``_build_key_suffixes`` drops
``_{tp_rank}_{tp_size}`` and ``_{pp_size}_{pp_rank}`` from ``kv_config_suffix``
exactly when ``canonical_kv_page is not None``, and both group logs of the boot
of record print ``#706 canonical KV page active: ... KV keys carry content only
(no tp/pp suffix)`` (P 10:32:14 PP0-2, D 10:33:08 TP0-2). So P and D SHARE the
suffix that carries the store's bulk (~94 % of it by the fork's own 940:60
page:draft ratio), and the fixtures below are built on the shared suffix rather
than on two private ones. What separates the two owners is not the suffix but
PROVENANCE -- which process wrote the file -- and that is what decides whether
this owner may unlink it.

This file is the desk-scale reproduction of that shape and the grading of the
fix: the cap is enforced against the bytes MEASURED in the directory (every
``.bin`` under it whatever its suffix, plus the staging files the ``.bin`` walk
cannot see, charged at ``_allocated_size``), an indexed page the SIBLING wrote
is counted and never unlinked (this store is Weg 2's handback carrier), the
wake re-scan that refreshes the measurement is WIRED rather than merely
defined, its blind verdict is VOTED at the group fence rather than raised on
one rank, the walk does not hold the store's write lock, an eviction run leaves
a proof line at INFO, and an evictor that cannot hold the cap refuses BY NAME.

Hermetic: a fake store directory, no torch collectives, no CUDA, no server.
"""

import ast
import logging
import os
import pathlib
import re
import tempfile
import types
import unittest

from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor
from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreIndexBlind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# THE SHARED #706 SUFFIX both groups write and both groups scan. One value,
# because that is the measured shape of the store of record; the geometry that
# produces it is graded in ``test_weg2_s5_store_carrier.py``.
SHARED = "_K"
# The group-private ``config_suffix`` tail, the ~6 % residue that does NOT
# cross groups. Kept so the two populations stay distinguishable in the
# fixtures the way they are distinguishable in ``stats()``.
P_PRIV = "_A"
D_PRIV = "_B"


def _dir_allocated_bytes(path: str, suffix: str = ".bin") -> int:
    """Files under ``path`` matching ``suffix``, in the evictor's own unit.

    ``max(st_blocks * 512, st_size)`` -- the #410 instrument, so the assertion
    and the thing it grades are measured the same way. Summing apparent size
    here would grade a cap enforced in allocated bytes against a different
    number, which is the instrument confusion the record had to unpick.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            if not name.endswith(suffix):
                continue
            st = os.stat(os.path.join(dirpath, name))
            blocks = getattr(st, "st_blocks", None)
            total += st.st_size if blocks is None else max(int(blocks) * 512, st.st_size)
    return total


def _owner(path: str, scan: tuple, cap: int, own_suffix: str = "", **kw):
    """An elected eviction owner of ``path`` scanning ``scan``."""
    return LRUFileEvictor(
        path,
        own_suffix or scan[0],
        tp_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        writes_shared_keys=True,
        scan_suffixes=scan,
        extra_config={"max_size": str(cap), "max_size_scope": "shared"},
        writer_count=1,
        **kw,
    )


def _write(path: str, stem: str, nbytes: int, ext: str = ".bin") -> None:
    with open(os.path.join(path, f"{stem}{ext}"), "wb") as f:
        f.write(b"\x07" * nbytes)


def _fill(owner, path: str, stems, page: int) -> int:
    """Reserve/write/commit each stem; return how many were admitted."""
    admitted = 0
    for stem in stems:
        if owner.reserve(stem, page, key=stem):
            _write(path, stem, page)
            owner.commit(stem)
            admitted += 1
    return admitted


def _weight_updater_source() -> tuple:
    """(path, source text) of ``weight_updater.py`` from the tree under test."""
    root = pathlib.Path(__file__).resolve()
    while root.name != "test" and root.parent != root:
        root = root.parent
    module = (
        root.parent
        / "python/sglang/srt/managers/scheduler_components/weight_updater.py"
    )
    assert module.is_file(), f"could not locate {module}"
    return module, module.read_text(encoding="utf-8")


class TestStoreCapBoundsTheDirectory(CustomTestCase):
    """The falsifier: two owners, one directory, one cap."""

    def test_two_owners_over_one_directory_hold_one_cap(self):
        """THE #1295 SHAPE AT DESK SCALE.

        Two elected owners over the same directory, each with the whole
        configured cap -- exactly ``P.log:507`` and ``D.log:933``, which print
        the same ``cap=30150131852 B ... 1 writer ranks`` on both groups --
        both attaching COLD, both writing under the shared #706 suffix. Driven
        in ALTERNATING awake phases with a wake re-scan at each, because that
        is Weg 2's actual form: a sleeping group performs no writes, so the
        wake is the only moment either owner's reading of the directory can be
        corrected. Before the fix each drives its own index to the cap and the
        directory lands at ~2x it with no refusal from either.

        NOTHING IS SWALLOWED HERE. The first cut of this test wrapped the wake
        in ``except Weg2StoreIndexBlind: pass`` -- which proved the bytes are
        bounded IF the wake survives, on a tree whose wake was built not to
        survive. The refusal fired only because that fixture gave the two
        owners DISJOINT suffixes, so each read the other's legitimate files as
        blindness. Under the canonical page they share the suffix, W8b reads
        ~100 %, and the shipped exception policy is graded on its own in
        ``TestTheWakeVerdictIsVotedNotRaised``.
        """
        page = 32768
        cap = 40 * page
        with tempfile.TemporaryDirectory() as d:
            owners = (
                (_owner(d, (SHARED, P_PRIV), cap, own_suffix=P_PRIV), "p"),
                (_owner(d, (SHARED, D_PRIV), cap, own_suffix=D_PRIV), "d"),
            )
            written = 0
            for phase in range(6):
                owner, tag = owners[phase % 2]
                owner.rescan()  # THE WAKE. No try/except: it must survive.
                stems = []
                for _ in range(15):
                    written += 1
                    stems.append(f"{tag}{written:05d}{SHARED}")
                _fill(owner, d, stems, page)
            on_disk = _dir_allocated_bytes(d)
            p_stats, d_stats = owners[0][0].stats(), owners[1][0].stats()
            self.assertLessEqual(
                on_disk,
                cap,
                f"two eviction owners over one directory each spent the whole "
                f"cap: {on_disk} B on disk against max_size={cap} B "
                f"({on_disk / cap:.2f}x); "
                f"P directory {p_stats['directory_bytes']} B, "
                f"D directory {d_stats['directory_bytes']} B",
            )

    def test_a_woken_owner_never_unlinks_the_siblings_handback_pages(self):
        """MUST_FIX 4: the fix must not destroy the carrier it bounds.

        Because the two groups SHARE the #706 suffix, a wake re-scan pulls the
        sibling's pages into this owner's ``_lru``, and
        ``_census_existing_files`` sorts oldest-``st_mtime``-first, so they
        sort FIRST. An unguarded ``_evict_locked`` therefore unlinks exactly
        the pages P wrote for D to load -- while P's own ``_total_bytes`` goes
        on counting them (a phantom cap on one side, a lost handback on the
        other). The boot of record already carries 13
        ``W53_Weg2StoreHandbackFailed`` WITHOUT this path.

        Here the sibling's pages are the OLDEST in the directory and the
        directory is driven well past the cap. Every one of them must still be
        on disk, and the cap must still hold -- by refusing this owner's writes
        by name, never by freeing the sibling's bytes.
        """
        page = 32768
        cap = 30 * page
        with tempfile.TemporaryDirectory() as d:
            # BOTH attach COLD, which is the boot of record's own shape
            # (``existing=0 B (0 entries)`` on P and on D). Attaching to a warm
            # directory ADOPTS what is there -- see
            # ``test_an_attaching_owner_adopts_what_is_already_there`` -- so the
            # sibling must write AFTER this owner has attached for its pages to
            # be attributable to it at all.
            sibling = _owner(d, (SHARED,), cap)
            me = _owner(d, (SHARED,), cap)
            sib_stems = [f"sib{i:04d}{SHARED}" for i in range(20)]
            self.assertEqual(_fill(sibling, d, sib_stems, page), 20)
            me.rescan()
            st = me.stats()
            self.assertEqual(
                st["foreign_indexed_bytes"],
                20 * page,
                "the sibling's pages were not attributed to the sibling",
            )
            self.assertEqual(
                st["reclaimable_bytes"],
                0,
                "an owner that can free nothing must not report reclaimable "
                "bytes (#715: never report as deliverable what the actuator "
                "cannot deliver)",
            )
            admitted = _fill(me, d, [f"mine{i:04d}{SHARED}" for i in range(20)], page)
            self.assertEqual(admitted, 20, "sanity: this owner had room of its own")
            for stem in sib_stems:
                self.assertTrue(
                    os.path.exists(os.path.join(d, f"{stem}.bin")),
                    f"{stem} -- the sibling group's handback page was unlinked "
                    f"by an owner that does not write it",
                )
            self.assertLessEqual(
                _dir_allocated_bytes(d),
                cap,
                "the cap did not hold once the sibling's pages became "
                "unevictable",
            )
            # The cap held by evicting THIS owner's own pages: it admitted 20
            # and keeps fewer, which is the only remaining way to be under a
            # 30-page cap with 20 of the sibling's pages untouchable.
            self.assertLess(
                me.stats()["reclaimable_bytes"],
                admitted * page,
                "nothing of this owner's own was evicted, so the cap was held "
                "by some other means -- check what was unlinked",
            )
            self.assertEqual(me.stats()["foreign_indexed_bytes"], 20 * page)

    def test_a_directory_the_sibling_alone_fills_starves_this_owner_by_name(self):
        """The bound this fix buys, and its PRICE, stated as a test.

        MUST_FIX 4 makes the sibling's pages untouchable, so a group that fills
        the directory on its own starves the other into named refusals instead
        of losing its own oldest pages. That is worse than one LRU over the
        directory ordered by ``st_mtime`` and better than 1.34x unbounded; the
        cross-owner order is the value fork this cut does not take, because
        under Weg 2 the pages at the head of that order are the handback the
        sibling has not loaded yet. What must NOT happen either way is a silent
        overshoot, so the starvation is graded: zero admitted, every sibling
        page still on disk, refusals counted and named.
        """
        page = 32768
        cap = 20 * page
        with tempfile.TemporaryDirectory() as d:
            sibling = _owner(d, (SHARED,), cap)
            me = _owner(d, (SHARED,), cap)
            sib_stems = [f"sib{i:04d}{SHARED}" for i in range(20)]
            self.assertEqual(_fill(sibling, d, sib_stems, page), 20)
            me.rescan()
            # LEVEL=INFO, and that is load-bearing: EVICTION-RUN is logger.info
            # (deliberately -- see test_an_eviction_run_leaves_a_proof_line...),
            # so a WARNING-level context cannot see it and the second assertion
            # below would pass vacuously. Measured: the mutant that drops
            # _foreign_indexed_bytes from the pre-check SURVIVED at WARNING and
            # dies here.
            with self.assertLogs(
                "sglang.srt.mem_cache.storage.file.lru_file_evictor", level="INFO"
            ) as cm:
                admitted = _fill(me, d, [f"m{i:04d}{SHARED}" for i in range(5)], page)
            self.assertEqual(admitted, 0, "an owner with nothing of its own wrote")
            self.assertTrue(any("CAP-UNHOLDABLE" in ln for ln in cm.output))
            # AND IT REFUSED WITHOUT TRYING. The pre-check exists so an owner
            # whose unreclaimable bytes already fill the cap does not walk its
            # whole index per refused page -- 655k entries at the size of the
            # store of record, once per page, for as long as the sibling holds
            # the directory. An eviction run that can free nothing still logs
            # EVICTION-RUN, so its ABSENCE here is the discriminator.
            self.assertFalse(
                any("EVICTION-RUN" in ln for ln in cm.output),
                f"the refusal ran an eviction pass that could not free a byte: "
                f"{[ln for ln in cm.output if 'EVICTION-RUN' in ln]}",
            )
            self.assertEqual(me.stats()["cap_refusals"], 5)
            for stem in sib_stems:
                self.assertTrue(os.path.exists(os.path.join(d, f"{stem}.bin")), stem)
            self.assertLessEqual(_dir_allocated_bytes(d), cap)

    def test_the_cap_counts_bytes_this_index_does_not_own(self):
        """The mechanism under the falsifier, isolated: the UNSCANNED residue.

        The ~6 % of the store that keeps a group-private ``config_suffix``.
        This owner's index cannot hold it at all, so before the fix
        ``_total_bytes`` is under the cap, nothing is evicted, and the
        directory stays over it -- the cap enforced against a subset of one
        filesystem. The own share is the majority here only so W8b (which
        still grades the INDEX, deliberately) does not refuse the fixture;
        the sibling-MAJORITY case is graded in
        ``test_a_woken_owner_never_unlinks_the_siblings_handback_pages``,
        where the shared suffix keeps coverage at 100 %.
        """
        page = 32768
        cap = 20 * page
        with tempfile.TemporaryDirectory() as d:
            for i in range(18):
                _write(d, f"{i:04d}{P_PRIV}", page)
            for i in range(10):
                _write(d, f"{i:04d}{D_PRIV}", page)
            self.assertEqual(_dir_allocated_bytes(d), 28 * page)
            p_owner = _owner(d, (P_PRIV,), cap)
            self.assertEqual(
                p_owner.stats()["foreign_bytes"],
                10 * page,
                "the sibling's bytes were not measured",
            )
            on_disk = _dir_allocated_bytes(d)
            self.assertLessEqual(
                on_disk,
                cap,
                f"attaching to a directory {28 * page} B over a {cap} B cap "
                f"evicted nothing: {on_disk} B still on disk, because this "
                f"index holds only its own "
                f"{p_owner.stats()['used_bytes']} B",
            )

    def test_staging_files_are_inside_the_cap(self):
        """SHOULD_FIX 8: ``.bin`` was never the whole directory.

        Every write passes through a ``<final>.tmp.<uuid>`` staging file and
        the canonical protocol leaves orphaned partials; both are reaped BY AGE
        AT ATTACH ONLY. Under Weg 2 the process attaches once and runs for
        hours, so between attaches that component sat on the cap's filesystem
        and outside ``seen_bytes`` entirely -- a second, independent answer to
        "can another component still exceed the cap": yes, this one.
        """
        page = 4096
        cap = 20 * page
        with tempfile.TemporaryDirectory() as d:
            for i in range(15):
                _write(d, f"orphan{i:04d}{SHARED}.bin.tmp.deadbeef", page, ext="")

            def _iter_staging():
                for name in os.listdir(d):
                    if not name.endswith(".bin"):
                        yield os.stat(os.path.join(d, name))

            owner = _owner(d, (SHARED,), cap, iter_staging=_iter_staging)
            st = owner.stats()
            self.assertEqual(
                st["staging_bytes"],
                15 * page,
                "staging/partial files are still outside the cap's population",
            )
            self.assertEqual(st["directory_bytes"], 15 * page)
            _fill(owner, d, [f"n{i:04d}{SHARED}" for i in range(20)], page)
            self.assertLessEqual(
                _dir_allocated_bytes(d, ".bin") + 15 * page,
                cap,
                f"the cap was held only over the .bin half: "
                f"{_dir_allocated_bytes(d, '.bin')} B of pages + {15 * page} B "
                f"of staging against max_size={cap} B",
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
                _write(d, f"{i:04d}{D_PRIV}", page)
            evictor = LRUFileEvictor(
                d,
                P_PRIV,
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
                evictor.reserve(f"0001{P_PRIV}", page, key="own"),
                "the per-rank path refused a write it has always admitted",
            )

    def test_an_attaching_owner_adopts_what_is_already_there(self):
        """The provenance rule's other half, so it cannot drift into a lease.

        At ATTACH there is no live sibling to attribute files to and adopting
        them is exactly the pre-#1295 behaviour of every single-owner store
        ("adopt whatever this rank already wrote while it was running
        unbounded"). Only a stem that appears BETWEEN this owner's censuses,
        without this owner writing it, is the sibling's. A rule that marked
        everything foreign at attach would make a warm single-owner restart
        unable to reclaim anything it wrote before the restart.
        """
        page = 4096
        cap = 40 * page
        with tempfile.TemporaryDirectory() as d:
            for i in range(10):
                _write(d, f"{i:04d}{SHARED}", page)
            owner = _owner(d, (SHARED,), cap)
            st = owner.stats()
            self.assertEqual(st["foreign_indexed_bytes"], 0)
            self.assertEqual(st["used_bytes"], 10 * page)
            self.assertEqual(st["reclaimable_bytes"], 10 * page)


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
            for i in range(20):
                _write(d, f"{i:04d}{P_PRIV}", page)
            for i in range(10):
                _write(d, f"{i:04d}{D_PRIV}", page)
            p_owner = _owner(d, (P_PRIV,), cap)
            with self.assertLogs(
                "sglang.srt.mem_cache.storage.file.lru_file_evictor", level="WARNING"
            ) as cm:
                self.assertFalse(p_owner.reserve(f"0001{P_PRIV}", page, key="k"))
            self.assertTrue(
                any("CAP-UNHOLDABLE" in line for line in cm.output),
                f"the refusal was not named: {cm.output}",
            )
            self.assertEqual(
                p_owner.stats()["cap_refusals"],
                1,
                "a refusal that is not counted cannot be read off a boot log",
            )

    def test_cap_and_min_free_refusals_are_two_populations(self):
        """SHOULD_FIX 7: the record's two zeros read as one number.

        ``min_free refusals 0`` and the cap's ``0 refusals`` were reported
        through different fields, one of which (``_refused_since_stop``) the
        watchdog ZEROES on recovery -- so a boot could refuse thousands of
        pages on the watermark and still read 0. Both counters exist, both are
        cumulative, and a cap refusal must not move the watermark counter.
        """
        page = 32768
        cap = 4 * page
        with tempfile.TemporaryDirectory() as d:
            for i in range(7):
                _write(d, f"{i:04d}{P_PRIV}", page)
            for i in range(4):
                _write(d, f"{i:04d}{D_PRIV}", page)
            owner = _owner(d, (P_PRIV,), cap)
            self.assertFalse(owner.reserve(f"x{P_PRIV}", page, key="x"))
            st = owner.stats()
            self.assertEqual(st["cap_refusals"], 1)
            self.assertEqual(
                st["min_free_refusals"],
                0,
                "a cap refusal was counted as a filesystem-watermark refusal",
            )

    def test_a_min_free_refusal_counts_on_the_min_free_counter(self):
        """The other direction of should_fix 7, so the pair cannot drift.

        The companion above proves a CAP refusal does not move the watermark
        counter. This proves the watermark refusal moves the watermark counter
        and not the cap one -- without it, folding the increment into
        ``_cap_refusals`` is invisible and the two populations silently become
        one again, which is the defect being fixed.

        Per-rank path (``writes_shared_keys=False``) deliberately: W8's launch
        gate refuses a shared-key store whose ``max_size + min_free`` exceeds
        the device, and the point here is the counter, not the gate.
        ``check_free_space`` is stubbed True so the refusal goes through the
        under-lock watermark check rather than the watchdog's latch -- both
        sites count, and this is the one a latch would hide.
        """
        page = 4096
        with tempfile.TemporaryDirectory() as d:
            evictor = LRUFileEvictor(
                d,
                P_PRIV,
                tp_rank=0,
                writes_shared_keys=False,
                extra_config={"max_size": str(100 * page)},
                writer_count=1,
            )
            evictor.set_limits(min_free_bytes=1 << 60)
            evictor.check_free_space = lambda: True
            self.assertFalse(evictor.reserve(f"z{P_PRIV}", page, key="z"))
            st = evictor.stats()
            self.assertEqual(
                st["min_free_refusals"],
                1,
                "a filesystem-watermark refusal was not counted as one",
            )
            self.assertEqual(
                st["cap_refusals"],
                0,
                "a watermark refusal was counted as a cap refusal: the two "
                "populations are one number again",
            )

    def test_the_backend_injects_the_staging_walk(self):
        """WIRED, not merely defined -- the trap that survived fix 1's sweep.

        ``_iter_staging_files`` on ``HiCacheFile`` is inert unless the backend
        hands it to the evictor. Every other test here injects its own walker,
        so all of them pass on a tree where the ctor call drops the keyword --
        present-but-unwired, the same class as #1295 itself and as the M8 that
        survived the first sweep. Graded at the construction site.
        """
        root = pathlib.Path(__file__).resolve()
        while root.name != "test" and root.parent != root:
            root = root.parent
        module = root.parent / "python/sglang/srt/mem_cache/hicache_storage.py"
        self.assertTrue(module.is_file(), f"could not locate {module}")
        source = module.read_text(encoding="utf-8")
        tree = ast.parse(source)
        ctors = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "LRUFileEvictor"
        ]
        self.assertTrue(ctors, "the backend no longer constructs an evictor")
        wired = any(
            kw.arg == "iter_staging"
            and isinstance(kw.value, ast.Attribute)
            and kw.value.attr == "_iter_staging_files"
            for c in ctors
            for kw in c.keywords
        )
        self.assertTrue(
            wired,
            "HiCacheFile does not hand _iter_staging_files to its evictor, so "
            "staging and partial bytes are outside the cap on the real backend "
            "however green the injected fixtures are",
        )
        self.assertIn(
            "def _iter_staging_files",
            source,
            "the staging walk itself is gone",
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
            evictor = _owner(d, (P_PRIV,), cap)
            for i in range(10):
                stem = f"{i:04d}{P_PRIV}"
                self.assertTrue(evictor.reserve(stem, page, key=stem))
                _write(d, stem, page)
                evictor.commit(stem)
            with self.assertLogs(
                "sglang.srt.mem_cache.storage.file.lru_file_evictor", level="INFO"
            ) as cm:
                stem = f"9999{P_PRIV}"
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
        module, source = _weight_updater_source()
        tree = ast.parse(source)
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

    def test_the_wake_walk_does_not_hold_the_store_write_lock(self):
        """MUST_FIX 5: 16 wakes per group x the whole store, under ``_lock``.

        ``_lock`` gates ``reserve``, ``commit``, ``abort``, ``touch`` and
        ``stats`` -- every store write. Holding it across a walk of 655,982
        files (~6.4 s INFERRED at that size from the #558 rate of 11.7 M
        entries in 114 s; an inference from a flat-directory measurement, not a
        measurement of this sharded walk) stalls the store for the duration,
        16 times per group by the boot's own ``WEG2-DORMANT cleared`` count,
        on a tree that already has a ``front_pyspy_FLIP-STALL`` artifact.

        Graded from INSIDE the walk: a non-blocking acquire of the very lock
        the writers need. ``threading.Lock`` is not reentrant, so this reads
        False on the pre-fix code, which held it.
        """
        page = 4096
        with tempfile.TemporaryDirectory() as d:
            for i in range(4):
                _write(d, f"{i:04d}{SHARED}", page)
            holder: dict = {}
            probed: list = []

            def _iter_existing():
                for name in sorted(os.listdir(d)):
                    if not name.endswith(".bin"):
                        continue
                    ev = holder.get("ev")
                    if ev is not None:
                        got = ev._lock.acquire(blocking=False)
                        probed.append(got)
                        if got:
                            ev._lock.release()
                    yield name[:-4], os.stat(os.path.join(d, name))

            ev = _owner(d, (SHARED,), 40 * page, iter_existing=_iter_existing)
            holder["ev"] = ev
            probed.clear()
            ev.rescan()
            self.assertTrue(probed, "the walk never ran, so nothing was graded")
            self.assertTrue(
                all(probed),
                f"the wake walk held the store write lock for "
                f"{probed.count(False)} of {len(probed)} files: every "
                f"reserve/commit/touch on this store blocks for the whole walk",
            )


class TestTheWakeVerdictIsVotedNotRaised(CustomTestCase):
    """MUST_FIX 1 + 2: the shipped exception policy, graded directly.

    Fix 2 shipped ``except Weg2StoreIndexBlind: raise`` under a docstring
    claiming a group-fatal escalation, and NO test covered that line -- the one
    test that ran it wrapped it in ``except ...: pass``.
    ``LRUFileEvictor.rescan`` returns early for a non-owner, so the raise could
    only ever fire on PP0 / TP0 while PP1-2 / TP1-2 cleared dormancy and woke
    on: the disagreement §0 forbids, shipped as the fix for it.
    """

    def _manager(self, backend):
        """A stand-in ``self`` carrying only what the method reads."""
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        controller = types.SimpleNamespace(storage_backend=backend)
        tree_cache = types.SimpleNamespace(cache_controller=controller)
        scheduler = types.SimpleNamespace(
            tree_cache=tree_cache, enable_hierarchical_cache=True
        )
        fake = types.SimpleNamespace(
            scheduler=scheduler, weg2_store_rescan_failure=""
        )
        return SchedulerWeightUpdaterManager._weg2_rescan_store_index, fake

    def test_a_blind_wake_records_the_verdict_and_does_not_raise(self):
        """The rank-local death must not happen; the verdict must survive."""

        class _Blind:
            def rescan_eviction_index(self):
                raise Weg2StoreIndexBlind("W8b: 0.0 % of the store is indexed")

        method, fake = self._manager(_Blind())
        method(fake)  # must NOT raise: a raise here kills the owner alone
        self.assertIn(
            "Weg2WakeRefused",
            fake.weg2_store_rescan_failure,
            "the blind verdict was neither raised nor recorded, so nothing "
            "downstream can stop the group",
        )
        self.assertIn("#1295", fake.weg2_store_rescan_failure)

    def test_an_unrelated_walk_error_leaves_the_wake_alone(self):
        """A stale index degrades the hit rate; refusing the wake is worse."""

        class _Broken:
            def rescan_eviction_index(self):
                raise OSError("EIO")

        method, fake = self._manager(_Broken())
        method(fake)
        self.assertEqual(
            fake.weg2_store_rescan_failure,
            "",
            "an OSError from the walk must not stop the group",
        )

    def test_a_clean_wake_records_nothing(self):
        class _Fine:
            def rescan_eviction_index(self):
                return {
                    "indexed_entries": 3,
                    "seen_entries": 3,
                    "indexed_bytes": 30,
                    "seen_bytes": 30,
                    "fraction": 1.0,
                    "foreign_bytes": 0,
                    "foreign_indexed_bytes": 10,
                    "staging_bytes": 0,
                }

        method, fake = self._manager(_Fine())
        method(fake)
        self.assertEqual(fake.weg2_store_rescan_failure, "")

    def test_the_recorded_verdict_reaches_the_resume_group_fence(self):
        """RECORDING IT IS HALF; the other half is that somebody votes it.

        ``_weg2_group_fence`` all-gathers an ``ok`` bit over the group's world
        cpu group and ANY False makes EVERY rank raise
        ``Weg2FlipRankDisagree`` (C15). That is the escalation, and it must be
        fed from the field the wake wrote -- otherwise the verdict is recorded
        into a variable nothing reads, which is the present-but-unwired class
        this whole ticket is an instance of.

        Source-level, deliberately: driving the real fence needs a live gloo
        group over three processes, which is a boot, not a desk test. What is
        graded here is that the resume leg reads the field, clears it, and
        hands it to the fence as the ok-bit -- plus a fallback raise for the
        world<=1 case the fence cannot answer.
        """
        _module, source = _weight_updater_source()
        tree = ast.parse(source)
        fn = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == "resume_memory_occupation"
            ),
            None,
        )
        self.assertIsNotNone(fn, "the Weg-2 wake entry point moved or was renamed")
        body = ast.get_source_segment(source, fn) or ""
        self.assertIn(
            "weg2_store_rescan_failure",
            body,
            "the resume leg never reads the field the wake writes",
        )
        fence_calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_weg2_group_fence"
        ]
        self.assertTrue(fence_calls, "the resume leg no longer joins a group fence")
        voted = False
        for call in fence_calls:
            for kw in call.keywords:
                if kw.arg == "ok" and "store_failure" in (
                    ast.get_source_segment(source, kw.value) or ""
                ):
                    voted = True
        self.assertTrue(
            voted,
            "the store verdict is recorded but never voted: the fence's ok-bit "
            "does not depend on it, so a blind index stops no rank",
        )
        self.assertTrue(
            re.search(r"raise Weg2WakeRefused\(\s*store_failure\s*\)", body),
            "no fallback for the case the fence cannot gather (world <= 1 / no "
            "cpu group), where a local raise IS the group-wide stop",
        )

    def test_the_wake_does_not_raise_the_blind_verdict_on_one_rank(self):
        """The mutant that would restore fix 2's defect, named.

        ``except Weg2StoreIndexBlind: raise`` inside the wake helper is the
        exact shape being removed; a bare re-raise there kills the eviction
        owner while its siblings wake on, whatever the docstring says.
        """
        _module, source = _weight_updater_source()
        tree = ast.parse(source)
        fn = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == "_weg2_rescan_store_index"
            ),
            None,
        )
        self.assertIsNotNone(fn, "the wake re-scan helper moved or was renamed")
        for handler in (n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)):
            names = ast.get_source_segment(source, handler.type) if handler.type else ""
            if "Weg2StoreIndexBlind" not in (names or ""):
                continue
            for stmt in handler.body:
                self.assertFalse(
                    isinstance(stmt, ast.Raise),
                    "the blind verdict is re-raised on the eviction owner's "
                    "rank; only that rank holds an index, so its siblings wake "
                    "on -- vote it at the group fence instead",
                )


class TestWarmStoreRefusalIsPinned(CustomTestCase):
    """The owed half, so it is a decision and not a mystery."""

    def test_a_warm_two_owner_store_is_refused_today(self):
        """PINNED, NOT FIXED.

        An owner attaching to a directory whose files carry a suffix its group
        does NOT scan grades W8b at 0.0 % and raises on a store nothing is
        wrong with. It never fired on the boot of record only because that
        store was COLD (``existing=0 B (0 entries)`` on both groups), i.e. the
        gate had a zero denominator, and the store tag is per-boot today.
        Re-pointing W8b at the #1295 bounded term removes this refusal and was
        written, run and REVERTED: it turns three deliberate launch refusals
        into runtime degradation. The day the retention tier is reused, this is
        the line to read.

        NOTE round 2: under the #706 canonical page the two groups SHARE the
        bulk suffix, so this fires on the group-private residue, not on the
        carrier. That narrows the exposure; it does not remove it.
        """
        page = 4096
        with tempfile.TemporaryDirectory() as d:
            for i in range(20):
                _write(d, f"{i:04d}{D_PRIV}", page)
            with self.assertRaises(Weg2StoreIndexBlind):
                _owner(d, (P_PRIV,), 100 * page)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
