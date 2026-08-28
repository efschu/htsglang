"""#558: sharding, the legacy layout, and a disk that fills mid-assembly.

STATE OF THE TICKET, pinned rather than asserted. Sharding itself already
exists in this tree -- ``page_shard`` spreads stems over 256 hex directories
with a ``zz`` fallback, new writes are sharded, and ``_existing_path`` serves
pre-sharding files read-through. So these tests are not "does sharding work"
but the three things that were NOT covered and that the canonical store (#706)
makes sharper:

1. THE CANONICAL LAYOUT LANDS IN THE SAME SCHEME. ``.part706`` and
   ``.slots706`` are derived from the final path, so they inherit its shard and
   never accumulate in the store root; the TTL sweeper walks both layouts.
2. MIXED-LAYOUT AMBIGUITY IS REFUSED. A stem present in BOTH layouts gives one
   content-addressed key two candidate files, and the read path silently
   prefers the sharded one -- while the flat copy may predate the current key
   format entirely. Refused at attach, loudly.
3. A FULL DISK IS THE INVISIBLE-UNTIL-COMPLETE PROTOCOL'S WORST CASE, so it is
   tested at each failure point rather than reasoned about: no ``.bin`` may
   ever appear from a write that did not finish, whatever fails and whenever.

The ENOSPC tests inject the failure at the exact syscalls the protocol uses
(``pwrite``, the marker write, ``fsync``), because "the disk is full" is not one
event -- it is three, with different amounts of work already spent.
"""

import errno
import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache import canonical_page_store as store
from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import (
    OutOfSpace,
    marker_path,
    page_is_complete,
    part_path,
    window_for_layers,
    write_slice,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    MixedLayoutError,
    audit_layout,
    page_shard,
)
from sglang.test.test_utils import CustomTestCase

ATTN_LAYER_IDS = list(range(3, 64, 4))
CELL = 64
SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
PP_CUT = [(0, 28), (28, 48), (48, 64)]
IDENTITY = "0123456789abcdef"


def _window(lo, hi):
    return window_for_layers(
        SPEC, ATTN_LAYER_IDS, [i for i in ATTN_LAYER_IDS if lo <= i < hi]
    )


def _payload(window, tag=10):
    buf = bytearray()
    for slot in window.slots:
        buf += bytes([(tag + slot) % 256]) * window.cell_bytes
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


def _backend(root, *, pp_rank=0, window=None):
    return HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=pp_rank,
            pp_size=3,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="Qwen3.6-27B",
            model_identity_hash=IDENTITY,
            canonical_kv_page=window,
        ),
        file_path=root,
    )


class TestShardingCoversBothLayouts(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        store.reset_space_cache()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(store.reset_space_cache)

    def test_canonical_sidecars_live_in_the_shard_not_the_root(self):
        """A partial page must not accumulate in the store root: that root is
        exactly what the #558 incident filled with 11.7M entries."""
        stage = _backend(self.root, window=_window(*PP_CUT[0]))
        stage.set("cafe01", _payload(stage.canonical_kv_page))
        roots = [
            e
            for e in os.listdir(self.root)
            if os.path.isfile(os.path.join(self.root, e))
        ]
        self.assertEqual(roots, [])
        shard = os.path.join(self.root, "ca")
        names = sorted(os.listdir(shard))
        self.assertTrue(any(n.endswith(".part706") for n in names), names)
        self.assertTrue(any(n.endswith(".slots706") for n in names), names)

    def test_a_completed_page_leaves_only_the_sharded_bin(self):
        windows = [_window(*cut) for cut in PP_CUT]
        for rank, window in enumerate(windows):
            _backend(self.root, pp_rank=rank, window=window).set(
                "cafe01", _payload(window)
            )
        shard = os.path.join(self.root, "ca")
        names = sorted(os.listdir(shard))
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].endswith(".bin"))

    def test_the_sweeper_reaps_orphans_in_both_layouts(self):
        """Flat orphans exist on any store that predates sharding, so the
        sweeper walks the root as well as the shards."""
        old = 7200
        for rel in ("legacy.bin.part706", os.path.join("ca", "sharded.bin.part706")):
            path = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"\x00")
            stamp = os.stat(path).st_mtime - old
            os.utime(path, (stamp, stamp))
        self.assertEqual(store.sweep_partials(self.root, older_than_s=3600), 2)

    def test_shard_of_a_synthetic_stem_falls_back(self):
        self.assertEqual(page_shard("ZZtop"), "zz")
        self.assertEqual(page_shard("ab1234"), "ab")


class TestLegacyFindability(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_a_pre_sharding_page_is_still_found(self):
        """Read-through migration: nothing moves, and the old file keeps
        serving. Written flat by hand, exactly as a pre-#558 store holds it."""
        plain = _backend(self.root)
        stem = plain._get_suffixed_key("cafe01")
        value = torch.arange(32, dtype=torch.uint8)
        with open(os.path.join(self.root, f"{stem}.bin"), "wb") as f:
            f.write(value.numpy().tobytes())
        # A second backend on the same directory: this is what a restart does.
        reader = _backend(self.root)
        self.assertTrue(reader.exists("cafe01"))
        out = torch.zeros(32, dtype=torch.uint8)
        self.assertIsNotNone(reader.get("cafe01", out))
        self.assertTrue(torch.equal(out, value))

    def test_a_stem_in_both_layouts_is_refused_at_attach(self):
        """One content-addressed key, two candidate files. The read path would
        silently prefer the sharded one, and the flat copy may predate the
        current key format entirely."""
        plain = _backend(self.root)
        stem = plain._get_suffixed_key("cafe01")
        with open(os.path.join(self.root, f"{stem}.bin"), "wb") as f:
            f.write(b"flat")
        os.makedirs(os.path.join(self.root, page_shard(stem)), exist_ok=True)
        with open(os.path.join(self.root, page_shard(stem), f"{stem}.bin"), "wb") as f:
            f.write(b"sharded")
        self.assertEqual(audit_layout(self.root), [stem])
        with self.assertRaises(MixedLayoutError) as cm:
            _backend(self.root)
        self.assertIn("both the flat and the sharded layout", str(cm.exception))

    def test_a_clean_store_audits_empty(self):
        _backend(self.root).set("cafe01", torch.arange(8, dtype=torch.uint8))
        self.assertEqual(audit_layout(self.root), [])
        _backend(self.root)  # attaches without raising


class TestFullDiskDuringCommit(CustomTestCase):
    """A full disk is this protocol's worst case: a page becomes visible by
    rename, so a failure at the wrong moment could publish a page with holes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.page = os.path.join(self.root, "cafe.bin")
        self.window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        store.reset_space_cache()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(store.reset_space_cache)

    def _enospc(self, name):
        """Inject ENOSPC at one syscall, and hand back the undo.

        Returned rather than only registered for cleanup: one test needs the
        disk to come BACK, and calling doCleanups() to get there would also tear
        down the temporary directory the test is still using."""
        real = getattr(os, name)

        def boom(*a, **kw):
            raise OSError(errno.ENOSPC, "No space left on device")

        setattr(os, name, boom)
        self.addCleanup(setattr, os, name, real)
        return lambda: setattr(os, name, real)

    def test_no_page_appears_when_pwrite_fails(self):
        self._enospc("pwrite")
        with self.assertRaises(OSError):
            write_slice(self.page, self.window, _payload(self.window))
        self.assertFalse(page_is_complete(self.page))
        self.assertFalse(os.path.exists(marker_path(self.page)))

    def test_no_page_appears_when_fsync_fails(self):
        """The nastiest point: every byte is written and coverage is complete,
        so the very next step would publish. It must not."""
        self._enospc("fsync")
        with self.assertRaises(OSError):
            write_slice(self.page, self.window, _payload(self.window))
        self.assertFalse(page_is_complete(self.page))

    def test_a_failed_marker_write_does_not_claim_coverage(self):
        """Partial page, and the marker write is what fails. The next writer
        must not believe those slots are present."""
        windows = [_window(*cut) for cut in PP_CUT]
        real_open = open

        def failing_open(path, mode="r", *a, **kw):
            if str(path).endswith(".slots706") and "w" in mode:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_open(path, mode, *a, **kw)

        import builtins

        builtins.open = failing_open
        self.addCleanup(setattr, builtins, "open", real_open)
        with self.assertRaises(OSError):
            write_slice(self.page, windows[0], _payload(windows[0]))
        builtins.open = real_open
        self.assertFalse(page_is_complete(self.page))
        # Coverage was never claimed, so the page is treated as untouched.
        self.assertEqual(
            store.missing_slots(self.page, SPEC), tuple(range(SPEC.num_attn_layers))
        )

    def test_the_page_still_completes_after_the_disk_frees_up(self):
        """The store is a cache: a failed write must cost the write, not the
        key. Once space returns, the same page assembles normally."""
        restore = self._enospc("pwrite")
        with self.assertRaises(OSError):
            write_slice(self.page, self.window, _payload(self.window))
        restore()
        result = write_slice(self.page, self.window, _payload(self.window))
        self.assertTrue(result.completed)
        self.assertTrue(page_is_complete(self.page))

    def test_a_write_below_the_floor_is_refused_before_anything_is_created(self):
        """Refuse at the door: nothing is created, so there is nothing to reap
        and no other writer's work is wasted."""
        huge = 1 << 62
        with self.assertRaises(OutOfSpace):
            write_slice(
                self.page,
                self.window,
                _payload(self.window),
                space_check=lambda need: store.ensure_space(self.root, need, huge),
            )
        self.assertFalse(os.path.exists(part_path(self.page)))
        self.assertFalse(page_is_complete(self.page))

    def test_the_floor_is_off_by_default(self):
        store.ensure_space(self.root, 1 << 20, 0)  # no raise


class TestAttachKeepsPreviousBootsWork(CustomTestCase):
    """2026-08-28 boot-3 store wipe: the attach-time TTL sweep reaped all
    16898 partial files boot 2 deposited into /tmp/hicache_flip0828 -- the
    ENTIRE store, because nothing had completed yet and the next boot came
    100 minutes (> TTL 3600 s) later. Attach on a same-format store must
    delete nothing resumable; a real format transition must be loud, never a
    silent wipe."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        store.reset_space_cache()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(store.reset_space_cache)

    def _age_partials(self, age_s=7200.0):
        aged = []
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                if name.endswith((".part706", ".slots706")):
                    path = os.path.join(dirpath, name)
                    old = os.stat(path).st_mtime - age_s
                    os.utime(path, (old, old))
                    aged.append(path)
        return aged

    def test_attach_on_a_same_format_store_deletes_nothing(self):
        stage = _backend(self.root, window=_window(*PP_CUT[0]))
        stage.set("cafe01", _payload(_window(*PP_CUT[0])))
        aged = self._age_partials()
        self.assertEqual(len(aged), 2)
        # The next boot's first attach, same geometry:
        _backend(self.root, pp_rank=1, window=_window(*PP_CUT[1]))
        for path in aged:
            self.assertTrue(os.path.exists(path), path)
        # The retained work PAYS: the next boot's stages complete the page.
        for rank, cut in enumerate(PP_CUT[1:], start=1):
            _backend(self.root, pp_rank=rank, window=_window(*cut)).set(
                "cafe01", _payload(_window(*cut))
            )
        names = sorted(os.listdir(os.path.join(self.root, "ca")))
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].endswith(".bin"))

    def test_attach_under_a_new_format_is_loud_not_silent(self):
        stage = _backend(self.root, window=_window(*PP_CUT[0]))
        stage.set("cafe01", _payload(_window(*PP_CUT[0])))
        self._age_partials()
        other_spec = CanonicalPageSpec(
            num_attn_layers=len(ATTN_LAYER_IDS),
            kv_bytes_per_token_per_attn_layer=CELL * 2,
        )
        other = window_for_layers(
            other_spec, ATTN_LAYER_IDS, [i for i in ATTN_LAYER_IDS if i < 28]
        )
        with self.assertLogs(
            "sglang.srt.mem_cache.canonical_page_store", level="WARNING"
        ) as logs:
            _backend(self.root, window=other)
        self.assertIn("geometry", "\n".join(logs.output))

    def test_attach_never_touches_completed_pages(self):
        for rank, cut in enumerate(PP_CUT):
            _backend(self.root, pp_rank=rank, window=_window(*cut)).set(
                "cafe01", _payload(_window(*cut))
            )
        shard = os.path.join(self.root, "ca")
        names = [n for n in os.listdir(shard) if n.endswith(".bin")]
        self.assertEqual(len(names), 1)
        path = os.path.join(shard, names[0])
        old = os.stat(path).st_mtime - 999999
        os.utime(path, (old, old))
        _backend(self.root, window=_window(*PP_CUT[0]))
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
