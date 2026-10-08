"""AP L15-12c-F2: the keep-span arm step of the L15 retain sleep hook.

Finding F2 (L15-RETAIN-E2E-REVIEW): the hook armed the keep byte spans
per allocation base and RAISED on rc != 0 AFTER retain_at_sleep had
already MOVED the held rows, reset_keep'd the tree, re-armed the
allocators and WRITTEN this rank's manifest file. The raise left the
manifest on disk claiming a hold while the pause discarded the held
pages: at the wake the hold signal (manifest + cap > 0) kept rows whose
physical pages were gone -> garbage KV served as a prefix hit. Worse
than a rank split.

Contract of arm_keep_spans (the scheduler hook keeps it a one-line call):
on the FIRST failing base (rc != 0 or any adapter exception) this rank
holds NOTHING from here on --
  (a) this rank's manifest file is unlinked (discard_manifest), so the
      wake sees no hold signal: this rank votes None while its peers
      vote a fingerprint, a MIXED verdict that the existing group rule
      turns into a group fallback (no wake-side code lives here).
      F-B (L15-FBFIX): an unlink failure (OSError on the directory)
      must not leave the stale hold claim either -- the record is then
      rewritten IN PLACE with a dead-pid tombstone (invalidate_manifest)
      that the wake's read reaps as surely as a real discard;
  (b) the keep sets already armed on the OTHER bases of this rank are
      cleared (set_keep_byte_spans(base, ()), best effort -- a raising
      or rc-failing clear must not mask the original failure). F-C
      (L15-FBFIX): a failed clear is COUNTED and the FAILED line
      carries clears_failed=%d, so the line never silently claims
      "holds nothing" while a base may still pin pages;
  (c) exactly ONE failure line is logged, naming the manifest state
      (discarded / invalidated / NOT invalidated). If both unlink and
      the rewrite fail the line says NOT invalidated -- it still does
      not raise: flush_cache has no handler for a raise here (the L15
      SETUP try ends before the arm call, scheduler.py) and the
      scheduler dispatcher does not catch handler exceptions (document
      at weight_updater release_memory_occupation: -> SIGQUIT), so a
      raise would kill this rank mid-sleep and split the group;
and it returns False WITHOUT raising, so the scheduler finishes the
sleep as a plain flush on this rank (tree reset + pool clear, the
non-retain path -- the hook reuses that code by clearing its retain
result).
"""

import os

from flliper.srt.pdflip import l15_manifest


def discard_manifest(path: str) -> bool:
    """Best-effort unlink of this rank's manifest file (the per-(group,
    rank) path retain wrote). Already-gone counts as success; any other
    OSError returns False -- the caller then tries invalidate_manifest
    before naming the record NOT invalidated (F-B)."""
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


# Dead-pid tombstone (F-B): a structurally VALID Manifest whose pid is 0.
# l15_manifest._pid_alive treats pid <= 0 as dead WITHOUT signalling, so
# the wake's read() reaps this record on the spot -- removes the file and
# reports absence -- and load_for_wake returns None: the exact no-hold
# vote a real discard produces, with no wake-side code and no exception.
# Keep pid 0 (never a live pid: the wake would vote on the tombstone's
# content). Validated against l15_manifest.from_json at import so a schema
# change cannot silently turn the tombstone into a ValueError at the wake.
_TOMBSTONE = l15_manifest.to_json(l15_manifest.Manifest(
    epoch=0, pid=0, spans=(), rows_by_rank=(), anchor_slots=0)).encode()
l15_manifest.from_json(_TOMBSTONE.decode())  # import-time schema guard


def invalidate_manifest(path: str) -> bool:
    """F-B (L15-FBFIX): the unlink-failure fallback -- overwrite the
    record IN PLACE with the dead-pid tombstone.

    os.unlink needs write permission on the DIRECTORY; rewriting an
    existing file needs write permission on the FILE only, so this can
    succeed exactly where discard_manifest failed. The wake then reaps
    the tombstone (read() -> pid dead -> remove + None). Returns False
    only if the rewrite itself hit an OSError; nothing is raised here
    (see the module docstring contract c: a raise mid-arm would split
    the rank group)."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_CREAT, 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(_TOMBSTONE)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        return False
    return True


def align_keep_ranges(ranges, granule, limit=None):
    """L15-FIX-KEEPALIGN (N3k 02.10. 01:07Z): byte ranges the native
    tms_set_keep_spans accepts.

    core.cpp set_keep_spans returns -3 unless every range is granule-aligned
    (lo % g == 0, hi % g == 0), non-empty, inside the allocation and the list
    is sorted and disjoint. The hook collects ROW-exact ranges
    (row * stride * elsize), so each range is widened OUTWARD to the granule
    (lo floored, hi ceiled -- keeping a little more is safe, keeping less
    would drop held rows), capped at ``limit`` (the allocation size rounded
    up to the granule) and overlapping/adjacent ranges are merged.
    """
    g = int(granule)
    if g <= 0:
        raise ValueError("granule must be > 0, got %r" % (granule,))
    out = []
    for lo, hi in sorted((int(a), int(b)) for a, b in ranges):
        if hi <= lo:
            continue
        a = (lo // g) * g
        b = -(-hi // g) * g
        if limit is not None:
            b = min(b, int(limit))
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _base_limit(base, granule):
    """The base allocation's byte size rounded up to the granule, or None
    when the object does not tell (desk fakes)."""
    try:
        n = int(base.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return None
    g = int(granule)
    return -(-n // g) * g


def _granule_of(base):
    from flliper.srt.pdflip.d_seat_vram import granule_for

    return int(granule_for(getattr(base, "device", "cuda")))


def arm_keep_spans(adapter, keep_by_base, manifest_path, *, rank, log=print,
                   granule=None, split_lookup=None):
    """Arm one adapter call per allocation base; degrade on failure.

    keep_by_base: the hook's {base_key: (base_buffer, [(lo, hi), ...])}
    dict, armed in iteration order. Returns True when every base armed
    with rc == 0 (manifest untouched); on the first failure runs the
    a/b/c contract from the module docstring and returns False.
    """
    armed = []
    rc_repr = None
    raw_b = 0
    kept_b = 0
    n_rng = 0
    for _key, (_base, _ranges) in keep_by_base.items():
        try:
            _g = int(granule) if granule is not None else _granule_of(_base)
            _aligned = align_keep_ranges(_ranges, _g, _base_limit(_base, _g))
            if split_lookup is not None:
                # L15-FIX-KEEP-SPLIT: the pause keeps only WHOLE extents of a
                # span-mapped allocation; a stock base or a range outside the
                # split hold region cannot be kept -> refuse (clean fallback)
                _whole = split_lookup(_base, _aligned)
                if _whole is None:
                    rc_repr = "not-split-or-outside-hold"
                    break
                _aligned = _whole
            raw_b += sum(max(0, int(h) - int(l)) for l, h in _ranges)
            kept_b += sum(h - l for l, h in _aligned)
            n_rng += len(_aligned)
            rc = adapter.set_keep_byte_spans(_base, _aligned)
        except Exception as exc:  # noqa: BLE001 - any arm error = no hold
            rc_repr = f"raise:{type(exc).__name__}"
            break
        if rc != 0:
            rc_repr = str(rc)
            break
        armed.append(_base)
    else:
        log(f"L15-KEEP-ALIGN rank={rank} bases={len(armed)} ranges={n_rng} "
            f"row_bytes={raw_b} kept_bytes={kept_b} "
            f"extra_mib={(kept_b - raw_b) / 1048576:.1f}")
        return True

    # Failure path: this rank holds nothing (docstring contract a/b/c).
    if discard_manifest(manifest_path):
        m_state = "manifest discarded"
    elif invalidate_manifest(manifest_path):
        m_state = "manifest invalidated (unlink failed, tombstone written)"
    else:
        m_state = f"manifest NOT invalidated at {manifest_path!r}"
    clears_failed = 0
    for _base in armed:
        try:
            rc_clear = adapter.set_keep_byte_spans(_base, ())
            if rc_clear not in (0, None):
                clears_failed += 1
        except Exception:  # noqa: BLE001 - best effort, counted (F-C)
            clears_failed += 1
    log(
        f"L15-RETAIN keep-arm FAILED rank={rank} rc={rc_repr}: {m_state}, "
        f"clears_failed={clears_failed}, this rank holds nothing"
    )
    return False
