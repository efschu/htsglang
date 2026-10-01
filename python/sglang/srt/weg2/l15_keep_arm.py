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
      turns into a group fallback (no wake-side code lives here);
  (b) the keep sets already armed on the OTHER bases of this rank are
      cleared (set_keep_byte_spans(base, ()), best effort -- a raising
      or rc-failing clear must not mask the original failure);
  (c) exactly ONE failure line is logged;
and it returns False WITHOUT raising, so the scheduler finishes the
sleep as a plain flush on this rank (tree reset + pool clear, the
non-retain path -- the hook reuses that code by clearing its retain
result).
"""

import os


def discard_manifest(path: str) -> bool:
    """Best-effort unlink of this rank's manifest file (the per-(group,
    rank) path retain wrote). Already-gone counts as success; any other
    OSError returns False (the caller logs and still flushes plain --
    with the pools cleared there is nothing left to hold anyway)."""
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def arm_keep_spans(adapter, keep_by_base, manifest_path, *, rank, log=print):
    """Arm one adapter call per allocation base; degrade on failure.

    keep_by_base: the hook's {base_key: (base_buffer, [(lo, hi), ...])}
    dict, armed in iteration order. Returns True when every base armed
    with rc == 0 (manifest untouched); on the first failure runs the
    a/b/c contract from the module docstring and returns False.
    """
    armed = []
    rc_repr = None
    for _key, (_base, _ranges) in keep_by_base.items():
        try:
            rc = adapter.set_keep_byte_spans(_base, _ranges)
        except Exception as exc:  # noqa: BLE001 - any arm error = no hold
            rc_repr = f"raise:{type(exc).__name__}"
            break
        if rc != 0:
            rc_repr = str(rc)
            break
        armed.append(_base)
    else:
        return True

    # Failure path: this rank holds nothing (docstring contract a/b/c).
    if not discard_manifest(manifest_path):
        log(f"L15-RETAIN keep-arm FAILED rank={rank}: manifest unlink "
            f"failed at {manifest_path!r}")
    for _base in armed:
        try:
            adapter.set_keep_byte_spans(_base, ())
        except Exception:  # noqa: BLE001 - best effort, keep the reason
            pass
    log(
        f"L15-RETAIN keep-arm FAILED rank={rank} rc={rc_repr}: manifest "
        "discarded, this rank holds nothing"
    )
    return False
