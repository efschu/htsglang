"""PARK-RETRACT-SPLIT (02.10.): where the flip park's ``retract`` phase goes.

y7y (NF, 999b64781a) FLIPCYCLE stage=park: retract 34-277 ms per park
(p50 ~115), the largest phase of the park handler -- and the park handler is
flip time (D>P starts at D's last decode token). ``retract_all(retain=True)``
runs, per running request, ``release_kv_cache`` -> ``cache_finished_req``
(insert + the forced host write-through of every new node: arena claim /
drop, KV D2H enqueue, mamba state kernel, PLE side rows, #1442 hand-off) and
``reset_for_retract``. The logs only show the sum. This splits it, rank-local
and without a collective, into:

  release  -- each request's ``cache_finished_req`` (wall, per rid)
  backup   -- the ``write_backup`` calls inside it (wall, count)
  write    -- of that, the ``cache_controller.write`` calls (wall, count)
  other    -- retract minus release (reset_for_retract, hisparse, eviction)

One line per park on every rank: ``WEG2-PARK-RETRACT-SPLIT``. Only the wall
clock is read (no device sync), so the split costs nothing measurable. The
wrappers are instance attributes set for the duration of the retract and
removed in ``finally``; the class methods stay untouched.
Switch ``SGLANG_WEG2_PARK_RETRACT_SPLIT`` (default on: an instrument only).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_PARK_RETRACT_SPLIT.get())
    except Exception:  # noqa: BLE001 - an instrument switch never breaks the park
        return False


class RetractSplit:
    """Wall-clock split of one park's retraction (see module doc)."""

    def __init__(self) -> None:
        self.release_ms: Dict[str, float] = {}
        self.backup_ms = 0.0
        self.backup_n = 0
        self.write_ms = 0.0
        self.write_n = 0
        self.total_ms = 0.0
        self._patched: List[tuple] = []

    # -- wrappers -----------------------------------------------------------------
    def _wrap(self, obj: Any, name: str, on_done: Callable[[Any, tuple, float], None]) -> None:
        orig = getattr(obj, name, None)
        if obj is None or not callable(orig):
            return
        had_own = name in getattr(obj, "__dict__", {})

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                on_done(args, kwargs, (time.perf_counter() - t0) * 1000.0)

        try:
            setattr(obj, name, timed)
        except Exception:  # noqa: BLE001 - a slotted / frozen object is not split
            return
        self._patched.append((obj, name, orig, had_own))

    def _on_release(self, args: tuple, kwargs: dict, ms: float) -> None:
        req = args[0] if args else kwargs.get("req")
        rid = str(getattr(req, "rid", "?"))
        self.release_ms[rid] = self.release_ms.get(rid, 0.0) + ms

    def _on_backup(self, args: tuple, kwargs: dict, ms: float) -> None:
        self.backup_ms += ms
        self.backup_n += 1

    def _on_write(self, args: tuple, kwargs: dict, ms: float) -> None:
        self.write_ms += ms
        self.write_n += 1

    def arm(self, tree_cache: Any) -> None:
        if tree_cache is None:
            return
        self._wrap(tree_cache, "cache_finished_req", self._on_release)
        self._wrap(tree_cache, "write_backup", self._on_backup)
        self._wrap(getattr(tree_cache, "cache_controller", None), "write", self._on_write)

    def disarm(self) -> None:
        while self._patched:
            obj, name, orig, had_own = self._patched.pop()
            try:
                if had_own:
                    setattr(obj, name, orig)
                else:
                    delattr(obj, name)
            except Exception:  # noqa: BLE001
                logger.warning("PARK-RETRACT-SPLIT could not restore %s.%s", type(obj).__name__, name)

    # -- report -------------------------------------------------------------------
    def line(self, epoch: int) -> str:
        release = sum(self.release_ms.values())
        other = max(0.0, self.total_ms - release)
        per = ",".join("%s:%.0f" % (rid, ms) for rid, ms in self.release_ms.items())
        return (
            "WEG2-PARK-RETRACT-SPLIT epoch=%d n=%d retract_ms=%.0f release_ms=%.0f [%s] "
            "backup_ms=%.0f backup_n=%d write_ms=%.0f write_n=%d backup_rest_ms=%.0f "
            "release_rest_ms=%.0f other_ms=%.0f (release = cache_finished_req per rid; backup = "
            "write_backup inside it; write = cache_controller.write inside that; the rests are "
            "what each level adds; sum release + other = retract)"
            % (
                epoch, len(self.release_ms), self.total_ms, release, per,
                self.backup_ms, self.backup_n, self.write_ms, self.write_n,
                max(0.0, self.backup_ms - self.write_ms),
                max(0.0, release - self.backup_ms), other,
            )
        )


def run_split(tree_cache: Any, retract: Callable[[], Any], epoch: int) -> Any:
    """Run ``retract`` (the park's ``retract_all``) under the split when the
    switch is on; the result is ``retract``'s, untouched."""
    if not enabled():
        return retract()
    split = RetractSplit()
    split.arm(tree_cache)
    t0 = time.perf_counter()
    try:
        return retract()
    finally:
        split.total_ms = (time.perf_counter() - t0) * 1000.0
        split.disarm()
        try:
            logger.info(split.line(epoch))
        except Exception:  # noqa: BLE001 - the line never breaks the park
            pass
