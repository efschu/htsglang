"""#1463/#1466: per-pass phase timers for the scheduler loops.

One tiny decorator, importable from every module that owns a phase of a
scheduler pass (request_receiver, scheduler, scheduler_pp_mixin) without a
circular import.  It records the wall time of ONE method call on the holder
(``setattr(self, attr, ms)``); the pass-level reader sums and resets.  A plain
function decorator: the #631 test family binds these methods onto a bare
SimpleNamespace one at a time, which a decorated function survives.
"""
import time


def timed(attr: str):
    def deco(fn):
        def wrapped(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(self, *a, **kw)
            finally:
                try:
                    setattr(self, attr, (time.perf_counter() - t0) * 1000.0)
                except Exception:  # noqa: BLE001
                    pass
        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        wrapped.__wrapped__ = fn
        return wrapped
    return deco


def read_ms(holder, attr: str) -> float:
    """The recorded ms (0.0 when never recorded) and reset it to 0."""
    try:
        v = float(getattr(holder, attr, 0.0) or 0.0)
        setattr(holder, attr, 0.0)
        return v
    except Exception:  # noqa: BLE001
        return 0.0
