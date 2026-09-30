# SPDX-License-Identifier: Apache-2.0
"""Item-size resolution for the #286 offload register.

The CPU-phase adapters booked graph rungs and drafter heads with 0 bytes;
this module supplies the real figures from the REGISTERED OBJECTS themselves
so that, once the code runs on GPU, the true number appears WITHOUT further
changes. The resolver is injectable: hermetic CPU tests feed fake tensors
(anything exposing ``numel()``/``element_size()``, including real torch
tensors on the ``meta`` or ``cpu`` device -- torch.cuda is never touched).

Recognized sources, in evaluation order:

* ``None``                       -> 0
* ``int``                        -> the value itself
* zero-arg callable              -> resolve its return value (lets adapters
                                    register a LIVE source that is still
                                    empty at registration time and refreshed
                                    later via ``refresh_sizes``)
* tensor-like                    -> ``numel() * element_size()``
* ``footprint_bytes`` attribute  -> the attribute (covers the #93
                                    ``_StateRecord`` of the tagged capture
                                    pools: measured paused bytes when
                                    available, else noted-tensor bytes)
* module-like                    -> sum over ``parameters()`` + ``buffers()``
* ``.model`` attribute           -> recurse (covers ModelRunner-shaped
                                    objects: the drafter head's runner)
* mapping                        -> sum over the values
* list/tuple/set/frozenset       -> sum over the elements

Anything unrecognized resolves to 0 -- an honest "unknown", never a guess.
The traversal is identity-deduplicating: an object reached twice (shared /
aliased tensors, cycles) contributes its bytes ONCE -- one allocation, one
count.
"""

from __future__ import annotations

from typing import Optional, Set

__all__ = ["resolve_size_bytes"]


def _tensor_nbytes(obj) -> Optional[int]:
    numel = getattr(obj, "numel", None)
    element_size = getattr(obj, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel()) * int(element_size())
        except Exception:
            return None
    return None


def resolve_size_bytes(source, _seen: Optional[Set[int]] = None) -> int:
    """Best-effort byte size of ``source`` (see module docstring). Cycle-safe
    and exception-safe: a source that cannot be measured contributes 0."""
    if source is None:
        return 0
    if isinstance(source, bool):
        return 0
    if isinstance(source, int):
        return max(0, source)
    if _seen is None:
        _seen = set()
    key = id(source)
    if key in _seen:
        return 0
    _seen.add(key)

    if callable(source) and not hasattr(source, "parameters"):
        try:
            return resolve_size_bytes(source(), _seen)
        except Exception:
            return 0

    nbytes = _tensor_nbytes(source)
    if nbytes is not None:
        return nbytes

    footprint = getattr(source, "footprint_bytes", None)
    if isinstance(footprint, int):
        return max(0, footprint)

    parameters = getattr(source, "parameters", None)
    if callable(parameters):
        total = 0
        try:
            for t in parameters():
                total += resolve_size_bytes(t, _seen)
            buffers = getattr(source, "buffers", None)
            if callable(buffers):
                for t in buffers():
                    total += resolve_size_bytes(t, _seen)
        except Exception:
            return 0
        return total

    model = getattr(source, "model", None)
    if model is not None:
        return resolve_size_bytes(model, _seen)

    if isinstance(source, dict):
        return sum(resolve_size_bytes(v, _seen) for v in source.values())
    if isinstance(source, (list, tuple, set, frozenset)):
        return sum(resolve_size_bytes(v, _seen) for v in source)

    return 0
