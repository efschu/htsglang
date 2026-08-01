# SPDX-License-Identifier: Apache-2.0
"""Split-GGUF shard-set resolution (#391 blocker 2).

llama.cpp's ``gguf-split`` writes a large export as ``<stem>-00001-of-000NN.gguf``
and records ``split.no`` / ``split.count`` / ``split.tensors.count`` in EVERY
part. The parts are not equivalent:

* the FIRST part carries the full KV block (architecture, geometry, tokenizer) --
  and, for an unsloth DeepSeek V4 export, **zero tensors**;
* every later part carries tensors and a six-entry KV block that does not even
  include ``general.architecture``.

sglang's GGUF path was written for one file. Pointed at part 1 of such an export
it built a correct model skeleton and loaded nothing into it -- no error, no
warning, a fluent-nonsense server. Pointed at a later part it could not find the
architecture at all.

This module is the single place that turns "the path the user passed" into "the
ordered set of files that make up this checkpoint", so that every reader in the
loader agrees on the same answer:

* :func:`resolve_gguf_shard_paths` -- the ordered set, validated;
* :func:`gguf_metadata_path` -- the part whose KV block is authoritative;
* :func:`iter_gguf_tensors` -- one tensor stream over the whole set.

A file that is not part of a split set resolves to a one-element list, and every
call site then does exactly what it did before this module existed.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: ``gguf-split``'s filename convention. The index is 1-based on disk while
#: ``split.no`` in the KV block is 0-based.
_SHARD_FILENAME_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$"
)

#: abspath -> resolved ordered shard list. A checkpoint does not change under a
#: running server, and the loader asks four to six times per process (name map,
#: extra-tensor probe, unquantized prefixes, weight iterator, plus the config
#: peek and the sibling reconciliation).
_RESOLVED_CACHE: Dict[str, Tuple[str, ...]] = {}


def _read_split_kv(path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """``(split.no, split.count, split.tensors.count)`` of one file, as ints.

    Any field the file does not carry comes back as None; a plain unsplit GGUF
    carries none of them.
    """
    import gguf

    reader = gguf.GGUFReader(path, "r")

    def field(key: str) -> Optional[int]:
        entry = reader.fields.get(key)
        if entry is None:
            return None
        try:
            return int(entry.contents())
        except (TypeError, ValueError):
            return None

    return field("split.no"), field("split.count"), field("split.tensors.count")


def resolve_gguf_shard_paths(gguf_file: str) -> List[str]:
    """Every file of ``gguf_file``'s split set, in shard order.

    Returns ``[gguf_file]`` unchanged for a file that is not split -- that is
    the overwhelmingly common case and it costs one header read.

    ``gguf_file`` may be ANY part of the set, not just the first: ``split.count``
    is recorded in all of them, so a user who points at part 3 gets the same
    answer as one who points at part 1.

    Raises ``RuntimeError`` rather than returning a partial set: a split export
    that silently loses a shard loads an incomplete model, which is the failure
    this whole module exists to prevent.
    """
    abspath = os.path.abspath(gguf_file)
    cached = _RESOLVED_CACHE.get(abspath)
    if cached is not None:
        return list(cached)

    _split_no, count, _tensors = _read_split_kv(abspath)
    if count is None or count <= 1:
        _RESOLVED_CACHE[abspath] = (abspath,)
        return [abspath]

    directory, filename = os.path.split(abspath)
    match = _SHARD_FILENAME_RE.match(filename)
    if match is None:
        raise RuntimeError(
            f"GGUF {abspath} declares split.count={count}, so it is one part of "
            "a split export, but its name does not follow llama.cpp's "
            "'<stem>-00001-of-000NN.gguf' convention and the sibling parts "
            "cannot be located. Restore the original filenames, or merge the "
            "export with 'llama-gguf-split --merge'."
        )

    total_from_name = int(match.group("total"))
    if total_from_name != count:
        raise RuntimeError(
            f"GGUF {abspath} is inconsistent with itself: the filename says it "
            f"is one of {total_from_name} parts, the KV block says "
            f"split.count={count}. Refusing to guess which is right."
        )

    stem = match.group("stem")
    paths = [
        os.path.join(directory, f"{stem}-{i + 1:05d}-of-{count:05d}.gguf")
        for i in range(count)
    ]

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(
            f"GGUF split export {stem} declares {count} parts but "
            f"{len(missing)} are missing from {directory}: "
            f"{[os.path.basename(p) for p in missing]}. Loading the parts that "
            "are present would produce a model with silently missing weights."
        )

    _validate_shard_set(paths, count)
    _RESOLVED_CACHE[abspath] = tuple(paths)
    logger.info(
        "GGUF split export: resolved %d parts for %s", count, os.path.basename(abspath)
    )
    return paths


def _validate_shard_set(paths: Sequence[str], count: int) -> None:
    """Every resolved part must agree that it is part ``i`` of ``count``.

    Catches a directory that holds two different exports whose names happen to
    collide, and a part that was replaced by one from another quantization.
    """
    declared_tensor_counts = set()
    for index, path in enumerate(paths):
        split_no, split_count, tensors_total = _read_split_kv(path)
        if split_count != count or split_no != index:
            raise RuntimeError(
                f"GGUF split export: {os.path.basename(path)} should be part "
                f"{index + 1} of {count} but its KV block says part "
                f"{(split_no + 1) if split_no is not None else '?'} of "
                f"{split_count if split_count is not None else '?'}. The files "
                "in this directory are not one consistent export."
            )
        if tensors_total is not None:
            declared_tensor_counts.add(tensors_total)
    if len(declared_tensor_counts) > 1:
        raise RuntimeError(
            "GGUF split export: the parts disagree about the total tensor "
            f"count ({sorted(declared_tensor_counts)}). The files in this "
            "directory are not one consistent export."
        )


def declared_tensor_count(gguf_file: str) -> Optional[int]:
    """``split.tensors.count`` -- how many tensors the whole set should hold, or
    None for a file that does not declare it (every unsplit GGUF)."""
    _split_no, _count, tensors_total = _read_split_kv(gguf_file)
    return tensors_total


def gguf_metadata_path(gguf_file: str) -> str:
    """The part of ``gguf_file``'s set whose KV block is authoritative.

    Only the first part of a split export carries the architecture, geometry and
    tokenizer; the later parts carry six ``split.*`` entries and nothing else. A
    caller that wants metadata must read this path, whichever part it was handed.
    """
    return resolve_gguf_shard_paths(gguf_file)[0]


def iter_gguf_tensors(paths: Iterable[str]) -> Iterator:
    """One ``ReaderTensor`` stream over an ordered set of GGUF files.

    The readers are held for the life of the generator on purpose: a
    ``ReaderTensor``'s ``.data`` is a view into its reader's memory map, and the
    consumer reads it lazily.
    """
    import gguf

    readers = [gguf.GGUFReader(path, "r") for path in paths]
    for reader in readers:
        yield from reader.tensors


def gguf_tensor_names(gguf_file: str) -> set:
    """Union of the tensor names across ``gguf_file``'s whole split set."""
    return {
        str(tensor.name)
        for tensor in iter_gguf_tensors(resolve_gguf_shard_paths(gguf_file))
    }
