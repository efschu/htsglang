"""1539 06b: in-tree stand-ins for the model directories under
``/spinning/llm_stuff/club-3090/models-cache/`` that are EMPTY on this box.

WHY.  1517 found seven weg2 test files red for one reason that has nothing to do
with the code they guard: the directories ``Qwen3.8-27B-INT8-gdncov-vocabembed``,
``Qwen3.8-27B-DFlash2-W8-lued`` (and the NF ones) are empty here (``ls -A`` shows
nothing, mtime 02.10. 13:22).  Every reader the tests reach (``config.json``,
the safetensors HEADERS) then fails with ``FileNotFoundError`` or a named W48 /
W108 / W163 refusal, which hides the launcher paths those tests pin.

WHAT THE FIXTURE IS.  ``fixtures/model_dirs_1539/<name>/`` holds the real
``config.json`` and the safetensors HEADERS of the real directory, verbatim and
without payload (the readers under test -- ``pp_cut.checkpoint_weight_terms``,
``dflash_pricing._safetensors_headers`` -- read nothing else).  They come from
the header snapshot the profil_s3_1003 task took of the real directories on
2026-10-03 (``PROVENANCE.json`` records sha256 per file and the proof that
mattered: summed tensor bytes + header bytes equal the on-disk shard bytes of
the real directory to the byte).  The NF directory (225,300 tensors, 31 MB of
headers) is kept as one xz archive of its 48 shard headers and unpacked once per
process into a private temp directory next to the ``config.json`` already in the
tree (``fixtures/profil_s3_1003/nextflash_int4mixed``, same sha256 as the
snapshot's).

THE LOST PATHS ARE NOT TOUCHED, AND THE TESTS KEEP THEIR PATH STRINGS.  The
empty directories stay as they are (an operator step).  The launcher compares
the string it was given (W161: ``--model`` against the census's recorded model;
W48 against the recorded boot argv), so a test that handed it a different path
would test something else.  :func:`overlay` therefore leaves every path string
alone and serves READS below a models-cache directory that is empty here from
the in-tree copy, while it is active.  A directory that has its own
``config.json`` (restored on the rig) is never overlaid: real data wins and the
overlay is a no-op for it.  Writes, and every path outside the three directories,
go to the real functions untouched.
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "fixtures", "model_dirs_1539")
CACHE = "/spinning/llm_stuff/club-3090/models-cache/"

VOCABEMBED = "Qwen3.8-27B-INT8-gdncov-vocabembed"
DFLASH_W8 = "Qwen3.8-27B-DFlash2-W8-lued"
NF_MINACHIST = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"

_NF_CONFIG_DIR = os.path.join(HERE, "fixtures", "profil_s3_1003", "nextflash_int4mixed")
_NF_HEADERS = os.path.join(ROOT, NF_MINACHIST + ".headers.tar.xz")
_NF_UNPACKED = []


def _unpacked_nf() -> str:
    """config.json (in-tree copy) + the 48 shard headers, unpacked once per process."""
    if not _NF_UNPACKED:
        d = tempfile.mkdtemp(prefix="model-dirs-1539-nf-")
        atexit.register(shutil.rmtree, d, ignore_errors=True)
        shutil.copy(os.path.join(_NF_CONFIG_DIR, "config.json"), os.path.join(d, "config.json"))
        with tarfile.open(_NF_HEADERS, "r:xz") as tf:
            tf.extractall(d, filter="data")
        _NF_UNPACKED.append(d)
    return _NF_UNPACKED[0]


def _stand_in(leaf: str) -> str:
    """The in-tree directory that stands in for ``models-cache/<leaf>``."""
    if leaf == NF_MINACHIST:
        return _unpacked_nf()
    return os.path.join(ROOT, leaf)


_LEAVES = (VOCABEMBED, DFLASH_W8, NF_MINACHIST)


def _active_leaves() -> dict:
    """models-cache leaf -> stand-in, for the leaves EMPTY on this box."""
    return {
        leaf: _stand_in(leaf) for leaf in _LEAVES
        if not os.path.exists(os.path.join(CACHE, leaf, "config.json"))
    }


def path(name: str) -> str:
    """The in-tree directory standing in for ``models-cache/<name>``."""
    return _stand_in(name)


def sha256_of(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def provenance() -> dict:
    with open(os.path.join(ROOT, "PROVENANCE.json"), encoding="utf-8") as fh:
        return json.load(fh)


@contextlib.contextmanager
def overlay():
    """Serve READS under the empty models-cache directories from the fixtures.

    Patched (and restored on exit): ``builtins.open`` / ``io.open`` (read modes
    only), ``glob.glob`` (results are mapped back to the models-cache spelling),
    ``os.listdir``, ``os.stat``, ``os.path.exists`` / ``isdir`` / ``isfile`` /
    ``getsize``.  A path is rewritten only when it is the directory itself or
    lies below it AND the directory is empty here (no ``config.json``).
    """
    import builtins
    import glob
    import io

    leaves = _active_leaves()
    if not leaves:
        yield
        return

    def mapped(p):
        if isinstance(p, os.PathLike):
            p = os.fspath(p)
        if isinstance(p, str) and p.startswith(CACHE):
            leaf, sep, rest = p[len(CACHE):].partition("/")
            stand_in = leaves.get(leaf)
            if stand_in is not None:
                return stand_in + sep + rest
        return p

    def unmapped(p):
        for leaf, stand_in in leaves.items():
            if p == stand_in or p.startswith(stand_in + "/"):
                return CACHE + leaf + p[len(stand_in):]
        return p

    real_open, real_io_open = builtins.open, io.open
    real_glob, real_listdir, real_stat = glob.glob, os.listdir, os.stat
    real_exists, real_isdir = os.path.exists, os.path.isdir
    real_isfile, real_getsize = os.path.isfile, os.path.getsize

    def _open(file, mode="r", *a, **kw):
        if not any(c in mode for c in "wax+"):
            file = mapped(file)
        return real_open(file, mode, *a, **kw)

    def _io_open(file, mode="r", *a, **kw):
        if not any(c in mode for c in "wax+"):
            file = mapped(file)
        return real_io_open(file, mode, *a, **kw)

    def _glob(pathname, *a, **kw):
        out = real_glob(mapped(pathname), *a, **kw)
        return [unmapped(x) for x in out] if mapped(pathname) != pathname else out

    saved = [
        (builtins, "open", real_open), (io, "open", real_io_open),
        (glob, "glob", real_glob), (os, "listdir", real_listdir), (os, "stat", real_stat),
        (os.path, "exists", real_exists), (os.path, "isdir", real_isdir),
        (os.path, "isfile", real_isfile), (os.path, "getsize", real_getsize),
    ]
    builtins.open, io.open, glob.glob = _open, _io_open, _glob
    os.listdir = lambda p=".", *a, **kw: real_listdir(mapped(p), *a, **kw)
    os.stat = lambda p, *a, **kw: real_stat(mapped(p), *a, **kw)
    os.path.exists = lambda p: real_exists(mapped(p))
    os.path.isdir = lambda p: real_isdir(mapped(p))
    os.path.isfile = lambda p: real_isfile(mapped(p))
    os.path.getsize = lambda p: real_getsize(mapped(p))
    try:
        yield
    finally:
        for mod, attr, orig in reversed(saved):
            setattr(mod, attr, orig)
