"""Transition shims for the package rename and the flip-subsystem rename (RENAME_PLAN 8.7 step 2).

Six things keep working across the ONE mechanical rename commit, in both directions:

* HTTP routes: every route under the old flip-front prefix also answers under the new one and
  vice versa (:func:`alias_aiohttp_routes`, :func:`alias_fastapi_routes`). Arms, probes, health
  checks and the README call the old state route; the renamed front registers the new one.
* The rig state directory ``~/.cache/<package>``: when the directory of the running package is
  missing but the one of the other name exists, the running name becomes a symlink to it
  (:func:`link_legacy_cache_dir`), and :func:`cache_file` reads a missing file from the other
  directory. Measured break without it: ``card_library.json`` -> "PRICE UNPRICED" (RENAME_PLAN 4.2).
* The operator directory under the gpu-arb root (boot state json, admin key, calibration, corridor
  sample): a host path, never renamed (:func:`operator_dir`).
* /dev/shm residue of boots under the other name: the launcher's sweep also lists the other
  spelling of its own name families (:func:`name_counterparts`).
* launcher flags: ``--<old token>-x`` and ``--<new token>-x`` both reach the running parser
  (:func:`canonical_flags`), so profiles and arms need not switch in the same step.
* processes of the other generation: the launcher's live-server census accepts both module
  names in argv (:func:`name_variants`) and both spellings of the boot token in another
  process's environment (:func:`env_name_variants`).

OLD AND NEW NAMES IN THIS FILE ARE WRITTEN SPLIT ON PURPOSE, and its prose names neither. The
mechanical pass (rename_to_flliper.py apply) rewrites every old token it sees; a literal pair
here would come out of it as new <-> new and the shim would be dead without any test turning
red. Which side is "running" is read from ``__name__`` at run time, so the same file is correct
before and after the rename (test_compat_shims.py runs this file through the rename rules).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: package names: the one this file was imported under, and the other one
_PKG_OLD = "sg" "lang"
_PKG_NEW = "fl" "liper"
#: route prefixes of the P/D-flip front (old, new)
ROUTE_PREFIXES: Tuple[str, str] = ("/we" "g2/", "/pd" "flip/")
#: name of the operator subdirectory under the gpu-arb root: a HOST path, kept
OPERATOR_SUBDIR = "we" "g2"


def running_package(name: Optional[str] = None) -> str:
    """The old package name before the mechanical commit, the new one after it."""
    return (name or __name__).split(".", 1)[0]


def other_package(name: Optional[str] = None) -> str:
    return _PKG_NEW if running_package(name) == _PKG_OLD else _PKG_OLD


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
def counterpart(path: str) -> Optional[str]:
    """old-prefix path <-> new-prefix path; None for every other path."""
    old, new = ROUTE_PREFIXES
    if path.startswith(old):
        return new + path[len(old):]
    if path.startswith(new):
        return old + path[len(new):]
    return None


def alias_aiohttp_routes(app) -> int:
    """Register the counterpart of every flip-front route on an aiohttp app (before it
    starts). Returns the number of aliases added; an existing counterpart is never
    overwritten."""
    have = set()
    todo = []
    for route in list(app.router.routes()):
        info = route.resource.get_info() if route.resource is not None else {}
        path = info.get("path") or info.get("formatter")
        if not path:
            continue
        have.add((route.method, path))
        cp = counterpart(path)
        if cp is not None:
            todo.append((route.method, cp, route.handler))
    n = 0
    for method, cp, handler in todo:
        if (method, cp) in have or method == "HEAD":
            continue
        app.router.add_route(method, cp, handler)
        have.add((method, cp))
        n += 1
    if n:
        logger.info("compat: %d flip-front route alias(es) %s <-> %s", n, *ROUTE_PREFIXES)
    return n


def alias_fastapi_routes(app) -> int:
    """Same for the FastAPI app of the http server (the PLE prefetch hint route)."""
    from fastapi.routing import APIRoute

    routes = [r for r in app.router.routes if isinstance(r, APIRoute)]
    have = {(r.path, m) for r in routes for m in (r.methods or ())}
    n = 0
    for r in routes:
        cp = counterpart(r.path)
        if cp is None:
            continue
        methods = sorted(m for m in (r.methods or ()) if (cp, m) not in have)
        if not methods:
            continue
        app.add_api_route(cp, r.endpoint, methods=methods, include_in_schema=False,
                          name=f"{r.name}__compat")
        have.update((cp, m) for m in methods)
        n += 1
    if n:
        logger.info("compat: %d http route alias(es) %s <-> %s", n, *ROUTE_PREFIXES)
    return n


# ---------------------------------------------------------------------------
# ~/.cache/<package>
# ---------------------------------------------------------------------------
def _cache_root(pkg: str, home: Optional[str] = None) -> str:
    return os.path.join(home or os.path.expanduser("~"), ".cache", pkg)


def link_legacy_cache_dir(home: Optional[str] = None, name: Optional[str] = None) -> Optional[str]:
    """If ``~/.cache/<running>`` is absent and ``~/.cache/<other>`` exists, make the running
    name a symlink to the other one, so every reader AND writer of the 33 hard-coded cache
    paths keeps seeing the rig's calibration files. Idempotent; never raises (a failure is
    logged and :func:`cache_file` still reads through). Returns the link path if one was made."""
    here = _cache_root(running_package(name), home)
    there = _cache_root(other_package(name), home)
    if os.path.lexists(here) or not os.path.isdir(there):
        return None
    try:
        os.symlink(there, here, target_is_directory=True)
    except OSError as e:
        logger.warning("compat: could not link %s -> %s (%s); files are read through cache_file()", here, there, e)
        return None
    logger.warning("compat: %s did not exist; linked it to %s (rig state from before the rename)", here, there)
    return here


def cache_file(*parts: str, home: Optional[str] = None, name: Optional[str] = None) -> str:
    """Path of a file under ``~/.cache/<running>``; if it is missing there but present under
    the other name, the other path (read-only fallback). Writers keep using the primary path."""
    primary = os.path.join(_cache_root(running_package(name), home), *parts)
    if os.path.exists(primary):
        return primary
    legacy = os.path.join(_cache_root(other_package(name), home), *parts)
    return legacy if os.path.exists(legacy) else primary


# ---------------------------------------------------------------------------
# launcher flags --<flip token>-x
# ---------------------------------------------------------------------------
#: the flip subsystem's flag token (old, new)
FLAG_TOKENS: Tuple[str, str] = ("we" "g2", "pd" "flip")


def canonical_flags(argv: Iterable[str], name: Optional[str] = None) -> List[str]:
    """Launcher argv with every ``--<other token>-x`` flag (also ``--<other token>-x=v``)
    spelled with the running tree's token. Profiles and arms carry the old flags
    (container profiles, 16k occurrences in the arms); the renamed launcher defines the new
    ones. Only a token that STARTS with the flag prefix is touched, never a value."""
    old, new = FLAG_TOKENS
    run, other = (old, new) if running_package(name) == _PKG_OLD else (new, old)
    src, dst = "--%s-" % other, "--%s-" % run
    out, seen = [], []
    for a in argv:
        if isinstance(a, str) and a.startswith(src):
            seen.append(a.split("=", 1)[0])
            a = dst + a[len(src):]
        out.append(a)
    if seen:
        logger.warning("compat: %d launcher flag(s) in the other spelling mapped %s* -> %s*: %s",
                       len(seen), src, dst, " ".join(sorted(set(seen))))
    return out


# ---------------------------------------------------------------------------
# /dev/shm name families
# ---------------------------------------------------------------------------
#: (old, new) tokens that the mechanical pass rewrites inside /dev/shm names
_NAME_TOKEN_PAIRS: Tuple[Tuple[str, str], ...] = ((_PKG_OLD, _PKG_NEW), ("we" "g2", "pd" "flip"))


def name_counterparts(prefixes: Iterable[str]) -> Tuple[str, ...]:
    """The other spelling of every name prefix that carries an old or a new token, in order and
    without the ones already listed. The launcher's residue sweep appends them to its own name
    families, so a renamed launcher still finds (and a pre-rename launcher already finds) the
    /dev/shm residue of a boot that ran under the other name (FL3 26.09., RENAME_PLAN 8.11)."""
    have = list(prefixes)
    out = []
    for p in have:
        q = p
        for old, new in _NAME_TOKEN_PAIRS:
            if old in q:
                q = q.replace(old, new)
            elif new in q:
                q = q.replace(new, old)
        if q != p and q not in have and q not in out:
            out.append(q)
    return tuple(out)


# ---------------------------------------------------------------------------
# processes of the other generation: module names in argv, env keys in /proc/<pid>/environ
# ---------------------------------------------------------------------------
def name_variants(name: str) -> Tuple[str, ...]:
    """``name`` itself first, then its other spelling when it carries a package or subsystem
    token: the stock server module of this tree and of the other one. The launcher's #1217 live
    server census matches argv against these, so a server started by a launcher of the other
    generation is still a live server and not a stranger that mentions nothing (FL5, RENAME_PLAN
    8.14 item 1)."""
    return (name,) + name_counterparts((name,))


def env_name_variants(name: str) -> Tuple[str, ...]:
    """``name`` itself first, then its other spelling under the env prefix pairs of
    ``name_compat.ENV_PREFIX_PAIRS`` (subsystem family before the generic one). For readers of
    ANOTHER process's environment (``/proc/<pid>/environ``): the package hook folds this
    process's own environment onto one spelling, but a process of the other generation carries
    only its own, so such a reader must accept both (the boot token of the #1217 census)."""
    from .name_compat import ENV_PREFIX_PAIRS

    for legacy, renamed in ENV_PREFIX_PAIRS:
        for have, other in ((legacy, renamed), (renamed, legacy)):
            if name.startswith(have) and len(name) > len(have):
                return (name, other + name[len(have):])
    return (name,)


# ---------------------------------------------------------------------------
# <gpu-arb>/<operator subdir> -- host path
# ---------------------------------------------------------------------------
def operator_dir(gpu_arb: str, *parts: str) -> str:
    """``<gpu_arb>/<operator subdir>[/parts...]``: the operator directory keeps its name through the rename."""
    return os.path.join(gpu_arb, OPERATOR_SUBDIR, *parts)
