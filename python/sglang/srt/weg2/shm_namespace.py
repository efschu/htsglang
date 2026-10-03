"""Auftrag 1000: the TEST NAMESPACE of the weg2 /dev/shm name families.

WHY (metal 03.10.2026, boot A, window 4dzm46): a broad agent pytest run (~190
weg2 test files) created real objects under ``/dev/shm`` in the SAME name
families a serving boot owns (``weg2-xchg-*``, ``sem.weg2-xchg-*``,
``weg2-xchg-bnc-*``, ``weg2-seq-*``).  The launcher's #1217 residue sweep
(``launcher.shm_residue_sweep``) counts every entry of those families as its
own and REFUSES the boot while a live process holds one -- so a pytest worker
holding a test region refused a real boot ("#1217 LIVE HOLDER").

THE RULE: a process that runs under pytest -- or that has
``SGLANG_WEG2_SHM_NAMESPACE`` set -- puts a ``test-<id>-`` prefix IN FRONT of
the whole name.  ``test-123-weg2-xchg-s4bx1`` matches no entry of
``launcher.SHM_OWN_PREFIXES``, so the sweep neither lists, archives, nor
holder-checks it (and ``launcher.is_test_shm_name`` says so explicitly).

RELEASE PATH BYTE-IDENTICAL: without the env and outside pytest
:func:`shm_prefix` returns ``""`` and every name is exactly the one it was
before this module existed (``tests: test_weg2_shm_namespace_1000.py`` pins the
literal names in a pytest-free subprocess).

SCOPE: names created under the real ``/dev/shm`` root, and the global POSIX
semaphore names.  A path under a private ``shm_root`` (a test's ``tmp_path``)
cannot collide with anything and keeps its historical spelling.

Stdlib + the env registry only: it is imported by the exchange modules on the
serving path.
"""

from __future__ import annotations

import os
import re
import sys

from sglang.srt.environ import envs

#: Every test-namespaced name starts with this, whatever the env value says.
#: ``launcher.is_test_shm_name`` keys on it.
TEST_NS_MARK = "test-"
#: The real tmpfs root.  Mirrors ``weight_exchange_region.SHM_ROOT`` (not
#: imported from there: that module imports THIS one).
REAL_SHM_ROOT = "/dev/shm"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "_pytest" in sys.modules


def namespace() -> str:
    """The namespace token, or ``""`` for the release spelling.

    * ``SGLANG_WEG2_SHM_NAMESPACE`` set: its value wins (``""`` = explicitly
      NO namespace, even under pytest -- the byte-identity test uses it);
    * unset and running under pytest: ``test-<pid>``, and the value is
      published into the environment so forked/spawned children (a pytest
      process's scheduler subprocess) compute the SAME token instead of their
      own pid;
    * otherwise ``""``.
    """
    raw = envs.SGLANG_WEG2_SHM_NAMESPACE.get()
    if raw is None:
        if not _under_pytest():
            return ""
        raw = f"{TEST_NS_MARK}{os.getpid()}"
        envs.SGLANG_WEG2_SHM_NAMESPACE.set(raw)
    if raw == "":
        return ""
    raw = _UNSAFE.sub("_", raw)
    return raw if raw.startswith(TEST_NS_MARK) else f"{TEST_NS_MARK}{raw}"


def shm_prefix() -> str:
    """``"test-<id>-"`` under a namespace, ``""`` otherwise."""
    ns = namespace()
    return f"{ns}-" if ns else ""


def shm_name(name: str) -> str:
    """``name`` with the namespace prefix in front (identity without one)."""
    return shm_prefix() + name


def shm_name_for_root(name: str, shm_root: str) -> str:
    """Like :func:`shm_name`, but only for the real ``/dev/shm`` root: a
    private root (``tmp_path``) keeps the historical spelling."""
    if os.path.normpath(str(shm_root)) != REAL_SHM_ROOT:
        return name
    return shm_name(name)


def posix_sem_name(name: str) -> str:
    """A POSIX named-semaphore name (``/<base>``): the prefix goes between the
    slash and the base, so glibc materialises ``/dev/shm/sem.test-<id>-<base>``.
    ``name`` may carry its leading slash."""
    p = shm_prefix()
    if not p:
        return name
    return "/" + p + name.lstrip("/")


def is_test_shm_name(entry: str) -> bool:
    """True for a ``/dev/shm`` directory entry created under a namespace
    (a plain file/dir ``test-...`` or a semaphore file ``sem.test-...``)."""
    return entry.startswith((TEST_NS_MARK, f"sem.{TEST_NS_MARK}"))
