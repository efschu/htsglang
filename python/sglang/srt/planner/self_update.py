# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""In-place self-update for the planner dashboard (versioned, roll-backable).

LAYOUT (``SGLANG_DASHBOARD_HOME``, default ``~/.local/share/sglang-dashboard``)::

    versions/<id>/          one installed code tree per version:
        dashboard_manifest.json   {version, git_hash, source, installed_at}
        python/sglang/...         the archived ``python/`` subtree
    current                 pointer FILE naming the active version id
    good/<id>               marker: this version passed a health check once
    switch_request.json     worker -> supervisor: "switch to <target>"
    last_switch.json        supervisor -> UI: outcome of the last switch

CODE / DATA SEPARATION (hard gate): a version switch only ever touches the
layout above — never the planner data stores. All persistent data lives in
the DATA dir (``SGLANG_PLANNER_DATA_DIR``, default ``~/.cache/sglang``),
which is shared by every installed version and by plain checkouts. Two
legacy stores used to default into the package directory (hicache savings,
measured energy results); :func:`planner_data_path` migrates them
copy-forward, once, idempotently — the legacy file is left in place as an
inert backup and is never read again.

DOWNGRADE SAFETY: the data dir carries a ``planner_data_schema.json`` stamp
(:data:`DATA_SCHEMA_VERSION`, written by :func:`stamp_data_schema`, never
downgraded). A dashboard that finds a NEWER stamp than it understands keeps
serving read-only: :func:`data_write_guard` returns a warning string and the
web UI refuses the store-writing endpoints with it.

SUPERVISOR (``--serve-supervised``): a minimal parent process starts the
worker (``--serve``) with ``PYTHONPATH`` pointing at the current version's
``python/`` tree, health-checks ``GET /`` (HTTP 200 within a timeout), and
marks the version good on success. On a switch request the worker exits with
:data:`RESTART_EXIT_CODE`; the supervisor moves the ``current`` pointer and
restarts. A failed health check rolls the pointer back to the last good
version automatically and restarts again. The supervisor itself runs from
the operator's launch checkout and is deliberately NOT swapped by updates:
it is small enough to stay stable across versions, and updating IT means
updating the launch checkout (``git pull``) and restarting the whole
process — document/runbook responsibility, not code. Plain ``--serve``
(the operator's existing start command) keeps working unchanged; switching
is refused in that mode with a clear message.

VERSION SOURCES: :class:`VersionSource` is the plug point.
:class:`LocalGitSource` (today) offers ``dashboard-v*`` tags plus the
current HEAD of the local fork checkout, installed via ``git archive`` of
the ``python/`` subtree. :class:`GitHubReleaseSource` (stub) has the same
interface and reports "not configured"; when it grows a real implementation
it must only touch the network on an explicit user click, never
automatically, and needs no PAT (public reads).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

__all__ = [
    "DASHBOARD_VERSION",
    "DATA_SCHEMA_VERSION",
    "RESTART_EXIT_CODE",
    "VersionInfo",
    "VersionSource",
    "LocalGitSource",
    "GitHubReleaseSource",
    "VersionStore",
    "default_sources",
    "runtime_identity",
    "planner_data_dir",
    "planner_data_path",
    "stamp_data_schema",
    "data_write_guard",
    "apply_health_result",
    "wait_health",
    "run_supervisor",
]

#: The dashboard's own version (semver-like). Bump on user-visible changes.
DASHBOARD_VERSION = "0.1.0"

#: Schema generation of the planner data stores as a WHOLE. Bump only when a
#: store format changes incompatibly; older dashboards then go read-only on
#: this data dir instead of silently writing an older format.
DATA_SCHEMA_VERSION = 1

#: Worker exit code that tells the supervisor "switch requested, restart me".
RESTART_EXIT_CODE = 43

MANIFEST_NAME = "dashboard_manifest.json"
SCHEMA_STAMP_NAME = "planner_data_schema.json"
TAG_PREFIX = "dashboard-v"

_DEFAULT_HOME = "~/.local/share/sglang-dashboard"
_DEFAULT_DATA_DIR = "~/.cache/sglang"


# ===========================================================================
# Data dir + legacy-store migration + schema stamp (code/data separation).
# ===========================================================================


def planner_data_dir(data_dir: Optional[str] = None) -> str:
    """The stable data directory shared by all dashboard versions."""
    return os.path.expanduser(
        data_dir
        or os.environ.get("SGLANG_PLANNER_DATA_DIR")
        or _DEFAULT_DATA_DIR
    )


def planner_data_path(
    name: str,
    legacy: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> str:
    """Resolve a store file to the DATA dir, migrating a legacy in-tree copy.

    One-time, idempotent: if ``legacy`` exists and the data-dir file does
    not, the legacy file is copied forward once. The legacy file is left in
    place as an inert backup (it is never read again); the returned data-dir
    path is authoritative from here on. A store must never live inside a
    version/code directory — a version switch must not be able to touch it.
    """
    root = planner_data_dir(data_dir)
    new = os.path.join(root, name)
    if legacy and os.path.exists(legacy) and not os.path.exists(new):
        try:
            os.makedirs(root, exist_ok=True)
            shutil.copy2(legacy, new)
        except OSError:
            # Read-only data dir: fall back to the legacy location rather
            # than breaking the caller; writes there will fail loudly.
            return legacy
    return new


def _schema_stamp_path(data_dir: Optional[str] = None) -> str:
    return os.path.join(planner_data_dir(data_dir), SCHEMA_STAMP_NAME)


def read_data_schema(data_dir: Optional[str] = None) -> Optional[dict]:
    try:
        with open(_schema_stamp_path(data_dir)) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def stamp_data_schema(data_dir: Optional[str] = None) -> dict:
    """Write/refresh the data-dir schema stamp. NEVER downgrades: a stamp
    newer than :data:`DATA_SCHEMA_VERSION` is left untouched (that is the
    downgrade-safety signal :func:`data_write_guard` reads)."""
    existing = read_data_schema(data_dir)
    if existing and int(existing.get("schema_version") or 0) >= DATA_SCHEMA_VERSION:
        # Same generation: leave the file byte-identical (restarts and
        # version switches must not touch the data dir); newer generation:
        # never downgrade the stamp.
        return existing
    stamp = {
        "schema_version": DATA_SCHEMA_VERSION,
        "written_by": DASHBOARD_VERSION,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    root = planner_data_dir(data_dir)
    os.makedirs(root, exist_ok=True)
    _atomic_write_json(_schema_stamp_path(data_dir), stamp)
    return stamp


def data_write_guard(data_dir: Optional[str] = None) -> Optional[str]:
    """None = writes allowed. A string = the data dir was last written by a
    NEWER dashboard (higher schema); return the read-only warning to show
    instead of silently writing an older store format."""
    existing = read_data_schema(data_dir)
    if not existing:
        return None
    found = int(existing.get("schema_version") or 0)
    if found <= DATA_SCHEMA_VERSION:
        return None
    return (
        f"data stores carry schema v{found} (written by dashboard "
        f"{existing.get('written_by', '?')}), but this dashboard "
        f"(v{DASHBOARD_VERSION}) only knows schema v{DATA_SCHEMA_VERSION}: "
        "store writes are disabled to protect the newer data. Upgrade the "
        "dashboard again to write, or keep it read-only."
    )


def _atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ===========================================================================
# Runtime identity: which code is THIS process running?
# ===========================================================================


def _git(repo: str, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _find_upwards(start: str, name: str, max_depth: int = 8) -> Optional[str]:
    d = os.path.abspath(start)
    for _ in range(max_depth):
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent
    return None


def code_repo_root() -> Optional[str]:
    """The git checkout this module runs from, if any (None for an
    installed version dir, which has no ``.git``)."""
    marker = _find_upwards(os.path.dirname(__file__), ".git")
    return os.path.dirname(marker) if marker else None


def runtime_identity() -> dict:
    """Version identity of the RUNNING code: an installed version dir is
    identified by its manifest; a git checkout by constant + live hash."""
    manifest = _find_upwards(os.path.dirname(__file__), MANIFEST_NAME)
    if manifest:
        try:
            with open(manifest) as f:
                m = json.load(f)
        except (OSError, ValueError):
            m = {}
        return {
            "version": m.get("version") or DASHBOARD_VERSION,
            "git_hash": m.get("git_hash"),
            "origin": "installed",
            "code_root": os.path.dirname(manifest),
        }
    repo = code_repo_root()
    return {
        "version": DASHBOARD_VERSION,
        "git_hash": _git(repo, "rev-parse", "--short", "HEAD") if repo else None,
        "origin": "checkout",
        "code_root": repo,
    }


# ===========================================================================
# Version sources.
# ===========================================================================


@dataclasses.dataclass
class VersionInfo:
    """One selectable version, independent of where it comes from."""

    id: str  # directory-safe id, e.g. "0.2.0" or "head-ab12cd3"
    label: str  # human label for the UI list
    origin: str  # "local-git" | "github-release"
    ref: str  # source-specific reference (git ref, release tag, ...)
    date: Optional[str] = None  # ISO date of the underlying commit/release
    git_hash: Optional[str] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


class VersionSource:
    """Interface every version source implements. ``configured`` sources
    list versions and can install them; unconfigured ones only explain
    themselves in the UI. Implementations must not touch the network
    except inside :meth:`list_versions` / :meth:`install` invoked by an
    explicit user action."""

    name = "abstract"

    @property
    def configured(self) -> bool:  # pragma: no cover - interface default
        return False

    @property
    def note(self) -> Optional[str]:
        """Why the source is unavailable (None when configured)."""
        return None

    def list_versions(self) -> List[VersionInfo]:
        raise NotImplementedError

    def install(self, version: VersionInfo, dest_dir: str) -> dict:
        """Materialize ``version`` into ``dest_dir`` (the version directory;
        the caller manages atomicity). Returns the manifest dict."""
        raise NotImplementedError


class LocalGitSource(VersionSource):
    """Versions = ``dashboard-v*`` tags of the local fork checkout, plus the
    checkout's current HEAD (as a dev version). Install = ``git archive``
    of the ``python/`` subtree into the version directory."""

    name = "local-git"

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or code_repo_root()

    @property
    def configured(self) -> bool:
        return bool(self.repo_root and os.path.isdir(self.repo_root))

    @property
    def note(self) -> Optional[str]:
        if self.configured:
            return None
        return (
            "no local git checkout found (running from an installed "
            "version dir without a configured repo)"
        )

    def list_versions(self) -> List[VersionInfo]:
        if not self.configured:
            return []
        out: List[VersionInfo] = []
        tags = _git(
            self.repo_root,
            "tag",
            "--list",
            TAG_PREFIX + "*",
            "--format=%(refname:strip=2)\t%(creatordate:iso-strict)",
        )
        for line in (tags or "").splitlines():
            tag, _, date = line.partition("\t")
            if not tag.startswith(TAG_PREFIX):
                continue
            ver = tag[len(TAG_PREFIX):]
            sha = _git(self.repo_root, "rev-parse", "--short", tag + "^{commit}")
            out.append(
                VersionInfo(
                    id=ver,
                    label=f"{ver} (tag {tag})",
                    origin=self.name,
                    ref=tag,
                    date=date or None,
                    git_hash=sha,
                )
            )
        head = _git(self.repo_root, "rev-parse", "--short", "HEAD")
        if head:
            branch = _git(self.repo_root, "rev-parse", "--abbrev-ref", "HEAD")
            date = _git(self.repo_root, "log", "-1", "--format=%cI", "HEAD")
            out.append(
                VersionInfo(
                    id=f"head-{head}",
                    label=f"HEAD {head} ({branch or 'detached'})",
                    origin=self.name,
                    ref="HEAD",
                    date=date,
                    git_hash=head,
                )
            )
        return out

    def install(self, version: VersionInfo, dest_dir: str) -> dict:
        if not self.configured:
            raise RuntimeError("local git source is not configured")
        os.makedirs(dest_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
            tar_path = tf.name
        try:
            with open(tar_path, "wb") as f:
                proc = subprocess.run(
                    ["git", "-C", self.repo_root, "archive", "--format=tar",
                     version.ref, "python/"],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    timeout=300,
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    "git archive failed: "
                    + proc.stderr.decode(errors="replace").strip()
                )
            with tarfile.open(tar_path) as tar:
                tar.extractall(dest_dir, filter="data")
        finally:
            os.unlink(tar_path)
        manifest = {
            "version": version.id,
            "label": version.label,
            "git_ref": version.ref,
            "git_hash": version.git_hash
            or _git(self.repo_root, "rev-parse", "--short",
                    version.ref + "^{commit}"),
            "source": self.name,
            "date": version.date,
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _atomic_write_json(os.path.join(dest_dir, MANIFEST_NAME), manifest)
        return manifest


class GitHubReleaseSource(VersionSource):
    """Stub for the fork's GitHub releases — the release feed does not exist
    yet. Same interface as :class:`LocalGitSource`; activation later via
    configuration. Design constraints for the real implementation: public
    reads only (no PAT), network access only on an explicit user click,
    download into the version dir + manifest exactly like the local source."""

    name = "github-release"

    def __init__(self, repo: Optional[str] = None):
        # e.g. "efschu/htsglang" once releases exist; None = unconfigured.
        self.repo = repo

    @property
    def configured(self) -> bool:
        return False  # becomes a config lookup once the feed exists

    @property
    def note(self) -> Optional[str]:
        return "no remote release source configured yet (fork has no releases)"

    def list_versions(self) -> List[VersionInfo]:
        return []

    def install(self, version: VersionInfo, dest_dir: str) -> dict:
        raise RuntimeError(self.note)


def default_sources() -> List[VersionSource]:
    return [LocalGitSource(), GitHubReleaseSource()]


# ===========================================================================
# Version store: the on-disk layout (versions/, current, good/, requests).
# ===========================================================================


class VersionStore:
    """Manages the dashboard-home layout. Every mutation here touches ONLY
    the home dir — never the planner data dir (tested)."""

    def __init__(self, home: Optional[str] = None):
        self.home = os.path.expanduser(
            home or os.environ.get("SGLANG_DASHBOARD_HOME") or _DEFAULT_HOME
        )
        self.versions_dir = os.path.join(self.home, "versions")
        self.good_dir = os.path.join(self.home, "good")
        self.current_file = os.path.join(self.home, "current")
        self.request_file = os.path.join(self.home, "switch_request.json")
        self.outcome_file = os.path.join(self.home, "last_switch.json")

    # -- installed versions -------------------------------------------------

    def version_dir(self, version_id: str) -> str:
        if not version_id or "/" in version_id or version_id.startswith("."):
            raise ValueError(f"invalid version id: {version_id!r}")
        return os.path.join(self.versions_dir, version_id)

    def manifest(self, version_id: str) -> Optional[dict]:
        try:
            with open(os.path.join(self.version_dir(version_id),
                                   MANIFEST_NAME)) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else None
        except (OSError, ValueError):
            return None

    def installed(self) -> List[dict]:
        """Manifests of every installed version, newest install first."""
        out = []
        try:
            ids = sorted(os.listdir(self.versions_dir))
        except OSError:
            return []
        for vid in ids:
            m = self.manifest(vid)
            if m:
                m = dict(m)
                m["id"] = vid
                out.append(m)
        out.sort(key=lambda m: m.get("installed_at") or "", reverse=True)
        return out

    def is_installed(self, version_id: str) -> bool:
        return self.manifest(version_id) is not None

    def python_root(self, version_id: str) -> Optional[str]:
        p = os.path.join(self.version_dir(version_id), "python")
        return p if os.path.isdir(p) else None

    def install(self, source: VersionSource, version: VersionInfo) -> dict:
        """Fill ``versions/<id>/`` atomically (stage + rename); an existing
        install of the same id is replaced only after a successful stage."""
        final = self.version_dir(version.id)
        os.makedirs(self.versions_dir, exist_ok=True)
        stage = tempfile.mkdtemp(
            prefix=f".stage-{version.id}-", dir=self.versions_dir
        )
        try:
            manifest = source.install(version, stage)
            if os.path.isdir(final):
                if self.current_id() == version.id:
                    raise RuntimeError(
                        f"version {version.id} is currently active; "
                        "switch away before reinstalling it"
                    )
                shutil.rmtree(final)
            os.replace(stage, final)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return manifest

    # -- current pointer + good markers -------------------------------------

    def current_id(self) -> Optional[str]:
        try:
            with open(self.current_file) as f:
                v = f.read().strip()
            return v or None
        except OSError:
            return None

    def set_current(self, version_id: str) -> None:
        if not self.is_installed(version_id):
            raise ValueError(f"version {version_id!r} is not installed")
        os.makedirs(self.home, exist_ok=True)
        tmp = self.current_file + ".tmp"
        with open(tmp, "w") as f:
            f.write(version_id + "\n")
        os.replace(tmp, self.current_file)

    def mark_good(self, version_id: str) -> None:
        os.makedirs(self.good_dir, exist_ok=True)
        path = os.path.join(self.good_dir, version_id)
        with open(path, "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")

    def is_good(self, version_id: str) -> bool:
        return os.path.exists(os.path.join(self.good_dir, version_id))

    def last_good(self, exclude: Optional[str] = None) -> Optional[str]:
        """Most recently health-passed INSTALLED version (marker mtime),
        excluding ``exclude`` (the version that just failed)."""
        try:
            names = os.listdir(self.good_dir)
        except OSError:
            return None
        best, best_mtime = None, -1.0
        for n in names:
            if n == exclude or not self.is_installed(n):
                continue
            try:
                mtime = os.path.getmtime(os.path.join(self.good_dir, n))
            except OSError:
                continue
            if mtime > best_mtime:
                best, best_mtime = n, mtime
        return best

    # -- switch request / outcome (worker <-> supervisor <-> UI) -------------

    def write_switch_request(self, target: str) -> None:
        os.makedirs(self.home, exist_ok=True)
        _atomic_write_json(
            self.request_file,
            {"target": target, "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        )

    def take_switch_request(self) -> Optional[dict]:
        try:
            with open(self.request_file) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        try:
            os.unlink(self.request_file)
        except OSError:
            pass
        return d if isinstance(d, dict) else None

    def record_switch(self, outcome: dict) -> None:
        os.makedirs(self.home, exist_ok=True)
        outcome = dict(outcome)
        outcome["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_write_json(self.outcome_file, outcome)

    def last_switch(self) -> Optional[dict]:
        try:
            with open(self.outcome_file) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else None
        except (OSError, ValueError):
            return None

    # -- retention -----------------------------------------------------------

    def cleanup_plan(self, keep: int = 3) -> List[str]:
        """Version ids that WOULD be removed: everything but the current
        version, the last good version, and the ``keep`` most recently
        installed of the rest. Pure — nothing is deleted here."""
        protected = {self.current_id(), self.last_good()}
        others = [m["id"] for m in self.installed() if m["id"] not in protected]
        return others[keep:]

    def cleanup(self, keep: int = 3) -> List[str]:
        removed = []
        for vid in self.cleanup_plan(keep):
            if vid == self.current_id():  # defense in depth
                continue
            shutil.rmtree(self.version_dir(vid), ignore_errors=True)
            try:
                os.unlink(os.path.join(self.good_dir, vid))
            except OSError:
                pass
            removed.append(vid)
        return removed


# ===========================================================================
# Supervisor: health check, auto-rollback decision, process loop.
# ===========================================================================


def apply_health_result(
    store: VersionStore, version_id: Optional[str], healthy: bool
) -> dict:
    """The auto-rollback DECISION, separated from process handling so it is
    testable without booting anything. Returns what happened:

    * healthy -> the version is marked good, no action.
    * unhealthy + a last-good version exists -> pointer moved back to it
      (``action: rollback``); the caller restarts the worker.
    * unhealthy + nothing to fall back to -> ``action: halt``.
    """
    if healthy:
        if version_id:
            store.mark_good(version_id)
        return {"healthy": True, "action": "none", "version": version_id}
    fallback = store.last_good(exclude=version_id) if version_id else None
    if fallback:
        store.set_current(fallback)
        return {
            "healthy": False,
            "action": "rollback",
            "version": version_id,
            "rolled_back_to": fallback,
        }
    return {
        "healthy": False,
        "action": "halt",
        "version": version_id,
        "rolled_back_to": None,
    }


def wait_health(
    url: str,
    timeout: float = 45.0,
    interval: float = 0.5,
    proc=None,
) -> bool:
    """True once ``GET url`` answers HTTP 200 within ``timeout`` seconds;
    False on timeout or when ``proc`` (the worker) exits early."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            # Died before ever answering — but a RESTART exit is a controlled
            # shutdown, not a health failure; the caller inspects the code.
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(interval)
    return False


def _terminate(proc) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_supervisor(
    host: str = "127.0.0.1",
    port: int = 8780,
    home: Optional[str] = None,
    health_timeout: float = 45.0,
) -> int:
    """The ``--serve-supervised`` loop. Worker = ``python -m
    sglang.srt.planner.cli --serve`` with PYTHONPATH pointing at the current
    version (or, with no version installed yet, at the launch checkout the
    supervisor itself runs from). See the module docstring for the contract;
    this function stays deliberately small and version-stable."""
    store = VersionStore(home)
    check_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{check_host}:{port}/"
    just_switched: Optional[dict] = None
    while True:
        current = store.current_id()
        env = dict(os.environ)
        env["SGLANG_DASHBOARD_SUPERVISED"] = "1"
        env["SGLANG_DASHBOARD_HOME"] = store.home
        pyroot = store.python_root(current) if current else None
        if pyroot:
            prev = env.get("PYTHONPATH")
            env["PYTHONPATH"] = pyroot + (os.pathsep + prev if prev else "")
        argv = [
            sys.executable,
            "-m",
            "sglang.srt.planner.cli",
            "--serve",
            "--host",
            host,
            "--port",
            str(port),
        ]
        print(
            f"[supervisor] starting worker: version="
            f"{current or 'launch checkout'}"
        )
        try:
            proc = subprocess.Popen(argv, env=env)
        except OSError as e:
            print(f"[supervisor] cannot start worker: {e}")
            return 1
        try:
            healthy = wait_health(url, timeout=health_timeout, proc=proc)
        except KeyboardInterrupt:
            _terminate(proc)
            return 0
        outcome = apply_health_result(store, current, healthy)
        if not healthy:
            _terminate(proc)
            store.record_switch(
                {
                    "event": "switch" if just_switched else "health_check",
                    **(just_switched or {"version": current}),
                    "healthy": False,
                    "action": outcome["action"],
                    "rolled_back_to": outcome.get("rolled_back_to"),
                    "note": f"no HTTP 200 on / within {health_timeout:.0f}s",
                }
            )
            just_switched = None
            if outcome["action"] == "rollback":
                print(
                    f"[supervisor] health check FAILED for {current}; "
                    f"rolled back to {outcome['rolled_back_to']}"
                )
                continue
            print(
                f"[supervisor] health check FAILED for "
                f"{current or 'launch checkout'} and no good version to "
                "fall back to; giving up"
            )
            return 1
        print(f"[supervisor] worker healthy ({url})")
        if just_switched:
            store.record_switch(
                {
                    "event": "switch",
                    **just_switched,
                    "healthy": True,
                    "note": "health check passed",
                }
            )
            just_switched = None
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            _terminate(proc)
            return 0
        if code == RESTART_EXIT_CODE:
            req = store.take_switch_request()
            target = (req or {}).get("target")
            if target and store.is_installed(target):
                previous = store.current_id()
                store.set_current(target)
                just_switched = {"from": previous, "to": target}
                store.record_switch(
                    {
                        "event": "switch",
                        **just_switched,
                        "healthy": None,
                        "note": "pointer moved; health check pending",
                    }
                )
                print(f"[supervisor] switching {previous} -> {target}")
            else:
                print(
                    "[supervisor] restart requested without a valid switch "
                    "target; restarting the current version"
                )
            continue
        print(f"[supervisor] worker exited with code {code}; stopping")
        return code


# ===========================================================================
# View helper: merge sources + store into the /api/version listing.
# ===========================================================================


def list_versions_view(
    store: VersionStore, sources: Optional[List[VersionSource]] = None
) -> Dict[str, object]:
    sources = default_sources() if sources is None else sources
    current = store.current_id()
    by_id: Dict[str, dict] = {}
    for src in sources:
        try:
            infos = src.list_versions()
        except Exception as e:  # pragma: no cover - defensive
            infos = []
            print(f"[self-update] listing {src.name} failed: {e}")
        for info in infos:
            by_id[info.id] = info.to_json()
    for m in store.installed():
        row = by_id.setdefault(
            m["id"],
            {
                "id": m["id"],
                "label": m.get("label") or m["id"],
                "origin": m.get("source") or "installed",
                "ref": m.get("git_ref"),
                "date": m.get("date"),
                "git_hash": m.get("git_hash"),
            },
        )
        row["installed_at"] = m.get("installed_at")
    rows = list(by_id.values())
    for row in rows:
        vid = str(row["id"])
        row["installed"] = store.is_installed(vid)
        row["current"] = vid == current
        row["good"] = store.is_good(vid)
    rows.sort(key=lambda r: (str(r.get("date") or "")), reverse=True)
    return {
        "versions": rows,
        "sources": [
            {"name": s.name, "configured": s.configured, "note": s.note}
            for s in sources
        ],
        "current_id": current,
    }
