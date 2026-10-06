"""AP0 reference harness of the profile planner (plan PLAN-PROFIL-PLANER-1006 section 2 stage B, R6/R9).

``propose()`` (AP-C) derives a candidate argv; THIS module is the ORACLE that tells what the launcher makes of it,
without a GPU, a container or a boot: it runs the launcher's REAL ``main()`` with ``--dry-run`` on a replayed NVML
inventory and returns the plan the launcher prints.  Nothing in here re-implements a solver or a refusal (R1/Q-710:
flags instead of code); every number in the result is a number the launcher itself printed.

Five parts, each usable alone:

(a) **NVML replay** -- :func:`replay_from_hardware_profile` (a ``flliper.hardware/1`` document, as
    ``rigmon.hardware_profile.build`` assembles it), :func:`replay_from_catalog` (synthetic cards from the dashboard's
    card catalog entries, ``rigdash.kartenplan_catalog.CATALOG`` rows, taken as plain dicts so this file stays free of the
    dashboard) and :func:`write_replay` produce the JSON that ``SGLANG_NVML_REPLAY_JSON`` reads
    (``registry/nvml.py:_replay_devices``; model: ``fixtures/xchg_launch_replay_0911/nvml_devices_1378.json``).
(b) **Dry-run runner** -- :func:`run_dry_run`, the hermetic form of ``deskq/work/hw1004/plan_dump.py``: private empty SHM
    dir, pinned quiet-host ``/proc/meminfo`` + cgroup (:func:`write_quiet_host`), the replay armed, the process environment
    scrubbed of every ``SGLANG_*``/``HTSGLANG_*`` variable the caller did not hand in, and restored afterwards (the
    launcher arms ``refusals`` and writes ``os.environ`` -- none of it may leak into the next run).
(c) **Plan-dump parser** -- :func:`parse_plan_dump` turns the normalised dump text into a dict of the resolved values (the
    argv the launcher builds for each group, the ``WEG2-*`` lines, ``FORCED-PAST`` codes, the exception).  It
    keeps every line; the keyed view is an index over them, so nothing the launcher said is lost.
(d) **Profile reader** -- :func:`profile_launch_input` evaluates a release ``.env`` (bash arrays, ``source`` chains, the
    ``_form`` environment) through ``weg2/profile_json.dump_env`` -- the one bash evaluator of the tree -- and returns the
    argv and the environment the entrypoint would hand the launcher.
(e) **Normalisation** -- :func:`normalise_dump` replaces what differs per run by construction (tree path, private temp
    dirs, timestamps, epoch, boot token, tree sha) and :func:`mask_live_box` the LIVE-BOX readings the dry run takes from
    the running machine (``plan_diff.py:6-17``); both are listed so a golden diff can never hide a value.

Two hermeticity seams added after review: the live evidence dir (W65 lists every ``boot_*.D.log`` of it; masked, and
``run_dry_run(evidence_dir=...)`` rebinds it for the proof that a new log changes nothing) and the process state a boot
leaves behind (:func:`_module_state_guard`); and the stand-in for a checkpoint that is not on the box
(:func:`snapshot_checkpoint` / :func:`materialize_checkpoint`: headers and sizes, which is all the launcher reads).

What this module does NOT do: choose any value (that is ``propose.py``), judge a refusal (``refusals.py``), start a
process other than the in-process ``launcher.main``.  GPU-free, NVML-free, Docker-free.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MIB = 1 << 20

#: the replay file name the launcher reads (``registry/nvml.py:ENV_NVML_REPLAY``)
ENV_NVML_REPLAY = "SGLANG_NVML_REPLAY_JSON"
#: the ``refusals`` marker variable ``refusals.arm(True)`` writes into ``os.environ``
ENV_FORCED_BOOT = "SGLANG_WEG2_FORCED_BOOT"
#: boot tag of every oracle run (appears in the dump as BOOT_TOKEN, normalised away)
ORACLE_TAG = "plangate"

# ---------------------------------------------------------------------------
# (a) NVML replay
# ---------------------------------------------------------------------------

#: the optional property fields of a replay row (``registry/nvml.py:IDENTITY_FIELDS``); a row without one replays it as
#: None = unknown, exactly like an NVML that did not answer
_IDENTITY_FIELDS = ("cc_major", "cc_minor", "bar1_total_bytes", "pcie_max_gen", "pcie_max_width",
                    "mem_bus_width_bits", "mem_clock_max_mhz")


def _fake_uuid(seed: str) -> str:
    """A stable NVML-shaped UUID for a synthetic card (``GPU-8-4-4-4-12``): the launcher keys cards by UUID."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    return "GPU-%s-%s-%s-%s-%s" % (h[:8], h[8:12], h[12:16], h[16:20], h[20:32])


def _fake_pci(i: int) -> str:
    return "0000:%02x:00.0" % (0x01 + int(i))


def _node_v(x: Any) -> Any:
    """Value of a ``flliper.hardware/1`` node (``{"v":.., "src":..}``) or a plain scalar; None when absent."""
    if isinstance(x, Mapping):
        return x.get("v")
    return x


def _int_or_none(x: Any) -> Optional[int]:
    return None if x is None else int(x)


def replay_row(index: int, *, uuid: str, name: str, total_mib: int, cc: Optional[Sequence[int]] = None,
               pci_bus_id: str = "", reserved_mib: int = 0, bar1_total_mib: Optional[int] = None,
               pcie_max_gen: Optional[int] = None, pcie_max_width: Optional[int] = None,
               mem_bus_width_bits: Optional[int] = None, mem_clock_max_mhz: Optional[int] = None) -> Dict[str, Any]:
    """One replay row in the exact field set of ``nvml_devices_1378.json`` (+ the optional properties when known).

    ``total_mib`` is MiB as NVML reports it (``total_bytes = total_mib << 20``, the 5090 fixture row is 32607 MiB);
    a property that is None is LEFT OUT of the row (replays as unknown), never written as a guess."""
    row: Dict[str, Any] = {
        "index": int(index),
        "uuid": str(uuid),
        "name": str(name),
        "total_bytes": int(total_mib) * MIB,
        "pci_bus_id": str(pci_bus_id or _fake_pci(index)),
        "reserved_bytes": int(reserved_mib) * MIB,
    }
    if cc:
        row["cc_major"], row["cc_minor"] = int(cc[0]), int(cc[1])
    if bar1_total_mib is not None:
        row["bar1_total_bytes"] = int(bar1_total_mib) * MIB
    for k, v in (("pcie_max_gen", pcie_max_gen), ("pcie_max_width", pcie_max_width),
                 ("mem_bus_width_bits", mem_bus_width_bits), ("mem_clock_max_mhz", mem_clock_max_mhz)):
        if v is not None:
            row[k] = int(v)
    return row


def replay_from_hardware_profile(profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Replay rows of a ``flliper.hardware/1`` document, in NVML INDEX order (the launcher orders cards itself).

    Uses what the profile holds as NVML identity: ``nvml_index``, ``uuid``, ``name``, ``pci_bus_id``, ``cc``,
    ``vram_total_mib``, ``bar1_total_mib``, ``pcie.max_gen/max_width``.  Only the value of a node is taken, and only when
    the node HAS one: a ``nicht gemessen`` node (``v`` null) leaves the field out (unknown), it is not filled."""
    if str(profile.get("schema", "")) != "flliper.hardware/1":
        raise ValueError("not a flliper.hardware/1 document (schema=%r)" % (profile.get("schema"),))
    cards = list(profile.get("cards") or [])
    if not cards:
        raise ValueError("hardware profile holds no card")
    rows = []
    for c in cards:
        total = _node_v(c.get("vram_total_mib"))
        if not total:
            raise ValueError("card %r of the hardware profile has no vram_total_mib value" % (c.get("uuid"),))
        pcie = c.get("pcie") or {}
        rows.append(replay_row(
            int(c["nvml_index"]), uuid=str(c["uuid"]), name=str(c["name"]), total_mib=int(total),
            cc=c.get("cc"), pci_bus_id=str(c.get("pci_bus_id") or ""),
            bar1_total_mib=_int_or_none(_node_v(c.get("bar1_total_mib"))),
            pcie_max_gen=_int_or_none(_node_v(pcie.get("max_gen"))),
            pcie_max_width=_int_or_none(_node_v(pcie.get("max_width")))))
    rows.sort(key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(len(rows))):
        raise ValueError("hardware profile NVML indices are not 0..N-1: %s" % [r["index"] for r in rows])
    return rows


def catalog_mem_clock_mhz(bw_gbs: Optional[float], bus_bits: Optional[int]) -> Optional[int]:
    """The ``mem_clock_max_mhz`` NVML would report for a nameplate bandwidth: the inverse of
    ``rigmon.hardware_profile``'s ``bus/8 * clock * 2 / 1000`` (datasheet GB/s -> NVML MHz, rounded).  None when
    either input is missing."""
    if not bw_gbs or not bus_bits:
        return None
    return int(round(float(bw_gbs) * 1000.0 / (int(bus_bits) / 8.0 * 2.0)))


def replay_from_catalog(entries: Sequence[Mapping[str, Any]], *, bar1_mib: Optional[int] = None,
                        total_mib_override: Optional[Mapping[str, int]] = None) -> List[Dict[str, Any]]:
    """Replay rows for synthetic cards: ``entries`` are card-catalog dicts (``rigdash.kartenplan_catalog.CATALOG`` rows:
    ``id, nvml_name, usable_mib, cc, mem_bw_gbs, bus_bits, pcie_native{gen,lanes}``), one per card, NVML index = list
    position.  UUID and PCI bus id are synthetic and stable (``_fake_uuid(index:id)``).

    The catalog carries no BAR1 size: ``bar1_mib`` (None = left out = unknown) applies to every card.  Every field the
    row gets is a catalog value; the synthetic card is a *datasheet* card -- the caller labels every value derived from
    it "unbelegt" (plan 1.3 A2)."""
    rows = []
    for i, e in enumerate(entries):
        cid = str(e.get("id") or e.get("nvml_name"))
        total = (total_mib_override or {}).get(cid, e.get("usable_mib"))
        if not total:
            raise ValueError("catalog card %r has no usable_mib" % cid)
        pn = e.get("pcie_native") or {}
        rows.append(replay_row(
            i, uuid=_fake_uuid("%d:%s" % (i, cid)), name=str(e["nvml_name"]), total_mib=int(total), cc=e.get("cc"),
            bar1_total_mib=bar1_mib, pcie_max_gen=pn.get("gen"), pcie_max_width=pn.get("lanes"),
            mem_bus_width_bits=e.get("bus_bits"),
            mem_clock_max_mhz=catalog_mem_clock_mhz(e.get("mem_bw_gbs"), e.get("bus_bits"))))
    return rows


def write_replay(devices: Sequence[Mapping[str, Any]], path: str) -> str:
    """Write ``devices`` as the replay JSON (list of rows, indent 2 like the fixture); returns ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(list(devices), fh, indent=2)
        fh.write("\n")
    return path


def read_replay(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# (b) dry-run runner
# ---------------------------------------------------------------------------

# A QUIET reading of the rig box, 2026-09-14T02:2xZ, pinned (test_weg2_hicache_disabled_1386._fake_quiet_host, #1390):
# the dry run must not read the running machine, or the host ledger refuses (W20) under a busy box.
_QUIET_MEMTOTAL_KB = 123_781_120
_QUIET_MEMAVAIL_KB = 118_833_964
_QUIET_CG_CURRENT_B = 14_064_181_248
_QUIET_CG_ANON_B = 5_015_404_544
_QUIET_CG_FILE_B = 8_809_852_928
_QUIET_CG_SHMEM_B = 577_536
_QUIET_CG_UNEVICTABLE_B = 36_864
_QUIET_CG_SLAB_RECLAIM_B = 188_900_232


def write_quiet_host(tmp: str) -> Tuple[str, str]:
    """Write the pinned quiet-box reading as real files under ``tmp``; returns ``(meminfo_path, cgroup_root)``
    (the launcher's ``MEMINFO_PATH`` / ``CGROUP_ROOT`` seams).  Same numbers as the #1390 fixture."""
    meminfo = os.path.join(tmp, "meminfo")
    with open(meminfo, "w") as f:
        f.write(
            f"MemTotal:       {_QUIET_MEMTOTAL_KB} kB\n"
            f"MemFree:        110046680 kB\n"
            f"MemAvailable:   {_QUIET_MEMAVAIL_KB} kB\n"
            f"Shmem:          {_QUIET_CG_SHMEM_B // 1024} kB\n"
            f"SwapTotal:      0 kB\n")
    cg = os.path.join(tmp, "cgroup")
    os.makedirs(cg, exist_ok=True)
    for name, text in (
            ("memory.current", f"{_QUIET_CG_CURRENT_B}\n"),
            ("memory.peak", f"{_QUIET_CG_CURRENT_B}\n"),
            ("memory.max", "max\n"),
            ("memory.events", "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n"),
            ("memory.stat", f"anon {_QUIET_CG_ANON_B}\nfile {_QUIET_CG_FILE_B}\nshmem {_QUIET_CG_SHMEM_B}\n"
                            f"unevictable {_QUIET_CG_UNEVICTABLE_B}\nslab_reclaimable {_QUIET_CG_SLAB_RECLAIM_B}\n")):
        with open(os.path.join(cg, name), "w") as f:
            f.write(text)
    return meminfo, cg


#: environment prefixes the hermetic run scrubs (the launcher and every module it imports read them at call time)
_SCRUB_PREFIXES = ("SGLANG_", "HTSGLANG_", "FLLIPER_")


class DryRunResult:
    """What one oracle run produced.  ``text`` is the NORMALISED stdout of ``launcher.main`` (``raw`` the verbatim one);
    ``rc`` the return value (None when ``main`` raised), ``exc_type``/``exc_msg`` the refusal that did,
    ``forced`` the ``refusals.forced_list()`` of the run (``[{code, text}]``), ``events`` nothing else."""

    __slots__ = ("rc", "exc_type", "exc_msg", "text", "raw", "forced", "argv", "exc_where")

    def __init__(self, rc, exc_type, exc_msg, text, raw, forced, argv, exc_where=""):
        self.rc, self.exc_type, self.exc_msg = rc, exc_type, exc_msg
        self.text, self.raw, self.forced, self.argv = text, raw, forced, argv
        #: AP-D: ``file:line in function`` of the innermost frame of the exception (empty = no exception); the verdict of a launcher
        #: CRASH (``IndexError`` inside the dry run) names where it happened, the golden header does not carry it
        self.exc_where = exc_where

    def header(self) -> str:
        """The first line of a golden dump: argv size, rc, exception -- the verdict of the run in one line."""
        return "# argv_n=%d rc=%r exc=%s: %s\n" % (
            len(self.argv), self.rc, self.exc_type, (self.exc_msg or "")[:300])

    def dump(self) -> str:
        """Header + normalised text: the golden-file content."""
        return self.header() + self.text


def _rebound(d: Any, replacements: Mapping[str, str]) -> Any:
    """A default value with the patched path constants swapped: a str itself, or the str elements of a tuple
    (``anchor_evidence_dirs: Sequence[str] = (EVIDENCE_DIR,)``); anything else is returned as it is."""
    if isinstance(d, str):
        return replacements.get(d, d)
    if isinstance(d, tuple) and any(isinstance(x, str) and x in replacements for x in d):
        return tuple(replacements.get(x, x) if isinstance(x, str) else x for x in d)
    return d


@contextlib.contextmanager
def _patched_default_args(module, replacements: Mapping[str, str]):
    """Rebind DEFAULT ARGUMENTS of the module's own functions whose default equals a patched constant.

    ``mock.patch.object(launcher, "STORE_ROOT", x)`` does not reach ``def sweep_store_residue(log, keep, dry, root=STORE_ROOT)``:
    the default was bound when the function was defined, so a dry run would list (and name in the plan) the LIVE store's
    residue under ``/spinning/hicache-weg2``.  The functions found by scanning, not by a hand list, so a new function with a
    default-bound path is covered too.  Restored on exit."""
    import inspect

    saved: List[Tuple[Any, tuple, Optional[dict]]] = []
    try:
        for _n, f in list(vars(module).items()):
            if not (inspect.isfunction(f) and f.__module__ == module.__name__):
                continue
            new = tuple(_rebound(d, replacements) for d in (f.__defaults__ or ()))
            newkw = {k: _rebound(v, replacements) for k, v in (f.__kwdefaults__ or {}).items()}
            if new != (f.__defaults__ or ()) or newkw != (f.__kwdefaults__ or {}):
                saved.append((f, f.__defaults__, f.__kwdefaults__))
                f.__defaults__ = new or None
                f.__kwdefaults__ = newkw or None
        yield
    finally:
        for f, d, kd in saved:
            f.__defaults__ = d
            f.__kwdefaults__ = kd


#: modules that hold PROCESS STATE the launcher's ``main`` writes (a rig-fingerprint cache, the active inventory view, the
#: weight-exchange geometry) are imported BEFORE the snapshot, so the guard below knows them from the first run on
_STATE_MODULES = ("sglang.srt.managers.corridor_guard", "sglang.srt.weg2.inventory_view",
                  "sglang.srt.weg2.weight_exchange_region", "sglang.srt.planner.pp_cut",
                  "sglang.srt.planner.pp_cut_launch")


@contextlib.contextmanager
def _module_state_guard(prefix: str = "sglang."):
    """Give every module-level variable of the ``sglang.*`` modules that exist at entry back its value at exit.

    ``launcher.main`` is a boot, not a pure function: it fills caches (``corridor_guard._RIG_FP_CACHE``), sets the active
    inventory view and the weight-exchange geometry.  Measured (2026-10-06): a ``--force`` run on TWO cards left the next
    THREE-card run in the same process refused (W40 PP-cut) instead of planned.  An oracle that answers the second question
    differently after the first is no oracle; propose() asks many.  A variable is restored by IDENTITY (rebound) and a
    list/dict/set by CONTENT (in place, the other modules hold the same object).  Variables the run ADDED stay (a lazy
    import's own globals); ``refusals`` and ``os.environ`` are reset by their own seams."""
    import importlib
    import sys

    for m in _STATE_MODULES:
        try:
            importlib.import_module(m)
        except Exception:  # noqa: BLE001 - a build without the module has no state of it to keep
            pass
    saved: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]] = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(prefix) or name == __name__:
            continue
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        ids: Dict[str, Any] = {}
        content: Dict[str, Any] = {}
        for k, v in list(d.items()):
            if k.startswith("__"):
                continue
            ids[k] = v
            if type(v) in (dict, list, set):         # plain containers only: a subclass (lazy mappings) is not ours to copy
                content[k] = v.copy()
        saved.append((mod, ids, content))
    try:
        yield
    finally:
        for mod, ids, content in saved:
            d = mod.__dict__
            for k, v in ids.items():
                if d.get(k, _MISSING) is not v:
                    d[k] = v
                c = content.get(k)
                if c is not None:
                    if type(v) is dict:
                        if v != c:
                            v.clear()
                            v.update(c)
                    elif type(v) is list:
                        if v != c:
                            v[:] = c
                    elif v != c:
                        v.clear()
                        v.update(c)


_MISSING = object()


def run_dry_run(launcher_argv: Sequence[str], devices: Sequence[Mapping[str, Any]], *, tree: str,
                env: Optional[Mapping[str, str]] = None, force: bool = False, tag: str = ORACLE_TAG,
                scratch: Optional[str] = None, replay_path: Optional[str] = None,
                keep_scratch: bool = False, farm_root: str = "",
                evidence_dir: Optional[str] = None) -> DryRunResult:
    """Run ``launcher.main(["--tree", tree, "--tag", tag, "--dry-run", (--force), *launcher_argv])`` hermetically.

    * ``devices``: replay rows (:func:`replay_from_hardware_profile` / :func:`replay_from_catalog` / the fixture),
      written to ``replay_path`` (default: inside ``scratch``) and armed through ``SGLANG_NVML_REPLAY_JSON`` -- the ONE
      seam all NVML readers of the launcher share.
    * ``env``: the process environment the entrypoint would give the launcher (``profile_launch_input(...).env``).
      Every ``SGLANG_*`` / ``HTSGLANG_*`` / ``FLLIPER_*`` variable of the surrounding process that ``env`` does not
      name is REMOVED for the run; the whole environment is restored afterwards (including what ``refusals.arm`` writes).
    * ``force``: ``--force`` (value refusals become ``FORCED-PAST`` lines; ``result.forced`` lists them).
    * ``evidence_dir``: None = the launcher's own ``EVIDENCE_DIR`` (the LIVE evidence dir: boot logs, records, censuses --
      the plan reads some of them; their enumeration is masked by :data:`LIVE_BOX_RULES`).  A path rebinds
      ``launcher.EVIDENCE_DIR`` and every default argument bound to it for the run (an import-time constant; the process
      environment is NOT touched: the group env lines of the plan stay those of the golden), and the
      text is normalised back to the live path, so a run on an overlay dir (the live dir's entries + one more log) is
      comparable to the golden -- that is how the test proves the golden does not move when the dir grows.
    * The module state ``refusals`` is reset before and after.

    ``launcher.main`` may raise (a refusal): that is a RESULT (``exc_type``), not an error of the oracle.  Only a failure
    of the harness itself (cannot write the scratch, import fails) propagates."""
    from unittest import mock

    from sglang.srt.weg2 import launcher, refusals

    own_scratch = scratch is None
    scratch = scratch or tempfile.mkdtemp(prefix="oracle-")
    os.makedirs(scratch, exist_ok=True)
    shm = os.path.join(scratch, "shm_EMPTY")
    host = os.path.join(scratch, "host")
    store = os.path.join(scratch, "store")
    for d in (shm, host, store):
        os.makedirs(d, exist_ok=True)
    rpath = write_replay(devices, replay_path or os.path.join(scratch, "nvml_replay.json"))
    meminfo, cg = write_quiet_host(host)

    run_env: Dict[str, str] = {k: v for k, v in os.environ.items() if not k.startswith(_SCRUB_PREFIXES)}
    run_env.update({k: str(v) for k, v in (env or {}).items()})
    run_env[ENV_NVML_REPLAY] = rpath
    run_env["CUDA_VISIBLE_DEVICES"] = ""
    # the launcher reads SGLANG_WEG2_STORE_ROOT ONCE at import (launcher.py:188/193): the module constants are patched
    store_root = run_env.setdefault("SGLANG_WEG2_STORE_ROOT", store)
    argv = ["--tree", tree, "--tag", tag, "--dry-run"] + (["--force"] if force else []) + list(launcher_argv)

    orig_roots = (launcher.SHM_DIR, launcher.MEMINFO_PATH, launcher.CGROUP_ROOT, launcher.STORE_ROOT)
    live_evidence = launcher.EVIDENCE_DIR
    ev_new = evidence_dir or live_evidence
    buf = io.StringIO()
    rc = exc = None
    forced: List[Dict[str, str]] = []
    try:
        with mock.patch.dict(os.environ, run_env, clear=True), \
                mock.patch.object(launcher, "SHM_DIR", shm), mock.patch.object(launcher, "MEMINFO_PATH", meminfo), \
                mock.patch.object(launcher, "CGROUP_ROOT", cg), \
                mock.patch.object(launcher, "STORE_ROOT", store_root), mock.patch.object(launcher, "STORE_ROOT_TOLD", True), \
                mock.patch.object(launcher, "EVIDENCE_DIR", ev_new), _module_state_guard(), \
                _patched_default_args(launcher, {orig_roots[0]: shm, orig_roots[1]: meminfo, orig_roots[2]: cg,
                                                 orig_roots[3]: store_root, live_evidence: ev_new}):
            refusals.arm(False)
            try:
                with contextlib.redirect_stdout(buf):
                    rc = launcher.main(argv)
            except BaseException as e:  # noqa: BLE001 - the refusal path raises; that is the result
                if isinstance(e, KeyboardInterrupt):
                    raise
                exc = e
            forced = refusals.forced_list()
            refusals.arm(False)
    finally:
        # refusals.arm(True) wrote ENV_FORCED_BOOT into os.environ INSIDE patch.dict -> restored with it; the module
        # state is reset here for the case that arm(False) above was skipped by an exception of the harness
        refusals.arm(False)
        # a replay path the library did not create is the caller's; ours goes with the scratch dir
        if own_scratch and not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    raw = buf.getvalue()
    text = normalise_dump(raw, tree=tree, host=host, shm=shm, store=store_root, scratch=scratch, tag=tag,
                          farm_root=farm_root, replay=rpath)
    if evidence_dir and evidence_dir != live_evidence:
        text = text.replace(evidence_dir, live_evidence)
        raw = raw.replace(evidence_dir, live_evidence)
    return DryRunResult(rc, type(exc).__name__ if exc else None, str(exc) if exc else "", text, raw, forced, argv,
                        _exc_where(exc, tree) if exc else "")


def _exc_where(exc: BaseException, tree: str) -> str:
    """``path:line in function`` of the innermost traceback frame of ``exc`` (the tree prefix cut to ``<TREE>``); '' when none."""
    import traceback

    try:
        fr = traceback.extract_tb(exc.__traceback__)
        if not fr:
            return ""
        last = fr[-1]
        fn = last.filename
        if tree and fn.startswith(tree):
            fn = "<TREE>" + fn[len(tree):]
        return "%s:%s in %s" % (fn, last.lineno, last.name)
    except Exception:  # noqa: BLE001 -- a diagnostic must never turn a result into an error
        return ""


# ---------------------------------------------------------------------------
# (e) normalisation: per-run-by-construction values, and the live-box readings
# ---------------------------------------------------------------------------

#: replaced on EVERY dump (they differ per run by construction): (name, regex, replacement)
_NORM_RULES: Tuple[Tuple[str, "re.Pattern[str]", str], ...] = (
    ("timestamp", re.compile(r"\[\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\]"), "[TS]"),
    ("epoch", re.compile(r"epoch=\d+(\.\d+)?"), "epoch=<EPOCH>"),
    ("boot-token", re.compile(r"BOOT_TOKEN=%s:\d+:\d+" % re.escape(ORACLE_TAG)), "BOOT_TOKEN=<TOKEN>"),
    ("tree-sha", re.compile(r"(tree=<TREE>|tree: <TREE>) @ [0-9a-f]{10}"), r"\1 @ <SHA>"),
    ("stamp", re.compile(r"stamp=\d{4}_\d{6}"), "stamp=<STAMP>"),
    # `(clean)` / `(DIRTY: <git status of the tree>)`: the tree's own state, not a plan value
    ("tree-state", re.compile(r"\((?:DIRTY: .*?|clean)\) stamp=", re.S), "(<TREE-STATE>) stamp="),
)

#: the LIVE-BOX readings the dry run takes from the running machine and its disks (``plan_diff.py:6-17``); masked for
#: the golden comparison only, never inside the dump itself
LIVE_BOX_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("shm-xchg-epoch", re.compile(r"/dev/shm/weg2-xchg-\d+")),
    ("box-state-at", re.compile(r"WEG2-DRY-RUN-BOX-STATE at=\S+")),
    ("host-live", re.compile(r"LIVE nonreclaim=[\d.]+ raw_current=[\d.]+ file_reclaimable=[\d.]+ GiB")),
    ("zfs-arc-size", re.compile(r"(c_max=[\d.]+ GiB) size=[\d.]+ GiB")),
    ("box-state-numbers", re.compile(r"memavail=[\d.]+ GiB anon=[\d.]+ GiB shmem=[\d.]+ GiB")),
    ("store-disk-free", re.compile(r"\bfree=[\d.]+ GiB")),
    ("disk-has-free", re.compile(r"which has [\d.]+ GiB free of")),
    ("l3-disk-free-gb", re.compile(r"against [\d.]+ GB free on")),
    ("foreign-load-now", re.compile(r"foreign_load_now=[\d.]+ GiB")),
    # W65 (``Weg2MeasuredAnchor``, ``launcher.py:16416``): when no D log of the form exists the launcher names EVERY
    # ``boot_*.D.log`` of the evidence dir with the reason it was rejected (``dual_w64.find_dual_d_measurement`` listing,
    # sorted by mtime; 834 names / 164 kB on 2026-10-06).  That list is the CONTENT OF THE LIVE EVIDENCE DIR, not a plan
    # value: any boot of the rig adds a name and would turn the reference red without a plan change.  Only the
    # enumeration after the colon is masked; ``W65 Weg2MeasuredAnchor: no anchor, heuristic path stands: no
    # boot_*.D.log of this form:`` stays in the text, so the day a D log of THIS form exists (an anchor is found, the
    # message changes, the plan moves) the diff goes red -- which is a real plan change.
    ("w65-d-log-enumeration", re.compile(r"(?<=no boot_\*\.D\.log of this form: ).*")),
    # the D-RANK VRAM fraction solve prints its own WALL TIME (``1295 Budget-Loesungen in 0.2 s``; 0.2-0.5 s between two
    # runs of the same plan on 2026-10-06): a clock reading, not a plan value.  The count before it stays.
    ("solver-wall-time", re.compile(r"(?<=Budget-Loesungen in )[\d.]+(?= s)")),
)


def normalise_dump(text: str, *, tree: str, host: str = "", shm: str = "", store: str = "", scratch: str = "",
                   tag: str = ORACLE_TAG, farm_root: str = "", replay: str = "") -> str:
    """Replace what differs per run by construction with tokens (tree path, private temp dirs, timestamps, epoch, boot
    token, tree sha, stamp).  Nothing else is touched: a value the launcher computed stays in the text."""
    # longest first: the scratch dir contains host/shm/store
    for needle, token in sorted(((tree, "<TREE>"), (host, "<HOST>"), (shm, "<SHM>"), (store, "<STORE>"),
                                 (scratch, "<SCRATCH>"), (farm_root, "<MODELS>"),
                                 (replay, "<NVML-REPLAY>")), key=lambda p: -len(p[0])):
        if needle:
            text = text.replace(needle, token)
    for _name, rx, rep in _NORM_RULES:
        text = rx.sub(rep, text)
    return text


def mask_live_box(text: str) -> Tuple[str, Dict[str, int]]:
    """Mask the live-box readings (:data:`LIVE_BOX_RULES`); returns ``(masked_text, {rule: hits})``."""
    counts: Dict[str, int] = {}
    for name, rx in LIVE_BOX_RULES:
        text, k = rx.subn(lambda m, name=name: "<%s>" % name + (m.group(1) if m.groups() else ""), text)
        counts[name] = k
    return text, counts


# ---------------------------------------------------------------------------
# (c) plan-dump parser
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^# argv_n=(\d+) rc=(\S+) exc=(\S*): ?(.*)$")
#: the launcher prefixes its log lines with ``[TS] WEG2-LAUNCH `` (timestamp token after normalisation)
_PREFIX_RE = re.compile(r"^(?:\[TS\]\s+)?(?:WEG2-LAUNCH\s+)?")
_FORCED_RE = re.compile(r"^FORCED-PAST (\S+) ?(.*)$")
_GROUP_ARGV_RE = re.compile(r"^group (P|D) argv: (.*)$")
_FRONT_ARGV_RE = re.compile(r"^front argv(?: \(dry\))?: (.*)$")
_KV_RE = re.compile(r"(?<![\w-])([A-Za-z_][\w.-]*)=([^\s;()\[\]]+)")
#: ``budget P group=P ordinal=0 nvml_idx=1 NVIDIA GeForce RTX 5090: 26064 MiB = total 32607 - ...``
_BUDGET_RE = re.compile(r"^budget (.+?) group=(\w+) ordinal=(\d+) nvml_idx=(\d+) (.+?): (\d+) MiB")
_TICKET_RE = re.compile(r"^#\d+[a-z]?(?:/#\d+[a-z]?)*$")


def _kind(body: str) -> str:
    """The marker a launcher line names: its first word (ticket tokens ``#114`` skipped) plus the following words that
    are all-uppercase (``PP-CUT SHIPPED:`` -> ``PP-CUT SHIPPED``, ``budget P group=..`` -> ``budget P``)."""
    toks = body.split()
    while toks and _TICKET_RE.match(toks[0]):
        toks.pop(0)
    if not toks:
        return ""
    out = [toks[0].rstrip(":")]
    for t in toks[1:4]:
        w = t.rstrip(":")
        if w and re.fullmatch(r"[A-Z][A-Z0-9-]*", w) and "=" not in t:
            out.append(w)
        else:
            break
    return " ".join(out)


def _kv(body: str) -> Dict[str, str]:
    """``key=value`` fields of a line (a value runs to the next blank; list values keep their commas, a trailing
    sentence comma/period/colon is dropped); the FIRST occurrence of a key wins."""
    out: Dict[str, str] = {}
    for k, v in _KV_RE.findall(body):
        out.setdefault(k, v.rstrip(",.:"))
    return out


def _flags(argv: Sequence[str]) -> Dict[str, str]:
    """flag -> value text of an argv (argparse: the LAST occurrence wins; a bare flag is ``""``; ``--f=v`` is split)."""
    out: Dict[str, str] = {}
    i = 0
    toks = [str(t) for t in argv]
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t.split("=", 1)
                out[k] = v
            elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[t] = toks[i + 1]
                i += 1
            else:
                out[t] = ""
        i += 1
    return out


def parse_plan_dump(dump: str) -> Dict[str, Any]:
    """The plan the launcher printed, as a dict of resolved values (an INDEX over the text; the text stays the golden).

    * ``header``: ``{argv_n, rc, exc_type, exc_msg}`` of the first line (:meth:`DryRunResult.header`).
    * ``lines``: every line of the plan, verbatim, in order (nothing is dropped).
    * ``kinds``: ``{marker: [line, ...]}`` -- lines grouped by their leading marker (:func:`_kind`).
    * ``kv``: ``{marker: [{key: value}, ...]}`` -- the ``key=value`` fields of EACH line of a marker.
    * ``budgets``: the ``budget <pass> group=.. ordinal=.. nvml_idx=.. <card>: <N> MiB`` lines as dicts.
    * ``pp_cut``: ``key=value`` fields of the ``PP-CUT SHIPPED`` line (``layers``, ``attn``, ``chosen_pool``, ...), or ``{}``.
    * ``group_argv``: ``{"P": [argv,...], "D": [...], "front": [...]}`` -- every printed launch line, split with shlex
      (the launcher prints each pass; the LAST one is what ships); ``group_flags``: ``{group: {flag: value}}`` of the last.
    * ``forced``: ``[{code, text}]`` of the ``FORCED-PAST`` lines.
    * ``w_codes``: ``{"W19": n, ...}`` -- the W-codes the plan names (``W<nn> Weg2...``), counted."""
    import shlex

    out: Dict[str, Any] = {"header": {}, "lines": [], "kinds": {}, "kv": {}, "budgets": [], "pp_cut": {},
                           "group_argv": {"P": [], "D": [], "front": []}, "group_flags": {}, "forced": [],
                           "w_codes": {}}
    for line in dump.splitlines():
        m = _HEADER_RE.match(line)
        if m and not out["header"] and not out["lines"]:
            out["header"] = {"argv_n": int(m.group(1)), "rc": m.group(2), "exc_type": None if m.group(3) in ("", "None") else m.group(3),
                             "exc_msg": m.group(4)}
            continue
        out["lines"].append(line)
        body = _PREFIX_RE.sub("", line.strip())
        kind = _kind(body)
        if kind:
            out["kinds"].setdefault(kind, []).append(line)
            out["kv"].setdefault(kind, []).append(_kv(body))
        for w in re.findall(r"\bW(\d+[a-z]?) Weg2[A-Za-z]+", body):
            out["w_codes"]["W" + w] = out["w_codes"].get("W" + w, 0) + 1
        fm = _FORCED_RE.match(body)
        if fm:
            out["forced"].append({"code": fm.group(1), "text": fm.group(2)})
        bm = _BUDGET_RE.match(body)
        if bm:
            out["budgets"].append({"pass": bm.group(1), "group": bm.group(2), "ordinal": int(bm.group(3)),
                                   "nvml_idx": int(bm.group(4)), "card": bm.group(5), "mib": int(bm.group(6))})
        if kind == "PP-CUT SHIPPED":
            out["pp_cut"] = _kv(body)
        for rx, key in ((_GROUP_ARGV_RE, None), (_FRONT_ARGV_RE, "front")):
            gm = rx.match(body)
            if gm:
                g, text = (gm.group(1), gm.group(2)) if key is None else ("front", gm.group(1))
                try:
                    out["group_argv"][g].append(shlex.split(text))
                except ValueError:
                    out["group_argv"][g].append(text.split())
    for g, lst in out["group_argv"].items():
        if lst:
            out["group_flags"][g] = _flags(lst[-1])
    return out


# ---------------------------------------------------------------------------
# (d) profile reader: release .env -> launcher argv + environment
# ---------------------------------------------------------------------------

#: ``/opt/htsglang/profiles/<line>/`` of the image -> host search dirs (first hit wins).  The census, the corridor sample
#: and the wake reference logs the release profiles name live there in the image and in these places on the rig.
DEFAULT_ASSET_DIRS = ("/spinning/gpu-arb/weg2/census", "/spinning/gpu-arb/weg2", "/spinning/evidence-665-f1",
                      "/spinning/gpu-arb/docker/profiles")
_IMAGE_PROFILES = "/opt/htsglang/profiles/"


class LaunchInput:
    """The launcher side of a release profile: ``argv`` (PROFILE_ARGS, placeholders and image paths resolved),
    ``env`` (what the entrypoint exports before the launcher starts: the file's top-level exports + ``profile_form_env``;
    ``profile_instr_env`` only with ``instruments="1"``), ``model``/``draft`` and ``vars`` (PROFILE_* scalars)."""

    __slots__ = ("argv", "env", "vars", "unresolved_paths", "source", "instruments")

    def __init__(self, argv, env, vars_, unresolved, source, instruments):
        self.argv, self.env, self.vars = argv, env, vars_
        self.unresolved_paths, self.source, self.instruments = unresolved, source, instruments

    @property
    def model(self) -> str:
        return str(self.vars.get("PROFILE_MODEL", ""))

    @property
    def draft(self) -> str:
        return str(self.vars.get("PROFILE_DRAFT", ""))


def resolve_image_path(token: str, asset_dirs: Sequence[str]) -> Tuple[str, bool]:
    """ONE image path ``/opt/htsglang/profiles/<line>/<file>`` (optionally with a prefix such as ``--flag=``) -> an
    existing host path.  Returns ``(path, found)``; an image path nothing resolves comes back UNCHANGED with
    ``found=False`` (the caller lists it, never silently keeps it)."""
    if _IMAGE_PROFILES not in token:
        return token, True
    pre, _, rest = token.partition(_IMAGE_PROFILES)
    rel = rest.split("/", 1)[1] if "/" in rest else rest        # drop the <line>/ component
    for d in asset_dirs:
        cand = os.path.join(d, rel)
        if os.path.exists(cand):
            return pre + cand, True
    return token, False


def _resolve_token(tok: str, asset_dirs: Sequence[str], miss: List[str]) -> str:
    """Resolve every image path inside ONE argv token (a token may hold a comma list, e.g. the wake reference logs)."""
    if _IMAGE_PROFILES not in tok:
        return tok
    parts = re.split(r"(,)", tok)
    out = []
    for p in parts:
        if _IMAGE_PROFILES in p:
            q, found = resolve_image_path(p, asset_dirs)
            if not found:
                miss.append(p)
            out.append(q)
        else:
            out.append(p)
    return "".join(out)


def profile_launch_input(env_path: str, *, instruments: str = "0", tag: str = ORACLE_TAG,
                         evidence_dir: str = "/spinning/evidence-665-f1", gpu_arb: str = "/spinning/gpu-arb",
                         asset_dirs: Sequence[str] = DEFAULT_ASSET_DIRS, runner=None) -> LaunchInput:
    """Evaluate the release profile ``env_path`` like the entrypoint and return what the launcher gets.

    Bash arrays are read by ``profile_json.dump_env`` (``declare -a`` + NUL records: ``PROFILE_ARGS`` keeps every token
    whole, quoted ``--extra-p '...'`` values included; a ``source "$(dirname ...)/27b-base.env"`` chain is evaluated by
    bash itself).  The entrypoint-supplied variables (``HTSGLANG_TAG``, ``SGLANG_WEG2_EVIDENCE_DIR``,
    ``SGLANG_WEG2_GPU_ARB``) are the dump's sentinels and are replaced by ``tag`` / ``evidence_dir`` / ``gpu_arb``.
    Image paths (``/opt/htsglang/profiles/<line>/<file>``) are resolved against ``asset_dirs`` (first hit wins); the ones
    that resolve nowhere stay as they are and are listed in ``unresolved_paths``.

    Not added here: ``--tree/--tag/--transport`` (the runner adds tree/tag) and ``--model``: a profile that names no model
    in PROFILE_ARGS (27b-base) gets it from the CALLER (``launch_argv(..., add_model=True)``), as plan_dump.py did."""
    from sglang.srt.weg2 import profile_json as PJ

    raw = PJ.dump_env(env_path, instruments, runner=runner)
    subst = {"@@HTSGLANG_TAG@@": tag, "@@SGLANG_WEG2_EVIDENCE_DIR@@": evidence_dir, "@@SGLANG_WEG2_GPU_ARB@@": gpu_arb}

    def sub(v: str) -> str:
        for k, val in subst.items():
            v = v.replace(k, val)
        return v

    miss: List[str] = []
    argv = [_resolve_token(sub(t), asset_dirs, miss) for t in raw["arrays"].get("PROFILE_ARGS", [])]
    env: Dict[str, str] = {}
    for k, v in raw["exports"]:
        env[k] = sub(v)
    for k, v in raw["form"]:
        env[k] = sub(v)
    if str(instruments) == "1":
        for k, v in raw["instr"]:
            env[k] = sub(v)
    vars_ = {k: sub(v) for k, v in raw["vars"].items()}
    # the instruments switch is itself read by the profile while sourced; the entrypoint exports it
    env["HTSGLANG_INSTRUMENTS"] = str(instruments)
    env["HTSGLANG_TAG"] = tag
    return LaunchInput(argv, env, vars_, miss, os.path.abspath(env_path), str(instruments))


#: where name farms live.  FIXED (not under the per-run scratch): launcher lines hash argv text (``WEG2-P-FORM key=``,
#: ``L3-PERSIST`` dir), so the farm path must be the same in every run or the golden moves with the temp dir.
DEFAULT_FARM_ROOT = "/tmp/planer_oracle_models"
_MC = "/spinning/llm_stuff/club-3090/models-cache/"
#: registry-named model dirs that are EMPTY on the rig box -> the sibling checkpoint of the same config
#: (``deskq/work/hw1004/plan_dump.py:41-52``; the calibration identity is the directory NAME, the sibling gives
#: ``config.json`` and the safetensors headers).  MEASURED 2026-10-06 (read-only ssh, Proxmox host): the siblings are NOT the
#: release checkpoints -- index total_size 29548245472 (gdncov-vocabembed) vs 30819147232 (gdncov sibling), draft safetensors
#: 2172742656 (DFlash2-W8-lued) vs 3848817896 (DFlash2 sibling).  A sibling is a LAST RESORT for a dry run that only needs
#: a plan to exist; a golden/reference run passes ``snapshots`` (``snapshot_checkpoint`` of the real directory), which
#: :func:`run_profile` prefers over any sibling.
DEFAULT_MODEL_SIBLINGS: Mapping[str, Sequence[str]] = {
    "Qwen3.8-27B-INT8-gdncov-vocabembed": (_MC + "Qwen3.8-27B-INT8-gdncov",),
    "Qwen3.8-27B-DFlash2-W8-lued": (_MC + "Qwen3.8-27B-DFlash2",),
}


#: files up to this size are copied into a checkpoint snapshot verbatim (config.json, tokenizer/generation configs, the
#: ``*.safetensors.index.json``); a bigger non-safetensors file is recorded by SIZE only
SNAPSHOT_COPY_MAX = 4 << 20
_SNAPSHOT_SCHEMA = "planer-oracle-checkpoint-snapshot/1"
#: the mtime every materialised stub file carries (2026-09-26 23:00:00 UTC: the host files' own day, a constant)
SNAPSHOT_MTIME_NS = 1790463600 * 10**9


#: stored blobs above this size are gzip-compressed (deterministic: mtime 0), so a snapshot of a 160 GB checkpoint (33 MB
#: of safetensors headers, a 25 MB index) is a few MB in the tree, not 58
SNAPSHOT_GZ_MIN = 64 << 10


def _store_blob(path: str, data: bytes) -> bool:
    """Write ``data`` to ``path`` (gzip to ``path + '.gz'`` above ``SNAPSHOT_GZ_MIN``); True when compressed."""
    if len(data) <= SNAPSHOT_GZ_MIN:
        with open(path, "wb") as out:
            out.write(data)
        return False
    with open(path + ".gz", "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as out:
        out.write(data)
    return True


def _load_blob(path: str, gz: bool) -> bytes:
    if gz:
        with gzip.open(path + ".gz", "rb") as fh:
            return fh.read()
    with open(path, "rb") as fh:
        return fh.read()


def snapshot_checkpoint(model_dir: str, out_dir: str, *, name: Optional[str] = None, source: Optional[str] = None) -> str:
    """Record what the launcher's dry run reads of a checkpoint directory, WITHOUT the weights: every ``*.safetensors``
    file as its header bytes (``8 + N``: the length word and the JSON) plus its size, every small other file verbatim,
    every big other file as a size.  The launcher takes per-tensor bytes from the headers and on-disk sizes from
    ``stat`` (``checkpoint_census`` / ``model_profile`` / ``form``), never from the data region, so a stub rebuilt by
    :func:`materialize_checkpoint` is the checkpoint as far as the plan can tell (round trip proven by the test).

    Run it on a box that HAS the checkpoint; the snapshot (header bytes only, a few MB) is committed as a fixture and the
    golden of that profile then runs anywhere.  Returns ``out_dir``; ``name`` (default: the directory name) is the
    registry name the stub is rebuilt under -- the calibration identity is the directory NAME.  ``source`` (default: the
    absolute ``model_dir``) is the provenance string written to the manifest; pass it when ``model_dir`` is a local header
    mirror of a checkpoint that lives elsewhere (e.g. read over ssh).  A ``*.safetensors.index.json`` is ALWAYS copied
    verbatim whatever its size (the launcher reads its ``weight_map``/``total_size``; a 25 MB index is real input)."""
    import struct

    src = os.path.abspath(model_dir)
    name = name or os.path.basename(src.rstrip("/"))
    if not os.path.isfile(os.path.join(src, "config.json")):
        raise FileNotFoundError("%s has no config.json: nothing to snapshot (an empty mount point?)" % src)
    files_dir = os.path.join(out_dir, "files")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(files_dir)
    entries: List[Dict[str, Any]] = []
    skipped_dirs: List[str] = []
    for fn in sorted(os.listdir(src)):
        path = os.path.join(src, fn)
        if os.path.isdir(path):
            skipped_dirs.append(fn)
            continue
        size = os.path.getsize(path)
        if fn.endswith(".safetensors"):
            with open(path, "rb") as fh:
                raw = fh.read(8)
                (n,) = struct.unpack("<Q", raw)
                hdr = raw + fh.read(n)
            if len(hdr) != 8 + n or 8 + n > size:
                raise ValueError("%s: header of %d bytes does not fit the file (%d bytes)" % (path, 8 + n, size))
            gz = _store_blob(os.path.join(files_dir, fn + ".hdr"), hdr)
            entries.append({"name": fn, "size": size, "kind": "header", "sha256_header": hashlib.sha256(hdr).hexdigest(),
                            **({"gz": True} if gz else {})})
        elif size <= SNAPSHOT_COPY_MAX or fn.endswith(".safetensors.index.json"):
            with open(path, "rb") as fh:
                gz = _store_blob(os.path.join(files_dir, fn), fh.read())
            entries.append({"name": fn, "size": size, "kind": "copy", **({"gz": True} if gz else {})})
        else:
            entries.append({"name": fn, "size": size, "kind": "zero"})
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": _SNAPSHOT_SCHEMA, "name": name, "source": source or src, "skipped_dirs": skipped_dirs,
                   "files": entries}, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return out_dir


def read_snapshot_manifest(snapshot_dir: str) -> Dict[str, Any]:
    with open(os.path.join(snapshot_dir, "manifest.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    if m.get("schema") != _SNAPSHOT_SCHEMA:
        raise ValueError("%s: not a checkpoint snapshot (schema=%r)" % (snapshot_dir, m.get("schema")))
    return m


def materialize_checkpoint(snapshot_dir: str, dest_parent: str) -> str:
    """Rebuild ``dest_parent/<name>`` from a :func:`snapshot_checkpoint`: real header bytes, files extended to the real
    size as SPARSE files (no disk use), copied files verbatim.  Idempotent: the directory is rebuilt from scratch."""
    m = read_snapshot_manifest(snapshot_dir)
    dest = os.path.join(dest_parent, m["name"])
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    for e in m["files"]:
        out = os.path.join(dest, e["name"])
        gz = bool(e.get("gz"))
        if e["kind"] == "copy":
            with open(out, "wb") as fh:
                fh.write(_load_blob(os.path.join(snapshot_dir, "files", e["name"]), gz))
        else:
            with open(out, "wb") as fh:
                if e["kind"] == "header":
                    fh.write(_load_blob(os.path.join(snapshot_dir, "files", e["name"] + ".hdr"), gz))
                fh.truncate(int(e["size"]))
        # a FIXED mtime: the launcher's L3 weights fingerprint (``l3_weights_fingerprint``) hashes (name, size, mtime_ns)
        # of the weight files, which names the store dir and the persisted identity in the plan -- a stub built "now"
        # would move that line on every run (measured 2026-10-06: two runs, two different dir hashes)
        os.utime(out, ns=(SNAPSHOT_MTIME_NS, SNAPSHOT_MTIME_NS))
    return dest


def ensure_model_dir(path: str, *, siblings: Sequence[str] = (), farm_root: str = DEFAULT_FARM_ROOT,
                     snapshot: str = "") -> str:
    """The model directory the dry run reads (``config.json``, safetensors headers).

    ``path`` itself when it has a ``config.json``.  Otherwise a symlink farm under ``farm_root/<basename(path)>`` (the
    calibration identity is the directory NAME: a registry-named dir whose files point at a sibling checkpoint of the SAME
    config, ``plan_dump.py:46-52``); ``siblings`` are tried in order.  Raises ``FileNotFoundError`` when neither has one --
    the oracle never invents a model.

    ``snapshot`` (a :func:`snapshot_checkpoint` directory of THIS checkpoint, i.e. of the very directory ``path`` names):
    when ``path`` is empty on this box the stub rebuilt from it is used under ``farm_root/<name>`` BEFORE any sibling --
    it is the checkpoint's own headers, a sibling is another checkpoint's."""
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    if snapshot and os.path.isfile(os.path.join(snapshot, "manifest.json")):
        if read_snapshot_manifest(snapshot)["name"] != os.path.basename(path.rstrip("/")):
            raise ValueError("snapshot %s is of %r, not of %r" % (snapshot, read_snapshot_manifest(snapshot)["name"],
                                                                 os.path.basename(path.rstrip("/"))))
        return materialize_checkpoint(snapshot, farm_root)
    for sib in siblings:
        if os.path.isfile(os.path.join(sib, "config.json")):
            farm = os.path.join(farm_root, os.path.basename(path.rstrip("/")))
            os.makedirs(farm, exist_ok=True)
            for name in os.listdir(sib):
                dst = os.path.join(farm, name)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(sib, name), dst)
            return farm
    raise FileNotFoundError("model dir %r has no config.json and no sibling %r does" % (path, list(siblings)))


class ProfileRun:
    """``run_profile`` result: the oracle ``result``, the profile's ``launch_input``, the final launcher ``argv`` (what
    ``launcher.main`` got after ``--tree/--tag/--dry-run``) and ``notes`` (every substitution the harness made --
    a farm for an empty model dir, an added ``--weg2-xchg-census-foreign`` -- so none is silent)."""

    __slots__ = ("result", "launch_input", "argv", "notes")

    def __init__(self, result, launch_input, argv, notes):
        self.result, self.launch_input, self.argv, self.notes = result, launch_input, argv, notes


def run_profile(env_path: str, devices: Sequence[Mapping[str, Any]], *, tree: str, force: bool = False,
                instruments: str = "0", extra_args: Sequence[str] = (), tag: str = ORACLE_TAG,
                scratch: Optional[str] = None, farm_root: str = DEFAULT_FARM_ROOT,
                siblings: Mapping[str, Sequence[str]] = DEFAULT_MODEL_SIBLINGS,
                asset_dirs: Sequence[str] = DEFAULT_ASSET_DIRS, evidence_dir: Optional[str] = None,
                snapshots: Optional[Mapping[str, str]] = None,
                launch_input: Optional["LaunchInput"] = None) -> ProfileRun:
    """Release profile -> launcher dry run on ``devices``: :func:`profile_launch_input`, the model name farms for model
    dirs that are empty on this box, :func:`run_dry_run`.  The ``plan_dump.py`` recipe as one call.

    Substitutions (all listed in ``notes``):

    * a profile that names no ``--model`` in PROFILE_ARGS (27b-base) gets ``--model PROFILE_MODEL``; with
      ``--spec-form DFLASH`` and no ``--dflash-draft-path`` also ``--dflash-draft-path PROFILE_DRAFT``;
    * a model/draft dir without ``config.json`` is replaced by the stub of its own header ``snapshots`` entry
      (``{registry dir name: snapshot dir}``, :func:`snapshot_checkpoint`) when there is one, else by its name farm
      (:func:`ensure_model_dir`) when ``siblings`` knows a sibling; otherwise it stays and the launcher's own refusal is
      the result;
    * a farm path is not the path the census was measured on: ``--weg2-xchg-census-foreign`` is added when a farm is used
      and the argv names a census (the flag the dual1i profile itself carries)."""
    # AP-C: ``launch_input`` = a ready LaunchInput (a ``propose()`` result: argv + env + the profile's model/draft vars) instead
    # of reading ``env_path``; everything below is the same recipe
    li = launch_input if launch_input is not None else profile_launch_input(
        env_path, instruments=instruments, tag=tag, asset_dirs=asset_dirs)
    notes: List[str] = []
    farmed = False

    def farm(p: str, what: str) -> str:
        nonlocal farmed
        try:
            bn = os.path.basename(p.rstrip("/"))
            q = ensure_model_dir(p, siblings=siblings.get(bn, ()), farm_root=farm_root,
                                 snapshot=(snapshots or {}).get(bn, ""))
        except FileNotFoundError as e:
            notes.append("%s: %s -- kept, the launcher decides" % (what, e))
            return p
        if q != p:
            farmed = True
            snap = (snapshots or {}).get(os.path.basename(p.rstrip("/")), "")
            notes.append("%s: %s has no config.json -> %s %s" % (
                what, p, "header snapshot stub" if snap and os.path.isfile(os.path.join(snap, "manifest.json"))
                else "name farm", q))
        return q

    argv = list(li.argv)
    pre: List[str] = []
    if "--model" not in argv and li.model:
        pre += ["--model", farm(li.model, "model")]
    elif "--model" in argv:
        i = len(argv) - 1 - argv[::-1].index("--model")
        nm = farm(argv[i + 1], "model")
        if nm != argv[i + 1]:
            argv[i + 1] = nm
    flags = _flags(argv)
    if "--dflash-draft-path" not in flags and li.draft and flags.get("--spec-form", "").upper() == "DFLASH":
        pre += ["--dflash-draft-path", farm(li.draft, "draft")]
    elif "--dflash-draft-path" in argv:
        i = len(argv) - 1 - argv[::-1].index("--dflash-draft-path")
        nd = farm(argv[i + 1], "draft")
        if nd != argv[i + 1]:
            argv[i + 1] = nd
    # the model/draft dirs are ALSO named inside the quoted --extra-p/--extra-d/--extra values (NF:
    # ``--speculative-draft-model-path $PROFILE_DRAFT``, which the launcher prices at W128 from the draft's own headers):
    # a dir that was stood in for is stood in for EVERYWHERE the argv (and the profile's environment) names it
    subs: Dict[str, str] = {}
    env = dict(li.env)
    for what, orig in (("model", li.model), ("draft", li.draft)):
        o = (orig or "").rstrip("/")
        if o and (any(o in t for t in pre + argv) or any(o in str(v) for v in env.values())):
            q = farm(o, what)
            if q != o:
                subs[o] = q
    if subs:
        def _sub(t: str) -> str:
            for o, q in subs.items():
                t = t.replace(o, q)
            return t
        pre, argv = [_sub(t) for t in pre], [_sub(t) for t in argv]
        env = {k: _sub(str(v)) for k, v in env.items()}
    if farmed and "--weg2-xchg-census" in argv and "--weg2-xchg-census-foreign" not in argv:
        pre += ["--weg2-xchg-census-foreign"]
        notes.append("--weg2-xchg-census-foreign added (the census names the registry path, the farm is another path)")
    final = pre + argv + list(extra_args)
    res = run_dry_run(final, devices, tree=tree, env=env, force=force, tag=tag, scratch=scratch, farm_root=farm_root,
                      evidence_dir=evidence_dir)
    return ProfileRun(res, li, final, notes)


def launch_input_doc(li: LaunchInput) -> Dict[str, Any]:
    """The comparable form of a :class:`LaunchInput`: ``{"argv": [...], "env": {...sorted...}, "model", "draft"}``."""
    return {"argv": list(li.argv), "env": dict(sorted(li.env.items())), "model": li.model, "draft": li.draft}


# ---------------------------------------------------------------------------
# golden helpers
# ---------------------------------------------------------------------------

def golden_text(result: DryRunResult) -> str:
    """The content stored as a golden file: the dump with the live-box readings masked (so the golden is stable per
    machine state) -- the mask rule names stay visible as ``<rule>`` tokens."""
    masked, _ = mask_live_box(result.dump())
    return masked


def diff_lines(a: str, b: str) -> List[str]:
    """Unified diff of two texts after masking BOTH (``plan_diff.py``): empty list = 0 diff lines."""
    import difflib

    ma, _ = mask_live_box(a)
    mb, _ = mask_live_box(b)
    return list(difflib.unified_diff(ma.splitlines(), mb.splitlines(), "golden", "dump", lineterm="", n=0))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m sglang.srt.weg2.propose_oracle golden --profile X.env --replay R.json --out D.txt [--tree T] [--force]``:
    write the golden dump of one profile (the file ``test_planer_referenz_n3_1006`` compares against), ``snapshot``
    the headers of a checkpoint dir (run where the checkpoint exists), ``launch`` a profile's argv/env, or ``parse``
    a dump file to JSON."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden", help="dry-run one release profile on a replay inventory and write the dump")
    g.add_argument("--profile", required=True)
    g.add_argument("--replay", required=True, help="NVML replay JSON (a list of rows)")
    g.add_argument("--out", required=True)
    g.add_argument("--tree", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
    g.add_argument("--force", action="store_true")
    g.add_argument("--instruments", default="0")
    g.add_argument("--checkpoint-snapshot", action="append", default=[], metavar="DIR",
                   help="a `snapshot` directory: the stub of that checkpoint stands in for an empty model dir "
                        "(repeatable; matched by the registry directory name)")
    sn = sub.add_parser("snapshot", help="record the headers (not the weights) of a checkpoint dir, to be committed")
    sn.add_argument("--model-dir", required=True)
    sn.add_argument("--out", required=True)
    sn.add_argument("--name", default=None, help="registry directory name (default: the model dir's name)")
    lp = sub.add_parser("launch", help="release profile -> {argv, env} JSON (image paths left as they are)")
    lp.add_argument("--profile", required=True)
    lp.add_argument("--out", required=True)
    lp.add_argument("--instruments", default="0")
    pz = sub.add_parser("parse", help="plan dump file -> JSON (stdout)")
    pz.add_argument("dump")
    ns = ap.parse_args(argv)
    if ns.cmd == "parse":
        with open(ns.dump, encoding="utf-8") as fh:
            print(json.dumps(parse_plan_dump(fh.read()), indent=1, ensure_ascii=False))
        return 0
    if ns.cmd == "snapshot":
        snapshot_checkpoint(ns.model_dir, ns.out, name=ns.name)
        m = read_snapshot_manifest(ns.out)
        print("%s: %d files (%d safetensors headers) -> %s" % (
            m["name"], len(m["files"]), sum(1 for e in m["files"] if e["kind"] == "header"), ns.out))
        return 0
    if ns.cmd == "launch":
        li = profile_launch_input(ns.profile, instruments=ns.instruments, asset_dirs=())
        with open(ns.out, "w", encoding="utf-8") as fh:
            json.dump(launch_input_doc(li), fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        print("%s: argv=%d env=%d -> %s" % (os.path.basename(ns.profile), len(li.argv), len(li.env), ns.out))
        return 0
    snaps = {read_snapshot_manifest(d)["name"]: d for d in ns.checkpoint_snapshot}
    run = run_profile(ns.profile, read_replay(ns.replay), tree=ns.tree, force=ns.force, instruments=ns.instruments,
                      snapshots=snaps)
    with open(ns.out, "w", encoding="utf-8") as fh:
        fh.write(golden_text(run.result))
    for n in run.notes:
        print("note:", n)
    print("%s: rc=%r exc=%s lines=%d -> %s" % (os.path.basename(ns.profile), run.result.rc, run.result.exc_type,
                                              run.result.text.count("\n"), ns.out))
    return 0 if run.result.exc_type is None else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
