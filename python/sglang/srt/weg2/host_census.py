"""Host census: WHO holds the cgroup's anon and shmem, per role and per class.

29.09. (z30w-park follow-up of the COLD_TIER_SHM post). The ledger priced
47.72 GiB against 79.46 GiB measured; after the mark, the arena and the store
were corrected ~10.5 GiB stayed unbooked, and every term of it is a MEASURED
quantity nobody wrote down:

* anon outside the ranks -- the front, the two ``sglang.launch_server`` main
  processes (tokenizer manager + HTTP server of P and of D), the two
  detokenizers, the torch-inductor compile workers the ranks fork, the PLE
  pread workers (creep_0929.jsonl 09:12:00Z: 1.13 + 1.13 + 1.04 + 1.09 +
  1.09 + 4 x 0.44 + 4 x 0.01 GiB Pss_Anon);
* shmem outside the store and the arena term -- the flip lane ring
  (/dev/shm/weg2-seq-*, 1.50 GiB), the arena's sidecar widths (the QSA index
  page, arena-49152.bin 0.26 GiB), the hand-off directory (0.23 GiB).

This module turns a process listing into those posts and keeps them as a
RECORD per model|form (max over the samples of a boot, max over boots), so the
ledger charges what that model in that form was seen to hold -- never a
constant. Shmem the classes cannot name is kept as ``unattributed_shm_gib``
and printed as ``ungebucht``, never folded into a post.

Pure but for /proc; the classification functions take text so they test
without a live box.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

GIB = float(1 << 30)
KIB = 1024

RECORD_NAME = "host_census_record.json"
CENSUS_MARKER = "WEG2-HOST-CENSUS"

#: the rank processes -- priced by the ledger's own heap/image terms, never here
RANK_ROLE = "rank"
#: non-rank roles, in the order the census line prints them
NONRANK_ROLES = (
    "front", "server_main", "launcher", "detokenizer", "inductor_compile_worker",
    "ple_pread_worker", "mp_resource_tracker", "other",
)


def classify_process(comm: str, cmd: str) -> str:
    """The role of one process from its ``comm`` and command line."""
    comm = str(comm or "")
    cmd = str(cmd or "")
    if comm.startswith("sglang::schedul") or "sglang::scheduler" in cmd:
        return RANK_ROLE
    if comm.startswith("sglang::detoken") or "sglang::detokenizer" in cmd:
        return "detokenizer"
    if "sglang.srt.weg2.front" in cmd:
        return "front"
    if "sglang.srt.weg2.launcher" in cmd:
        return "launcher"
    if "sglang.launch_server" in cmd:
        return "server_main"
    if "compile_worker" in cmd:
        return "inductor_compile_worker"
    if "ple_pread_worker" in cmd:
        return "ple_pread_worker"
    if "resource_tracker" in cmd:
        return "mp_resource_tracker"
    return "other"


#: shmem classes the ledger BOOKS elsewhere (store = cold_tier_shm, arena
#: KV/draft/mamba = arena term, l3idx = l3_index) -- named, not charged here
BOOKED_ELSEWHERE = ("store", "arena_booked", "l3idx")
#: shmem classes this census charges as their own posts
CHARGED_CLASSES = ("seq_ring", "arena_sidecar", "arena_handoff")


def shm_class(path: str, store_dir: str = "", arena_booked_widths: Sequence[int] = ()) -> str:
    """The class of one shmem path (a /dev/shm or store file, or an anonymous
    shared mapping). ``arena_booked_widths``: the page widths the ledger's
    arena term already prices (KV, draft, mamba); any other arena width is a
    sidecar (the QSA index page)."""
    p = str(path or "")
    if store_dir and (p == store_dir or p.startswith(store_dir.rstrip("/") + "/")):
        return "store"
    if p.startswith("/dev/zero") or p.startswith("/memfd:") or p.startswith("/SYSV") or p == "[anon_shmem]":
        return "anon_shared"
    if "/weg2-seq-" in p:
        return "seq_ring"
    if "/weg2-xchg-" in p:
        return "xchg"
    if "/weg2-bar1-" in p:
        return "bar1"
    if "/weg2-arena-" in p:
        if p.endswith("-l3idx") or "-l3idx/" in p:
            return "l3idx"
        if "/handoff" in p:
            return "arena_handoff"
        m = re.search(r"/arena-(\d+)\.bin", p)
        if m and int(m.group(1)) not in set(int(w) for w in arena_booked_widths):
            return "arena_sidecar"
        return "arena_booked"
    if "/sgl-cold-" in p:
        return "store"
    return "other_tmpfs"


def roles_from_procs(procs: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    """Pss_Anon GiB per role from rows with ``comm``, ``cmd``, ``pss_anon_k``
    (the creep sampler's shape, and :func:`sample_live`'s)."""
    out: Dict[str, float] = {}
    for p in procs:
        role = classify_process(str(p.get("comm", "")), str(p.get("cmd", "")))
        out[role] = out.get(role, 0.0) + float(p.get("pss_anon_k", 0) or 0) * KIB / GIB
    return out


def census_from_creep(rec: Mapping[str, object], *, store_gib: float,
                      arena_booked_widths: Sequence[int] = (), source: str = "") -> Dict[str, object]:
    """A census from one creep_*.jsonl row (creep_sample.py: Pss per pid, du of
    /dev/shm, cgroup memory.stat). The store is not under /dev/shm there, so
    its size comes in as ``store_gib`` from the map that dimensions it."""
    stat = rec.get("stat") or {}
    classes: Dict[str, float] = {"store": float(store_gib)}
    for ln in rec.get("shm_du") or []:
        parts = str(ln).split()
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].startswith("/dev/shm/"):
            continue
        path = parts[1]
        if "/weg2-arena-" in path:
            continue      # arena dir + its l3idx are broken down by arena_sub below
        c = shm_class(path, arena_booked_widths=arena_booked_widths)
        classes[c] = classes.get(c, 0.0) + int(parts[0]) * KIB / GIB
    for ln in rec.get("arena_sub") or []:
        parts = str(ln).split("\t")
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        c = shm_class(parts[1], arena_booked_widths=arena_booked_widths)
        classes[c] = classes.get(c, 0.0) + int(parts[0]) * KIB / GIB
    cg_shmem = float(stat.get("shmem", 0) or 0) / GIB
    named = sum(v for k, v in classes.items() if k != "anon_shared")
    return {
        "at": str(rec.get("ts", "")),
        "source": source or f"creep {rec.get('name', '')} {rec.get('ts', '')}",
        "roles_anon_gib": roles_from_procs(rec.get("procs") or []),
        "shm_classes_gib": classes,
        "cg_anon_gib": float(stat.get("anon", 0) or 0) / GIB,
        "cg_shmem_gib": cg_shmem,
        "unattributed_shm_gib": max(0.0, cg_shmem - named),
    }


# -- live sampling (the front's periodic thread) -----------------------------

_SMAPS_HEAD = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s+(\S+)\s+\S+\s+\S+\s+\d+\s*(.*)$")


def smaps_shm_pss(text: str) -> Dict[str, int]:
    """Pss bytes of every SHARED mapping in one ``/proc/<pid>/smaps`` text,
    keyed by path (anonymous shared memory as ``/dev/zero (deleted)`` etc.).
    Private mappings are skipped: they are anon, not shmem."""
    out: Dict[str, int] = {}
    path, shared = "", False
    for ln in text.splitlines():
        m = _SMAPS_HEAD.match(ln)
        if m:
            perms, path = m.group(1), m.group(2).strip()
            shared = len(perms) >= 4 and perms[3] == "s"
            continue
        if shared and ln.startswith("Pss:"):
            kb = int(ln.split()[1])
            if kb:
                out[path] = out.get(path, 0) + kb * KIB
    return out


def tmpfs_mounts(mounts_text: str) -> List[str]:
    """Mount points whose pages are shmem (tmpfs/ramfs), longest first."""
    out = []
    for ln in mounts_text.splitlines():
        parts = ln.split()
        if len(parts) >= 3 and parts[2] in ("tmpfs", "ramfs"):
            out.append(parts[1])
    return sorted(set(out), key=len, reverse=True)


def is_shmem_path(path: str, mounts: Sequence[str]) -> bool:
    """A shared mapping is shmem when it is anonymous shared memory or a file
    on a tmpfs; a MAP_SHARED file on disk is page cache and not counted."""
    p = str(path or "").replace(" (deleted)", "")
    if shm_class(p) == "anon_shared":
        return True
    return any(p == m or p.startswith(m.rstrip("/") + "/") for m in mounts)


def _read(path: str) -> str:
    with open(path) as fh:
        return fh.read()


def sample_live(*, cgroup_root: str = "/sys/fs/cgroup", store_dir: str = "",
                arena_booked_widths: Sequence[int] = (), reader=_read) -> Dict[str, object]:
    """One census of the live cgroup: Pss_Anon per role (smaps_rollup), shmem
    Pss per class (smaps, shared mappings only), cgroup anon/shmem."""
    pids = [int(x) for x in reader(os.path.join(cgroup_root, "cgroup.procs")).split() if x.strip()]
    mounts = tmpfs_mounts(reader("/proc/mounts"))
    procs: List[Dict[str, object]] = []
    classes: Dict[str, float] = {}
    for pid in pids:
        try:
            comm = reader(f"/proc/{pid}/comm").strip()
            cmd = reader(f"/proc/{pid}/cmdline").replace("\0", " ").strip()
            roll = reader(f"/proc/{pid}/smaps_rollup")
            m = re.search(r"^Pss_Anon:\s+(\d+) kB", roll, re.M)
            procs.append({"pid": pid, "comm": comm, "cmd": cmd,
                          "pss_anon_k": int(m.group(1)) if m else 0})
            for path, b in smaps_shm_pss(reader(f"/proc/{pid}/smaps")).items():
                if not is_shmem_path(path, mounts):
                    continue
                c = shm_class(path.replace(" (deleted)", ""), store_dir, arena_booked_widths)
                classes[c] = classes.get(c, 0.0) + b / GIB
        except (OSError, ValueError):
            continue    # a process that exited between listing and read
    stat = reader(os.path.join(cgroup_root, "memory.stat"))

    def _stat(k: str) -> float:
        m = re.search(rf"^{k}\s+(\d+)$", stat, re.M)
        return int(m.group(1)) / GIB if m else 0.0

    cg_shmem = _stat("shmem")
    named = sum(v for k, v in classes.items() if k != "anon_shared")
    return {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "live /proc smaps(_rollup) + memory.stat",
        "roles_anon_gib": roles_from_procs(procs),
        "shm_classes_gib": classes,
        "cg_anon_gib": _stat("anon"),
        "cg_shmem_gib": cg_shmem,
        # anonymous shared memory (TMS images, parked drafts, memfd) has no
        # path to name it by -- it stays inside `ungebucht`, its size listed
        # as the class anon_shared so the next reader knows what it holds
        "unattributed_shm_gib": max(0.0, cg_shmem - named),
    }


# -- the record ---------------------------------------------------------------

def census_key(model_digest: str, form: str) -> str:
    return f"{model_digest or '?'}|{form or '?'}"


def load_record(path: str) -> Dict[str, object]:
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def merge_into_record(path: str, key: str, census: Mapping[str, object]) -> Dict[str, object]:
    """Max-merge one census into the record under ``key`` (a peak is what the
    ledger must fund) and write it atomically. Returns the merged entry."""
    data = load_record(path)
    old = data.get(key) if isinstance(data.get(key), dict) else {}
    ent: Dict[str, object] = {
        "roles_anon_gib": dict(old.get("roles_anon_gib") or {}),
        "shm_classes_gib": dict(old.get("shm_classes_gib") or {}),
        "unattributed_shm_gib": float(old.get("unattributed_shm_gib") or 0.0),
        "samples": int(old.get("samples") or 0) + 1,
        "sources": list(old.get("sources") or [])[-4:] + [str(census.get("source", ""))],
        "last_at": str(census.get("at", "")),
    }
    for field in ("roles_anon_gib", "shm_classes_gib"):
        for k, v in dict(census.get(field) or {}).items():
            ent[field][k] = max(float(ent[field].get(k, 0.0)), float(v))
    ent["unattributed_shm_gib"] = max(ent["unattributed_shm_gib"],
                                      float(census.get("unattributed_shm_gib") or 0.0))
    # 29.09. NF1d (W21 84.16): the fields above are max-merged EACH ON ITS OWN,
    # so their sum is a sum of maxima from different samples (store 41.69 of
    # 09291559 + ungebucht 6.26 of z30w, whose store was 39.20). The peak of the
    # SUM is kept beside them, with the store/arena of that same sample, so the
    # ledger can cap its shmem claim at what one instant actually held.
    for _f in ("shm_total_max_gib", "shm_total_store_gib", "shm_total_arena_gib", "shm_total_source",
               "shm_rest_max_gib", "shm_rest_source"):
        if _f in old:
            ent[_f] = old[_f]
    _tot = census.get("cg_shmem_gib")
    # 30.09. LEDGER-FIXPOINT: the shmem OUTSIDE the store and the arena at the
    # SAME instant, max over the samples -- the ledger adds THIS arm's own
    # store/arena to it instead of charging an old instant's larger store.
    if _tot is not None:
        _c = dict(census.get("shm_classes_gib") or {})
        _rest = max(0.0, float(_tot) - float(_c.get("store", 0.0)) - float(_c.get("arena_booked", 0.0)))
        if _rest > float(ent.get("shm_rest_max_gib", -1.0)):
            ent["shm_rest_max_gib"] = _rest
            ent["shm_rest_source"] = f"{census.get('source', '')} {census.get('at', '')}".strip()
    if _tot is not None and float(_tot) > float(ent.get("shm_total_max_gib", -1.0)):
        _cls = dict(census.get("shm_classes_gib") or {})
        ent["shm_total_max_gib"] = float(_tot)
        ent["shm_total_store_gib"] = float(_cls.get("store", 0.0))
        ent["shm_total_arena_gib"] = float(_cls.get("arena_booked", 0.0))
        ent["shm_total_source"] = f"{census.get('source', '')} {census.get('at', '')}".strip()
    data[key] = ent
    tmp = f"{path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return ent


def ledger_terms(entry: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """The ledger posts of one record entry. Empty entry -> all 0 and
    ``census_source`` says UNMEASURED (the ledger prints it, never guesses)."""
    if not entry:
        return {"nonrank_anon_gib": 0.0, "seq_ring_gib": 0.0, "arena_sidecar_gib": 0.0,
                "arena_handoff_gib": 0.0, "unbooked_shm_gib": 0.0, "xchg_measured_gib": None,
                "arena_measured_gib": None, "other_tmpfs_gib": 0.0,
                "census_roles": {},
                "census_source": "UNMEASURED (no host census record for this model|form)"}
    roles = {k: float(v) for k, v in dict(entry.get("roles_anon_gib") or {}).items() if k != RANK_ROLE}
    cls = dict(entry.get("shm_classes_gib") or {})
    return {
        "nonrank_anon_gib": sum(roles.values()),
        "seq_ring_gib": float(cls.get("seq_ring", 0.0)),
        "arena_sidecar_gib": float(cls.get("arena_sidecar", 0.0)),
        "arena_handoff_gib": float(cls.get("arena_handoff", 0.0)),
        "unbooked_shm_gib": float(entry.get("unattributed_shm_gib") or 0.0),
        # 29.09.: the exchange carrier's own shmem as MEASURED (weg2-xchg-*
        # files). With a record present the ledger charges this, not the
        # priced bounce region (see host_ledger.charge_terms).
        "xchg_measured_gib": float(cls.get("xchg", 0.0)),
        # 29.09. (two-sided replay of 27B 09291331): the L2 arena file as
        # MEASURED, and the tmpfs no class names -- both are shmem the cgroup
        # holds; charge_terms books what no priced post already carries.
        "arena_measured_gib": (float(cls["arena_booked"]) if "arena_booked" in cls else None),
        "other_tmpfs_gib": float(cls.get("other_tmpfs", 0.0)),
        "census_roles": roles,
        "census_source": f"record: {int(entry.get('samples') or 0)} sample(s), last {entry.get('last_at', '?')}",
        # 29.09. NF1d: the peak of the SUM (one instant), None when never sampled
        "shm_total_max_gib": (float(entry["shm_total_max_gib"])
                              if entry.get("shm_total_max_gib") is not None else None),
        "shm_total_store_gib": float(entry.get("shm_total_store_gib") or 0.0),
        "shm_total_arena_gib": float(entry.get("shm_total_arena_gib") or 0.0),
        "shm_total_source": str(entry.get("shm_total_source") or ""),
        # 30.09. LEDGER-FIXPOINT: max over samples of (total - store - arena), None before
        "shm_rest_max_gib": (float(entry["shm_rest_max_gib"])
                             if entry.get("shm_rest_max_gib") is not None else None),
    }


BACKFILL_TAG = "WEG2-HOST-CENSUS SHM-TOTAL"


def _memts_span_and_shmem_max(path: str) -> Optional[Tuple[str, str, float, str]]:
    """``(first_ts, last_ts, shmem_max_gib, ts_of_max)`` of one memts CSV
    (launcher.start_memts: host /proc/meminfo per 5 s), None when unreadable."""
    import csv

    try:
        with open(path) as fh:
            rows = [r for r in csv.DictReader(fh)
                    if r.get("ts_utc") and str(r.get("shmem_kb", "")).isdigit()]
    except OSError:
        return None
    if not rows:
        return None
    top = max(rows, key=lambda r: int(r["shmem_kb"]))
    return rows[0]["ts_utc"], rows[-1]["ts_utc"], int(top["shmem_kb"]) / (1024 * 1024), top["ts_utc"]


def backfill_shm_total(path: str, key: str, evidence_dir: str) -> Tuple[Optional[Mapping[str, object]], str]:
    """29.09. NF1d: the record's peak of the SUM (``shm_total_max_gib``), when it
    is missing, rebuilt from the MEASURED source of the boot that last sampled
    into this key: its memts CSV (the one whose time span holds the entry's
    ``last_at``), host Shmem per sample, max over the boot. Written back with
    provenance and returned; ``(entry, line)``.

    The key IS the identity (checkpoint digest | form, what the front of that
    boot wrote under), so the sample is this model's and this form's. memts
    carries no store/arena PER SAMPLE: the instant's store/arena are taken as
    the entry's class maxima -- an UPPER-bound assumption for the store at that
    instant, named on the line. Host Shmem >= cgroup shmem: the total is an
    upper bound too. No entry, a field already there, or no CSV whose span
    holds ``last_at``: the entry unchanged (the ledger then keeps today's
    refusing sum) and the line says which.
    """
    import glob

    data = load_record(path)
    ent = data.get(key) if isinstance(data.get(key), dict) else None
    if ent is None or ent.get("shm_total_max_gib") is not None:
        return ent, ""
    at = str(ent.get("last_at") or "")
    hit = None
    for csv_path in sorted(glob.glob(os.path.join(evidence_dir, "docker_*", "memts_weg2_*.csv"))
                           + glob.glob(os.path.join(evidence_dir, "memts_weg2_*.csv"))):
        span = _memts_span_and_shmem_max(csv_path)
        if span and at and span[0] <= at <= span[1]:
            hit = (csv_path, span)
            break
    if hit is None:
        return ent, (f"{BACKFILL_TAG} NOT BACKFILLED: no memts CSV under {evidence_dir} spans the "
                     f"record's last_at={at or '?'} -- the census shmem is charged as the sum of its "
                     f"per-field maxima (refusing direction)")
    csv_path, (_t0, _t1, tot, ts) = hit
    cls = dict(ent.get("shm_classes_gib") or {})
    ent = dict(ent)
    ent["shm_total_max_gib"] = float(tot)
    ent["shm_total_store_gib"] = float(cls.get("store", 0.0))
    ent["shm_total_arena_gib"] = float(cls.get("arena_booked", 0.0))
    ent["shm_total_source"] = (f"backfill from memts {csv_path} sample {ts} (host Shmem, upper bound; "
                               f"store/arena of that instant = the record's class maxima, upper-bound "
                               f"assumption: memts has no per-sample classes)")
    data[key] = ent
    persisted = "written back"
    try:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError as e:
        persisted = f"NOT written back ({type(e).__name__}: {e}), used for this pricing only"
    return ent, (f"{BACKFILL_TAG} BACKFILLED shm_total_max={tot:.2f} GiB "
                 f"store@={ent['shm_total_store_gib']:.2f} arena@={ent['shm_total_arena_gib']:.2f} "
                 f"-- {ent['shm_total_source']}; {persisted}")


def census_line(key: str, terms: Mapping[str, object]) -> str:
    roles = dict(terms.get("census_roles") or {})
    rl = " ".join(f"{r}={roles[r]:.2f}" for r in NONRANK_ROLES if roles.get(r))
    return (f"{CENSUS_MARKER} key={key} nonrank_anon={float(terms['nonrank_anon_gib']):.2f} GiB "
            f"[{rl or 'none'}] seq_ring={float(terms['seq_ring_gib']):.2f} "
            f"arena_sidecar={float(terms['arena_sidecar_gib']):.2f} "
            f"arena_handoff={float(terms['arena_handoff_gib']):.2f} GiB (charged, both moments); "
            f"ungebucht={float(terms['unbooked_shm_gib']):.2f} GiB shmem no class names "
            f"(NOT charged, printed; pathless shmem -- a share of it may be what the "
            f"ratchet/draft-park/anchor terms already price) -- {terms.get('census_source', '')}")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Seed/extend the record from a creep_*.jsonl row (the boot is gone, its
    census is not): --from-creep FILE --name SUBSTR --store-gib X --key K
    --record PATH [--widths W,W]."""
    import argparse

    ap = argparse.ArgumentParser(prog="host_census")
    ap.add_argument("--from-creep", required=True)
    ap.add_argument("--name", required=True, help="substring of the creep row's name (the boot tag)")
    ap.add_argument("--at", default="", help="exact ts of the row; default: the last row carrying shm_du")
    ap.add_argument("--store-gib", type=float, required=True, help="the store as its map dimensions it")
    ap.add_argument("--widths", default="", help="arena widths the arena term prices, comma list")
    ap.add_argument("--key", required=True)
    ap.add_argument("--record", required=True)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(a.from_creep) if l.strip()]
    rows = [r for r in rows if a.name in str(r.get("name", "")) and r.get("shm_du")
            and (not a.at or r.get("ts") == a.at)]
    if not rows:
        print("no matching creep row")
        return 2
    rec = rows[-1]
    c = census_from_creep(rec, store_gib=a.store_gib,
                          arena_booked_widths=[int(w) for w in a.widths.split(",") if w.strip()],
                          source=f"creep {os.path.basename(a.from_creep)} {rec.get('name', '')} {rec.get('ts', '')}")
    ent = merge_into_record(a.record, a.key, c)
    print(census_line(a.key, ledger_terms(ent)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
