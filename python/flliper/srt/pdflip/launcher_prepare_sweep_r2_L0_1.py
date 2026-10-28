"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import os
import time

from flliper.srt.compat_shims import name_counterparts as _name_counterparts
from flliper.srt.compat_shims import operator_dir as _operator_dir
from flliper.srt.managers import corridor_guard
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import rank_state as rank_state_mod
from flliper.srt.pdflip import ring_table
from flliper.srt.pdflip import weight_exchange_region
from flliper.srt.pdflip.launcher_d_leaf import _SPEC_FORM
from flliper.srt.pdflip.launcher_d_leaf import _alive
from flliper.srt.pdflip.launcher_d_leaf import _credit_counter_rows
from flliper.srt.pdflip.launcher_d_leaf import boot_state_dir
from flliper.srt.pdflip.launcher_d_leaf import derive_x_star
from flliper.srt.pdflip.launcher_d_leaf import dflash_placement
from flliper.srt.pdflip.launcher_d_leaf import dflash_produce_on_p
from flliper.srt.pdflip.launcher_d_mid import DFLASH_PRODUCE_ENV
from flliper.srt.pdflip.launcher_d_p_r2_L0 import EVIDENCE_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import GPU_ARB
from flliper.srt.pdflip.launcher_d_p_r2_L0 import SPEC_NUM_DRAFT_TOKENS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_MAMBA_BLOB_BYTES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_ROOT
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_log_plan import StoreDiskPlan
from flliper.srt.pdflip.launcher_p_l3_leaf import L3_PERSIST_PREFIX
from flliper.srt.pdflip.launcher_p_l3_leaf import _front_leg_form
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_mid import _serialised_cards
from flliper.srt.pdflip.launcher_p_mid import measure_x_inputs
from flliper.srt.pdflip.launcher_refuse_proc_leaf import DUPLEX_GATE
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _proc_cmdline
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _split_decisions
from flliper.srt.pdflip.launcher_refuse_proc_leaf import spec_form_is_dflash
from flliper.srt.pdflip.launcher_types import BootState
from flliper.srt.pdflip.launcher_types import BubbleMeasurement
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import DepthDecision
from flliper.srt.pdflip.launcher_types import GroupSpec
from flliper.srt.pdflip.launcher_types import HostRingPlan
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_types import PdFlipRingFormUnproven
from flliper.srt.pdflip.launcher_types import XSeed
from flliper.srt.pdflip.launcher_vision_spec_mid import user_reserve_rank_env
from flliper.srt.pdflip.launcher_xchg_vision_leaf import _tree_bytes
import math
import shutil
import signal
import socket
import subprocess

SHM_DIR = "/dev/shm"
#: C18: where the per-card host-ring files live under the MAP_SHARED form.  A
#: tmpfs, because the granules must be shared PAGES (both co-located rank
#: processes map the same file), not a disk-backed file.  THIS COMMENT IS THE
#: HOST RING'S, NOT THE STORE'S -- the store above is disk and shares nothing.
HOST_RING_DIR = "/dev/shm/pdflip-hostring"
#: #1264 ``--draft-kv-on-p``.  THE WHOLE PRODUCER FORM OF GROUP P, in one
#: place, so that ``on`` and ``off`` differ by the presence of ONE list and
#: not by five ``if``s scattered through ``argv_p``.
#:
#: The four speculative flags are group D's, BYTE-FOR-BYTE (they hash into the
#: drafter identity, W5); ``--speculative-draft-kv-only`` is the silencer that
#: turns that head into a draft-KV PRODUCER instead of a verifier.
#: ``--max-total-tokens`` rides WITH this list (``argv_p`` appends it right
#: behind these flags, only in the ``on`` form) and its VALUE is no longer a
#: constant: until 2026-09-09 it was ``P_MAX_TOTAL_TOKENS = 428000``, derived
#: once from boot weg2zr2's last-stage numbers minus the draft residue.  On
#: boot weg2sn5pre that constant was the BINDING term of group P's pool: P's
#: own sizing gave 463,289 tokens on the binding rank against the solver's
#: priced 463,763 (0.1 %%), and the cap cut the realised pool to 428,000
#: (-7.7 %%) -- BELOW the pool floor the boot claimed to hold.  The cap is
#: now the SHIPPED CUT'S PRICED POOL (``PCutFacts.pool_tokens``), the number
#: the #1286 pool model already charges the head against (``stage_fixed``
#: on the last stage), so realised = min(P's own sizing, priced) and the
#: floor binds the REALISED pool by construction.  Still head-scoped: in the
#: ``off`` form there is no head to make room for and the rg6 baseline
#: carried no such flag at all.  A term whose premise is the head belongs to
#: the head.
#: #1241 (refuter MF-6). THE BOOT'S SPEC AND KV FACTS, ONE WRITER.
#:
#: These five values were literals in FIVE places: the P draft-KV flag tuple,
#: ``common_flags``, ``argv_d``, and BOTH ``PlanInputs`` blocks that price a
#: weight vector. The argv half and the pricing half agreed by coincidence of
#: typing, not by construction -- so a change to the shipped draft-token count
#: (which #1242 is already moving) would have gone on pricing the OLD
#: configuration, silently, on every boot including the default one. That is a
#: second bookkeeping of the boot's own argv, which is the shape
#: UPSTREAM-MINIMAL forbids. One name each, read by everybody.
SPEC_ALGORITHM = "NEXTN"
SPEC_NUM_STEPS = 2
SPEC_EAGLE_TOPK = 1
#: The measured device figures for this dataset, SECTION 1as sec. 0 of
#: ``gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md`` (2026-09-09 08:41Z, 2 GiB of
#: /dev/urandom -- ``/dev/zero`` on a COMPRESSED dataset scored 10.5 GB/s and
#: measured nothing).  Only the two ``oflag/iflag=direct`` rows may be cited:
#: buffered write 3016.5 MB/s and any log-file read are ARC/compression
#: artefacts.  Used to PRICE reads, never to gate them.
STORE_DISK_WRITE_MBPS = 1346.9
STORE_DISK_READ_MBPS = 2754.9
#: THE DERIVATION IS NOT RE-GROUNDED, so a derived depth is not shipped
#: (#1240 FOLLOW FIX 1, MUST_FIX 3 of BOOT_weg2pp2_0907.md). MEASURED against
#: this workload, cited to that record's DEPTH VERDICT table: depth 1 cost
#: +6.9 % on TTFT p50 (3.215 vs 3.008 s, A/A floor 1.7 %) and depth 2 +9.6 %
#: (3.298), the bubble share on the binding stage did NOT fall (21.8/24.6 % at
#: depth 0, 26.8 % at depth 1, 24.1 % at depth 2, floor 2.8 pts), duty fell 15
#: points, and at the ~60 k deep point all depths sat inside the 3.35 % floor
#: = null. The #692 mechanism overlaps a rank's output exchange with its next
#: forward; the measurement says this workload's gap is not that exchange --
#: consistent with the depth line's own stall_share=0.221 against
#: bubble_share=0.466, i.e. over half the bubble was already queue starvation,
#: which depth cannot touch, and the remainder is evidently not exchange-bound
#: either. Re-grounding means a MEASURED exchange-bound share to derive from;
#: until one exists the arithmetic is printed and the shipped depth is 0. An
#: explicit --p-microbatch-depth still wins, priced exactly as before.
DEPTH_DERIVATION_GROUNDED = False
#: S104 review A2 (nf-review-s104-1020): a scheduler process that still lives this long after the teardown's TERM
#: is SIGKILLed by name. Only the L3 write-behind owner (D TP0, P PP0) handles SIGTERM at all
#: (mem_cache/l3_final_drain.py: drain bounded by FINAL_DRAIN_S = 3 s, then SIGTERM to itself); every other rank
#: keeps the default action. A Python handler runs only between bytecodes: an owner whose main thread is blocked in C
#: (an NCCL wait on a dead peer, a CUDA sync) never runs it and used to live until the 60 s SIGKILL below, holding
#: its card. FINAL_DRAIN_S + 2 s: the drain's own bound plus its join and the re-sent SIGTERM.
S104_SCHED_KILL_S = 5.0

def spec_flags(*, producer: bool) -> List[str]:
    """The speculative flag family of one group. Under NEXTN this is the
    constant family (P adds the producer silencer); under DFLASH the external
    draft, byte-identical on P and D (they hash into the drafter identity),
    D additionally with the compact draft cache window."""
    if spec_form_is_dflash():
        flags = [
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", str(_SPEC_FORM["draft_path"]),
            "--speculative-num-draft-tokens", str(int(_SPEC_FORM["block"])),
        ]
        if producer:
            flags.append("--speculative-draft-kv-only")
        else:
            flags += ["--speculative-draft-window-size", str(int(_SPEC_FORM["window"]))]
            if dflash_placement() == "solo":
                flags += ["--speculative-draft-placement", "solo"]
        return flags
    flags = [
        "--speculative-algorithm", SPEC_ALGORITHM,
        "--speculative-num-steps", str(SPEC_NUM_STEPS),
        "--speculative-eagle-topk", str(SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens", str(SPEC_NUM_DRAFT_TOKENS),
    ]
    if producer:
        flags.append("--speculative-draft-kv-only")
    return flags

def spec_form_env(group: str) -> Dict[str, str]:
    """Environment the form needs on one group: D's window pool under DFLASH,
    and (solo placement) the solo host's compact draft cache; P's draft-KV
    producer switch under DFLASH (--dflash-produce-on-p, ALWAYS written so the
    rank never falls back to its unset default). Nothing under NEXTN (the NF
    environment stays byte-identical)."""
    if spec_form_is_dflash() and group == "D":
        env = {"FLLIPER_DFLASH_WINDOW_POOL": "1"}
        if dflash_placement() == "solo":
            env["FLLIPER_DFLASH_SOLO_COMPACT"] = "1"
        return env
    if spec_form_is_dflash() and group == "P":
        return {DFLASH_PRODUCE_ENV: "1" if dflash_produce_on_p() else "0"}
    return {}

def resolve_x(override: Optional[int], evidence_dir: str, floor_tokens: int,
              accept: Optional[Callable[[str], bool]] = None) -> XSeed:
    """X and its PROVENANCE LINE, naming the three inputs and their source.

    Order: an explicit ``--tp-prefill-max-tokens`` wins (A1-4: derived with
    the flag as override), else the newest front log on this rig that
    carries all three instruments, else the recorded PRE-BARLINK pair.

    #1299: THE SCAN READS EVERY LOG UNTIL ONE QUALIFIES -- it used to stop
    after the newest EIGHT, and a log that parsed to ``None`` consumed a slot
    exactly like one that measured.  Measured consequence, four boots of one
    day: dec1 (07:32Z) and both sb5h (10:25Z, 10:31Z) found
    ``weg2sb5f...front.log`` at scan positions 1 and 5 and seeded X=8,742 from
    it; by dec2b (12:50Z) and shadow boot B (12:56Z) the same file had drifted
    to positions 8 and 9 behind eight barren logs -- refused or decode-only
    boots that print a front log but never drain P -- and both fell through to
    the recorded pair at X=22,556.  The seed tripled on two INDEPENDENT lines
    at the same wall-clock while the file it should have read sat unchanged on
    disk, because the cap counts files examined, not files that qualified.

    The two ``.P.log`` scans in this same launcher
    (:func:`newest_p_log_with_bubble` and the mean-prefix scan beside it)
    already walk their whole sorted list and skip what does not carry the line;
    this is that shape, and it deletes a hand number rather than adding one.

    Both provenance lines now carry the scan's own denominator: how many logs
    were skipped before the seed, or how many were examined before the
    fallback.  Without it "no front log on this rig carried all three
    instruments" reads as a fact about the rig when it was only a fact about
    the eight the scan looked at.

    UNIFY S4: ``accept`` is this boot's calibration identity
    (:func:`calib_log_accept_of`, the same filter the design-prefix and P-log
    scans take): a front log of another checkpoint, form or line never seeds
    X -- a 27B front log used to seed NF's X and the other way round (NF
    inventory, risk (c)). ``None`` (a desk caller) = every log, as before.
    Refused logs are counted in the provenance line, like the barren ones.
    """
    if override is not None:
        return XSeed(max(floor_tokens, int(override)), (
            f"X={max(floor_tokens, int(override))} source=flag "
            f"(--tp-prefill-max-tokens, operator override; floor {floor_tokens})"
        ), True)
    logs: List[str] = []
    try:
        logs = sorted(
            (os.path.join(evidence_dir, n) for n in os.listdir(evidence_dir) if n.endswith(".front.log")),
            key=lambda q: os.path.getmtime(q),
            reverse=True,
        )
    except OSError:
        logs = []
    skipped = 0
    refused = 0
    if accept is not None:
        kept = [q for q in logs if accept(q)]
        refused = len(logs) - len(kept)
        logs = kept
    refused_note = (f"; refused {refused} front log(s) of another checkpoint/form/line "
                    f"(calibration identity)" if accept is not None else "")
    for path in logs:
        got = measure_x_inputs(path, floor_tokens)
        if got is None:
            skipped += 1
            continue
        flip_s, r_d, r_p, n_f, n_d, n_p = got
        try:
            x = derive_x_star(flip_s, r_d, r_p, floor_tokens)
        except ValueError:
            skipped += 1
            continue
        return XSeed(x, (
            f"X={x} source=boot:{os.path.basename(path)} "
            f"X*=2*flip_s/(1/r_D-1/r_P) flip_s={flip_s:.2f} (median of {n_f}) "
            f"r_D={r_d:.0f} tok/s (median of {n_d} D single prefills) "
            f"r_P={r_p:.0f} tok/s (median of {n_p} P leg 1s) floor={floor_tokens}; "
            f"skipped {skipped} newer front log(s) carrying no complete instrument set"
            f"{refused_note}"
        ), True)
    rec_flip_s = _pconst("X_RECORDED_FLIP_S")
    rec_r_d = _pconst("X_RECORDED_R_D_TOKS")
    rec_r_p = _pconst("X_RECORDED_R_P_TOKS")
    x = derive_x_star(rec_flip_s, rec_r_d, rec_r_p, floor_tokens)
    return XSeed(x, (
        f"X={x} source=recorded PRE-BARLINK (examined {len(logs)} front log(s) on this rig, "
        f"none carried all three instruments{refused_note}) X*=2*flip_s/(1/r_D-1/r_P) flip_s={rec_flip_s} "
        f"r_D={rec_r_d:.0f} tok/s r_P={rec_r_p:.0f} tok/s "
        f"floor={floor_tokens} -- record 1l/1o weg2zr2, conservative corner of the measured "
        f"range; post-barlink r_D 1354 tok/s pushes X far higher and post-flipcost pulls it "
        f"back, so this is a fallback, never a table (O4)"
    ), False)

def refuse_if_front_unbindable(log: Log, host: str, port: int, dry: bool) -> None:
    """Pre-spawn probe: can the front bind ``host:port`` at all?

    aiohttp binds with SO_REUSEADDR, so the probe does the same. The case this
    exists for: the temporary LAN forwarder ``pdflip-lan-forward.socket``
    (2026-09-10) listens on 192.168.0.101:30030; a wildcard bind on the same
    port then fails with EADDRINUSE. Without this probe the front would die
    120 s into the boot after P and D are already up. Never on a dry run --
    a dry run is taken while the serving base still holds the port.
    """
    if dry:
        log(f"front bind probe: skipped on dry run ({host}:{port})")
        return
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as e:
        raise PdFlipLaunchRefused(
            f"front cannot bind {host}:{port} ({e}); a listener already holds the port. If it is the "
            f"temporary LAN forwarder: systemctl disable --now pdflip-lan-forward.socket (the front "
            f"serves the LAN itself on {host}); otherwise: ss -tlnp | grep :{port}"
        ) from e
    finally:
        probe.close()
    log(f"front bind probe: {host}:{port} bindable")

def stale_deadman_sweep(log: Log, ports: Sequence[int], dry: bool) -> None:
    out = subprocess.run(["pgrep", "-af", "boot_deadman.sh"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, argv = line.partition(" ")
        if any(f" {p}" in argv for p in ports) and "boot_deadman.sh" in argv:
            if dry:
                log(f"DRY-RUN: would kill stale deadman pid {pid} ({argv})")
            else:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    log(f"stale deadman pid {pid} killed (argv: {argv}) -- it watched a previous boot")
                except OSError:
                    pass

def store_sidecar_factor_of(profile: Optional[str] = None) -> float:
    """UNIFY S3: the sidecar factor from ``profile``'s store census row (the
    same arithmetic as :data:`STORE_SIDECAR_FACTOR`, which is the 27B row)."""
    kv = _pconst("STORE_CENSUS_KV_PAGES", profile) * _pconst("STORE_CENSUS_KV_PAGE_BYTES", profile)
    mamba = _pconst("STORE_CENSUS_MAMBA_BLOBS", profile) * STORE_MAMBA_BLOB_BYTES
    draft = _pconst("STORE_CENSUS_DRAFT_PAGES", profile) * host_ledger.DRAFT_PAGE_BYTES
    return (kv + mamba + draft) / float(kv)

def prepare_store(log: Log, plan: StoreDiskPlan, dry: bool) -> str:
    """Make this boot's store directory -- a PLAIN DIRECTORY, freshly.

    No mount and no umount: the store is the upstream file backend's own
    directory on the ZFS dataset.  "Fresh per boot" is kept exactly as the
    tmpfs form had it, but by NAMING the directory after the boot tag instead
    of by re-mounting: a previous boot's store is a different directory, so it
    can never be inherited by accident, and sweeping it is an explicit act
    (:func:`sweep_store_residue`) with its bytes printed.
    """
    if dry:
        log(
            f"DRY-RUN: would create store directory {plan.directory} "
            f"(max_size {plan.max_size_bytes / host_ledger.GIB:.2f} GiB, "
            f"min_free {plan.min_free_bytes / host_ledger.GIB:.2f} GiB)"
        )
        return plan.directory
    os.makedirs(plan.directory, exist_ok=True)
    return plan.directory

def sweep_store_residue(log: Log, keep: str, dry: bool, root: str = STORE_ROOT) -> int:
    """Remove PREVIOUS boots' store directories under :data:`STORE_ROOT`.

    The tmpfs form got "fresh store per boot" from the umount+mount pair.  On
    disk the equivalent act is explicit, and it is deliberately scoped to
    ``root`` and to directories that are not ``keep``: nothing outside this
    line's own root is listed, stat'ed or removed -- the same rule
    :func:`shm_residue_sweep` follows for /dev/shm, for the same reason.
    Returns the bytes reclaimed (0 in dry-run, which is stated, not implied).
    """
    try:
        names = sorted(os.listdir(root))
    except OSError:
        log(f"store residue: {root} unreadable -- NOT swept, and not read as empty")
        return 0
    # L3P: a persistent store (``l3-*``, this model's or another's) is never
    # residue -- only the per-boot tag directories are.
    stale = [n for n in names if os.path.join(root, n) != keep
             and not n.startswith(L3_PERSIST_PREFIX)
             and os.path.isdir(os.path.join(root, n))]
    if not stale:
        log(f"store residue: none under {root} (fresh store per boot)")
        return 0
    freed = 0
    detail = []
    for n in stale:
        p = os.path.join(root, n)
        alloc, _apparent, count = _tree_bytes(p)
        freed += alloc
        detail.append(f"{n}={alloc}B/{count}f")
    if dry:
        log(
            f"DRY-RUN store residue: would remove {len(stale)} previous store dir(s) "
            f"under {root} freeing {freed / host_ledger.GIB:.2f} GiB ({', '.join(detail)})"
        )
        return 0
    for n in stale:
        shutil.rmtree(os.path.join(root, n), ignore_errors=True)
    log(
        f"store residue: removed {len(stale)} previous store dir(s) under {root}, "
        f"freed {freed / host_ledger.GIB:.2f} GiB ({', '.join(detail)}) -- DISK, not "
        f"host RAM; nothing here was ever charged to the reap ledger"
    )
    return freed

def store_read_cost_line(plan: StoreDiskPlan, cap_tokens: int) -> str:
    """PRICE a store read from disk, once, at boot -- the read-side cost line.

    Bandwidth: the measured ``iflag=direct`` read-back of INCOMPRESSIBLE bytes,
    SECTION 1as sec. 0.  The buffered figure and every log-file read are ARC
    and compression artefacts and are not cited.

    The bound it is checked against is the one that already exists: #1238 fix 7
    ``Scheduler._deferred_prefetch_bound_s`` = ``prefetch_timeout_base + pages
    x prefetch_timeout_per_page``, with the defaults of
    ``PrefetchTimeoutConfig`` (base 2.0 s, 1.0 s per 1024 tokens) and
    ``page_size=1`` from :func:`common_flags`.  The backend's configured
    ``max`` (60 s) is printed beside it because the linear form is not clipped
    on this path and a reader should see both ceilings.
    """
    read_bps = STORE_DISK_READ_MBPS * 1e6
    cap_bytes = int(cap_tokens) * plan.cell_bytes
    cap_s = cap_bytes / read_bps
    pool_s = plan.p_pool_bytes / read_bps
    bound_cap = 2.0 + int(cap_tokens) / 1024.0
    bound_pool = 2.0 + plan.p_pool_tokens / 1024.0
    return (
        f"PDFLIP-STORE READ COST: per-request cap {int(cap_tokens)} tokens x "
        f"{plan.cell_bytes} B = {cap_bytes / host_ledger.GIB:.2f} GiB prices at "
        f"{cap_s:.2f} s of device time at the MEASURED {STORE_DISK_READ_MBPS:.1f} MB/s "
        f"direct read (SECTION 1as sec. 0, /dev/urandom, iflag=direct on the ZFS "
        f"compressed dataset -- the 3016.5 MB/s buffered write and every log-file read "
        f"are ARC/compression artefacts and are NOT cited). A P-POOL-SIZED PREFIX, the "
        f"case the flip actually asks for, is {plan.p_pool_tokens} tokens = "
        f"{plan.p_pool_bytes / host_ledger.GIB:.2f} GiB = {pool_s:.2f} s. Against the "
        f"#1238 fix 7 deferral bound (_deferred_prefetch_bound_s = base 2.0 s + 1.0 "
        f"s/KiToken at page_size=1, unclipped on that path): {bound_cap:.1f} s for the "
        f"cap and {bound_pool:.1f} s for the P pool -- covered "
        f"{bound_cap / cap_s:.0f}x and {bound_pool / pool_s:.0f}x. Even against the "
        f"backend's configured PrefetchTimeoutConfig.max of 60.0 s both reads fit "
        f"({60.0 / cap_s:.0f}x, {60.0 / pool_s:.0f}x). Writes price at "
        f"{STORE_DISK_WRITE_MBPS:.1f} MB/s and ride the backup thread "
        f"(cache_controller backup_thread_func), off the scheduler thread."
    )

def sweep_dead_credit_counters(log: Log, credit_dir: str = "",
                               dry: bool = False) -> List[str]:
    """Remove credit counters of DEAD boots at LAUNCH.  Never a live holder.

    FIX 3 round 3.  Teardown is exactly what a CRASHED boot does not run, and a
    crash is how three weg2rg2 counters were still on this rig when weg2rg3
    launched.  The composed epoch makes them harmless; this makes them absent,
    and the printed line is the evidence a reader wants instead of an inference.

    The holder test is the file's own ``publisher_pid``: a counter whose
    publisher is ALIVE belongs to a boot in flight, and is left alone and named
    -- the same rule ``presence_sweep`` applies to its own residue.
    """
    rows = _credit_counter_rows(credit_dir)
    if not rows:
        log("PDFLIP-VRAM-CREDIT residue: none")
        return []
    live = [n for _p, n, _b, pid in rows if pid and _alive(pid)]
    dead = [(p, n, b) for p, n, b, pid in rows if not (pid and _alive(pid))]
    if live:
        log(f"PDFLIP-VRAM-CREDIT residue: {len(live)} counter(s) held by a LIVE "
            f"publisher, left untouched: {live}")
    if dry:
        log(f"PDFLIP-VRAM-CREDIT DRY-RUN: would remove {len(dead)} dead-epoch "
            f"counter(s): {[n for _p, n, _b in dead]}")
        return []
    removed: List[str] = []
    for path, name, boot in dead:
        try:
            os.unlink(path)
        except OSError:
            continue
        removed.append(f"{name} (boot {boot or 'pre-token'})")
    log(f"PDFLIP-VRAM-CREDIT residue swept: {len(removed)} dead-epoch counter(s) "
        f"{removed} -- a crashed boot runs no teardown, so LAUNCH sweeps too")
    return removed

def sweep_xchg_semaphores(
    log: Log,
    shm_dir: str = SHM_DIR,
    dry: bool = False,
    unlink: Optional[Callable[[str], int]] = None,
) -> List[str]:
    """#1273 S3: ``sem_unlink`` the exchange semaphores a dead boot left behind.

    ORDERING IS LOAD-BEARING AND IS NOW STRUCTURAL: this function is called
    from INSIDE :func:`shm_residue_sweep`, between its refusal phase and its
    move phase.  It was previously a second call in ``main()`` after that
    sweep, and in that position it could never unlink anything -- the residue
    sweep had already ``shutil.move``d every ``sem.pdflip-xchg-*`` file into the
    archive, so this re-listed ``/dev/shm``, found nothing and logged
    ``residue: none`` on every boot, residue or not.  Both halves of the
    position matter: after the refusals (``shm_residue_sweep`` is the one
    authority that refuses the boot while another ``launch_server`` is alive,
    and unlinking a LIVE boot's handshake is worse than leaving a dead one's
    behind), and before the move.

    WHY A SEPARATE SWEEP AT ALL, given that ``sem.pdflip-xchg-*`` is in
    :data:`SHM_OWN_PREFIXES` and the generic sweep would archive the files:
    a POSIX named semaphore is destroyed by ``sem_unlink``, not by moving its
    backing file.  Renaming the file leaves the kernel object referenced by
    any process that still has it open, and a later ``sem_open(O_CREAT)`` on
    the same name would create a SECOND object while the first still holds
    waiters.  The prefix entry is the visibility half (a leaked handshake is
    named and its bytes counted); this is the correctness half.

    ``unlink`` is the one seam a hermetic test needs: ``sem_unlink`` acts on
    the machine's real ``/dev/shm`` namespace whatever ``shm_dir`` says, so a
    test that drove the real syscall would have to sweep a live boot's names
    to prove anything.  The default IS the real syscall.
    """
    try:
        names = sorted(os.listdir(shm_dir))
    except OSError:
        log(f"#1273 xchg semaphores: {shm_dir} unreadable -- NOT swept, and not read as empty")
        return []
    prefix = f"sem.{weight_exchange_region.REGION_PREFIX}"
    # rename transition (compat_shims): a dead boot under the other name left its handshake too
    stale = [n for n in names if n.startswith((prefix,) + _name_counterparts((prefix,)))]
    if not stale:
        log("PDFLIP-XCHG-SEM residue: none")
        return []
    if dry:
        log(f"PDFLIP-XCHG-SEM DRY-RUN: would sem_unlink {len(stale)} name(s): {stale}")
        return []
    if unlink is None:
        lib = weight_exchange_region._libc()

        def unlink(posix_name: str) -> int:
            return lib.sem_unlink(posix_name.encode("ascii"))

    gone: List[str] = []
    for entry in stale:
        posix_name = "/" + entry[len("sem."):]
        if unlink(posix_name) == 0:
            gone.append(posix_name)
    log(f"PDFLIP-XCHG-SEM residue swept: {len(gone)}/{len(stale)} sem_unlink'ed {gone} "
        f"-- a crashed boot runs no teardown, so LAUNCH sweeps too")
    return gone

def prepare_xchg_region(log: Log, boot_nonce: str, hook_mode: int,
                        dry: bool = False) -> Dict[str, object]:
    """#1273 S3: create the staging region + its 24 semaphores, return the env.

    ONCE PER BOOT, keyed by the BOOT NONCE and not by a flip epoch: the region
    is ftruncate'd to 385 MiB and (S4) cudaHostRegister'ed once, and neither
    belongs on a 1.5 s flip's critical path.  The per-flip stamp is
    ``XchgRegion.begin_flip('<boot>.<flip>')`` inside the ranks; see that
    method for why the two identities may not be conflated.

    TODO(#1273 S6): the caller that decides ``--pdflip-weight-source exchange``
    lives in S6.  It calls this once, before either group starts, and merges
    the returned ``env`` into BOTH groups' environment -- one region, six
    ranks, found by the same two names.  Under ``ring`` this is not called at
    all and no region file exists, which is what makes the ring form byte-for-
    byte today's path.
    """
    if dry:
        path = weight_exchange_region.region_path(boot_nonce)
        log(f"PDFLIP-XCHG-REGION DRY-RUN: would create {path} "
            f"({weight_exchange_region.REGION_BYTES} bytes) and 24 semaphores")
        return {"path": path, "boot": str(boot_nonce), "env": {}, "sems": 0, "line": ""}
    return weight_exchange_region.prepare_region(boot_nonce, hook_mode=hook_mode, log=log)

def teardown_xchg_region(log: Log, boot_nonce: str) -> Dict[str, int]:
    """#1273 S3 / spec section 3.8: unlink the 24 names and remove the region.

    Both halves matter -- teardown alone leaves a crashed boot's region
    behind, which is the case :func:`sweep_xchg_semaphores` and the
    :data:`SHM_OWN_PREFIXES` entry close at the NEXT launch.
    """
    return weight_exchange_region.teardown_region(boot_nonce, log=log)

def prepare_host_ring(cards: List[Card], log: Log, tag: str, form: str,
                      evidence_dir: str, boot_stem: str, dry: bool, *,
                      leg_form: str = "", pcie_directional: Optional[bool] = None,
                      duplex_probe: str = "", tree: str = "",
                      py: str = "",
                      p_argv: Optional[Sequence[str]],
                      weights_cpu_backup_armed: bool = True) -> HostRingPlan:
    """C20 + C18: solve the table, print L6, REFUSE by name, then arm the region.

    #1369 CLOSEOUT (user order 2026-09-14, "DIE 48GB MUESSEN WEG. UND ZWAR
    PRONTO"): ``weights_cpu_backup_armed=False`` (the operator's explicit
    ``--pdflip-weights-cpu-backup off``, or ``auto`` under an armed,
    authoritative exchange) short-circuits BEFORE ``ring_table.solve`` is
    ever called -- no table is measured, no per-card file is created.  This
    is the gap ``pdflip_weights_cpu_backup_ring_kw`` (``choose_host_ledger``,
    below in this module) does NOT close: that function zeroes the LEDGER'S
    PRICE for a ring whose bytes it is handed; this function is what
    actually calls ``os.ftruncate`` on a REAL per-card file sized from a
    REAL measured table, regardless of what any later pricing call does
    with the number.  Byte for byte with the launcher's own established
    idiom for "this mechanism has no subject on this arm" --
    :func:`prepare_weight_exchange` returns ``None`` outright for
    ``weight_source != "exchange"`` rather than pricing an unused exchange
    at zero -- this returns an unarmed, ``table=None`` plan rather than
    solving and then discarding a table nothing will read.

    ``table=None`` is safe here, not merely convenient: it is the SAME state
    :class:`HostRingPlan`'s own properties, ``choose_host_ledger`` ->
    ``pdflip_weights_cpu_backup_ring_kw``, and
    :func:`refuse_unless_same_form_source` already handle for the R22
    "could not measure" case -- none of the three raises or overcharges on
    it, and none of the three needs to know WHY the table is absent.

    Order is load-bearing: the inequalities are checked and the per-card files
    are created BEFORE either group starts, so a configuration that cannot walk
    the corridor never reaches a rank (W32/W34 rather than a mid-flip W31).

    TWO inequalities, and WHICH ONE A CARD IS CHECKED AGAINST IS THE CARD'S OWN
    PROPERTY (FIX 1 round 1).  R5's corridor is checked on every card (W32).
    The SERIAL form's ``H(c) >= image_W(c) + max_tag_S(c)`` is checked on the
    cards where the two legs cannot actually overlap (W34) -- see
    :func:`_serialised_cards`, which reads the leg form from the front and the
    key from the module that owns it, per card.  ``pcie_directional`` is the
    test override for that second fact: ``True`` = every card's key splits,
    ``False`` = none does, ``None`` = ask per card.

    NO TABLE IS NOT A FALLBACK: with ``table`` None this returns an un-armed,
    unpriceable plan and the ledger refuses the launch (W20).  See
    :class:`HostRingPlan`.

    ``p_argv`` is REQUIRED (keyword-only, no default) -- #1379.  This
    function has exactly ONE production caller (the ``main()`` step-1b call
    below), which has always built and passed ``form_argv_p`` unconditionally;
    the ``= None`` default that used to sit here served no caller and was only
    ever a landmine for a future refactor that could drop the argument and
    silently disarm the W48 form gate with no signal at the call site. Compare
    :func:`ring_table.solve`, which DOES keep its own ``p_argv=None`` default,
    because its own unit-test callers deliberately omit it to test some other
    property (see its docstring) -- the two functions have different caller
    populations and therefore different defaults, not one.
    """
    leg_form = leg_form or _front_leg_form()
    # C12/C13 + A1-4: the per-card duplex ratio, SOLVED from the step-0 probe's
    # own lines and published to the ranks as launcher output.  Printed per card
    # BEFORE either group starts, beside H(c), because the split's benefit is
    # per card and the flip's critical path is the card that does not get it.
    duplex, duplex_why = ring_table.solve_duplex(duplex_probe) if duplex_probe else (None, "no --duplex-probe given")
    plan = HostRingPlan(form=form, duplex=duplex)
    if duplex is None:
        plan.lines.append(
            f"PDFLIP-PCIE-DUPLEX UNMEASURED: {duplex_why} -- no card has a RATIO "
            "this boot, so none of them has an expectation about what the split "
            "buys.  Under gathered legs every key still SPLITS, and that is a "
            "correctness decision, not an optimisation: with one key the sleep "
            "leg blocks in the ring's acquire holding the key the wake leg needs "
            "(boot weg2rg2, W31 after 120.189 s -> W29 -> W4).  The predecessor "
            "of this line said an unmeasured card is 'slower, never wrong'; the "
            "metal refuted that."
        )
    else:
        plan.lines.extend(duplex.format_lines(cards, DUPLEX_GATE))
    for ln in plan.lines:
        log(ln)
    logged = len(plan.lines)
    if not weights_cpu_backup_armed:
        # #1369 CLOSEOUT: no subject, no measurement, no file -- see this
        # function's own docstring for why `table=None` here is the SAME
        # safe state three other functions already handle for the R22
        # "could not measure" case, reached here for a different, NON-
        # REFUSAL reason. Deliberately NOT worded like the A1-3 refusal
        # text below (never "REFUSES", never a W-code): a boot that reaches
        # this line is not failing, it is doing exactly what
        # `--pdflip-weights-cpu-backup off` (or `auto` under an armed,
        # authoritative exchange) asked for.
        plan.lines.append(
            "PDFLIP-HOST-RING ABSENT-BY-DESIGN: --pdflip-weights-cpu-backup "
            "resolved to off for this boot (explicit, or auto under an "
            "armed, authoritative exchange) -- the weights region never "
            "opens with enable_cpu_backup=True on this arm "
            "(weight_exchange.weights_cpu_backup_armed(), model_runner.py's "
            "own gate), so a per-card ring file here would be bytes nothing "
            "ever maps. No table is measured, no directory is made, no file "
            "is created or ftruncated. This is NOT a refusal."
        )
        for ln in plan.lines[logged:]:
            log(ln)
        return plan
    table, reason = ring_table.solve(cards, evidence_dir, boot_stem or None, p_argv=p_argv)
    if table is None:
        plan.lines.append(
            "PDFLIP-HOST-RING R22: no measured per-card byte table -- " + reason +
            ".  The planner REFUSES to guess H, and THIS BOOT REFUSES: the OLD "
            "flip form is not a fallback here, because its own host charge (one "
            "image plus one tag in flight) is solved from the same table, so the "
            "ledger has no host weights term and stops the launch by name (W20 "
            "PdFlipHostLedgerRefused, exit 2) before either group starts.  What "
            "makes a boot priceable again is a predecessor boot in "
            + (evidence_dir or "the evidence dir") +
            " that logged PDFLIP-CHUNK-BYTES / PDFLIP-FLIP-TAG lines and its own "
            "NVML -> CUDA ordinal map, or --ring-table-boot naming one."
        )
        for ln in plan.lines[logged:]:
            log(ln)
        return plan
    plan.table = table
    # FIX 3 round 3: every NEWER boot the solver skipped, with its reason, one
    # line each -- on SUCCESS, not only on failure.  Without them the log shows
    # a well-formed table and no way to ask why that boot and not the newest.
    # #1264 (B): PRINT THE CHOSEN SOURCE, not only the rejected ones.  The
    # predecessor took `reason.split("\n")[1:]`, i.e. it kept every SKIPPED line
    # and threw away line 0 -- the one that names the boot the table was
    # actually solved from, the form it was measured in, and (since #1264)
    # which boot's dormant sample was joined to it.  So a launch log could say
    # why four boots were NOT used and never say which one was, and the weg2t2b
    # ring drift had to be reconstructed from the ledger's arithmetic instead of
    # read off.  Line 0 first, then the skipped ones.
    _reason_lines = reason.split("\n")
    if _reason_lines and _reason_lines[0].strip():
        plan.lines.append("PDFLIP-HOST-RING SOURCE " + _reason_lines[0].strip())
    plan.lines.extend(ln for ln in _reason_lines[1:] if ln.strip())
    plan.lines.extend(table.format_l6())          # L6
    for ln in plan.lines[logged:]:
        log(ln)
    if form not in ("auto", "MAP_SHARED"):
        head = (
            f"W33 PdFlipRingFormUnproven: --ring-form {form or 'none'} selects no "
            "registration form, so the host ring cannot be armed -- AND THERE IS NO "
            "FALLBACK TO RUN INSTEAD (spec Amendment A1-3: the OLD flip form is "
            "INFEASIBLE on this host budget, W20 at every rung on the default "
            "table, boot weg2rg1).  This boot REFUSES, exit 2, before either group "
            "starts.  MAP_SHARED (cudaHostRegister on a /dev/shm MAP_SHARED file) "
            "is the form the step-0 probe PROVED on this rig "
            "(WEG2_BUILD_DECISIONS_0906 section 1p, 2026-09-07T23:13:23Z: all three "
            "gates pass, verdict BUILD-MAP_SHARED, duplex arm KEEP); it is the only "
            "form built, and the memfd candidate that probe retired is deleted."
        )
        plan.lines.append(head)
        log(head)
        raise PdFlipRingFormUnproven(head)
    # TWO launch checks, ONE rule for what a failure does: the ring cannot be
    # armed, and per Amendment A1-3 there is nothing to fall back to, so the
    # launcher REFUSES BY NAME and exits 2 -- whether or not a form was asked
    # for explicitly.  The predecessor downgraded silently to "the OLD flip form
    # runs (spec section 10.4)" unless --ring-form named a form; boot weg2rg1
    # then died of W20 two steps later, having told the operator a fallback had
    # run that never did.
    #
    # W32 is spec R5's corridor inequality -- the one that makes a blocking ring
    # deadlock-free when the legs are GATHERED (spec C9).  It is checked always.
    checks = [
        # L6 AGAINST AN INDEPENDENT SOURCE (train fix 5).  Listed FIRST because
        # it is the only one of the three that does not read the census that
        # sized the ring: it compares span1 against the weight bytes the
        # checkpoint says this boot's stages will load.  boot weg2tr2 passed
        # every check below it and then died at its first sleep with the runtime
        # saying "the launch check (L6) was violated" -- because L6 was checking
        # a premise against itself.  A stale table now stops here, by name,
        # before either group starts.
        ("W49 PdFlipRingWeightsUnderSized",
         "span1 is below the checkpoint's own weight bytes for that PP stage on "
         "{n} card(s) -- the ring cannot hold what the stage loads, so its FIRST "
         "pause blocks and the group dies on the acquire budget, which is boot "
         "weg2tr2 exactly",
         table.ckpt_refusals(),
         ring_table.PdFlipRingWeightsUnderSized),
        ("W32 PdFlipRingCreditRefused",
         "the R5 corridor inequality (H(c) >= image_W(c) - device_credit(c) + "
         "max_tag_S(c) + max_tag_W(c), spec C20) fails on {n} (card x direction) "
         "case(s); a negative slack is a flip that would wedge, not one that "
         "would be slow",
         table.refusals(),
         ring_table.PdFlipRingCreditRefused),
    ]
    # W34 IS PER CARD (FIX 1 round 1).  The serial requirement
    # H(c) >= image_W(c) + max_tag_S(c) applies to a card wherever the two legs
    # cannot actually overlap ON THAT CARD, and there are two ways for that to
    # be true:
    #
    #   * the FRONT does not gather at all (``FLIP_LEG_FORM != "interleave"``,
    #     which no state of this tree produces -- a stale worktree on
    #     PYTHONPATH, a partial rebase).  Then every card is serialised.
    #   * the CARD's key does not split (A1-4: 5090 1.759 and nvml2 1.687 reach
    #     R17's gate, nvml0 measured 1.316 and does not).  Each leg holds that
    #     key across its whole tag loop, so on that card the gathered pair is
    #     serialised again and R5's premise -- W's per-tag releases fund S's
    #     acquires -- is false there.
    #
    # The predecessor checked only the first and printed the second, and boot
    # weg2rg2 is what that cost: nvml0 passed R5 on both directions, the L6 line
    # said SERIAL FORM slack=1122/-2986 MiB on the SAME card, the launcher armed
    # anyway, and the first flip wedged for 120.189 s into W31 -> W29 on all
    # three P ranks -> group-fatal W4.  A refusal 120 s into a flip that has
    # already mutated VRAM is not the refusal this gate exists to give.
    card_uuids = [c.uuid for c in table.cards]
    splits = _split_decisions(
        card_uuids, duplex.ratios if duplex else {}, leg_form, pcie_directional
    )
    # THE DECISION IS PUBLISHED HERE AND NOWHERE ELSE (R19: launcher output).
    # The ranks build their key from this same string, so the inequality this
    # module checks and the key the metal takes cannot be two different facts.
    # THE FORMAT NAMES ITS VERSION IN A ROW OF ITS OWN (FIX 3 round 3, finding
    # 1).  A reader that knows the key refuses by name (W36) on an absent or
    # unknown version instead of resolving a single key on every card; a reader
    # that does NOT know it drops that one row on float() and still resolves
    # every real card.  FIX 2's "v2|" PREFIX had neither property -- it mangled
    # exactly the first card's uuid, which on this rig is the card carrying the
    # co-located pair, and was therefore a regression against the un-versioned
    # string it replaced.
    from flliper.srt.managers import pdflip_memory_saver as _saver
    plan.duplex_env = ",".join(
        [f"{_saver.PCIE_DUPLEX_VERSION_KEY}={_saver.PCIE_DUPLEX_FORMAT}"]
        + [
            f"{u}={('%.3f' % duplex.ratios[u]) if duplex and u in duplex.ratios else ''}"
            f":{'split' if splits[u] else 'single'}"
            for u in card_uuids
        ]
    )
    probe_notes: List[str] = []
    serialised = _serialised_cards(card_uuids, plan.duplex_env, leg_form,
                                   py=py, tree=tree, notes=probe_notes)
    plan.lines.extend(probe_notes)
    for ln in probe_notes:
        log(ln)
    checks_logged = len(plan.lines)
    for c in table.cards:
        is_serial = c.uuid in serialised
        plan.lines.append(
            f"PDFLIP-HOST-RING CHECK card={c.uuid} nvml{c.nvml_index} {c.name} "
            f"key={'SINGLE (both legs serialise here)' if is_serial else 'SPLIT per direction'} "
            f"duplex_ratio={(duplex.ratios.get(c.uuid) if duplex else None)} "
            f"gate={DUPLEX_GATE} leg_form={leg_form} -- checked against "
            + ("the R5 corridor (W32) AND the SERIAL requirement "
               "H >= image_W + max_tag_S (W34), because the two legs cannot "
               "overlap on this card"
               if is_serial else
               "the R5 corridor (W32) alone, because both legs can be in flight "
               "on this card at once")
        )
        # #1284: SAY WHAT THE WAIVER COSTS, per card, at launch.
        #
        # boot weg2sb5e armed with `serialised_cards=none` and printed
        # `SERIAL FORM ... slack=-1608/-2509`, `-881/-2986`, `-982/-2622` on
        # its three L6 rows -- the serial requirement failed on EVERY card and
        # the waiver was silent about it, because the CHECK line above says
        # only WHICH inequality was used, never what the unused one would have
        # demanded. Thirty flips later the wake leg did not run, the sleeping
        # leg met exactly that shortfall, and the C++ acquire burned its full
        # 110 s budget before dying with a line that blames L6.
        #
        # The debt is stated, not funded: paying it costs +8117 MiB of shmem on
        # these numbers and the host ledger's chosen arm is already ON its
        # store floor, so funding it would trade this wedge for a W21 refusal.
        # What covers it is the runtime guard named on the line.
        if not is_serial and c.serial_debt_mib > 0:
            plan.lines.append(
                f"PDFLIP-HOST-RING SERIAL-DEBT card={c.uuid} nvml{c.nvml_index} "
                f"debt_mib={c.serial_debt_mib} H={c.h_mib} "
                f"need_serial_d2p={c.need_serial_d2p_mib} "
                f"need_serial_p2d={c.need_serial_p2d_mib} -- this card is armed "
                "under the INTERLEAVE waiver, so R5's corridor was checked and "
                "the serial requirement H >= image_W + max_tag_S was NOT. If "
                "the two legs ever fail to overlap on this card the ring is "
                f"short by AT LEAST {c.serial_debt_mib} MiB and the sleeping "
                "leg cannot complete however long it waits -- at least, because "
                "image_W + max_tag_S prices a serial order in which S places "
                "only its largest tag, while a peer that releases NOTHING makes "
                "the requirement image_W + image_S, which no ring this box can "
                "fund. R5's premise is a RACE, not an "
                "invariant -- the front's own gather carries no happens-before "
                "between the sleeping leg's first acquire and the waking leg's "
                "first release. COVERED AT RUNTIME by W51 PdFlipHostRingUnfunded "
                "(pdflip/ring_guard.py), which emits the PDFLIP-RING NEED series per "
                "tag and refuses in ~2 s when free stops rising, instead of "
                "wedging for the C++ acquire budget (weg2sb5e: 110 s). A "
                "non-zero debt here with no W51 in the sleeping leg is weg2sb5e."
            )
    for ln in plan.lines[checks_logged:]:
        log(ln)
    if serialised:
        why_serial = (
            f"front.FLIP_LEG_FORM is {leg_form!r}, not 'interleave' -- this "
            "launcher is running against a front that does not gather its legs, "
            "which no state of this tree produces (C9), so EVERY card is "
            "serialised"
            if leg_form != "interleave" else
            f"the legs are gathered (C9) but on {len(serialised)} of "
            f"{len(table.cards)} card(s) -- {', '.join(serialised)} -- the "
            "measured duplex ratio does not reach R17's gate, so a sleep-D2H and "
            "the co-located wake-H2D take the SAME per-card PCIe key and each "
            "holds it across its whole tag loop; on those cards the pair is "
            "serialised whatever the front does"
        )
        checks.append(
            ("W34 PdFlipRingNeedsInterleave",
             why_serial + "; the serial requirement H(c) >= image_W(c) + "
             "max_tag_S(c) therefore applies THERE and fails on {n} "
             "(card x direction) case(s).  S blocks in acquire holding the "
             "per-card key, W cannot issue the release that would fund it "
             "because it is queued on that same key, and the acquire budget "
             "expires into W31 -> W29 on every rank of the group -> "
             "group-fatal W4.  That is boot weg2rg2, observed, not predicted",
             table.serial_refusals(only=serialised),
             ring_table.PdFlipRingNeedsInterleave))
    for name, why, bad, exc in checks:
        if not bad:
            continue
        head = (f"{name}: " + why.format(n=len(bad)) +
                ".  The host ring is NOT armed, and per spec Amendment A1-3 "
                "there is NO fallback flip form to run on this host budget "
                "(the OLD form refuses W20 at every rung on the default table, "
                "boot weg2rg1) -- so this boot REFUSES by name and exits 2, "
                "BEFORE either group starts.")
        plan.lines.append(head)
        plan.lines.extend(bad)
        for ln in plan.lines[-(len(bad) + 1):]:
            log(ln)
        raise exc(head + "\n" + "\n".join(bad))
    plan.epoch = int(time.time())
    plan.dir = f"{HOST_RING_DIR}-{tag}"
    if not dry:
        os.makedirs(plan.dir, exist_ok=True)
    for c in table.cards:
        path = os.path.join(plan.dir, f"{c.uuid}.ring")
        size = ring_table.MIB * 2 + c.h_mib * ring_table.MIB  # header granule + data
        if not dry:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.ftruncate(fd, size)
            finally:
                os.close(fd)
        plan.lines.append(
            f"PDFLIP-HOST-RING file {path} size={size // ring_table.MIB} MiB "
            f"(2 MiB header granule + H {c.h_mib} MiB) "
            + ("DRY-RUN: would be" if dry else "")
            + " created and ftruncated BEFORE either group starts"
        )
    plan.env_map = table.env_map()
    plan.form = "MAP_SHARED"          # 'auto' resolves to the one proven form
    plan.armed = True
    plan.lines.append(
        f"PDFLIP-HOST-RING ARMED form={plan.form} leg_form={leg_form} "
        f"serialised_cards={serialised or 'none'} "
        # #1284: this parenthetical used to read "each checked against the
        # SERIAL requirement, not R5's corridor" UNCONDITIONALLY, which is the
        # exact opposite of what the per-card CHECK lines say whenever
        # `serialised` is empty -- and it was empty on weg2sb5e, the boot that
        # then died of the un-checked serial requirement. A summary line that
        # contradicts its own detail lines sends every reader to the wrong
        # half of the arithmetic. It now states which inequality was used and
        # what debt that leaves.
        + (f"(SERIAL requirement H >= image_W + max_tag_S checked on "
           f"{len(serialised)}/{len(table.cards)} card(s); the rest were armed "
           f"on R5's corridor alone, carrying "
           f"{sum(c.serial_debt_mib for c in table.cards if c.uuid not in serialised)}"
           f" MiB of SERIAL-DEBT covered at runtime by W51 -- see the "
           f"PDFLIP-HOST-RING CHECK and SERIAL-DEBT lines per card) "
           if len(serialised) < len(table.cards) else
           "(SERIAL requirement H >= image_W + max_tag_S checked on every card "
           "-- see the PDFLIP-HOST-RING CHECK line per card) ")
        + f"epoch={plan.epoch} dir={plan.dir} "
        f"Sigma H={table.total_h_bytes // ring_table.MIB} MiB "
        f"Sigma span1={table.total_span1_bytes // ring_table.MIB} MiB granule=2 MiB "
        f"register_est_ms={int(table.total_span1_bytes / float(2**30) * 44)} "
        "(38-49 ms/GiB measured on the step-0 probe at ONE size point, 4 GiB -- "
        "an extrapolation, and it is a LAUNCH-moment charge: span 1 is "
        "registered at P's first pause, R7) "
        f"-- provenance: {table.provenance()}"
    )
    for ln in plan.lines[-(len(table.cards) + 1):]:
        log(ln)
    return plan

def user_reserve_env_entries(env: Dict[str, str], operator_extra: Optional[Dict[str, str]],
                             user_reserve_by_card: Optional[Dict[str, int]],
                             cards: Optional[Sequence["Card"]],
                             booked_rest_mib: Optional[Sequence[Optional[int]]]) -> None:
    """``build_env``'s reserve block (#236): publish :func:`user_reserve_rank_env` into ``env``.

    An operator value stated through ``--env-p/--env-d`` (``operator_extra``) is
    RESPECTED, not stripped (review SOLL-3): it is explicit operator input and the
    launcher's effective mapping is then not published over it; ``rank_reserve_kwargs``
    names that case in the log. Otherwise both variables are popped first (launcher
    OUTPUT, R19), so a shell value never describes a boot it did not resolve. (A helper
    and not inline: ``test_pdflip_hicache_draft_tier_0924`` keeps the word "group" out of
    ``build_env``'s code after the draft-tier pop.)
    """
    names = {corridor_guard.USER_RESERVE_ENV, corridor_guard.USER_RESERVE_MASKED_ENV}
    if names & set(operator_extra or {}):
        return
    for name in names:
        env.pop(name, None)
    env.update(user_reserve_rank_env(user_reserve_by_card, cards, booked_rest_mib))

def rank_state_dir_for(spec: GroupSpec) -> str:
    """Where group ``spec``'s ranks write their RankState: rankstate/<G> inside the
    boot's state directory (§2.2), else next to the group log (Phase 1)."""
    d = boot_state_dir()
    if d:
        return os.path.join(d, "rankstate", spec.name)
    return rank_state_mod.rank_state_dir_for_log(spec.log)

def write_p_respawn_spec(spec: GroupSpec, ns, log: Log, dry: bool,
                         card_need_mib: Optional[Dict[int, int]] = None) -> str:
    """S7b P-RESTART (Auftrag #214): the exec record of group P -- the argv/env/cwd launch_group actually
    exec'd, never re-planned -- for the front's P restart (pdflip/dual_p_restart.py, stage 2): same cards, same
    planner cut. Dual layout only; '' = not written (the front's restart is then unarmed and says so). The file
    is mode 0600 (the env is the group's real one, the log copy is the redacted one)."""
    if dry or not getattr(ns, "dual_layout", False) or not spec.pid:
        return ""
    from flliper.srt.pdflip import dual_p_restart as _s7b

    if not _s7b._on(os.environ, _s7b.RESTART_ENV):
        log(f"{_s7b.MARK} P-RESPAWN-SPEC not written: {_s7b.RESTART_ENV}=0 (stage 1 only, P is not restarted)")
        return ""

    path = _s7b.respawn_path(GPU_ARB, getattr(ns, "tag", "") or "untagged")
    try:
        _s7b.write_spec(path, argv=list(spec.argv), env=dict(spec.env), cwd=spec.cwd, log=spec.log,
                        port=int(spec.port), tag=str(getattr(ns, "tag", "") or ""), pid=int(spec.pid),
                        card_need_mib=card_need_mib)
    except OSError as e:
        log(f"{_s7b.MARK} P-RESPAWN-SPEC not written ({type(e).__name__}: {e}) -- the front's P restart is unarmed")
        return ""
    log(f"{_s7b.MARK} P-RESPAWN-SPEC {path}: argv {len(spec.argv)} items, env {len(spec.env)} keys (secrets "
        f"masked), cwd {spec.cwd or '-'}, mode 0600, P's planned peak per card (budget + overhead, MiB) "
        f"{card_need_mib or 'NONE -- the restart will refuse to spawn'} -- the front's P restart (stage 2) "
        f"re-execs exactly this")
    return path

def _stop_helper_pid(pid: int, expect: str = "boot_deadman") -> str:
    """TERM a helper: its whole process group when it leads one AND its argv
    still carries ``expect`` (a recycled pid that happens to lead some other
    group is never group-signalled), else the pid alone -- the pre-#135
    behaviour. Returns 'group', 'pid' or 'gone'."""
    try:
        if os.getpgid(pid) == pid and expect in _proc_cmdline(pid):
            os.killpg(pid, signal.SIGTERM)
            return "group"
    except OSError:
        pass
    try:
        os.kill(pid, signal.SIGTERM)
        return "pid"
    except OSError:
        return "gone"

def served_dormant_growth(cards: List[Card], profile: Optional[str] = None) -> Tuple[Optional[List[int]], str]:
    """``P_DORMANT_SERVED_GROWTH_MIB`` of the profile, per card ordinal, or
    ``(None, "")`` where the profile never measured it (27B today: its budgets
    stay byte-identical until a PDFLIP-DORMANT-SERVED record lands)."""
    try:
        vals = [int(v) for v in _pconst("P_DORMANT_SERVED_GROWTH_MIB", profile)]
    except KeyError:
        return None, ""
    if len(vals) != len(cards):
        raise PdFlipLaunchRefused(
            f"P_DORMANT_SERVED_GROWTH_MIB has {len(vals)} entries for {len(cards)} cards; "
            f"re-measure with python -m flliper.srt.pdflip.dormant_residue")
    return vals, "PDFLIP-DORMANT-SERVED record"

def _vfl_effective_draft(row) -> str:
    """Review S3: D's EFFECTIVE draft (the spec form main resolved, with an --extra-d draft-token override), not the
    profile row's; the row's draft kind:steps only where the spec form cannot be read."""
    try:
        extra = _SPEC_FORM.get("d_extra_draft_tokens")
        if spec_form_is_dflash():
            return (f"DFLASH:block={int(_SPEC_FORM['block'])}:window={int(_SPEC_FORM['window'])}"
                    + (f":tokens={extra}" if extra else ""))
        return f"{SPEC_ALGORITHM}:steps={SPEC_NUM_STEPS}:tokens={extra or SPEC_NUM_DRAFT_TOKENS}"
    except Exception:  # noqa: BLE001
        dr = getattr(row, "draft", None) if row is not None else None
        return "" if dr is None else f"{getattr(dr, 'kind', '')}:{getattr(dr, 'steps', '')}"

def record_line_spec(ns) -> str:
    """UNIFY S7: the one-string identity the front's ``--record-line`` parses
    (27B line_identity.spec form: model|repo|head|evidence)."""
    tree = getattr(ns, "tree", "") or ""
    return (f"model={getattr(ns, 'model', '')}|repo={os.path.abspath(tree) if tree else ''}"
            f"|head=HEAD|evidence={EVIDENCE_DIR}")

def solve_p_depth(
    measured: Optional[BubbleMeasurement],
    pool_tokens: float,
    attn_counts: Sequence[int],
    kv_mib_per_token_per_attn_layer: float,
    hidden_size: int,
    chunk_tokens: int,
    cap_tokens: int,
    gapped_layer_set: str = "",
    dtype_bytes: int = 2,
    pinned_depth: Optional[int] = None,
) -> DepthDecision:
    """Group P's microbatch depth, DERIVED from the previous boot's own bubble.

    THE MECHANISM. ``init_pp_loop_state`` reads this knob twice: it widens
    ``pp_loop_size = pp_size + depth``, and -- the half that matters for the
    bubble -- ``_event_loop_pp_body`` moves
    ``_pp_commit_send_output_work_and_preprocess_output_tensors`` from AFTER
    ``_pp_launch_batch`` to BEFORE it when the depth is non-zero. At depth 0
    a rank's output exchange for pass *i-1* therefore serialises with its
    launch of pass *i*, and that serialisation is exactly the host time the
    PP-BUBBLE meter measures between two forwards.

    THE DERIVATION. A stage busy ``forward/(gap+forward)`` of the time needs
    ``ceil(1/(1-stall_share))`` passes in flight to stay busy across the gap.
    One of those passes is the one it is forwarding, so the FLAG -- which
    counts passes BEYOND the ring's own -- is that number minus one. Stated
    rather than folded in, because it is where this function departs from the
    briefing's ``depth = ceil(1/(1-bubble_share))``: that form provisions one
    extra in-flight pass beyond the gap it has to cover, and every extra pass
    is charged to the KV pool below. Both terms are printed, so a reader who
    wants the other convention can see the arithmetic rather than infer it.

    THE #692 GATE, RE-READ FOR WEG 2. ``DESIGN_691_bubble_levers.md`` gated
    this lever on "measure it against the seam, not just against throughput",
    for two named costs. They do not survive equally here:

    * *"more live KV on cards already failing seam funding at 179 MiB"* -- does
      NOT apply. That was the one-process flip, where the live set at the seam
      had to be FUNDED to survive a cutover. In Weg 2 group P is its own
      process group and carries nothing across: its sleep releases ``kv_cache``
      and the weight tags wholesale to the memory saver (``sleep_group``), and
      the carrier between the phases is the HiCache store, which D reads. There
      is no funded live set for depth to grow. What depth does cost is priced
      here instead, in the only budget it actually touches: P's own KV pool,
      against ``--max-kv-per-request``.
    * *"a deeper pipeline has more state to quiesce, so seam entry takes
      longer"* -- DOES apply, unchanged, and is not priced here. The front's
      quiesce witness is ``/flush_cache`` returning 200 only when every
      in-flight term is zero (``front.py`` witness B), so ``depth`` extra
      in-flight chunks are ``depth`` extra chunk-forwards of drain before P can
      sleep. At the measured 369.6 ms per full chunk on the binding stage that
      is sub-second per unit of depth against a 15-17 s flip, which is why it
      is named and left to the flip's own measurement rather than converted
      into a second, unmeasured budget here.

    Refuses W42 rather than lowering the depth: a silently reduced depth is a
    hand number wearing a derivation.
    """
    kv_mib_per_token = [
        max(1, int(a)) * float(kv_mib_per_token_per_attn_layer) for a in attn_counts
    ]
    # The activation the extra pass keeps alive: one PPProxyTensors
    # hidden-states frame per stage boundary, [chunk, hidden] in the model
    # dtype. It is NOT in pp_cut.PhasePoolModel (which prices weights, mamba
    # state, the arming floor and KV only), so it is the one genuinely new
    # term -- and it is converted into pool tokens rather than charged against
    # a second MiB budget, because the pool model has already spent every free
    # MiB into tokens and charging both would be two books for one byte.
    act_mib_per_pass = float(chunk_tokens) * float(hidden_size) * float(dtype_bytes)
    act_mib_per_pass /= 1024.0 * 1024.0

    if measured is None:
        passes = 1
    else:
        stall = measured.stall_share
        passes = int(math.ceil(1.0 / (1.0 - stall))) if stall < 1.0 else 0
    depth = max(0, passes - 1)
    derived_depth = depth
    pinned = pinned_depth is not None
    if pinned:
        if int(pinned_depth) < 0:
            raise PdFlipLaunchRefused(
                "W42 PdFlipDepthUnfunded: --p-microbatch-depth %d is negative; "
                "the flag counts in-flight passes." % int(pinned_depth)
            )
        # A pin replaces the derivation, never the PRICE: it is announced and
        # then funded on exactly the same axis, so an override cannot buy a
        # depth the pool cannot hold.
        depth = int(pinned_depth)
        passes = depth + 1

    gapped_layout = bool(gapped_layer_set)
    # ONE ZERO, ONE CAUSE. Both rules below produce depth 0, so they are
    # branches of one ``elif`` and the acting one is RECORDED: a reader
    # grepping ``P-DEPTH solver:`` for why the depth is 0 gets the rule that
    # ran, not two candidate explanations. The LAYOUT is tested first because
    # it is the structural bound -- a gapped map cannot run more than one pass
    # whatever the measurement later says -- while the retraction is a verdict
    # on the derivation that a re-grounded #692 will lift.
    lowered_by = ""
    if depth > 0 and gapped_layout and not pinned:
        # DERIVED, not lowered. The bubble measurement asks for more passes,
        # the layout admits one, and the PP-CUT solver already RANKED this
        # candidate at one pass (CutCandidate: a gapped map's stages do not
        # overlap, so its makespan was the SUM). Taking the depth the layout
        # admits is therefore consistent with the number that chose it; the
        # line says so rather than printing a bare 0.
        depth, passes = 0, 1
        lowered_by = "layout"
    elif depth > 0 and not pinned and not DEPTH_DERIVATION_GROUNDED:
        # RETRACTED BY MEASUREMENT, not lowered by taste. The arithmetic above
        # is unchanged and is printed; what does not ship is its OUTPUT, because
        # BOOT_weg2pp2_0907.md measured this knob against this workload and it
        # lost on every column that has a floor (see DEPTH_DERIVATION_GROUNDED).
        # A derivation whose premise the metal refuted is not a default, and
        # keeping it would be a hand number wearing a derivation just as much as
        # silently lowering one would be.
        depth, passes = 0, 1
        lowered_by = "retraction"

    # Only a PIN can still be non-zero against a gapped map: the derived path
    # was taken to 0 by the branch above.
    if depth > 0 and gapped_layout:
        raise PdFlipLaunchRefused(
            "W43 PdFlipDepthGapped: the layer set %r puts group P on a "
            "GAPPED layer set, and the PINNED --pp-async-batch-depth %d would "
            "die at scheduler_pp_mixin.init_pp_loop_state, which refuses the "
            "pair outright: under a gapped set every stage must be inside the "
            "SAME forward, because each stage's next layer is another's "
            "previous one, and async depth lets a rank enter the next forward "
            "while a peer is still in the last one's output exchange. Refused "
            "here so the launcher says it, rather than three ranks discovering "
            "it after the weights are loaded. A DERIVED depth is taken down to "
            "0 by the layout instead (the solver priced this candidate at one "
            "pass); an OVERRIDE is refused, because an override the boot cannot "
            "run is not a tuning choice." % (gapped_layer_set, depth)
        )

    price_rows = depth * int(chunk_tokens)
    price_mib = tuple(
        depth * (float(chunk_tokens) * k + act_mib_per_pass) for k in kv_mib_per_token
    )
    # The stage that converts worst is the one with the FEWEST attention
    # layers: a token costs it less MiB, so the same MiB of crossing frame
    # costs it MORE tokens. The pool is a MIN over stages, so that stage is
    # the one that binds it.
    price_tokens_per_pass = max(
        float(chunk_tokens) + act_mib_per_pass / k for k in kv_mib_per_token
    )
    price_tokens = depth * price_tokens_per_pass if depth > 0 else 0.0
    pool_after = float(pool_tokens) - price_tokens

    if depth > 0 and pool_after < float(cap_tokens):
        # WHO ASKED FOR THIS DEPTH decides the first and last sentence. While
        # DEPTH_DERIVATION_GROUNDED is False a derived depth is always 0, so
        # the only depth that can reach this floor today is a PIN -- and
        # opening with "the measured bubble asks for" then attributed the
        # operator's own override to a measurement, printed that measurement's
        # stall_share as if it had produced the number, and closed with a
        # sentence ("a depth the bubble did not ask for is a hand number")
        # that refutes itself on exactly this path. Both branches below are
        # live: the derived one the day #692 re-grounds.
        if pinned:
            asked = (
                "--p-microbatch-depth %d was PINNED (the measurement derived "
                "%d), and %d passes in flight do not fit group P's pool"
                % (depth, derived_depth, passes)
            )
            closing = (
                "The pin is NOT quietly lowered to what fits: an override "
                "that silently becomes another number is a hand number "
                "wearing a pin. Re-pin the depth the pool can hold (0 is a "
                "pin), or fund this one."
            )
        else:
            asked = (
                "the measured bubble asks for --pp-async-batch-depth %d (%d "
                "passes in flight at stall_share %.3f), but group P's pool "
                "cannot fund it"
                % (
                    depth,
                    passes,
                    0.0 if measured is None else measured.stall_share,
                )
            )
            closing = (
                "The depth is NOT quietly lowered to what fits: a depth the "
                "bubble did not ask for is a hand number. Pin depth 0 to "
                "accept today's behaviour."
            )
        raise PdFlipLaunchRefused(
            "W42 PdFlipDepthUnfunded: %s. Pool is %d tokens; EACH extra pass "
            "costs %d KV rows plus a %.1f MiB crossing frame per stage = %d "
            "pool tokens at the worst-converting stage, so depth %d costs %d "
            "in total, leaving %d against the %d-token floor "
            "(--max-kv-per-request) -- short by %d. Raise the per-rank "
            "budgets or lower --max-kv-per-request. %s"
            % (
                asked,
                int(pool_tokens),
                int(chunk_tokens),
                act_mib_per_pass,
                int(price_tokens_per_pass),
                depth,
                int(price_tokens),
                int(pool_after),
                int(cap_tokens),
                int(float(cap_tokens) - pool_after),
                closing,
            )
        )

    return DepthDecision(
        depth=depth,
        derived_depth=derived_depth,
        passes_in_flight=passes,
        pinned=pinned,
        measured=measured,
        pool_tokens=float(pool_tokens),
        pool_after=pool_after,
        cap_tokens=int(cap_tokens),
        price_rows=price_rows,
        price_tokens=price_tokens,
        price_mib=price_mib,
        act_mib_per_pass=act_mib_per_pass,
        chunk_tokens=int(chunk_tokens),
        gapped_layout=gapped_layout,
        lowered_by=lowered_by,
        price_tokens_per_pass=price_tokens_per_pass,
    )

def state_path(state: BootState) -> str:
    return _operator_dir(GPU_ARB, f"boot_{state.tag}.json")

def s104_scheduler_pids(pids, cmdline=None) -> List[int]:
    """The scheduler processes among ``pids`` (process title ``flliper::scheduler...``, set by
    ``run_scheduler_process``) -- read BEFORE the TERM."""
    cmdline = cmdline or _proc_cmdline
    out = []
    for p in sorted(pids):
        try:
            if "::scheduler" in (cmdline(int(p)) or ""):
                out.append(int(p))
        except Exception:  # noqa: BLE001 - an unreadable pid is not named
            pass
    return out

def s104_escalate_schedulers(sched, bound_s: float = S104_SCHED_KILL_S, *, alive=None, kill=None,
                             clock=time.monotonic, sleep=time.sleep, log=print) -> List[int]:
    """Wait at most ``bound_s`` for the TERMed scheduler pids, then SIGKILL the ones still alive, named
    (``S104 TEARDOWN-SCHED-KILL``). Returns the pids killed. Nothing else in the teardown changes: the front and
    the HTTP servers keep their 60 s graceful bound."""
    alive = alive or _alive
    kill = kill or (lambda p: os.kill(p, signal.SIGKILL))
    sched = list(sched or [])
    if not sched:
        return []
    t0 = clock()
    while clock() - t0 < bound_s and any(alive(p) for p in sched):
        sleep(0.1)
    left = [p for p in sched if alive(p)]
    for p in left:
        try:
            kill(p)
        except OSError:
            pass
    if left:
        log(f"S104 TEARDOWN-SCHED-KILL pids={left} after {bound_s:.1f} s (a scheduler still alive after the "
            f"TERM: the L3 write-behind owner's drain is bounded at 3 s -- a main thread blocked in C never ran "
            f"its handler; SIGKILL instead of holding the card until the 60 s bound)", flush=True)
    return left
