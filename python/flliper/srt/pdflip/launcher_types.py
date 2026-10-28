"""Module-level classes moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
import time

from dataclasses import dataclass
from dataclasses import field
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import ring_table
import subprocess
import threading

#: The record line the retraction above is read from. Named so the number and
#: its provenance cannot drift apart.
DEPTH_RETRACTION_RECORD = "/spinning/gpu-arb/weg2/BOOT_weg2pp2_0907.md"

class MixedRateUnits(ValueError):
    """X* was asked for from two rates that are not the same measurement."""

class XSeed(NamedTuple):
    """X, its provenance line, and whether a MEASUREMENT produced it.

    ``measured`` is load-bearing, not decoration.  The carrier floor is 1.25x X
    (:func:`carrier_census.route_floor`), so an X this rig never measured
    decides through W45 whether a boot may start at all -- a fallback acting as
    an actuator, which the standing law forbids.  Carried as a field rather
    than sniffed back out of ``provenance`` so the two cannot drift (#1299).
    """

    tokens: int
    provenance: str
    measured: bool

@dataclass(frozen=True)
class EarlyReadFact:
    """One fact group D states TWICE, and the reader that makes both necessary.

    #1235's inventory called these "double statements of one fact" and read
    them as second bookkeeping to delete. THEY ARE NOT, and the ordering is the
    proof: ``ServerArgs.__post_init__`` reads ``os.environ`` for these keys in
    ``_handle_dcp_validation`` (server_args.py:9631), ``_handle_uneven_tp``
    (server_args.py:12025) and ``uneven_perf.apply_auto_performance``
    (uneven_perf.py:4648, reached FROM _handle_uneven_tp), while the FLAG only
    publishes itself into the environment later, in
    ``_handle_environment_variables`` -> ``_publish_promoted_781_flags``
    (server_args.py:18271), called at server_args.py:7186 -- after all three.

    So the environment variable is the statement that reaches those readers and
    the flag is the statement that reaches everything downstream plus the argv
    a human reads. Dropping either one changes behaviour or hides it.

    What was WRONG is that they were two independent literals in two functions
    that could drift apart silently. Here they are one row, consumed by both
    ``build_env`` and the argv builders, so the pair cannot disagree.

    ``groups`` records an EXISTING asymmetry rather than quietly fixing it:
    the uneven-DCP keys are exported to BOTH groups' environments while only
    group D carries the flags. Group P runs tp_size=1, where the weighted-DCP
    path has nothing to weight, and no boot has ever been run with the export
    removed -- so it is preserved exactly and named here for the boot that can
    measure it, instead of being changed by a slice that cannot.
    """

    env_key: str
    env_value: str
    flag: Tuple[str, ...]
    groups: str
    reader: str

class PdFlipLaunchRefused(RuntimeError):
    pass

class PdFlipDKvStageMaxRefused(PdFlipLaunchRefused):
    """#251c: the operator's FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS does not fit
    the launcher's stage table (one entry per seat count 1..cap, each a stage
    index of the table). Named, never quietly replaced by the default -- an
    A/B arm that silently runs the other arm measures nothing."""

class PdFlipDKvStageWavesRefused(PdFlipDKvStageMaxRefused):
    """#251d: the stage form lowers a batch's capture floor below what the
    overflow-wave cap told in --env-d serves -- the capture would refuse the
    boot on TP0. Named here instead (a subclass, so d_seat_table_lines lets it
    through like W168)."""

class PdFlipTokenCutNotWired(PdFlipLaunchRefused):
    """#239 S2: ``--d-kv-token-cut`` (form ``kv=qsa_forma_dcp``) is PLANNED
    -- the D solve prices the token cut, the dry run prints it -- but the
    runtime half is not built: a Form A worker attends nothing (seam F5, S3
    wires the worker attention over its KV share). A real boot is refused
    here, before any rank loads, instead of booting a KV split nobody reads.
    """

class PdFlipTokenCutFlipNotWired(PdFlipLaunchRefused):
    """#239 S3h: a FLIP boot under the token cut (``kv=qsa_forma_dcp`` with P)
    in a form whose P->D hand-off is not wired.

    The wired form (S3h, 28.09.): the hand-off rides the ONE arena L2 and the
    canonical page store. P writes whole geometry-neutral pages (#706); a D
    worker that owns rows reads exactly its extents of them through its own
    L2 binding (S4b part 3, ArenaMHAHostPool with owner rows) and its KV
    window of the file backend (part 2); tail adopt is per owner (part 5);
    the claim of every wake-path prefetch is the group MIN (the worker votes
    in the min arm only, ``cache_controller.claim_vote_min_only``); the wake
    credit is taken per rank against its own card at run time (the KV resume
    EARLY only when the card funds it, else after the legs). A flip boot
    WITHOUT a host tier (``--pdflip-disable-hicache``) has no carrier for the
    cut's rows and stops here by name; ``--d-only`` (H87) boots either way.
    """

class PdFlipStoreDiskRefused(PdFlipLaunchRefused):
    """W57: the disk cannot fund ``max_size + min_free`` for this boot's store.

    RENUMBERED W52 -> W57 by the serve-next4 train (2026-09-09): W52 was
    already ``PdFlipNoServiceableRoute`` (``front.py``, #1290) at the base this
    branch merges into.  The number is a label, the NAME is the identity
    (``front.py``'s ``X_REFUSAL_MARKER`` note); grep the name.
    """

class PdFlipStoreArcRefused(PdFlipLaunchRefused):
    """W58: the ZFS ARC is uncapped and the reap margin does not cover the store.

    RENUMBERED W53 -> W58 by the serve-next4 train (2026-09-09): W53 was
    already ``PdFlipStoreHandbackFailed`` (``front.py``, #1291) at the base.
    """

@dataclass
class Card:
    nvml_index: int
    uuid: str
    name: str
    total_mib: int
    #: Bytes the driver holds back out of ``total_mib`` and never hands to any
    #: allocation (425 MiB on this rig's 3080s, 518 on the 5090).  Carried so
    #: the free/used readers below can be carve-out honest.  NOTE: ``total_mib``
    #: is still the full board and the budget arithmetic still spends against
    #: it -- that the awake budget omits this term is a SEPARATE open finding
    #: (pdflip/refute/lens2.md sec 3), deliberately not changed here.
    reserved_mib: int = 0
    #: HW-GENERIC 1002: the NVML properties identity and order are decided on
    #: (pdflip/card_identity.py). ``None`` = NVML did not report it; a card
    #: without ``cc`` never passes ``resolve_cards`` (arch gate, HW-ARCH).
    cc: Optional[Tuple[int, int]] = None
    bar1_total_mib: Optional[int] = None
    pcie_max_gen: Optional[int] = None
    pcie_max_width: Optional[int] = None
    mem_bus_width_bits: Optional[int] = None
    mem_clock_max_mhz: Optional[int] = None

@dataclass
class GroupSpec:
    name: str
    port: int
    argv: List[str]
    log: str
    env: Dict[str, str]
    pid: int = 0
    proc: Optional[subprocess.Popen] = None
    #: S7b P-RESTART: the cwd the group was exec'd in (launch_group), for the respawn record.
    cwd: str = ""

@dataclass
class BootState:
    tag: str
    tip: str
    tree: str
    stamp: str
    cards: List[Dict] = field(default_factory=list)
    cvd: str = ""
    logs: Dict[str, str] = field(default_factory=dict)
    pids: Dict[str, int] = field(default_factory=dict)
    helper_pids: List[int] = field(default_factory=list)
    #: H52: the mem time series supervisor (session leader, pgid == pid);
    #: also in ``helper_pids``. teardown() stops it by process group.
    memts_pid: int = 0
    #: DUAL-TP3PP3 --dual-mps on: the private MPS daemon's root dir
    #: (``/tmp/pdflip-dual-mps-<tag>``); empty when no daemon was started. teardown()
    #: reads it from the state json (a fresh process) and stops the daemon there.
    dual_mps_root: str = ""
    #: #1236: WHERE this boot's store directory is, on disk. Replaces
    #: ``store_mount`` + ``store_gib`` -- there is no mount and no RAM size.
    store_dir: str = ""
    store_max_size_bytes: int = 0
    store_min_free_bytes: int = 0
    #: What the host ledger reports instead of a store size: hard bound minus
    #: predicted run peak. ``None`` when no cgroup sample allowed a prediction.
    reap_headroom_gib: Optional[float] = None
    budgets: Dict[str, List[int]] = field(default_factory=dict)
    dc_measured_p: Dict[str, int] = field(default_factory=dict)
    dc_expect_d: Dict[str, int] = field(default_factory=dict)
    ledger_lines: List[str] = field(default_factory=list)
    ring_lines: List[str] = field(default_factory=list)
    #: #1273 S7: the W71 residency rows, empty on the default `ring` arm --
    #: an empty list there means "the exchange did not run", never "it was
    #: checked and found nothing".
    xchg_lines: List[str] = field(default_factory=list)
    ring_form: str = ""
    ring_dir: str = ""
    #: The boot nonce (``HostRingPlan.epoch``), so teardown can unlink THIS
    #: boot's VRAM credit counters and no other boot's (FIX 3 round 3).
    ring_epoch: str = ""
    #: #1275 fix 2: this boot's admin key FILE (never the key), so `--teardown`
    #: can unlink it. A per-boot secret that outlives its boot is a stale file
    #: naming a server that no longer exists.
    admin_key_file: str = ""
    argv: Dict[str, str] = field(default_factory=dict)
    t_ready: Dict[str, float] = field(default_factory=dict)
    sleep_p_ms: float = 0.0
    deviations: List[str] = field(default_factory=list)
    carrier_max_tokens: int = 0
    weight_chunks: int = 0
    tms_so: str = ""
    p_depth: int = 0
    #: #1233 fix 5: the pre-boot cgroup sample, oom_kill baseline included, so
    #: a later death can be attributed by diff instead of retroactively.
    cgroup: Dict[str, Optional[int]] = field(default_factory=dict)
    #: P's DERIVED PP layer split (empty = refused, reason in the log line).
    p_stage_layers: List[int] = field(default_factory=list)
    #: #1233 fix 8: what the /dev/shm orphan sweep found and freed.
    shm_sweep: Dict[str, object] = field(default_factory=dict)
    #: #1233 fix 8: the DORMANT-IMAGE sample taken at group P's first sleep --
    #: the term the next boot's ledger prices its image from.
    dormant_image_p: Dict[str, object] = field(default_factory=dict)
    #: fnFL2 H57: the power limit per card this boot ran under (NVML, read
    #: only) -- ``power_limit.state_dict``: the POWER-LIMIT line, and per
    #: ``nvmlN`` limit/max/default W and the max SM clock. Every time or rate a
    #: later reader takes from this boot is only comparable at this limit.
    power_limits: Dict[str, object] = field(default_factory=dict)
    #: YaRN x2 (28.09.): die W132-FR_P-KAPPUNG dieses Boots (vorher/nachher je
    #: Stufe, gekappte Stufen, Zeilen); leer = keine Stufe gekappt.
    p_fr_cap: Dict[str, object] = field(default_factory=dict)
    #: #239 S3f: the D ownership solve of this boot (``--d-kv-token-cut
    #: owned``): stated and solved ``--rank-moe-ratio``, cut, FR_D, miss time
    #: per rank against Form A; empty = no owned solve.
    d_owner_solve: Dict[str, object] = field(default_factory=dict)

@dataclass
class HostRingPlan:
    """What C18 publishes, and why it may publish nothing.

    ``armed`` False means the ring is NOT in this boot, and THERE IS NO
    FALLBACK -- spec Amendment A1-3, measured, not argued:

        the OLD flip form is INFEASIBLE on this host budget on the default
        table (W20 at every rung: host weights 38.77 GiB run / 33.12 GiB
        launch, boot weg2rg1) -- no Weg-2 boot may pin --ring-table-boot
        weg2zr2 to get past it; the ring with gathered legs is the only route.

    So every un-armed arm below REFUSES BY NAME and the launcher exits 2.  The
    predecessor of this text said "the host ring is NOT armed and the OLD flip
    form runs (spec section 10.4)" in the W32, W33 and W34 arms, and the metal
    had refuted it twenty minutes before that commit was written: those lines
    are printed at ``prepare_host_ring`` (launcher.py step 1b) and the arm that
    decides whether the un-armed form is fundable is ``host_ledger.choose`` one
    step LATER, so the sentence was an assertion made before its own check --
    and the check said no.  A refusal that names a fallback must HAVE one.

    Two reasons remain, and both are printed:

    * ``table`` is not None -- W33 (no proven registration form), W32, or W34
      (the launcher found itself without gathered legs).  The table exists, so
      the un-armed charge CAN be priced; it just cannot be funded.
    * ``table`` is None -- R22, no measured table at all.  Then it cannot even
      be priced, and the ledger refuses by name (W20) for that reason instead.
    """

    form: str = ""
    dir: str = ""
    env_map: str = ""
    #: THE BOOT NONCE, and it must be unique per launch on EVERY arm of
    #: `prepare_host_ring`, not only on the armed one. `str(plan.epoch)` is
    #: what `prepare_xchg_env` publishes as FLLIPER_PDFLIP_XCHG_BOOT: the shm
    #: region name, the flip token's boot half, the credit-counter scope and
    #: `xchg_manifest.boot_token()`. The armed arm sets `int(time.time())`
    #: (below, next to the ring files); the #1369 ABSENT-BY-DESIGN arm --
    #: every boot since 2026-09-14 -- returned early with the old default 0,
    #: so weg2xsn28..xsn64 ALL ran as boot "0". MEASURED on weg2xsn64: the
    #: manifest writer's W68 ratchet ("a file of THIS boot with a DIFFERENT
    #: inventory") refused every rank, because xsn63's stale manifests carried
    #: token "0" too and the inventory had changed (the tower left). Every
    #: earlier boot passed only because the inventory never changed; and
    #: `load_manifests(boot_token=...)`, the filter that is "not a hope",
    #: could not exclude a stale file either. Same semantics the front already
    #: uses for TMS_HOST_RING_EPOCH's fallback (front.py: `str(int(time.time()))`).
    epoch: int = field(default_factory=lambda: int(time.time()))
    armed: bool = False
    table: Optional["ring_table.RingTable"] = None
    lines: List[str] = field(default_factory=list)
    #: C12/C13: the per-card duplex table and the ``FLLIPER_PDFLIP_PCIE_DUPLEX``
    #: string built from it.  Launcher OUTPUT (spec R19), solved from the
    #: step-0 probe record; empty means no card splits its lock key.
    duplex: Optional["ring_table.DuplexTable"] = None
    duplex_env: str = ""

    @property
    def host_weights_bytes(self) -> int:
        """The RUN-moment host weights charge, from the same measured table for
        both forms.

        Armed: ``Sigma_c H(c)``.  The region is fungible, so the tag in flight
        is drawn from the SAME per-card bytes the dormant image occupies and
        there is nothing to add (spec C19 / section 7).

        Un-armed (the OLD serial per-tag form): one image plus ONE tag in
        flight -- ``Sigma_c H(c) + max_k Sigma_c bytes(k, c)``.  This is the
        parent's ``backup_resident + chunk_gib`` model with the stale constant
        replaced by the measured table and the AVERAGE chunk replaced by the
        measured largest one.  It is ONE tag because the front sends one
        ``/release_memory_occupation`` per tag; the sum is over cards because
        that tag's shards land on every card at once.  Charging ``Sigma_c
        max_k`` instead -- each card's own largest tag, summed -- prices a step
        the flip never takes (the maxima are different tags) and cost the
        DEFAULT path 5.6 GiB it does not spend, which was enough to make every
        arm of the ladder refuse with W20.
        """
        if self.table is None:
            return 0
        total = self.table.total_h_bytes
        if not self.armed:
            total += self.table.max_step_total_mib * ring_table.MIB
        return total

    @property
    def host_weights_span1_bytes(self) -> int:
        return 0 if self.table is None else self.table.total_span1_bytes

    @property
    def provenance(self) -> str:
        if self.table is None:
            return "no measured table"
        # A1-3: un-armed names no form that runs (FIX 3 round 3, same class as
        # the two sentences fix 2 retracted).  The table is still charged and
        # printed, because the refusal's arithmetic is what the operator reads.
        form = (f"ring form {self.form}" if self.armed
                else "NO ARMED FORM -- this boot refuses (A1-3)")
        return f"{self.table.provenance()}; charged for the {form}"

class LaunchGuard:
    """#1361a: the host latch, running from LAUNCH instead of from READY.

    THE HOLE IT CLOSES, measured on boot weg2xsn27: group D emitted its
    ``PDFLIP-XCHG-PLAN dir=d2h`` at 00:12:42 and the kernel reaped the boot at
    ~00:13:04 -- while the FRONT's guard loop printed its first
    ``PDFLIP-HOST WATERMARK`` line at 00:13:46, **64 s late**.  The front is not
    slow; it is started by this launcher AFTER both groups, so the most
    dangerous window of the whole boot -- the startup sleep leg, where D loads
    weights into anon WHILE writing its dormant image into shmem -- is
    structurally unguarded no matter how fast that loop ticks.

    So the guard runs HERE, in the launcher, from the first ``launch_group``
    until the front is up and takes over.  Same latch class, same cushion
    terms, same reading (``read_cgroup_pressure`` once per tick, feeding both
    the cushion test and the level), so the two halves of the boot's protection
    cannot disagree about what they graded.

    A DAEMON THREAD AND NOT A TASK: the launcher is synchronous here and blocks
    in ``wait_ready`` for minutes at a time; an asyncio task would not get a
    tick in exactly the window this exists for.
    """

    def __init__(self, log, reap_mark_gib: float, period_s: float = 0.5) -> None:
        self._log = log
        # 12.10.2026 (NF W98 HOST-WARN, user rule 06.10.): "Ledger, Launcher und
        # Startskripte warnen nur" -- this guard used to hand a W98 verdict to
        # main(), which raised PdFlipLaunchHostLatched and tore the spawned groups
        # down. It now warns (marker HOST-RAM-WARN W98) and the boot goes on.
        self._latch = host_ledger.RateLatch(reap_mark_gib=reap_mark_gib, warn_only=True)
        self._period = float(period_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        #: kept for the call sites that still read it; since 12.10.2026 always
        #: ``None`` -- the guard no longer produces a verdict that stops a boot.
        self.verdict: Optional[str] = None
        self.warns = 0
        self.ticks = 0

    def start(self) -> "LaunchGuard":
        self._thread = threading.Thread(
            target=self._run, name="pdflip-launch-guard", daemon=True)
        self._thread.start()
        self._log(
            f"PDFLIP-HOST LAUNCH-GUARD armed period={self._period:.2f}s "
            f"reap_mark={self._latch.reap_mark_gib:.2f} GiB -- the cushion latch runs "
            f"from THIS moment, not from READY. Boot weg2xsn27's front printed its "
            f"first WATERMARK line 64 s after the kernel had already reaped it "
            f"(#1361a); the startup sleep leg is the window this closes"
        )
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                pr = host_ledger.read_cgroup_pressure()
                nr = pr.get("nonreclaim_gib")
                if nr is not None:
                    f, sh = pr.get("file_gib"), pr.get("shmem_gib")
                    # W98 z30w: in the Docker form /proc/meminfo is the HOST's;
                    # the pool and the ceiling come from this cgroup.
                    free, free_src, ceiling = host_ledger.latch_free_pool(pr)
                    line = self._latch.observe(
                        time.time(), float(nr),
                        cushion_gib=(
                            None if f is None or sh is None
                            else float(f) - float(sh)
                        ),
                        shmem_gib=None if sh is None else float(sh),
                        free_gib=free,
                        ceiling_gib=ceiling,
                        free_source=free_src,
                    )
                    self.ticks += 1
                    if line is not None:
                        self._log(line)
                        if (host_ledger.HOST_WARN_MARKER in line
                                and "CLEARED" not in line and "RATE-GAP" not in line):
                            self.warns += 1
            except Exception as e:  # noqa: BLE001 -- a guard never kills its host
                self._log(f"PDFLIP-HOST LAUNCH-GUARD read failed: {e}")
            self._stop.wait(self._period)

    def stop(self, why: str = "") -> Optional[str]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._log(
            f"PDFLIP-HOST LAUNCH-GUARD disarmed ticks={self.ticks} "
            f"warns={self.warns} (warn-only, nothing was stopped)"
            + (f" ({why})" if why else "")
        )
        return self.verdict

class PdFlipDualLayoutRefused(PdFlipLaunchRefused):
    """DUAL-TP3PP3: a flag combination the dual layout cannot serve."""

class PdFlipDualPSleepShareRefused(SystemExit):
    """W-DUAL-P-SLEEP-SHARE: --dual-p-sleep on with --dual-share. Not resident, P's
    weights load inside the memory-saver's private MemPool; the union bind frees
    P's own copies there and a private pool never returns them (gmps9 boot death,
    P-PP0 'per-rank budget leaves no GPU memory for the KV cache')."""

class PdFlipDEarlyStartUnreviewed(PdFlipLaunchRefused):
    """W185: an explicit early D start on a profile whose D expectation is
    still the legacy term (``d_expect_from_p_records`` False): D would be
    planned from D's own reserve standing in for P's residue (NF z30r3:
    540-960 MiB per card short; 27B: ~1 GiB on the 5090), and the gate only
    checks planned <= measured, it never re-solves D."""

@dataclass(frozen=True)
class DOperatingPointRow:
    """One operating point's vector and what the runtime's own model prices it at."""

    position: str
    #: What the vector is proportional to, named so a reader never has to guess
    #: which of the two measured rates produced it.
    score_name: str
    scores: Tuple[float, ...]
    weights: Tuple[int, ...]
    attn_heads: Tuple[int, ...]
    gdn_heads: Tuple[int, ...]
    #: Lockstep round cost from PerfCostModel, in the unit that method returns.
    #: ``None`` = the model could not price it, and the row then says so rather
    #: than carrying a number that looks measured.
    round_ms: Optional[float]
    round_unit: str
    world_pool_tokens: Optional[int]
    #: ``PerfCostModel.predict_capacity``'s own verdict that this vector fits
    #: at all. ``None`` = the capacity call did not run (unpriced geometry).
    feasible: Optional[bool] = None
    #: The FUNDED context ``min(sum_r P_r, 64 * min_r P_r)`` -- bounded by the
    #: SMALLEST rank, which is the whole reason it is carried next to the
    #: world pool: a vector that collapses one rank barely moves ``sum(p)``
    #: and can halve this.
    funded_ctx_tokens: Optional[int] = None
    note: str = ""
    #: #1293. WHICH AXIS THE ATTENTION FAMILY'S COMPUTE RIDES on this boot's
    #: geometry -- ``"token"`` under replicated-KV uneven DCP, ``"head"``
    #: otherwise. It is not decoration: it names the grid the W62 saturation
    #: verdict below was taken against, and the two grids differ by 16x on this
    #: rig (4 kv-head units vs 64 token units), which is the difference between
    #: a position being representable and being refused.
    attn_axis: str = "head"
    #: The DCP token partition of ``_CP_TOKEN_UNITS`` units (largest-remainder,
    #: the same partitioner and the same units the runtime's own "Uneven DCP:
    #: auto-set dcp_size" install uses). Empty tuple on the head axis, where
    #: there is no token vector to report.
    attn_token_units: Tuple[int, ...] = ()
    #: Floor-aware re-normalisation statement: which family/axis had a rank
    #: pinned at the partitioner's one-unit floor, and how far the REALISED
    #: shares then sit from the requested ratio. Empty when no grid binds.
    axis_note: str = ""

@dataclass(frozen=True)
class DTpRatioDecision:
    """Group D's weight-vector objective, the argv it produces, and its price."""

    objective: str
    tune: str
    flags: Tuple[str, ...]
    line: str
    #: #1241 slice (2). The three-vector provenance line. Empty only when the
    #: rows could not be built at all, which the line itself then says.
    op_line: str = ""
    rows: Tuple[DOperatingPointRow, ...] = ()

@dataclass(frozen=True)
class DTokenVectorDecision:
    """Group D's KV-token ownership vector: what is shipped, and why."""

    flags: Tuple[str, ...]
    line: str

@dataclass(frozen=True)
class BubbleMeasurement:
    """One stage's PP-BUBBLE totals, summed over every window in one log.

    Summed, never averaged: each window line carries its own numerator AND its
    own denominator, so a mean of the printed shares would weight a 12-forward
    window like a 2-forward one. Sum of numerators over sum of denominators is
    the only aggregation that keeps the denominator honest.
    """

    source: str
    rank: int
    windows: int
    forward_ms: float
    bubble_ms: float
    starved_ms: float
    n_forwards: int

    @property
    def denominator_ms(self) -> float:
        """``gap + forward`` -- the denominator the emitting line names."""
        return self.bubble_ms + self.forward_ms

    @property
    def bubble_share(self) -> float:
        d = self.denominator_ms
        return 0.0 if d <= 0.0 else self.bubble_ms / d

    @property
    def forward_share(self) -> float:
        return 1.0 - self.bubble_share

    @property
    def stall_share(self) -> float:
        """The part depth can move.

        ``starved_ms`` is the part of the gap in which this rank visited the
        loop with NOTHING to launch (fix 1r/2). Depth overlaps a rank's output
        exchange with its next forward; it cannot manufacture a chunk that the
        queue never supplied. Charging starvation to depth would buy in-flight
        slots against a supply problem -- and pay for them out of the KV pool.
        """
        d = self.denominator_ms
        if d <= 0.0:
            return 0.0
        return max(0.0, self.bubble_ms - self.starved_ms) / d

@dataclass(frozen=True)
class DepthDecision:
    """Group P's ``--pp-async-batch-depth`` and the ONE provenance line for it."""

    depth: int
    passes_in_flight: int
    pinned: bool
    measured: Optional[BubbleMeasurement]
    pool_tokens: float
    pool_after: float
    cap_tokens: int
    price_rows: int
    price_tokens: float
    price_mib: Tuple[float, ...]
    act_mib_per_pass: float
    chunk_tokens: int
    #: What the bubble ARITHMETIC asks for, before the retraction. Kept beside
    #: the shipped :attr:`depth` rather than replaced by it: the derivation is
    #: the thing that has to be re-grounded, so it stays visible and the day a
    #: measured exchange-bound share exists, ``DEPTH_DERIVATION_GROUNDED``
    #: flips and this number ships without a second derivation being written.
    derived_depth: int = 0
    #: True when the chosen layout is a GAPPED layer set. The depth is then 0
    #: BY THE LAYOUT, not lowered: a gapped map admits exactly one pass in
    #: flight, and that is the number the PP-CUT solver already priced the
    #: candidate at. Printed so the 0 is read as a derivation, not a default.
    gapped_layout: bool = False
    #: WHICH rule lowered the shipped depth below the derivation, recorded at
    #: the point it acts rather than inferred afterwards from the numbers:
    #: ``""`` (none acted -- the derivation shipped, or there was nothing to
    #: lower), ``"retraction"`` (the #692 derivation is not re-grounded) or
    #: ``"layout"`` (a gapped map admits exactly one pass). The two rules are
    #: branches of ONE ``elif`` in :func:`solve_p_depth`, so at most one can
    #: act, and :meth:`line` prints exactly one cause for a 0. Inferring it
    #: from ``derived_depth > depth`` cannot tell them apart, which is how the
    #: line came to print both.
    lowered_by: str = ""
    #: What ONE extra pass costs in pool tokens at the worst-converting stage.
    #: :attr:`price_tokens` is ``depth * price_tokens_per_pass``; the two are
    #: kept apart so no sentence can price "one extra pass" with the total
    #: (fix 3, review finding 2: the W42 text did exactly that at depth 2).
    price_tokens_per_pass: float = 0.0

    @property
    def retracted(self) -> bool:
        """True when the RETRACTION is the rule that lowered the depth."""
        return self.lowered_by == "retraction"

    @property
    def lowered_by_layout(self) -> bool:
        """True when the GAPPED LAYOUT is the rule that lowered the depth."""
        return self.lowered_by == "layout"

    def line(self) -> str:
        """THE one line. Format is load-bearing: a reader greps ``P-DEPTH solver:``."""
        if self.measured is None:
            src = (
                "no PP-BUBBLE measurement found, so bubble_share is unknown and "
                "the depth stays at today's behaviour"
            )
            shares = "bubble_share=n/a forward_share=n/a"
        else:
            m = self.measured
            src = (
                "from %s rank=%d (binding stage, largest forward total) over %d "
                "window(s), n=%d forwards; starved_ms=%.1f of bubble_ms=%.1f is "
                "queue starvation and is NOT charged to depth, stall_share=%.3f"
                % (
                    m.source,
                    m.rank,
                    m.windows,
                    m.n_forwards,
                    m.starved_ms,
                    m.bubble_ms,
                    m.stall_share,
                )
            )
            shares = "bubble_share=%.3f forward_share=%.3f" % (
                m.bubble_share,
                m.forward_share,
            )
        if self.retracted:
            src = (
                "the #692 DERIVATION IS RETRACTED and NOT re-grounded: it asks "
                "for depth %d, and %s (DEPTH VERDICT) measured that depth "
                "against this workload -- +6.9 %% on TTFT p50 at depth 1, "
                "+9.6 %% at depth 2, the bubble share did NOT fall (21.8/24.6 "
                "-> 26.8 -> 24.1 %%) and the deep point was inside its floor. "
                "The shipped depth is 0 until a MEASURED exchange-bound share "
                "exists to derive from; an explicit --p-microbatch-depth still "
                "wins and is still priced. " % (self.derived_depth, DEPTH_RETRACTION_RECORD)
                + src
            )
        if self.lowered_by_layout:
            src = (
                "the chosen layout is a GAPPED layer set, which admits exactly "
                "ONE pass in flight (scheduler_pp_mixin.init_pp_loop_state "
                "refuses the pair: under a gapped set each stage's next layer "
                "is another's previous one, so every stage must be inside the "
                "same forward). The depth is 0 BY THE LAYOUT -- the same one "
                "pass the PP-CUT solver priced that candidate at, which is why "
                "it is a derivation and not a lowered hand number. " + src
            )
        elif self.gapped_layout:
            # The bound is real and worth printing, but it did NOT act: the
            # depth was already at or below one pass. Saying "BY THE LAYOUT"
            # here would name a rule that never ran, which is the same
            # instrument-text defect as printing two causes for one 0.
            src = (
                "the chosen layout is a GAPPED layer set, which admits exactly "
                "ONE pass in flight; the depth was already 0, so the layout "
                "bound did not have to lower anything. " + src
            )
        # The bracket names TWO pass counts and says which one ships. The
        # shipped count is depth+1 by definition; the derived count is the
        # formula's output and ships only when no rule lowered it (fix 3,
        # review finding 1: printing the formula label over the shipped
        # value read "ceil(1/(1-stall_share))=1" on a retracted depth whose
        # formula gave 2).
        if self.measured is None:
            derived = "derived passes=n/a (no measurement)"
        else:
            derived = "derived passes=ceil(1/(1-stall_share))=%d" % (
                self.derived_depth + 1
            )
            if self.lowered_by:
                derived += " (NOT shipped: %s)" % self.lowered_by
            elif self.pinned and self.depth != self.derived_depth:
                derived += " (NOT shipped: pin overrides)"
        return (
            "PDFLIP P-DEPTH solver:%s%s %s depth=%d price_rows=%d/stage "
            "price_mib=%s MiB/stage pool_after=%d (constraint pool >= %d) "
            "[passes_in_flight (shipped)=%d = depth+1; %s; each extra pass "
            "costs %d KV rows plus a %.1f MiB crossing frame per stage = %d "
            "pool tokens at the stage that converts worst, %d in total for "
            "depth %d; %s]"
            % (
                " PINNED (user override)" if self.pinned else "",
                " RETRACTED (derived %d, shipped %d)" % (self.derived_depth, self.depth)
                if self.retracted
                else "",
                shares,
                self.depth,
                self.price_rows,
                ",".join("%.1f" % v for v in self.price_mib),
                int(self.pool_after),
                int(self.cap_tokens),
                self.passes_in_flight,
                derived,
                self.chunk_tokens,
                self.act_mib_per_pass,
                int(self.price_tokens_per_pass),
                int(self.price_tokens),
                self.depth,
                src,
            )
        )

@dataclass(frozen=True)
class PCutFacts:
    """What the solved cut knows that the depth price also needs.

    Handed on rather than re-derived: the pool, the attention split and the KV
    cell are already solved once here, and a second derivation beside them
    would be a second set of books for one physical fact.
    """

    stage_ratio: str
    attn_stage_ratio: str
    pool_tokens: float
    attn_counts: Tuple[int, ...]
    kv_mib_per_token_per_attn_layer: float
    hidden_size: int
    cap_tokens: int
    #: The solver's chosen ownership map in ``--pp-layer-set`` syntax, and
    #: whether it is GAPPED. Empty string = a contiguous cut, which needs no
    #: map at all: the count form expresses it and every path below stays
    #: byte-identical to what it was.
    layer_set: str = ""
    gapped: bool = False
    #: FIX 2: THE REALIZED PER-STAGE LAYER COUNTS OF THIS CUT -- the one fact
    #: the flip-order map needs and the only one that may not be re-derived
    #: from a score vector somewhere else. Not a convenience field: it is
    #: already ROUND-TRIPPED here against the runtime's own authority
    #: (``derive_pp_layer_split`` for the count form, ``parse_pp_layer_sets``
    #: for a gapped map) and a cut that fails that round trip is refused, so
    #: this is the split the boot provably runs. ``main`` reads it instead of
    #: asking ``p_stage_layers`` about the incumbent again.
    layer_counts: Tuple[int, ...] = ()

class PdFlipDFormOffTheMap(PdFlipLaunchRefused):
    """W170 (#239 rc12z29c-Blocker 2): group D is about to start with a MoE
    ownership (--rank-moe-ratio) or FR_D (--rank-moe-resident-fraction /
    FLLIPER_MOE_RESIDENT_EXPERT_FRACTION) other than the one the Platztausch
    map was built from. P loaded its common prefix and store slots from that
    map, so D would load a form the map does not describe and die at load
    ('Platztausch-Karte nennt N residente Zeilen ...'), minutes later."""

class PFrCap(NamedTuple):
    fractions: Tuple[float, ...]
    stages: Tuple[int, ...]
    lines: Tuple[str, ...]
    refusal: str

class PdFlipRingFormUnproven(ring_table.PdFlipRingRefused):
    """W33: ``--ring-form`` named no form this tree has a proven build for.

    A subclass of the ring's own refusal base, so it inherits ``cli()``'s
    handler -- the one named line and exit 2 -- without being enumerated
    anywhere (FIX 2's lesson, applied to the new member rather than repeated).
    """

class PdFlipLaunchHostLatched(PdFlipLaunchRefused, host_ledger.PdFlipHostLedgerRefused):
    """#1361 21a-fix: the launch guard's latch, as BOTH a launch and a host refusal.

    The launch-guard latch fires after both groups are spawned, so whatever it
    raises must run through the teardown funnel -- a stopped boot that leaves
    six ranks holding three cards is worse than the breach it prevented.  Only
    `PdFlipLaunchRefused` reaches that funnel.

    But the host ledger's own callers catch `PdFlipHostLedgerRefused`, and the
    latch IS a host-ledger verdict; narrowing it to a launch type would have
    moved the defect rather than fixed it.  So it is both, exactly as
    :class:`PdFlipChunkCardMismatch` is both a launch refusal and a ValueError --
    the #1294 precedent, not a new pattern.  No layer inversion: this module
    already imports `host_ledger`, never the other way round.
    """

class PdFlipChunkCardMismatch(PdFlipLaunchRefused, ValueError):
    """W59: ``chunk_tag_cards()``'s ``card_of_stage`` length does not match
    the PP stage count (#1294).  RENUMBERED W52 -> W59 (serve-next4 train,
    2026-09-09); the emitter is ``pdflip_memory_saver.py``.

    Before this class existed the site raised a bare ``ValueError``, which is
    reachable from ``main()`` after the dry-return (``launcher.py`` calls
    ``chunk_tag_cards`` while building the flip's chunk-tag-to-card map) and
    left the flliper groups alive on the cards as an uncaught traceback --
    both funnels (this one and the pre-#1248 one) missed it because it was
    neither in ``REFUSALS`` nor named ``PdFlip*``.  Subclasses BOTH
    ``PdFlipLaunchRefused`` (so ``cli()``'s ``except REFUSALS`` catches it, the
    same as every other launch refusal) AND ``ValueError`` (so the
    pre-existing ``test_pdflip_flip_order_1233.py::test_card_list_that_does_not_match_the_stages_is_refused``
    -- which asserts ``ValueError`` and is not itself in scope here -- keeps
    passing unmodified: the exception TYPE this call raises is intentionally
    made richer, not swapped for an unrelated one, so no consumer of the old
    type loses its match).
    """

class PdFlipCarrierFloorUnreachable(PdFlipLaunchRefused):
    """W60: ``carrier_census.route_floor()``'s bisection could not bracket a
    prompt length that exceeds the SHORT bound (#1294).  RENUMBERED W54 -> W60
    (serve-next4 train, 2026-09-09): W54 is ``PdFlipCorridorBudgetUnpriced``.

    Before this class existed the site raised a bare ``RuntimeError`` for
    what its own comment calls unreachable "for any sane divisor" --
    reachable from ``main()`` after the dry-return via
    ``_cc.route_floor(x_tokens)`` -- and left the flliper groups alive on the
    cards as an uncaught traceback, missed by both funnels for the same
    reason as ``PdFlipChunkCardMismatch`` above.  No pre-existing test asserts
    on the specific ``RuntimeError`` this branch used to raise (checked:
    every ``route_floor`` call site in
    ``test_pdflip_1246_carrier_census.py`` exercises the normal
    ``(floor, why)`` return, none the unreachable branch), so no
    multiple-inheritance is needed here -- ``PdFlipLaunchRefused`` already
    subclasses ``RuntimeError``, so any hypothetical ``except RuntimeError``
    elsewhere still matches.
    """
