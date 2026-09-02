"""#719: rebind the once-assigned HiCache pool bindings at the flip cutover.

#718 established the hazard and cut it: ``HiCacheController`` captures its
device pool once, the flip runs a second pool, and device-tier I/O in the
phase that did not build the binding is silent corruption -- so it is disarmed
there. That leaves the device tier usable in ONE phase. This module is the
other half: move the binding at the cutover so it is usable in BOTH.

THREE READERS, AND WHY A PARTIAL REBIND IS WORSE THAN NONE. The pool identity
is captured in three places, each assigned once and never reassigned:

  1. ``HiCacheController`` -- ``mem_pool_device`` (the object both device
     transfers name directly), ``mem_pool_device_hybrid``, and
     ``mem_pool_device_allocator``;
  2. the radix cache -- ``hybrid_kv_cache`` / ``kvcache`` and the HOST pool
     built from it;
  3. the scheduler -- ``token_to_kv_pool_allocator``.

If one moves and another does not, the readers disagree about which pool a row
id names, and the disagreement is invisible: every call still succeeds, against
different memory. That is strictly worse than the #718 state, where the tier is
merely off. So a rebind here is ALL THREE OR NONE, and it is verified after the
fact by generation, not by inspection.

A REBIND IS NOT A POINTER SWAP, and this is the finding that shapes the module.
The host pool is CONSTRUCTED FROM the device pool
(``hybrid_pool_assembler.build_kv_host_pool(kv_pool=...)``), so its
``layer_num`` and its buffer sizes are that phase's. Under the flip the two
phases have different per-rank layer counts -- a PP stage holds 7/5/4 of the 16
attention layers while the TP stack holds all 16 -- so repointing the
controller at the other phase's device pool while its host pool still describes
this phase's is a NEW corruption wearing the fix's clothes: the copy would run,
with matching row ids and mismatched widths.

Therefore the rebind REFUSES unless it is handed a host pool whose shape
matches the incoming device pool. Supplying one is a boot-time question (a
second host pool costs host RAM, which DESIGN_706 C1 shows is the binding
constraint on this box: the flip's weight images leave ~5.37 GB for the whole
tier). Until a phase-matched host pool exists, this refuses and #718's disarm
stands -- which is the honest state, and a loud one.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#719 hicache-rebind]"

# The phase whose pools the bindings were built with. The boot topology is the
# PP prefill stack (phase_flip_boot), so that is where every binding starts.
BOOT_PHASE = "pp"


class RebindRefused(RuntimeError):
    """The rebind cannot be performed safely. Never downgraded to a warning."""


class RebindIncoherent(RuntimeError):
    """A rebind left readers on different generations. Never recoverable."""


@dataclasses.dataclass(frozen=True)
class PhasePools:
    """The pool triple ONE phase owns.

    ``host_pool`` is part of the identity on purpose: it is the piece that
    cannot be shared between phases, so a binding that does not name its own
    host pool is not a binding.
    """

    phase: str
    device_pool: Any
    host_pool: Any
    allocator: Any
    # #706 x #719 (0828 specimen): the WRAPPED hybrid pool of the phase, kept
    # beside the unwrapped one because only the wrapper knows its layers by
    # GLOBAL id (full_attention_layer_id_mapping / mamba_pool.mamba_map) --
    # the inputs the canonical-window rebuild reads at this very cutover.
    # None keeps the unwrapped pool as the stamp target (pre-0828 shape).
    device_pool_hybrid: Any = None

    def layer_num(self) -> Optional[int]:
        return getattr(self.device_pool, "layer_num", None)

    def host_layer_num(self) -> Optional[int]:
        return getattr(self.host_pool, "layer_num", None)


class BindingState:
    """Which phase's pools the HiCache readers are currently bound to.

    Generation is the coherence primitive: every reader stamped by a rebind
    carries the generation it was stamped with, so "did everyone move?" is a
    comparison rather than an inspection of three different object graphs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = BOOT_PHASE
        self._generation = 0
        # W35: generation -> the host pool bound at that generation. Recorded
        # by the same advance() that mints the generation, so there is exactly
        # one place where a generation and its pool are associated. A release
        # produced under an older binding can then be freed against the pool it
        # actually came from instead of the one that happens to be bound now.
        self._pools: dict = {}

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def generation(self) -> int:
        return self._generation

    def reset(self) -> None:
        with self._lock:
            self._phase = BOOT_PHASE
            self._generation = 0

    def advance(self, phase: str, host_pool=None) -> int:
        with self._lock:
            prior_generation = self._generation
            prior_phase = self._phase
            self._phase = str(phase)
            self._generation += 1
            if host_pool is not None:
                self._pools[self._generation] = host_pool
            generation = self._generation
        # #1061: record generation -> phase for the producer-phase census, at
        # the one place that mints generations (the census docstring names
        # this ``advance()`` as its intended feed). The OUTGOING pair is
        # recorded too, so generation 0 / the boot phase is renderable after
        # the first cutover; ``note_generation`` is idempotent. Ungated on the
        # census knob deliberately: one dict write per CUTOVER, and a census
        # armed later (tests, a restarted reader) still needs the history.
        try:
            from sglang.srt.mem_cache.producer_phase_census import note_generation

            note_generation(prior_generation, prior_phase)
            note_generation(generation, str(phase))
        except Exception:  # noqa: BLE001 - an instrument may never break a rebind
            pass
        return generation

    def host_pool_at(self, generation):
        """The host pool bound at ``generation``, or None if unknown."""
        if generation is None:
            return None
        return self._pools.get(int(generation))


_STATE = BindingState()


def binding_state() -> BindingState:
    return _STATE


def bound_phase() -> str:
    """The phase whose pools the readers currently name."""
    return _STATE.phase


def check_shapes(incoming: PhasePools) -> None:
    """Refuse a binding whose host pool does not describe its device pool.

    The single most dangerous rebind is the one that succeeds: matching row
    ids, mismatched widths, and a copy that runs.
    """
    dev = incoming.layer_num()
    host = incoming.host_layer_num()
    if dev is None or host is None:
        raise RebindRefused(
            f"{LOG_PREFIX} incoming '{incoming.phase}' binding does not expose "
            f"layer counts (device={dev}, host={host}); refusing to rebind onto "
            "pools whose shapes cannot be compared."
        )
    if int(dev) != int(host):
        raise RebindRefused(
            f"{LOG_PREFIX} the '{incoming.phase}' phase's device pool holds "
            f"{dev} attention layers but its host pool describes {host}. A host "
            "pool is BUILT FROM a device pool, so this pair belongs to "
            "different phases -- rebinding onto it would copy matching row ids "
            "at mismatched widths, which is a new corruption rather than a fix. "
            "A phase-matched host pool has to be built at boot (it costs host "
            "RAM, the binding constraint on this box) before the rebind can "
            "arm; until then the #718 disarm is the correct state."
        )


def check_pool_coverage(readers: dict, incoming: PhasePools) -> None:
    """Refuse a rebind whose host tier describes FEWER pools than the bound one.

    #718/#847. ``check_shapes`` above compares ONE SCALAR -- the anchor entry's
    layer count -- where the invariant is STRUCTURAL: the incoming tier must be
    able to describe every pool the outgoing one did. A KV-only ``HostPoolGroup``
    passes the scalar check while dropping MAMBA entirely, and the consumers then
    split in two: ``move_indices`` dereferences the index set the resolver could
    not fill (the W38-B crash), and ``HostPoolGroup.load_to_device_per_layer``
    skips the pool it does not know -- moving the KV while the recurrent state
    stays behind, with the tree marking the prefix resident. The second is a
    wrong ANSWER and it is the one a crash-only fix would have hidden.

    A refusal here is the cheap outcome and the designed one: ``rebind`` raises,
    the #718 device tier stays DISARMED for that phase, and every read-through
    misses. A miss is recomputed; a narrowed tier is not.

    Only pools the reader ALREADY names are required -- a WIDER incoming tier is
    fine, and a reader that names no host tier at all (the scheduler names the
    allocator only) is not turned into a coverage claim.
    """
    incoming_names = set(getattr(incoming.host_pool, "entry_map", None) or ())
    for name, obj in readers.items():
        current = getattr(obj, "mem_pool_host", None)
        bound_names = set(getattr(current, "entry_map", None) or ())
        missing = bound_names - incoming_names
        if missing:
            raise RebindRefused(
                f"{LOG_PREFIX} the '{incoming.phase}' host tier describes "
                f"{sorted(str(n) for n in incoming_names)} but reader '{name}' "
                f"is bound to a tier describing "
                f"{sorted(str(n) for n in bound_names)}: "
                f"{sorted(str(n) for n in missing)} would lose its host "
                "backing. Every transfer built for those pools would then be "
                "either unresolvable (a None index set reaching move_indices) "
                "or silently skipped (the KV moves, the state does not, and "
                "the tree still calls the prefix resident) -- a crash or a "
                "wrong answer. A phase host tier has to be built with the full "
                "pool set before this rebind can arm; until then the #718 "
                "disarm is the correct state and a read-through miss is the "
                "correct cost."
            )


def rebind(readers: dict, incoming: PhasePools) -> int:
    """Point every reader at ``incoming``. ALL THREE OR NONE.

    ``readers`` maps a name to the object holding a binding; each is stamped
    with the new generation. The stamping is what makes the coherence check
    afterwards meaningful -- a reader nobody rebound keeps the old generation
    and is therefore detectable, which a "looks right" inspection is not.
    """
    check_shapes(incoming)
    check_pool_coverage(readers, incoming)
    missing = [name for name, obj in readers.items() if obj is None]
    if missing:
        raise RebindRefused(
            f"{LOG_PREFIX} cannot rebind: reader(s) {', '.join(sorted(missing))} "
            "are absent. A rebind that moves some readers and not others leaves "
            "them naming different memory for the same row id, which is worse "
            "than not rebinding at all."
        )

    generation = _STATE.advance(incoming.phase, incoming.host_pool)
    for name, obj in readers.items():
        try:
            _stamp(obj, incoming, generation)
        except Exception as e:
            # One reader failed mid-way: the set is now split, and there is no
            # safe way to serve from it. Roll the state forward to an
            # impossible generation so the coherence check below cannot pass.
            _STATE.advance(f"{incoming.phase}-broken")
            raise RebindIncoherent(
                f"{LOG_PREFIX} rebinding reader '{name}' failed ({e}); the "
                "reader set is now split and device-tier I/O must stay off."
            ) from e
    logger.info(
        "%s rebound %d reader(s) to the '%s' pools (generation %d, %s layers).",
        LOG_PREFIX,
        len(readers),
        incoming.phase,
        generation,
        incoming.layer_num(),
    )
    return generation


def _stamp(obj: Any, incoming: PhasePools, generation: int) -> None:
    """Move ONE reader onto the incoming pools and record the generation.

    Attributes are set only where they already exist: the three readers hold
    different subsets (the controller names the device pool and the allocator,
    the radix cache names the device pool and its host pool, the scheduler
    names the allocator), and inventing an attribute on a reader that never had
    one would create a fourth, silent binding.
    """
    hybrid = incoming.device_pool_hybrid
    if hybrid is None:
        hybrid = incoming.device_pool
    mapping = {
        "mem_pool_device": incoming.device_pool,
        # #706: the WRAPPER, deliberately. The controller's own __init__ keeps
        # the wrapped pool in `mem_pool_device_hybrid` because only the
        # wrapper knows its layers by GLOBAL id, and the canonical-window
        # rebuild at this very cutover reads exactly that. Stamping the
        # unwrapped pool here was the sibling defect of the 0828
        # frozen-window specimen: it silently removed the rebuild's layer-id
        # source at the one moment the rebuild must run.
        "mem_pool_device_hybrid": hybrid,
        "mem_pool_device_allocator": incoming.allocator,
        "mem_pool_host": incoming.host_pool,
        "hybrid_kv_cache": incoming.device_pool,
        "kvcache": incoming.device_pool,
        "kv_cache": incoming.device_pool,
        "token_to_kv_pool_allocator": incoming.allocator,
        "token_to_kv_pool_host": incoming.host_pool,
    }
    for attr, value in mapping.items():
        if value is not None and hasattr(obj, attr):
            setattr(obj, attr, value)
    obj.hicache_binding_generation = generation


def coherence_check(readers: dict) -> int:
    """Verify every reader is on the CURRENT generation. Raises otherwise.

    Run after a rebind and before device-tier I/O is re-armed. The failure this
    catches -- one reader left behind -- produces no error of its own: every
    call still succeeds, against different memory.
    """
    expected = _STATE.generation
    stale = {}
    for name, obj in readers.items():
        got = getattr(obj, "hicache_binding_generation", 0)
        if int(got) != int(expected):
            stale[name] = got
    if stale:
        raise RebindIncoherent(
            f"{LOG_PREFIX} reader(s) {stale} are not on generation {expected}. "
            "They would name different memory for the same row id than their "
            "peers, and every call against them would still succeed -- which is "
            "why this is checked rather than assumed."
        )
    return expected


def readers_of(scheduler: Any) -> dict:
    """The three readers, by name, from a scheduler.

    Named rather than discovered: a reader this function forgets is a reader
    the rebind silently leaves behind, so the list is explicit and the
    coherence check is run against the same dict that performed the rebind.
    """
    tree_cache = getattr(scheduler, "tree_cache", None)
    return {
        "scheduler": scheduler,
        "tree_cache": tree_cache,
        "cache_controller": getattr(tree_cache, "cache_controller", None),
    }


def phase_pools_for(scheduler: Any, phase: str) -> PhasePools:
    """The incoming phase's triple, assembled from what the boot built.

    The device pool and allocator come from the phase's own worker stack. The
    HOST pool is looked up in ``scheduler.phase_flip_host_pools`` -- a mapping
    the BOOT must provide, because a host pool cannot be derived from a device
    pool after the fact: it is allocated from it, and on this box that
    allocation is the binding constraint (DESIGN_706 C1). Its absence is the
    honest reason a rebind cannot arm yet, and it is reported as such.
    """
    stacks = getattr(scheduler, "phase_flip_stacks", None)
    if phase == "tp":
        worker = getattr(stacks, "tp_worker", None)
    else:
        worker = getattr(scheduler, "tp_worker", None)
    runner = getattr(worker, "model_runner", None)
    device_pool = getattr(runner, "token_to_kv_pool", None)
    allocator = getattr(runner, "token_to_kv_pool_allocator", None) or getattr(
        scheduler, "token_to_kv_pool_allocator", None
    )
    inner = getattr(device_pool, "full_kv_pool", device_pool)
    host_pools = getattr(scheduler, "phase_flip_host_pools", None) or {}
    host_pool = host_pools.get(phase)
    if host_pool is None:
        raise RebindRefused(
            f"{LOG_PREFIX} the '{phase}' phase has no host pool. A host pool is "
            "ALLOCATED FROM its device pool, so the flip's second phase needs "
            "its own, built at boot and charged to the pinned host budget; on "
            "this box that budget is the binding constraint (the flip's weight "
            "images leave ~5.37 GB for the whole tier). Until the boot provides "
            "scheduler.phase_flip_host_pools['" + phase + "'], the #718 disarm "
            "is the correct state and this rebind stays refused."
        )
    return PhasePools(
        phase=phase,
        device_pool=inner,
        host_pool=host_pool,
        allocator=allocator,
        device_pool_hybrid=device_pool,
    )


def settle_pending_releases(scheduler: Any) -> int:
    """Free every queued host release AGAINST THE BINDING IT WAS ENQUEUED ON.

    THE W35 CRASH, and the reason this is a SETTLE and not a discard.
    Measured on metal 2026-08-25, all three ranks, seconds after the first
    rebind this tree has ever armed:

        AssertionError: Double-free detected: slots not currently allocated:
          [0, 1, 2, ...]
        check_hicache_events -> drain_storage_control_queues -> _drain_release
          -> HostPoolGroup.free

    ``cc.host_mem_release_queue`` holds bare index tensors naming host slots
    allocated from the OUTGOING pool. ``rebind`` re-points ``mem_pool_host``
    to the incoming one. The next ordinary scheduler round then drains those
    entries and frees them against a pool that never handed those ids out.

    THE CRITERION, DECIDED BY READING RATHER THAN GUESSED. Dropping stale
    entries would be correct only if the outgoing pool died with them. It does
    not: ``_stamp`` only re-points readers, nothing tears the outgoing pool
    down, and ``scheduler.phase_flip_host_pools`` deliberately holds BOTH
    phases for process life so the flip can alternate. The outgoing pool
    therefore survives WITH LIVE ALLOCATIONS and comes back on the next flip --
    so a dropped release is a real, recurring host-slot LEAK in a pool that is
    about to be used again. Route, do not drop.

    SETTLING BEATS ROUTING-AT-DRAIN, and it needs no second stamp scheme.
    Routing later would mean stamping every entry at enqueue and carrying a
    generation->pool map -- a parallel bookkeeping scheme beside the #719
    generation, i.e. exactly the second-copy defect that cost W32. Here the
    invariant is made true BY CONSTRUCTION instead: the binding changes in
    exactly one place, so at this instant every queued entry belongs to the
    binding still installed. Drain it now and no entry can ever outlive its
    binding. The #719 generation stays the single coherence primitive and
    gains a second consumer rather than a rival.

    LOUD IN BOTH WRONG DIRECTIONS. A queue this cannot settle is REFUSED, not
    silently skipped: the caller turns that into a refused rebind, which is
    safe by construction (the binding does not move, so #718 keeps the device
    tier disarmed -- the state that held before the feature existed). Silently
    proceeding would trade a loud crash for a silent corruption, and silently
    discarding would trade it for a silent leak.

    Returns how many index blocks were settled, for the seam's log line.
    """
    tree_cache = getattr(scheduler, "tree_cache", None)
    cc = getattr(tree_cache, "cache_controller", None)
    if cc is None:
        return 0
    outgoing = getattr(cc, "mem_pool_host", None)
    queue = getattr(cc, "host_mem_release_queue", None)
    if outgoing is None or queue is None:
        return 0

    # The auxiliary per-component queues are NOT settled here, and that is a
    # refusal rather than an omission: their entries route through per-pool
    # allocators this function does not resolve, and guessing that routing on
    # a free path is how a loud crash becomes a silent corruption. Non-empty
    # means the rebind must not proceed.
    extras = getattr(cc, "extra_host_mem_release_queues", None) or {}
    pending_extra = {
        str(name): q.qsize()
        for name, q in extras.items()
        if getattr(q, "qsize", None) and q.qsize() > 0
    }
    if pending_extra:
        raise RebindRefused(
            f"{LOG_PREFIX} cannot settle auxiliary host release queue(s) "
            f"{pending_extra} before rebinding. Their entries name slots in "
            "the OUTGOING pool and route through per-component allocators this "
            "settle step does not resolve; draining them against the incoming "
            "binding is the W35 double-free and dropping them leaks host slots "
            "in a pool that returns on the next flip. Refusing leaves the "
            "binding where it is, which keeps #718 disarmed -- the state that "
            "held before this feature existed."
        )

    settled = 0
    blocks = []
    while True:
        try:
            blocks.append(queue.get_nowait())
        except Exception:  # noqa: BLE001 - Empty, and any queue that mimics it
            break
    if not blocks:
        return 0
    try:
        import torch

        outgoing.free(torch.cat(blocks, dim=0))
        settled = len(blocks)
    except Exception as exc:  # noqa: BLE001 - a failed settle must be loud
        raise RebindRefused(
            f"{LOG_PREFIX} failed to settle {len(blocks)} queued host release "
            f"block(s) against the outgoing pool ({exc}). The rebind is refused "
            "rather than run over an unsettled queue."
        ) from exc
    logger.info(
        "%s settled %d queued host release block(s) against the OUTGOING pool "
        "before rebinding, so no release can outlive the binding it was "
        "enqueued on (W35 double-free).",
        LOG_PREFIX,
        settled,
    )
    return settled


def _quiesce_storage_io(cc: Any) -> bool:
    """#1025: STOP THE STORAGE IO THREADS ACROSS THE POOL SWAP.

    THE SAME LAW AS ``settle_pending_releases``, ONE QUEUE OVER. W35 settled
    the RELEASE queue before the swap because "every queued release belongs to
    the binding still installed at this instant". The prefetch TRANSFER queue
    has exactly that property and was never settled: ``prefetch_io_aux_func``
    pulls an operation out of ``prefetch_buffer`` and runs ``_page_transfer``
    on it, and that operation names host indices and a page geometry minted
    under the binding that was installed when it was OPENED. The swap moves
    the pools underneath it mid-transfer.

    MEASURED, on every recent no-grid boot of this strand (2026-08-29), and
    invisible to twenty sections of analysis because it lands on a WORKER
    THREAD's stderr rather than in a counted marker:

        boot 3e06c24514  2 thread deaths   boot be4647950f  7
        boot 73b66f4a4a  8                 boot de3857b1f4  6
        boot 3f6c1b0799  5                 boot 0922e207ec  0

    in two flavours, both at a cutover second, both the same root:
      * ``StrayHostIndexError: HICACHE-INDEX REFUSED (#718 class, index axis):
        set_from_flat_data_page ... handed index range [42315, 42315] outside
        [0, 30518)`` -- indices minted against the wider outgoing host tier.
      * ``RuntimeError: shape '[2, 16, 1, 4, 256]' is invalid for input of
        size 8192`` -- a page shaped for the outgoing layer count reshaped
        under the incoming one.

    AND THE DEATH IS PERMANENT AND RANK-LOCAL, which is what makes it a root
    rather than noise. ``prefetch_io_aux_func`` catches ``Empty`` and nothing
    else, so the first stale operation kills the thread for the life of the
    process: from that instant this rank completes NO prefetch, every
    read-through misses, and its prefix match diverges from the peers that
    still have a live pipeline -- the shape ``raenge-nie-uneins-crash-stop``
    forbids, arriving through a queue instead of through a verdict.

    WHY QUIESCE RATHER THAN ROUTE. The release side was fixed by stamping the
    generation onto the operation and routing the free to the pool that
    generation names (``host_pool_for_generation``). Doing the same for
    transfers would need a THIRD consumer taught to read a stamp -- second
    bookkeeping beside the pool binding, which is the defect class this
    campaign is deleting. The user's own cutover law says the opposite and is
    simpler: at the flip everything is nulled and re-entered
    (``cutover-full-reset-reentry``). An operation that cannot be in flight
    while the geometry moves needs no stamp at all.

    NO NEW MECHANISM: ``_stop_storage_threads`` / ``_start_storage_threads``
    already exist for exactly this (runtime detach/attach), and the stop half
    already joins ``prefetch_io_aux_thread`` -- the half ``reset()`` hand-rolls
    and omits. This uses them and adds nothing.

    Returns True only when the caller MUST resume. A stop that did not
    complete returns False deliberately: threads are still alive on the old
    buffer, and starting a second set on top of them would double the very
    hazard being removed. That state is logged loudly and left as the honest
    pre-#1025 behaviour.
    """
    if cc is None or not getattr(cc, "enable_storage", False):
        return False
    stop = getattr(cc, "_stop_storage_threads", None)
    if stop is None or getattr(cc, "_start_storage_threads", None) is None:
        return False
    try:
        stop()
    except Exception as exc:  # noqa: BLE001 - a refused quiesce is reported, not raised
        logger.error(
            "%s #1025 STORAGE IO NOT QUIESCED for the cutover (%s). The "
            "prefetch transfer thread is still alive on the OUTGOING binding "
            "while the pools swap, which is the state that kills it with a "
            "StrayHostIndexError or a page-shape RuntimeError and leaves this "
            "rank with no prefetch pipeline at all. Not restarted here: a "
            "second set of threads on the same buffer would double the "
            "hazard.",
            LOG_PREFIX,
            exc,
        )
        return False
    return True


def _resume_storage_io(cc: Any, quiesced: bool) -> None:
    """#1025: bring the storage IO pipeline back on the INCOMING binding.

    Always paired with ``_quiesce_storage_io`` through a ``finally``, because
    the one outcome worse than a stale transfer is a rank that silently stops
    prefetching for the rest of the boot -- which is precisely the state the
    thread deaths produced, and precisely what nobody counted.

    Restarting also RESURRECTS a pipeline an earlier stale operation already
    killed: ``_start_storage_threads`` builds fresh threads and fresh queues,
    so a rank that lost its aux thread at cutover N is whole again at cutover
    N+1 instead of staying blind for the rest of the process.
    """
    if not quiesced:
        return
    try:
        cc.storage_stop_event.clear()
        cc._start_storage_threads()
        logger.info(
            "%s #1025 storage IO resumed on the incoming binding: fresh "
            "prefetch/backup threads and queues, nothing in flight from the "
            "outgoing one.",
            LOG_PREFIX,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "%s #1025 STORAGE IO DID NOT RESUME after the cutover (%s). This "
            "rank now completes NO prefetch for the rest of the process: "
            "every read-through misses, every re-admitted prefix is "
            "recomputed in full, and its match diverges from the peers whose "
            "pipeline is alive.",
            LOG_PREFIX,
            exc,
        )


def rebind_for_cutover(scheduler: Any, incoming_phase: str) -> Optional[int]:
    """Flag-gated rebind at the cutover. Returns the generation, or None.

    ``None`` means the feature is off -- the default, and byte-identical to not
    calling it. A refusal RAISES rather than returning None, because the two
    outcomes mean opposite things: off is a choice, refused is a configuration
    that asked for the rebind and cannot have it. The caller decides what to do
    with the raise at the seam; it must not become a silent no-op here.
    """
    server_args = getattr(scheduler, "server_args", None)
    if not getattr(server_args, "phase_flip_rebind_hicache", False):
        return None
    readers = readers_of(scheduler)
    incoming = phase_pools_for(scheduler, incoming_phase)
    # #1025: NOTHING MAY BE IN FLIGHT WHILE THE POOLS MOVE. The stop joins the
    # prefetch transfer thread; the resume in the `finally` below rebuilds the
    # pipeline on the incoming binding whatever happens in between. See
    # `_quiesce_storage_io` for the measurement and for why this is a quiesce
    # rather than a third generation-routing scheme.
    _cc = readers.get("cache_controller")
    _quiesced = _quiesce_storage_io(_cc)
    try:
        return _rebind_for_cutover_inner(
            scheduler, incoming_phase, readers, incoming
        )
    finally:
        _resume_storage_io(_cc, _quiesced)


def _rebind_for_cutover_inner(
    scheduler: Any, incoming_phase: str, readers: Any, incoming: Any
) -> Optional[int]:
    """The rebind proper, bracketed by the #1025 storage-IO quiesce."""
    # W35: settle BEFORE the swap. Every queued release belongs to the binding
    # still installed at this instant; after `rebind` it would be freed against
    # a pool that never allocated it.
    settle_pending_releases(scheduler)
    generation = rebind(readers, incoming)
    coherence_check(readers)
    # #1068 (G2): the write-through ring is the complement of the prefetch
    # budget PROPERTY of the pool just bound; rebuild it here, AFTER the
    # coherence check, and print the budget now in force (L3).
    rebuild_staging_ring_after_rebind(readers, scheduler)
    # #861: THE FOURTH PARTICIPANT. The draft half is registered here and
    # nowhere else, for the same reason the other three are rebound here: its
    # host indices are 1-to-1 with the target host pool's, and that pool has
    # just moved. Registering it at boot -- which is what Scheduler.__init__
    # tried to do, against the deliberately-nulled boot-phase drafter -- names
    # generation 0's slot space for a consumer that runs at generation n.
    #
    # AFTER coherence_check, deliberately: a draft half armed against a target
    # binding that itself failed to move is the partial-rebind hazard this
    # module's own docstring calls "strictly worse than the #718 state".
    #
    # A refusal here RAISES (the caller treats it as a refused rebind); a
    # phase that simply owns no drafter DISARMS quietly, which is the ordinary
    # tp->pp leg.
    from sglang.srt.mem_cache.kv_cache_builder import (
        rebind_hicache_draft_for_phase,
    )

    # ITS OWN try/except, AND THE REASON IS A LOG LINE THAT ACCUSED THE WRONG
    # PARTICIPANT (#861c). In W37-C the draft leg raised on every cutover after
    # the first, the exception propagated out of this function, and the seam's
    # handler printed "#719 HiCache rebind refused" six times -- for a TARGET
    # rebind that had ALREADY succeeded two lines above, generation and
    # coherence check included. A reader chasing "the rebind was refused" was
    # chasing a rebind that happened.
    #
    # The target rebind is COMMITTED at this point and must be reported as
    # such. A draft-half refusal is a strictly smaller failure -- the device
    # tier stays armed for the target, only the draft piggyback is off -- so it
    # is logged as itself and the committed generation is returned.
    try:
        rebind_hicache_draft_for_phase(scheduler, incoming_phase)
    except Exception as exc:  # noqa: BLE001 - a draft refusal is not a rebind refusal
        logger.error(
            "%s the TARGET rebind to '%s' COMMITTED at generation %d; only the "
            "#861 draft half was refused (%s). Draft-piggyback L2/L3 traffic is "
            "off until the next cutover; target device-tier I/O is unaffected.",
            LOG_PREFIX,
            incoming.phase,
            generation,
            exc,
        )
    # #706 x #719 (0828 specimen): the canonical storage windows are DERIVED
    # QUANTITIES of the pool geometry this rebind just swapped -- the KV
    # window from the host pool's flat page and the device pool's global
    # layer ids, the mamba window from the current pool's layer map and this
    # rank's head shard. They were built ONCE at storage attach; without this
    # rebuild every canonical read after the first cutover refuses
    # deterministically ("read target holds 32768 bytes but this KV page
    # window is 8192 bytes", 128x on boot_943bx_0828), the prefetch completes
    # 0, and the #928 anchor re-prefills everything.
    #
    # A failure here is logged, not raised: the target rebind is COMMITTED,
    # the old windows stay installed, and the backend's own probe
    # compatibility check (`_canonical_probe_mismatch`) keeps that state an
    # honest cold miss instead of a broken promise.
    try:
        cc = readers.get("cache_controller")
        rebuild = getattr(cc, "rebind_canonical_windows", None)
        if rebuild is not None:
            rebuild(incoming.phase)
    except Exception as exc:  # noqa: BLE001 - a window refusal is not a rebind refusal
        logger.error(
            "%s the '%s' rebind COMMITTED at generation %d, but the #706 "
            "canonical storage windows could not be re-derived from the "
            "incoming pools (%s). The previous windows stay installed; every "
            "canonical probe now answers 0 (honest miss) until the next "
            "successful rebuild.",
            LOG_PREFIX,
            incoming.phase,
            generation,
            exc,
        )
    return generation


def rebuild_staging_ring_after_rebind(readers: dict, scheduler: Any) -> None:
    """#1068 (G2): rebuild the tree's staging ring against the pool the
    readers were just bound to, then emit L3 from the live budget property.

    A tree without the hook (plain RadixCache) is skipped; a failure is
    logged and never raised, because the target rebind is COMMITTED at this
    point and a ring refusal is a strictly smaller failure than an
    un-reported torn binding.
    """
    tree = readers.get("tree_cache")
    rebuild = getattr(tree, "rebuild_staging_write_ring", None)
    if rebuild is None:
        return
    try:
        rebuild(getattr(scheduler, "server_args", None))
    except Exception as exc:  # noqa: BLE001 - a ring refusal is not a rebind refusal
        logger.error(
            "%s the rebind COMMITTED, but the #810 staging ring could not be "
            "rebuilt against the incoming host pool (%s). Write-through runs "
            "without its bound until the next successful rebuild.",
            LOG_PREFIX,
            exc,
        )
        return
    from sglang.srt.mem_cache.prefetch_budget import log_prefetch_limit

    log_prefetch_limit(readers.get("cache_controller"), site="rebind_for_cutover")


def current_generation() -> int:
    """#760/#719: the generation a host write-back must still be riding.

    THE STAMP HAS TO BE READ TWICE, and the second read is the point. A
    write-back is ENQUEUED against one binding and CONSUMED later; the flip
    rebinds in between. Both crash specimens died three seconds AFTER a
    ``pp_to_tp`` cutover completed, which is an already-queued copy reaching a
    pointer table that no longer describes the pool it was built from.

    ``check_shapes`` cannot catch that one. Under ``layer_first`` the host
    layout equals the device layout, so a stale binding is SHAPE-IDENTICAL to
    the live one -- the shape guard passes by construction and stays silent,
    which is exactly what the #760 evidence records (guard armed on three
    ranks, zero refusals, SIGSEGV anyway).
    """
    return binding_state().generation


def host_pool_for_generation(generation):
    """The host pool a release stamped at ``generation`` must be freed against.

    W35 DURABLE FIX, replacing a one-shot settle. `settle_pending_releases`
    drained the queue at the rebind instant, which cannot see entries that do
    not exist yet -- and three producers keep filling that queue AFTER the
    rebind: `_drain_revoke` (prefetch revocation), the prefetch transfer thread
    (`_page_transfer` -> `append_host_mem_release`), and the direct path. Each
    manufactures a release naming slots from the pool bound when the OPERATION
    was opened, not the pool bound now.

    So the generation travels with the operation (`StorageOperation` stamps
    itself at construction) and the release is routed to the pool that
    generation names. Same #719 authority, one more consumer -- not a second
    stamp scheme, which is the defect that cost W32.
    """
    return _STATE.host_pool_at(generation)


def write_back_stamp_is_current(stamped: int) -> bool:
    """True when a write-back stamped at ``stamped`` may still be consumed."""
    if stamped is None:
        return False
    return int(stamped) == int(current_generation())
