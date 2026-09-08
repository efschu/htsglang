"""Weg-2 slice S1 -- the three things the upstream memory saver does not do.

Upstream owns the sleep/wake mechanism itself and Weg 2 uses it unchanged:
``TorchMemorySaverAdapter`` (``srt/utils/torch_memory_saver_adapter.py``), the
``release_memory_occupation`` / ``resume_memory_occupation`` RPCs, the tag
regions in ``memory_pool.py`` / ``model_runner.py``, and
``update_weights_from_disk`` as the backup-OFF wake source.  Nothing here
duplicates any of that -- a fork-owned twin of an upstream mechanism is a
defect, so this module adds exactly the three pieces upstream has no equivalent
for:

* **W12 ``Weg2MemorySaverInactive``** -- the refusal.  ``create(enable=False)``
  returns a no-op adapter whose ``pause()``/``resume()`` are literally ``pass``,
  so every sleep is a success value with no action.  Upstream's own
  ``check_validity()`` only warns and the release path never calls it.
* **The sleep-acceptance census** (design (S) 2.4 step 11) -- the runtime
  instrument that answers "the dormant group holds nothing" on the flip path.
  It reads the canonical device registry (``srt/registry/nvml.py``) and the
  arena's own counters (``arena_census()``); it keeps no counters of its own.
* **The per-physical-GPU PCIe serialisation lock** (design (S) 2.7) -- lifted
  from #89 hibernate's ``park_weights_to_disk`` (*adopt the lock, not the
  module*): a sleep-D2H and a wake-H2D must never overlap on one card's link,
  which halves both on the x4-linked 3080.  This is a DIFFERENT lock from the
  L2 ring's ``flock`` and is named separately on purpose.

Every wait here is bounded and every expiry is a refusal, never a longer wait.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

#: Where the PCIe serialisation lock files live.  ``/dev/shm`` is a tmpfs that
#: every rank on this host shares, which is exactly the scope of the lock: one
#: physical GPU, all processes on this box.
PCIE_LOCK_DIR_ENV = "SGLANG_WEG2_PCIE_LOCK_DIR"
DEFAULT_PCIE_LOCK_DIR = "/dev/shm"

#: Deliberately NOT ``weg2-l2-<uuid>`` -- that name belongs to the L2 host ring
#: backing file and its own ``flock``.  Two locks, two names, so a reader never
#: has to guess which one a path refers to.
PCIE_LOCK_PREFIX = "weg2-pcie-serialize"

#: C12: WHERE THE PER-CARD DUPLEX RATIO COMES FROM.  Launcher output, never
#: operator input (spec R19), in the same shape as ``TMS_HOST_RING_MAP``:
#: ``<uuid>=<ratio>,<uuid>=<ratio>,...``.  The launcher solves it from the
#: step-0 probe's own measured lines (``ring_table.solve_duplex``) and every
#: rank reads only its OWN card's row.  Absent means "not measured on this
#: boot", and then no card splits -- the conservative direction, because a
#: split that is not earned re-creates exactly the overlap R9 forbids.
PCIE_DUPLEX_ENV = "SGLANG_WEG2_PCIE_DUPLEX"

#: Spec R17's gate, verbatim: "if concurrent aggregate < 1.5x serial, S1's
#: direction split is a null".  A threshold ruled by the spec, not a size
#: guessed here -- and the numbers it grades are measured, per card.  The
#: step-0 probe of 2026-09-07T23:13Z measured 1.759 on the 5090 (nvml1) and
#: 1.687 on nvml2, both DUPLEX-OK, against 1.316 on nvml0 (the x4 slot) which is
#: DUPLEX-NULL.
#:
#: FIX 2 round 2, finding 4: DUPLEX-NULL DOES NOT MEAN ONE KEY ON THIS TREE, and
#: the sentence that said it did is RETRACTED, not softened.  The ratio is
#: an expectation about THROUGHPUT; whether the key splits is a CORRECTNESS
#: decision, and under the gathered legs of C9 the launcher decides ``split``
#: for EVERY card whatever its ratio (``launcher._split_decisions``:
#: ``if leg_form == "interleave": return {uuid: True ...}``) -- because on a
#: one-key card the gathered pair serialises into a deadlock, which is boot
#: weg2rg2's W31 -> W29 -> W4.  What this threshold still grades is the VALUE of
#: the split: Amendment A1-4, the benefit is PER CARD and the flip's critical
#: path is exactly the card that does not get it.
DUPLEX_SPLIT_MIN_RATIO = 1.5

#: The two direction tokens the key may carry.  ``d2h`` is a sleep's copy to the
#: host, ``h2d`` a wake's copy back -- named from the DEVICE's point of view,
#: which is the direction the PCIe link sees.
PCIE_DIRECTIONS = ("d2h", "h2d")

#: A sleep-D2H or wake-H2D of a 27 GiB shard runs ~2.1 s measured (campaign (a),
#: 2026-09-06, n=9).  The default deadline allows a full transfer of the sibling
#: plus slack; the caller may shorten it.
DEFAULT_PCIE_LOCK_TIMEOUT_S = 120.0

#: The S1 sleep-acceptance criterion in its DELTA form: how much of what this
#: process held before the pause must be gone after it.
#:
#: PROVENANCE, and it is a chosen number, not a measured one.  Campaign (a)
#: measured, on this rig, a genuine sleep going 30,154 -> 1,294 MiB per-process
#: (CAMPAIGN_a_0906.md §2 arm 1, n=9 steady cycles over two cold boots): the
#: sleep RELEASED 95.7 % and RETAINED 4.3 %.  A no-op sleep -- the exact
#: condition W12 and this census exist to catch -- releases 0 %.  Half is the
#: midpoint between those two, i.e. ~11x the measured retained margin, so the
#: criterion cannot be tripped by allocator noise and cannot be passed by a
#: no-op.  It is deliberately NOT a ceiling on residency: the dormant floor
#: D_c is UNMEASURED until S3 (register U2), and inventing a MiB ceiling here
#: would be a number the tree cannot defend.  When S3 measures D_c its launcher
#: passes ``expected_max_resident_bytes`` and the ceiling form takes over --
#: both forms may be supplied, and then both must hold.
WEG2_SLEEP_MIN_RELEASED_FRACTION = 0.5

#: The tag set a Weg-2 sleep releases, and therefore the only request shape the
#: DELTA form's denominator is the right one for.  The floor is a fraction of
#: this process's WHOLE device residency (NVML per-process bytes), so it may
#: only grade a release that actually targets the whole of it: the weights
#: image AND the KV pool (which carries the mamba/GDN anchors under the same
#: tag -- memory_pool.py:1017, there is no separate mamba tag).  Anything less
#: -- the #89 park's ``tags=["weights"]``, a kv-only release -- releases a
#: PROPER SUBSET and cannot be graded against the whole; see
#: :func:`sleep_acceptance_census`.
WEG2_SLEEP_TAGS = frozenset((GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS))


class Weg2MemorySaverInactive(RuntimeError):
    """W12: the memory-saver adapter is a no-op, so every sleep is a lie."""


class Weg2WakeRefused(RuntimeError):
    """W4: a wake leg failed.  VRAM has been mutated; there is no unwound state.

    Group-fatal STOP, never an automatic retry.  The recovery lane is an
    operator-driven full teardown and relaunch of that group (design (S) 2.7).
    """


class Weg2DormantRefused(RuntimeError):
    """W25 (S1 boot killer K2, WEG2_BUILD_DECISIONS_0906.md section 1g).

    A group whose ``kv_cache`` tag is paused holds NO request-index pool, NO KV
    pool and NO mamba pool: ``prepare_for_extend`` on an admitted request
    walks into ``write_req_to_token_pool_triton`` on released VMM pages and the
    Triton launcher rejects the pointer, which kills the whole group.  Measured
    2026-09-07 05:56:56 on weg2s1: one health-probe generation, 26 s into a
    dormant dwell, took the group down.  The front never routes to a dormant
    group; this refusal is the OTHER half -- anything that reaches the port
    anyway (a probe, an operator curl, a stale client) gets a named abort at
    the admission seam, BEFORE ``prepare_for_extend``, and the group lives.
    """


class Weg2PcieLockTimeout(RuntimeError):
    """The PCIe serialisation lock was not acquired inside its deadline."""


# ---------------------------------------------------------------------------
# W12
# ---------------------------------------------------------------------------


def assert_memory_saver_active(adapter: Any, *, context: str) -> None:
    """Refuse unless ``adapter`` will actually release memory.

    ``adapter.enabled`` is upstream's own property and the single authority:
    ``_TorchMemorySaverAdapterNoop`` returns False, and the real adapter
    returns ``_memory_saver is not None and _memory_saver.enabled``, so a
    present-but-disabled library is caught too.  No second bookkeeping.

    Called from the launcher at boot (once per rank) and from the first sleep,
    which is the last moment before a no-op release would return success.
    """
    if adapter is not None and bool(getattr(adapter, "enabled", False)):
        return
    raise Weg2MemorySaverInactive(
        f"W12 Weg2MemorySaverInactive at {context}: the torch-memory-saver "
        f"adapter reports enabled=False, so pause()/resume() are no-ops that "
        f"return success and no VRAM is ever released. Launch this group with "
        f"--enable-memory-saver (and check that torch_memory_saver imported "
        f"and its hook mode is 'preload'). Refusing rather than sleeping."
    )


def checkpoint_quantization(model_config: Any, server_args: Any) -> Optional[str]:
    """The quantization actually in force for this process's checkpoint.

    ONE definition, two users (the launch arm and the wake), so a launcher edit
    cannot make the two disagree.  ``ModelConfig.quantization`` is the merged
    value -- the CLI flag AND the config.json ``quantization_config`` the loader
    auto-detects -- so a checkpoint that carries its own quant config without
    ``--quantization`` is covered; ``server_args`` is the fallback for the
    moments where no model config is reachable yet.
    """
    for holder in (model_config, server_args):
        if holder is None:
            continue
        value = getattr(holder, "quantization", None)
        if value:
            return str(value)
    return None


def assert_backup_off_wake_refill_is_defined(
    *, quantization: Optional[str], context: str
) -> None:
    """W4: refuse a backup-OFF wake whose refill would re-run a repacking pass.

    The backup-OFF arm (record 1b round-2 Q2, option (ii)) refills the weights
    with ``update_weights_from_disk``, which ends in
    ``loader.load_weights_and_postprocess`` -- ``model.load_weights(iter)``
    followed by ``quant_method.process_weights_after_loading(module)`` for every
    module (``model_loader/loader.py:921-931``).  Neither half is idempotent on
    a quantized checkpoint, and the FIRST half is the one that breaks:

    * the post-load pass REPLACES the parameter rather than writing into it --
      ``layer.weight = Parameter(weight.t(), requires_grad=False)``
      (``compressed_tensors_w8a8_int8.py:151,159``, and the same shape in the
      AWQ/FP8 schemes).  DESK-MEASURED on this tree with the reference
      checkpoint's own scheme: a ``ModelWeightParameter`` of shape ``(4, 8)``
      carrying a ``weight_loader`` comes back a plain ``Parameter`` of shape
      ``(8, 4)`` with no ``weight_loader``
      (``test_quantized_post_load_replaces_the_weight_parameter``).
    * so the wake's ``model.load_weights(iter)`` resolves
      ``getattr(param, "weight_loader", default_weight_loader)`` to the DEFAULT
      loader, which asserts ``param.size() == loaded_weight.size()``
      (``weight_utils.py:1709``) against a transposed parameter and raises.
      ``model_runner.py:2857-2865`` then re-runs the same failing load as its
      rollback, OUTSIDE any ``try``.

    There is no second lane in this tree: #89's ``HibernateModelLoader`` is a
    ``BaseModelLoader``, which ``update_weights_from_disk`` rejects outright
    (``model_runner.py:2827-2829``), and it supports GGUF only.

    So this arm is UNDEFINED for a quantized checkpoint, and the honest form is
    a named refusal at the earliest decidable moment rather than a wake that
    commits the VMM pages and then fails inside the loader.  The two lanes that
    ARE defined, both named in the message: launch the group with
    ``--enable-weights-cpu-backup`` (the TMS restore carries the post-transform
    bytes and no reload runs at all), or serve an unquantized checkpoint.
    """
    if not quantization:
        return
    raise Weg2WakeRefused(
        f"W4 Weg2WakeRefused at {context}: this group runs the backup-OFF wake "
        f"arm (--enable-memory-saver without --enable-weights-cpu-backup) on a "
        f"{quantization!r} checkpoint. That arm refills the weights with "
        f"update_weights_from_disk, whose load_weights_and_postprocess writes "
        f"the checkpoint tensors into parameters a previous "
        f"process_weights_after_loading has already REPLACED with transposed, "
        f"weight_loader-less ones (loader.py:921-931; "
        f"compressed_tensors_w8a8_int8.py:151,159), so the refill raises inside "
        f"the loader and the model_runner rollback re-runs the same failing "
        f"load. Weights would be left committed and undefined. Launch this "
        f"group with --enable-weights-cpu-backup (the TMS restore carries the "
        f"post-transform bytes and no reload runs), or serve an unquantized "
        f"checkpoint. Refusing rather than waking into undefined weights."
    )


# ---------------------------------------------------------------------------
# The per-physical-GPU PCIe serialisation lock (design (S) 2.7, from #89)
# ---------------------------------------------------------------------------


def _sanitize(uuid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", uuid)


def duplex_ratios(published: Optional[str] = None) -> Dict[str, float]:
    """``{card uuid: measured duplex ratio}`` as the launcher published it.

    Parsed permissively and REPORTED as empty when nothing was published -- an
    unparsable row is dropped rather than defaulted, because a defaulted ratio
    is exactly the hand number spec section 10.3 forbids and it would open the
    split on a card nobody measured.
    """
    return _parse_duplex(published)[0]


#: FIX 2 round 2, finding 2.  The wire format of :data:`PCIE_DUPLEX_ENV` carries
#: its own version, and a reader that does not know the version REFUSES.  The
#: predecessor's format (``<uuid>=<ratio>:split``) was a silent superset of the
#: one before it (``<uuid>=<ratio>``): an older parser did ``float("1.759:split")``,
#: dropped every row, resolved an EMPTY table, and therefore gave one key on ALL
#: THREE cards -- strictly worse than the boot the split exists to fix, with no
#: line saying so anywhere.  A format that degrades to the unsafe answer is not a
#: format; the version token is what makes the degrade a named stop instead.
PCIE_DUPLEX_FORMAT = "v2"


class Weg2DuplexDecisionRefused(RuntimeError):
    """W36 -- this rank cannot resolve the launcher's per-card key decision.

    Two shapes, one refusal, both of them "the launcher and the rank do not
    agree on what the key is", which is the precondition for boot weg2rg2's
    deadlock (S blocked in ``acquire`` holding the key W needs, W31 at the
    120 s budget, W29 on every rank, group-fatal W4 -- 120 s into a flip that
    has already mutated VRAM):

    * an UNKNOWN FORMAT VERSION -- a newer launcher against an older tree on
      ``PYTHONPATH``, the partial-rebase case the launcher's own W34 comment
      names;
    * DECISIONS PUBLISHED BUT NONE PARSED -- a non-empty published string that
      yields no ``split``/``single`` row at all.  Falling back to R17's ratio
      gate there is precisely the silent degrade: the ratio gate answers a
      question about VALUE and would leave the x4 card on one key.

    There is no fallback and none is claimed: the launcher's own launch check
    resolves the key through this same module (``_serialised_cards``), so this
    refusal is normally raised BEFORE either group starts, and never mid-flip.
    """


def duplex_splits(published: Optional[str] = None) -> Dict[str, bool]:
    """``{card uuid: does its key split}`` as the launcher DECIDED it.

    FIX 1 round 1.  The ratio is an expectation about throughput; whether the
    key splits is a CORRECTNESS decision, and after boot weg2rg2 the two are no
    longer the same question.  The launcher makes that decision once, from the
    ratio AND from the leg form, and publishes it here beside the ratio so the
    rank that builds the key and the launch check that prices it read one
    answer.

    FIX 2 round 2: a published string this reader cannot resolve is
    :class:`Weg2DuplexDecisionRefused`, never a fall back to the ratio gate.
    """
    return _parse_duplex(published)[1]


def _parse_duplex(published: Optional[str]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """``[v2|]<uuid>=<ratio>[:split|:single]`` rows -> (ratios, decisions).

    The ratio may be empty (``<uuid>=:split``): a card the step-0 probe never
    measured still needs a decision, and under gathered legs that decision is
    ``split`` -- see :func:`pcie_lock_separates_directions`.

    Refuses (W36) on an unknown version token, and on a non-empty string that
    produced no decision at all.  The EMPTY string is not a refusal: it is the
    honest "nothing was published on this boot", and it resolves to no split,
    which the launcher's own check then reads as SERIALISED and refuses at
    launch (W34).
    """
    raw = os.environ.get(PCIE_DUPLEX_ENV, "") if published is None else published
    raw = (raw or "").strip()
    if "|" in raw:
        version, _, raw = raw.partition("|")
        version = version.strip()
        if version != PCIE_DUPLEX_FORMAT:
            raise Weg2DuplexDecisionRefused(
                f"W36 Weg2DuplexDecisionRefused {PCIE_DUPLEX_ENV} carries format "
                f"{version!r}, and this tree reads {PCIE_DUPLEX_FORMAT!r} -- the "
                "launcher and this rank would build DIFFERENT PCIe keys, which is "
                "the single-key deadlock of boot weg2rg2 (W31 -> W29 -> W4).  No "
                "fallback: relaunch with one tree on PYTHONPATH."
            )
    ratios: Dict[str, float] = {}
    splits: Dict[str, bool] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        uuid, _, value = item.partition("=")
        uuid = uuid.strip()
        ratio_text, _, decision = value.partition(":")
        decision = decision.strip()
        if decision in ("split", "single"):
            splits[uuid] = decision == "split"
        elif decision:
            continue
        try:
            ratios[uuid] = float(ratio_text)
        except ValueError:
            continue
    if raw and not splits:
        raise Weg2DuplexDecisionRefused(
            f"W36 Weg2DuplexDecisionRefused {PCIE_DUPLEX_ENV} is set "
            f"({raw[:200]!r}) but names no split/single decision for any card.  "
            "R17's ratio gate is NOT the fallback here: it grades whether the "
            "split is worth having, not whether the single key is safe, and "
            "under gathered legs the single key deadlocks (boot weg2rg2). "
            "No fallback: relaunch with one tree on PYTHONPATH."
        )
    return ratios, splits


def pcie_lock_separates_directions(
    nvml_uuid: str,
    *,
    ratios: Optional[Dict[str, float]] = None,
    splits: Optional[Dict[str, bool]] = None,
) -> bool:
    """Does the lock KEY separate the two directions ON THIS CARD?

    THE FACT LIVES WITH THE KEY (review nb1 of the ring fix-2 tip): this used to
    be a hand-typed module boolean sitting beside :func:`pcie_lock_path`, which
    had no direction parameter at all -- so the launcher's gate read a constant
    while the function that owns the key could not have honoured it.  Now the
    key is derived from the same answer this returns, and the launcher asks per
    card.

    THE PUBLISHED DECISION WINS, AND IT IS NOT THE RATIO (FIX 1 round 1).  R17's
    ratio answers "is the split worth it"; boot weg2rg2 answered a different
    question the hard way -- "is the single key SAFE" -- and it is not.  Each
    leg holds the per-card key across its whole tag loop (weight_updater's
    sleep-D2H and wake-H2D blocks), so on a card where both legs resolve to ONE
    key the gathered pair of C9 is serialised again: whichever leg wins the key
    runs to completion, and if that is the SLEEP leg it blocks in the ring's
    ``acquire`` with free = H - image_W = 0 while the wake leg that would fund
    it is queued on the key it holds.  That is W31 Weg2HostRingExhausted
    need=24 free=12 MiB after 120.189 s on nvml0, then W29 on all three P ranks,
    then group-fatal W4 -- observed, 2026-09-08 03:06Z.  R5's corridor, which
    the launcher checks and which passes 6/6 on both tables, has as its PREMISE
    that W's per-tag releases fund S's acquires; the single key falsifies it.

    So under gathered legs the launcher publishes ``split`` for every card and
    that decision is read here.  It costs nothing measured: the x4 card's own
    1.316 is a CONCURRENT-OVER-SERIAL ratio, i.e. concurrency is 32 % faster
    there too -- R17's 1.5 gate only says the split is not worth a mechanism of
    its own, which is a statement about value, never about safety (A1-4 says
    that card gets no benefit from the split, not that it must not have one).

    With NOTHING published the ratio gate stands -- the no-launcher case, and
    the conservative reading of it.  With something published that this reader
    cannot resolve, :func:`_parse_duplex` refuses by name (W36, FIX 2 round 2):
    the ratio gate is not a fallback for a decision that exists and could not
    be read.
    """
    decisions = duplex_splits() if splits is None else splits
    if nvml_uuid in decisions:
        return decisions[nvml_uuid]
    table = duplex_ratios() if ratios is None else ratios
    ratio = table.get(nvml_uuid)
    return ratio is not None and ratio >= DUPLEX_SPLIT_MIN_RATIO


def pcie_lock_path(
    nvml_uuid: str,
    *,
    lock_dir: Optional[str] = None,
    direction: Optional[str] = None,
    ratios: Optional[Dict[str, float]] = None,
    splits: Optional[Dict[str, bool]] = None,
) -> str:
    """Path of the PCIe serialisation lock for one physical GPU.

    C12: the key is ``<uuid>.<d2h|h2d>`` on a card whose measured duplex ratio
    earns the split, and the bare ``<uuid>`` everywhere else -- so two holders
    of the SAME direction on one card still serialise (that is one link
    direction and it is genuinely shared), while opposite directions stop
    excluding each other on the cards where the metal says they need not.
    """
    directory = lock_dir or os.environ.get(PCIE_LOCK_DIR_ENV, DEFAULT_PCIE_LOCK_DIR)
    key = _sanitize(nvml_uuid)
    if direction:
        if direction not in PCIE_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {PCIE_DIRECTIONS}, not {direction!r}"
            )
        if pcie_lock_separates_directions(nvml_uuid, ratios=ratios, splits=splits):
            key = f"{key}.{direction}"
    return os.path.join(directory, f".{PCIE_LOCK_PREFIX}-{key}.lock")


def _resolve_uuid(nvml_uuid: Optional[str]) -> str:
    """The card key, or a refusal -- never a CUDA context bought to answer.

    ``current_device_uuid()`` falls back to torch when the pin is not readable
    from the environment, and that fallback INITIALISES CUDA.  The canonical
    holder of that guard is the flight recorder
    (``srt/mem_ledger/flight_recorder.card_pin_unresolvable_without_cuda``);
    this call is a USE of it, not a copy.  On the sleep path the context always
    exists, so the guard never fires there -- it exists because
    :func:`sleep_acceptance_census` is public and the S3 launcher calls it
    PRE-LAUNCH per rank, which is exactly the moment where a context created by
    the instrument would corrupt the very number the instrument reports.
    """
    if nvml_uuid is not None:
        return nvml_uuid
    from sglang.srt.mem_ledger.flight_recorder import (
        card_pin_unresolvable_without_cuda,
    )

    unresolved = card_pin_unresolvable_without_cuda()
    if unresolved is not None:
        raise RuntimeError(unresolved)

    from sglang.srt.registry import nvml as nvml_registry

    return nvml_registry.current_device_uuid()


def resolve_pcie_lock_key() -> str:
    """This process's physical-GPU key for the PCIe lock.

    Public on purpose: a caller that wants to degrade to "unserialised" when
    the card key cannot be resolved must be able to resolve the key in a
    ``try`` of its own, and then take the lock OUTSIDE that handler -- so a
    :class:`Weg2PcieLockTimeout` propagates by construction rather than by the
    ordering of two ``except`` clauses.
    """
    return _resolve_uuid(None)


@contextmanager
def pcie_transfer_lock(
    *,
    nvml_uuid: Optional[str] = None,
    lock_dir: Optional[str] = None,
    timeout_s: float = DEFAULT_PCIE_LOCK_TIMEOUT_S,
    poll_s: float = 0.01,
    label: str = "transfer",
    direction: Optional[str] = None,
    ratios: Optional[Dict[str, float]] = None,
) -> Iterator[str]:
    """Serialise host<->device transfers per PHYSICAL GPU.

    Two co-located ranks (Weg 2 puts one P rank and one D rank on each card)
    would otherwise overlap a sleep-D2H with a wake-H2D and halve both on the
    x4-linked 3080.  Keyed on the NVML UUID, so co-located ranks agree on the
    key without any registry between them.

    ``flock`` is held on this open file description only, so two threads or two
    processes contend correctly and the lock dies with the holder.  Bounded:
    the deadline's expiry raises, it never waits longer.

    C12: pass ``direction`` (``"d2h"`` on a sleep, ``"h2d"`` on a wake) and the
    key splits ON THE CARDS WHOSE MEASURED DUPLEX RATIO EARNS IT -- see
    :func:`pcie_lock_path`.  Passing no direction keeps the old single key,
    which is what every caller that is not a flip leg wants.
    """
    uuid = _resolve_uuid(nvml_uuid)
    path = pcie_lock_path(uuid, lock_dir=lock_dir, direction=direction, ratios=ratios)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    deadline = time.monotonic() + float(timeout_s)
    waited_s = 0.0
    handle = open(path, "w")
    try:
        t_wait = time.perf_counter()
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise Weg2PcieLockTimeout(
                        f"PCIe serialisation lock {path} not acquired within "
                        f"{timeout_s:.1f}s while waiting to {label} on GPU "
                        f"{uuid}; the sibling rank on this card is still "
                        f"transferring. Refusing rather than overlapping."
                    ) from None
                time.sleep(poll_s)
        waited_s = time.perf_counter() - t_wait
        t_held = time.perf_counter()
        try:
            yield path
        finally:
            held_s = time.perf_counter() - t_held
            logger.info(
                "WEG2-PCIE-LOCK %s on %s dir=%s key=%s waited=%.3fs held=%.3fs "
                "(denominator: one physical GPU, all processes on this host; "
                "a key WITHOUT a direction suffix means this card did not reach "
                "R17's 1.5x gate and both directions share it by measurement)",
                label,
                uuid,
                direction or "none",
                os.path.basename(path),
                waited_s,
                held_s,
            )
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# C14 -- the per-card VRAM credit (spec section 4 C14, F5)
# ---------------------------------------------------------------------------


class Weg2FlipRankDisagree(RuntimeError):
    """W29 -- one rank of the group could not complete this leg, so NO rank may.

    The law (memory ``raenge-nie-uneins``, spec section 10.5): a state-changing
    decision is taken once for the whole group, and a detected disagreement is a
    STOP, never a compensation.  The flip's legs are exactly that shape -- the
    front reads one rank's HTTP answer as "the GROUP moved these pages" and
    immediately moves the OTHER group's pages onto the same cards, so a leg that
    succeeded on two ranks of three is worse than one that failed on all three.

    Raised on EVERY rank, with the failing rank and tag named, from the ok-bit
    gather that spec C15 puts AFTER the existing ``monitored_barrier`` (R15:
    gloo's ``all_gather_object`` has no timeout of its own, so the barrier --
    which NAMES a non-joiner -- must run first).
    """


class Weg2VramCreditRefused(RuntimeError):
    """W35 -- the waking rank needs device bytes the sleeping rank will not free.

    NAMED TWICE ON PURPOSE.  Spec section 6 lists this refusal as ``W30
    Weg2VramCreditRefused``; the builder briefing of 2026-09-08 renames it
    ``W35``.  It is ONE refusal, and both codes appear once each in the message
    so that section 9.3's trap-safe count finds it under either name and neither
    count is doubled.
    """


#: Deliberately a THIRD name beside ``weg2-pcie-serialize`` (this module) and
#: ``weg2-l2-`` (the host ring): three mechanisms, three prefixes, so a path in
#: a log never has to be guessed at.
VRAM_CREDIT_PREFIX = "weg2-vram-credit"
VRAM_CREDIT_DIR_ENV = "SGLANG_WEG2_VRAM_CREDIT_DIR"


def credit_epoch(boot_nonce: Any, flip_index: Any) -> str:
    """The credit counter's epoch: ``<boot>.<flip>``, a token, never a number.

    FIX 2 round 2, finding 1 -- A CROSS-BOOT REGRESSION, and the whole reason
    this function exists rather than the bare flip counter.  FIX 1 dated the
    counter with ``front.Front.self.epoch``, which is initialised to 0 at every
    front start and incremented once per flip.  The counter FILE outlives the
    front: it lives at ``/dev/shm/.weg2-vram-credit-<uuid>.json`` and nothing
    unlinked it, so boot A that ran K flips left epoch K-1 on disk with
    ``leg_complete: true`` and a whole image of credit -- and boot B's flip K-1
    read it as ITS OWN counter, because the two integers are equal.  Spec
    section 9.1 demands >= 4 flips per direction, so every boot walks straight
    through the range its predecessor left behind.  Measured on this rig
    2026-09-08: three terminal files from boot weg2rg2, one carrying
    ``{"epoch": 12, "credit_bytes": 14589886464, "leg_complete": true}``.  That
    is C14's failure verbatim (spec section 10.5): W is told the peer funded
    bytes nobody released, ``resume(tag)`` maps and H2Ds them, CUDA OOM,
    rank-local.  ``int(time.time())`` -- what FIX 1 replaced -- never collided
    across boots; a per-front counter always will.

    THE EPOCH MUST NAME THE FLIP AND THE BOOT, so it is composed of both.  The
    boot half is the launcher's own ring epoch (``HostRingPlan.epoch``, already
    published to every rank as ``TMS_HOST_RING_EPOCH``): one boot nonce for the
    boot, not a second one invented here.  The flip half is the front's
    counter.  Composed as a STRING and compared as a string -- no field width to
    overflow, no arithmetic to get wrong, and an old boot's bare integer
    ``12`` can never equal ``"1757308800.12"``.
    """
    return f"{boot_nonce}.{flip_index}"


def vram_credit_path(nvml_uuid: str, *, credit_dir: Optional[str] = None) -> str:
    """Path of the per-card credit counter.  Keyed exactly like the PCIe lock.

    NOT keyed by boot: the epoch inside the file is (see :func:`credit_epoch`),
    and the launcher's teardown unlinks the file with the ring files it already
    removes.  Both halves matter -- teardown alone leaves a CRASHED boot's file
    behind, and that is the case the epoch closes.
    """
    directory = credit_dir or os.environ.get(
        VRAM_CREDIT_DIR_ENV, os.environ.get(PCIE_LOCK_DIR_ENV, DEFAULT_PCIE_LOCK_DIR)
    )
    return os.path.join(directory, f".{VRAM_CREDIT_PREFIX}-{_sanitize(nvml_uuid)}.json")


class VramCredit:
    """The device-side mirror of the host ring's bitmap, per physical GPU.

    WHY IT EXISTS.  C9 puts both flip legs in flight at once, so on a card with
    two co-located ranks the waking rank may need device bytes that only the
    sleeping rank's release frees.  The host side of that corridor is funded by
    the ring's own bitmap (spec R11).  The device side has no such structure:
    without this counter the waking rank either wins the race or dies of a CUDA
    OOM, which is a rank-local silent failure (spec section 10.5).

    THE PREDICATE IS THE PEER'S OWN LEG, NOT A CLOCK (spec section 10.9, no new
    timeout constant).  S publishes what it released, tag by tag, and marks
    ``leg_complete`` when its whole leg is done; W waits until the credit covers
    what it needs, and the moment S's leg is complete WITHOUT that cover the
    answer is known and W refuses by name.  The absolute bound is the caller's
    existing budget -- the group fence's -- passed in, never a constant of this
    module.

    MONOTONE WITHIN A LEG: :meth:`publish` only ever adds, so a concurrent
    reader can never see the counter go backwards mid-leg.  :meth:`begin_leg`
    resets it and stamps a new epoch, which is the ONLY non-monotone step and is
    taken by the publisher before it releases anything.

    Second bookkeeping?  No: these bytes are not recorded anywhere else.  The
    NVML free figure is the card's, not the peer's intent, and it cannot say
    "the peer has finished and will free no more" -- which is the whole
    terminating predicate.
    """

    def __init__(self, nvml_uuid: str, *, credit_dir: Optional[str] = None) -> None:
        self.uuid = nvml_uuid
        self.path = vram_credit_path(nvml_uuid, credit_dir=credit_dir)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # -- state file, always under flock on ONE open fd -----------------------

    @contextmanager
    def _locked(self, exclusive: bool) -> Iterator[Any]:
        handle = open(self.path, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()

    @staticmethod
    def _load(handle: Any) -> Dict[str, Any]:
        handle.seek(0)
        text = handle.read()
        if not text.strip():
            return {}
        try:
            state = json.loads(text)
        except ValueError:
            # A torn or foreign file is treated as NO CREDIT, never as credit:
            # the failure direction of an unreadable counter must be "wait and
            # then refuse", not "proceed as if funded".
            return {}
        return state if isinstance(state, dict) else {}

    @staticmethod
    def _store(handle: Any, state: Dict[str, Any]) -> None:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(state))
        handle.flush()
        os.fsync(handle.fileno())

    # -- the sleeping rank's side -------------------------------------------

    def begin_leg(self, epoch: Any, *, pid: Optional[int] = None) -> None:
        """S opens a leg: the counter is zeroed and stamped.

        The stamp is stored as a TOKEN (:func:`credit_epoch`), so a previous
        BOOT's file -- whose stamp is that boot's flip counter -- can never
        compare equal to this boot's.
        """
        with self._locked(True) as handle:
            self._store(
                handle,
                {
                    "epoch": str(epoch),
                    "credit_bytes": 0,
                    "leg_complete": False,
                    "publisher_pid": int(pid if pid is not None else os.getpid()),
                    "tags": [],
                },
            )

    def publish(self, tag: str, released_bytes: int) -> int:
        """S adds the device bytes one tag's release freed.  Returns the total."""
        with self._locked(True) as handle:
            state = self._load(handle)
            total = int(state.get("credit_bytes", 0)) + max(0, int(released_bytes))
            state["credit_bytes"] = total
            tags = list(state.get("tags", []))
            tags.append(str(tag))
            state["tags"] = tags
            self._store(handle, state)
            return total

    def leg_complete(self) -> None:
        """S closes its leg.  This is W's terminating predicate."""
        with self._locked(True) as handle:
            state = self._load(handle)
            state["leg_complete"] = True
            self._store(handle, state)

    # -- the waking rank's side ---------------------------------------------

    def read(self) -> Dict[str, Any]:
        with self._locked(False) as handle:
            return self._load(handle)

    def wait_for(
        self,
        need_bytes: int,
        *,
        budget_s: float,
        tag: str,
        free_bytes_now: Optional[int] = None,
        poll_s: float = 0.01,
        epoch: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """W waits for the peer to fund ``need_bytes``, or refuses by name.

        ``epoch`` IS LOAD-BEARING (FIX 1 round 1, finding 4).  The per-card file
        is shared by the two co-located ranks and alternates writers across
        flips, and C9 issues both legs in ONE ``asyncio.gather``, so there is no
        happens-before between S's :meth:`begin_leg` and W's first call here: at
        flip N+1 the state on disk is typically flip N's TERMINAL state --
        ``leg_complete`` with a whole image of credit.  A state whose epoch is
        not this flip's is therefore NO CREDIT and NO TERMINATING PREDICATE: it
        is waited past, exactly as an empty counter is, so a stale full credit
        cannot license a resume of bytes nobody freed and a stale complete flag
        cannot refuse a flip nothing is wrong with.  ``None`` accepts any epoch
        and is for callers that own the ordering themselves (the tests).

        FIX 2 round 2: the epoch is a TOKEN naming the BOOT and the flip
        (:func:`credit_epoch`), because the file outlives the boot and the flip
        counter restarts at 0 with every front -- see that function for the
        leftovers this rig was carrying.

        Returns a record (never raises) in the two cases where nothing is owed:

        * the card ALREADY has the bytes -- ``free_bytes_now >= need_bytes``.
          This is what makes a single-group boot cost nothing and wait for
          nobody: with no co-located peer the card is not short, so there is no
          credit to wait for and none is invented.  It is also the honest test:
          the credit exists to cover a SHORTFALL, and where there is none the
          peer's leg is irrelevant.
        * the peer funded it before the first poll.

        Raises :class:`Weg2VramCreditRefused` when the peer's leg completes
        without cover, and when ``budget_s`` expires -- a bounded wait whose
        expiry is a refusal, never a longer wait.
        """
        need = max(0, int(need_bytes))
        if need == 0:
            return {"waited_s": 0.0, "reason": "this tag needs no device bytes"}
        if free_bytes_now is not None and int(free_bytes_now) >= need:
            return {
                "waited_s": 0.0,
                "free_bytes": int(free_bytes_now),
                "reason": (
                    "the card already holds the bytes, so no peer release funds "
                    "this tag and none is waited for"
                ),
            }
        deadline = time.monotonic() + float(budget_s)
        t0 = time.perf_counter()
        stale_epoch = None
        while True:
            state = self.read()
            if epoch is not None and state and str(state.get("epoch")) != str(epoch):
                # Not this flip's counter -- and "this flip" means THIS BOOT's
                # flip: the comparison is on the composed token of
                # :func:`credit_epoch`, so a previous boot's leftover file is
                # waited past for the same reason and by the same line as a
                # previous flip's.  Neither its bytes nor its leg_complete flag
                # say anything about the leg now in flight.
                stale_epoch = state.get("epoch")
                state = {}
            credit = int(state.get("credit_bytes", 0))
            if credit >= need:
                return {
                    "waited_s": time.perf_counter() - t0,
                    "credit_bytes": credit,
                    "reason": "the peer's releases funded this tag",
                }
            if bool(state.get("leg_complete")):
                raise Weg2VramCreditRefused(
                    f"W35 Weg2VramCreditRefused (spec section 6 lists it as W30) "
                    f"card={self.uuid} tag={tag} credit={credit // MIB} MiB "
                    f"requested={need // MIB} MiB peer_leg_complete=True "
                    f"free_bytes_now={free_bytes_now} -- the sleeping rank has "
                    f"finished its whole leg and will release nothing further, so "
                    f"these bytes are never coming; refusing by name rather than "
                    f"waiting out the budget or walking into a CUDA OOM"
                )
            if time.monotonic() >= deadline:
                raise Weg2VramCreditRefused(
                    f"W35 Weg2VramCreditRefused (spec section 6 lists it as W30) "
                    f"card={self.uuid} tag={tag} credit={credit // MIB} MiB "
                    f"requested={need // MIB} MiB peer_leg_complete=False "
                    f"budget={budget_s:.0f}s EXPIRED -- the peer neither funded "
                    f"this tag nor completed its leg within the caller's own "
                    f"budget (this module owns no timeout constant of its own)"
                    + (
                        f"; the only state on the counter was epoch {stale_epoch}, "
                        f"not this flip's {epoch} -- a PREVIOUS leg's, ignored "
                        "rather than read as funding"
                        if stale_epoch is not None else ""
                    )
                )
            time.sleep(poll_s)


def vram_credit(card: Optional[str] = None, *, credit_dir: Optional[str] = None) -> VramCredit:
    """The credit counter of ``card``, or of THIS process's card.

    The key is :func:`resolve_pcie_lock_key`'s, so the credit and the PCIe lock
    name the same physical GPU by construction and neither can drift onto a
    different card than the other.
    """
    return VramCredit(card or resolve_pcie_lock_key(), credit_dir=credit_dir)


# ---------------------------------------------------------------------------
# Sleep-acceptance census (design (S) 2.4 step 11)
# ---------------------------------------------------------------------------


@dataclass
class SleepAcceptanceCensus:
    """What this process still holds on this card, right after a sleep.

    Every field names its instrument.  ``accepted`` is False whenever the
    instrument could not read -- a census that cannot measure must not report
    a pass, or it becomes the ``cutover_participants.py`` failure mode with a
    number attached.
    """

    pid: int
    nvml_uuid: Optional[str]
    #: The tag set the graded release declared, as the caller resolved it.
    #: ``None`` means the caller declared none, i.e. the population is this
    #: process's whole device residency.  Printed on the line: a verdict that
    #: cannot be attributed to the request that produced it is not one.
    tags: Optional[Tuple[str, ...]]
    #: NVML ``nvmlDeviceGetComputeRunningProcesses`` for THIS pid, in bytes.
    proc_used_bytes: Optional[int]
    nvml_free_bytes: Optional[int]
    nvml_total_bytes: Optional[int]
    #: The caller's pre-pause reading of the SAME instrument, and the two
    #: criteria it may grade against.  ``None`` everywhere means the caller
    #: supplied no criterion, and then ``accepted`` is False by construction:
    #: a reading without a criterion is a number, not a verdict.
    before_bytes: Optional[int]
    released_bytes: Optional[int]
    min_released_fraction: Optional[float]
    #: False when ``min_released_fraction`` was supplied but the declared tag
    #: set is a PROPER SUBSET of :data:`WEG2_SLEEP_TAGS`, so the whole-process
    #: floor is not the criterion in force.  The line then prints the supplied
    #: value together with the reason it does not grade, never a bare number
    #: that reads as a criterion that ran.
    delta_form_in_force: bool
    expected_max_resident_bytes: Optional[int]
    #: Sums over ``kv_vmm_backing.arena_census()`` rows for this process.
    arena_reserved_bytes: int
    arena_backed_bytes: int
    arena_retained_bytes: int
    arena_rows: int
    #: Tri-state.  True/False only when a live ``KvVmmArena`` was actually
    #: censused: False means some arena still owns unmapped handles, and NVML
    #: charges the process for that ADDRESS SPACE, so the freed-memory reading
    #: is not one.  ``None`` means there was NO live arena to read -- the
    #: permanent state under Weg 2, where ``--enable-vram-dial`` is refused
    #: (W13) and the phase flip is deleted (S0/S7).  Reporting that case as
    #: True would be an assertion the instrument never took.
    retain_handles_asserted: Optional[bool]
    accepted: bool
    refusal_reason: Optional[str]
    denominator: str

    @property
    def proc_used_mib(self) -> Optional[float]:
        if self.proc_used_bytes is None:
            return None
        return self.proc_used_bytes / MIB

    def format_line(self) -> str:
        used = "n/a" if self.proc_used_bytes is None else f"{self.proc_used_mib:.1f}"
        free = (
            "n/a"
            if self.nvml_free_bytes is None
            else f"{self.nvml_free_bytes / MIB:.1f}"
        )
        if self.arena_rows == 0:
            # No live KvVmmArena in this process: print n/a, never a 0.0 MiB
            # that reads like a measurement, and never `retain_handles=True`,
            # which would be an assertion nothing was asserted against.
            arena = (
                "arena_backed=n/a arena_retained=n/a retain_handles=n/a "
                "(no live KvVmmArena in this process)"
            )
        else:
            arena = (
                f"arena_backed={self.arena_backed_bytes / MIB:.1f} MiB "
                f"arena_retained={self.arena_retained_bytes / MIB:.1f} MiB "
                f"retain_handles={self.retain_handles_asserted}"
            )
        before = (
            "n/a" if self.before_bytes is None else f"{self.before_bytes / MIB:.1f}"
        )
        released = (
            "n/a" if self.released_bytes is None else f"{self.released_bytes / MIB:.1f}"
        )
        ceiling = (
            "n/a"
            if self.expected_max_resident_bytes is None
            else f"{self.expected_max_resident_bytes / MIB:.1f}"
        )
        if self.min_released_fraction is None:
            floor = "n/a"
        elif self.delta_form_in_force:
            floor = f"{self.min_released_fraction:.2f}"
        else:
            floor = (
                f"n/a (partial tag set: {list(self.tags or ())}; the floor is a "
                f"fraction of this process's whole device residency)"
            )
        tags = (
            "n/a (whole-process residency)"
            if self.tags is None
            else f"{list(self.tags)}"
        )
        return (
            f"[weg2 sleep-acceptance] uuid={self.nvml_uuid} pid={self.pid} "
            f"tags={tags} "
            f"proc_used={used} MiB nvml_free={free} MiB "
            f"proc_used_before={before} MiB released={released} MiB "
            f"min_released_fraction={floor} ceiling={ceiling} MiB "
            f"rows={self.arena_rows} {arena} accepted={self.accepted} "
            f"reason={self.refusal_reason or '-'} denominator={self.denominator}"
        )


_ProcessBytes = Union[Dict[int, int], Callable[[str], Dict[int, int]], None]


def sleep_acceptance_census(
    *,
    nvml_uuid: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    before_bytes: Optional[int] = None,
    min_released_fraction: Optional[float] = None,
    expected_max_resident_bytes: Optional[int] = None,
    _process_bytes: _ProcessBytes = None,
    _memory_info: Optional[Tuple[int, int]] = None,
    _arena_census: Optional[Dict[int, Dict[str, int]]] = None,
    _pid: Optional[int] = None,
) -> SleepAcceptanceCensus:
    """Read what this rank still holds on its card after ``pause()``, and GRADE it.

    Read-only and never raises: an instrument that can fail a boot is not an
    instrument (same contract as ``arena_census()``).  Every failure to read
    lands in ``refusal_reason`` and forces ``accepted=False``.

    The NVML half carries the verdict, and therefore it needs a criterion.  Two
    are accepted and both, if supplied, must hold:

    * the DELTA form -- ``before_bytes`` (the caller's pre-pause reading of this
      same instrument) plus ``min_released_fraction``.  This is the S1 form:
      the dormant floor D_c is unmeasured until S3, so there is no honest
      ceiling yet, but "the sleep released nothing" is decidable without one.
    * the CEILING form -- ``expected_max_resident_bytes``, the declared dormant
      ceiling.  S3's launcher supplies it once D_c is measured.

    ``tags`` is the POPULATION the verdict is about, and it is printed.  The
    delta floor is a fraction of this process's WHOLE device residency, so it
    may only grade a release that targets the whole of it
    (:data:`WEG2_SLEEP_TAGS`: weights AND kv_cache, the latter carrying the
    mamba/GDN anchors under the same tag).  A declared PROPER SUBSET -- the #89
    park's ``tags=["weights"]``, or a kv-only release -- suppresses the delta
    form, because both errors are reachable on this one endpoint: a weights-only
    park need not clear half of an awake residency that also carries KV, graphs,
    activations and the CUDA context (false FAIL), and a kv-only release on a
    rank whose KV exceeds half its residency clears the floor with the entire
    weights shard still resident (false PASS).  With the delta form suppressed
    the verdict rests on the ceiling form, or -- with no ceiling either -- on an
    explicit refusal.  ``tags=None`` means the caller declared no restriction,
    and then the whole-process denominator is the right one.

    Supplying NEITHER is itself refused: with no criterion the function would
    report ``accepted=True`` for a rank still holding its entire shard, i.e.
    for the exact silent-no-op condition it exists to catch, and a boot
    postmortem would quote that pass.  A gate that cannot fail is not a gate.

    The payload it reads -- NVML per-process bytes, card free/total, and the KV
    arena's own counters -- has a canonical holder in this tree:
    ``srt/mem_ledger/flight_recorder`` (``_nvml_view`` / ``_kv_arena_view``).
    This function is a VERDICT WRAPPER over that same payload, not a second
    collector: it shares the recorder's card-pin guard (see
    :func:`_resolve_uuid`) and adds only ``accepted`` / ``refusal_reason`` /
    ``denominator``.  A boot postmortem grades residency off the recorder's
    field names; the ``[weg2 sleep-acceptance]`` line is the flip path's
    verdict, not a second set of numbers to reconcile.

    The leading-underscore parameters are injection seams for the hermetic
    tests; production calls pass none of them.
    """
    pid = os.getpid() if _pid is None else int(_pid)
    reasons = []
    declared_tags: Optional[Tuple[str, ...]] = (
        None if tags is None else tuple(str(tag) for tag in tags)
    )
    partial_tag_set = declared_tags is not None and not WEG2_SLEEP_TAGS.issubset(
        set(declared_tags)
    )

    uuid = nvml_uuid
    if uuid is None:
        try:
            uuid = _resolve_uuid(None)
        except Exception as exc:  # pragma: no cover - depends on the rig
            uuid = None
            reasons.append(f"device uuid unresolved ({exc})")

    proc_used: Optional[int] = None
    proc_count = 0
    if uuid is not None:
        try:
            table = _process_bytes
            if table is None:
                from sglang.srt.registry import nvml as nvml_registry

                table = nvml_registry.process_bytes_on_uuid(uuid)
            elif callable(table):
                table = table(uuid)
            proc_count = len(table)
            if pid in table:
                proc_used = int(table[pid])
            else:
                reasons.append(f"pid {pid} not among NVML's {proc_count} process(es)")
        except Exception as exc:
            reasons.append(f"per-process read failed ({exc})")

    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    if uuid is not None:
        try:
            info = _memory_info
            if info is None:
                from sglang.srt.registry import nvml as nvml_registry

                mem = nvml_registry.memory_info_for_uuid(uuid)
                info = (mem.total_bytes, mem.free_bytes)
            total_bytes, free_bytes = int(info[0]), int(info[1])
        except Exception as exc:
            reasons.append(f"card memory read failed ({exc})")

    rows = _arena_census
    if rows is None:
        try:
            from sglang.srt.mem_cache.kv_vmm_backing import arena_census

            rows = arena_census()
        except Exception as exc:
            rows = {}
            reasons.append(f"arena census unavailable ({exc})")

    reserved = backed = retained = 0
    for row in (rows or {}).values():
        reserved += int(row.get("reserved", 0))
        backed += int(row.get("backed", 0))
        retained += int(row.get("retained", 0))

    arena_rows = len(rows or {})
    # Tri-state, never a fabricated pass: with zero live arenas there is
    # nothing to assert against, so `retain_handles` is n/a and `accepted` is
    # decided by the NVML half ALONE.  Weg 2 makes the zero-row case permanent
    # (W13 refuses --enable-vram-dial, S0/S7 delete the phase flip), so a
    # `retain_handles_asserted=True` printed over an empty _LIVE_ARENAS would
    # be quoted in every boot postmortem as a check that never ran.
    retain_handles_asserted: Optional[bool] = None if arena_rows == 0 else retained == 0
    if retain_handles_asserted is False:
        reasons.append(
            f"arena retain_handles is in force: {retained / MIB:.1f} MiB of "
            f"unmapped-but-owned ADDRESS SPACE stays charged to this process "
            f"by NVML, so this reading is address space, not free memory"
        )
    if backed:
        reasons.append(f"arena still backs {backed / MIB:.1f} MiB of device memory")

    # --- the NVML half's criterion.  Without one this whole function is a
    # printer: it reported accepted=True at 30,154 MiB and accepted=True at
    # 1,294 MiB -- the two ends of campaign (a)'s own 28.9 GiB swing -- and the
    # number in its test was decorative.
    delta_supplied = before_bytes is not None and min_released_fraction is not None
    # The floor's denominator is the WHOLE process residency, so a release that
    # declared a proper subset of the sleep tags is not gradeable by it -- in
    # either direction (see the docstring).  Suppressed, never silently applied.
    delta_form = delta_supplied and not partial_tag_set
    ceiling_form = expected_max_resident_bytes is not None
    released_bytes: Optional[int] = None
    if not delta_form and not ceiling_form:
        if delta_supplied and partial_tag_set:
            reasons.append(
                "no residency criterion in force: this release declared the "
                f"partial tag set {list(declared_tags or ())}, and the delta "
                "floor is a fraction of this process's WHOLE device residency, "
                "which a release of only those tags need neither clear nor be "
                "graded by -- this reading is a number, not a verdict"
            )
        else:
            reasons.append(
                "no residency criterion supplied (neither before_bytes + "
                "min_released_fraction nor expected_max_resident_bytes) -- this "
                "reading is a number, not a verdict"
            )
    elif proc_used is not None:
        if delta_form:
            released_bytes = int(before_bytes) - proc_used
            floor = int(float(min_released_fraction) * int(before_bytes))
            if released_bytes < floor:
                reasons.append(
                    f"the sleep released {released_bytes / MIB:.1f} MiB of the "
                    f"{int(before_bytes) / MIB:.1f} MiB this process held "
                    f"before the pause, below the required "
                    f"{float(min_released_fraction):.0%} "
                    f"({floor / MIB:.1f} MiB) -- a sleep that frees nothing "
                    f"returns success and holds the whole shard"
                )
        if ceiling_form and proc_used > int(expected_max_resident_bytes):
            reasons.append(
                f"this process still holds {proc_used / MIB:.1f} MiB, above "
                f"the declared dormant ceiling "
                f"{int(expected_max_resident_bytes) / MIB:.1f} MiB"
            )

    accepted = proc_used is not None and not reasons
    criterion_denominator = (
        (
            "delta form SUPPRESSED (partial tag set "
            f"{list(declared_tags or ())}), no other criterion"
            if delta_supplied and partial_tag_set
            else "no criterion"
        )
        if not delta_form and not ceiling_form
        else ", ".join(
            part
            for part in (
                (
                    f"delta form (>= {float(min_released_fraction):.0%} of "
                    f"{int(before_bytes) / MIB:.1f} MiB released)"
                    if delta_form
                    else ""
                ),
                (
                    f"ceiling form (<= "
                    f"{int(expected_max_resident_bytes) / MIB:.1f} MiB)"
                    if ceiling_form
                    else ""
                ),
            )
            if part
        )
    )
    arena_denominator = (
        "no live KvVmmArena, so the arena half is n/a and acceptance rests on "
        "the NVML half alone"
        if arena_rows == 0
        else f"{arena_rows} live arena row(s)"
    )
    tag_denominator = (
        "no tag set declared, so the population is this process's whole device "
        "residency"
        if declared_tags is None
        else f"declared tags {list(declared_tags)}"
    )
    denominator = (
        f"NVML per-process bytes for pid={pid} on uuid={uuid} over "
        f"{proc_count} compute process(es); {tag_denominator}; "
        f"arena_rows={arena_rows} "
        f"({arena_denominator}); criterion: {criterion_denominator}"
    )
    return SleepAcceptanceCensus(
        pid=pid,
        nvml_uuid=uuid,
        tags=declared_tags,
        proc_used_bytes=proc_used,
        nvml_free_bytes=free_bytes,
        nvml_total_bytes=total_bytes,
        before_bytes=None if before_bytes is None else int(before_bytes),
        released_bytes=released_bytes,
        min_released_fraction=(
            None if min_released_fraction is None else float(min_released_fraction)
        ),
        delta_form_in_force=delta_form,
        expected_max_resident_bytes=(
            None
            if expected_max_resident_bytes is None
            else int(expected_max_resident_bytes)
        ),
        arena_reserved_bytes=reserved,
        arena_backed_bytes=backed,
        arena_retained_bytes=retained,
        arena_rows=arena_rows,
        retain_handles_asserted=retain_handles_asserted,
        accepted=accepted,
        refusal_reason="; ".join(reasons) if reasons else None,
        denominator=denominator,
    )


#: The one string a dormant refusal carries, so a log grep for the marker and
#: the client-visible error name the same event.
# ---------------------------------------------------------------------------
# #1233 ONE-BACKUP FLIP: chunked weight tags (record section 1h, 2026-09-07)
# ---------------------------------------------------------------------------
#
# Upstream pauses/resumes the weights as ONE tag, so at a flip the sleeping
# group's whole cpu backup lands next to the waking group's whole backup and
# the host holds TWO images (boot weg2ls1b2: 28.8 + 27.15 GiB -> host OOM).
# The user's design statement is "only ONE layout lies in host RAM": the
# weights region is tagged per group of layers at construction, so the front
# can interleave D.pause(weights_k) with P.resume(weights_k) and the host
# never holds more than one image plus one chunk (the torch_memory_saver
# resume frees the chunk's host image, PATCH.md).
#
# The tag is the CURRENT torch_memory_saver tag at cudaMalloc time, so the
# only mechanism here is `tms_set_current_tag` inside the already-open
# weights region -- no second allocator, no pool, no bookkeeping of what
# landed where (torch_memory_saver's own metadata map is the ledger).  The
# tag family is `weights_<k>` for the layer chunks plus the base
# GPU_MEMORY_TYPE_WEIGHTS for everything outside a layer (embeddings, head,
# norms, the NEXTN draft, rotary caches); release/resume treat the whole
# family as "the weights" (weight_updater.py).  Two envs, both set by the
# launcher from the checkpoint's num_hidden_layers, both unset = the stock
# single tag:
#
#   SGLANG_WEG2_WEIGHT_CHUNK_LAYERS   layers per chunk (L)
#   SGLANG_WEG2_WEIGHT_CHUNKS         number of chunk tags (N); a layer id
#                                     beyond N*L (the NEXTN draft layer)
#                                     clamps to the last chunk
#
# Granularity caveat, stated so nobody measures it as a defect: the caching
# allocator packs allocations < 10 MiB into shared 20 MiB segments and reuses
# freed blocks across layers, and torch_memory_saver tags at SEGMENT
# granularity, so a few MiB per chunk carry a neighbour's tag.  Every tag of
# the family is paused before the group computes again and resumed before
# it computes again, so this only blurs the per-chunk byte count, never the
# content.  Chunk bytes are therefore MEASURED (RssShmem delta per chunk),
# never derived.

WEIGHT_CHUNK_ENV_LAYERS = "SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"
WEIGHT_CHUNK_ENV_COUNT = "SGLANG_WEG2_WEIGHT_CHUNKS"
WEIGHT_CHUNK_PREFIX = GPU_MEMORY_TYPE_WEIGHTS + "_"
_LAYER_ID_IN_NAME = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def weight_chunk_geometry() -> Tuple[int, int]:
    """(layers_per_chunk, chunk_count) from the env; (0, 0) = chunking OFF."""
    try:
        layers = int(os.environ.get(WEIGHT_CHUNK_ENV_LAYERS, "0") or 0)
        count = int(os.environ.get(WEIGHT_CHUNK_ENV_COUNT, "0") or 0)
    except ValueError:
        return 0, 0
    if layers <= 0 or count <= 0:
        return 0, 0
    return layers, count


def weight_chunk_tag(layer_id: int) -> Optional[str]:
    """The chunk tag of a layer, or None when chunking is off."""
    layers, count = weight_chunk_geometry()
    if layers <= 0:
        return None
    return f"{WEIGHT_CHUNK_PREFIX}{min(int(layer_id) // layers, count - 1)}"


def weights_family_tags(chunk_count: Optional[int] = None) -> list:
    """Every tag the sleep/wake path treats as 'the weights', chunks FIRST and
    the base tag LAST -- the order the front pauses them in, so the base tag
    (the remainder: embeddings, head, draft, buffers) closes the sleep."""
    if chunk_count is None:
        chunk_count = weight_chunk_geometry()[1]
    return [f"{WEIGHT_CHUNK_PREFIX}{k}" for k in range(int(chunk_count))] + [
        GPU_MEMORY_TYPE_WEIGHTS
    ]


def is_weights_family_tag(tag: Any) -> bool:
    return isinstance(tag, str) and (
        tag == GPU_MEMORY_TYPE_WEIGHTS or tag.startswith(WEIGHT_CHUNK_PREFIX)
    )


def layer_id_from_module_name(name: str) -> Optional[int]:
    m = _LAYER_ID_IN_NAME.search(name or "")
    return int(m.group(1)) if m else None


def _tms_cdll_in_region():
    """torch_memory_saver's C entry points, ONLY while a region is open on
    this thread; None otherwise (no saver, not initialised, or outside a
    region -- an allocation there is not tracked, so a tag is meaningless)."""
    try:
        import torch_memory_saver as _tms  # noqa: WPS433

        impl = _tms.torch_memory_saver._impl
    except Exception:  # noqa: BLE001 -- not installed / not the real lib
        return None
    if impl is None:
        return None
    cdll = impl._binary_wrapper.cdll
    if not cdll.tms_get_interesting_region():
        return None
    return cdll


@contextmanager
def weight_chunk_scope(layer_id: Optional[int]) -> Iterator[Optional[str]]:
    """Tag every allocation inside as the chunk of ``layer_id``.

    Valid ONLY inside the weights region (model construction and the
    post-load pass, both under model_runner's region(GPU_MEMORY_TYPE_WEIGHTS)):
    on exit the current tag is restored to the base weights tag, which is
    what the region set.  No-op when chunking is off, when no region is open,
    or when ``layer_id`` is None (a module outside any layer).
    """
    tag = None if layer_id is None else weight_chunk_tag(layer_id)
    cdll = None if tag is None else _tms_cdll_in_region()
    if cdll is None:
        yield None
        return
    cdll.tms_set_current_tag(tag.encode("utf-8"))
    try:
        yield tag
    finally:
        cdll.tms_set_current_tag(GPU_MEMORY_TYPE_WEIGHTS.encode("utf-8"))


DORMANT_REFUSAL_MARKER = "W25 Weg2DormantRefused"


def dormant_refusal_message(*, rid: str, context: str) -> str:
    """The abort text for a request admitted to a sleeping group.

    Pure: no device access, no scheduler state.  ``context`` names the seam
    that refused (generate / embedding) so a postmortem can tell them apart.
    """
    return (
        f"{DORMANT_REFUSAL_MARKER}: this group is ASLEEP (kv_cache paused via "
        f"release_memory_occupation) and refuses {context} request {rid!r} at "
        "the admission seam. A sleeping group has no KV/mamba/req-index pool; "
        "admitting the request would fault in write_req_to_token_pool_triton "
        "on released VMM pages and kill the group (S1 boot killer K2). Route "
        "to the awake group via the Weg-2 front on :30030, or wake this group "
        "first (resume_memory_occupation)."
    )
