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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)
from sglang.srt.environ import envs

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
#: NUTZER-ORDER 14.09. ("man sollte diesen timeout deutlich verringern -- der
#: timeout hilft nicht beim problem, sondern laesst nur mehr zeit verstreichen,
#: bis man den fail erkennt"): the 600 s raise (this session's first attempt)
#: was the WRONG dimension. The per-copy lock (d491500c71) made the holds
#: 0.045-0.123 s; a lock wait that exceeds a short budget means a stuck
#: holder, and failing FAST with the named Weg2PcieLockTimeout is worth more
#: than hanging. 120 s stays.
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

#: Item `dormant`: the smallest allocation that may be routed into the graph
#: tag's private pool.  NOT a tuning knob -- it is #102's ``MIN_TAGGED_BYTES``
#: (``adaptive_graph_memory.py:269``) and it is a CORRECTNESS gate with a
#: measured reason: the caching allocator splits large blocks, so a tagged
#: segment can carry a free tail of up to ~2 MiB, a later sub-2MiB allocation
#: of another tag can be served from that tail, and once the first tag is
#: paused the tail is UNMAPPED -- first touch is then an illegal memory access
#: (observed live on the 5-state high-accept boot: state k2's ~1.5 MiB
#: custom_mask in paused k5's segment tail).  Allocations at or above this size
#: always get their own segment.  Same number, same reason, not a second one.
WEG2_GRAPH_SCRATCH_MIN_BYTES = 2 * 1024 * 1024

#: Item `dormant` FIX 2, finding 1: the name of THIS rank's Weg-2 group, or the
#: empty string on every engine that is not a Weg-2 group at all.  Set by
#: ``weg2/launcher.build_env`` for both groups and by nothing else, which is
#: exactly why it can be the discriminator: ``--enable-memory-saver`` cannot,
#: because upstream engines set it too.
#:
#: There is NO server-args equivalent to read instead.  ``weg2_group`` is read
#: once (``weight_updater._weg2_group_name``) and ASSIGNED NOWHERE in either
#: tree -- `grep -rn 'weg2_group' --include=*.py .` returns 18 lines, all of
#: them reads or unrelated ``_weg2_group_*`` method names -- so that getattr
#: has always fallen through to its ``"?"`` default on every rank of every
#: boot.  This env is the fact that was believed to exist there.
WEG2_GROUP_ENV = "SGLANG_WEG2_GROUP"

#: Item `dormant`: whether this rank's sleep also releases the CUDA-graph tag.
#: Resolved ONCE per process, deliberately: the sleep adds the tag to
#: ``offload_tags`` and the wake removes it, so a resolver that could change
#: answer between the two legs would raise ``KeyError`` on a tag that was never
#: paused.  A cached read cannot drift.
_GRAPH_TAG_ARMED: Optional[bool] = None

#: Item `dormant` FIX 2: the cached :data:`WEG2_GROUP_ENV` reading, same reason
#: as above -- both legs of a flip must get the same answer.
_WEG2_GROUP_NAME: Optional[str] = None

#: Item `dormant`: ONE private ``torch.cuda.MemPool`` for the graph tag's
#: non-capture scratch, created on first use and kept for the process lifetime
#: (#102's rule: a tag's free space must only ever be visible to allocations of
#: that same tag).
_GRAPH_SCRATCH_POOL: Any = None

#: Item `dormant` FIX 3, finding 2: how often each conjunct of
#: :func:`weg2_graph_tag_armed` refused, by reason.  The line is rate-limited
#: (see :func:`_refuse_graph_tag`), so this is the DENOMINATOR the log's
#: ``occurrence=`` field is drawn from -- a reader must never take "one line"
#: for "one refusal".
_GRAPH_TAG_REFUSALS: Dict[str, int] = {}


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


class Weg2XchgWakeSourceGapRefused(RuntimeError):
    """W106 -- a tag about to lose its host-ring backup has no OTHER wake
    source, checked at the SLEEP leg, before the pause that would make it
    true.

    #1369/#1394 (DESK10, Paket B).  User order 2026-09-14: the checkpoint on
    disk is the fallback, and a correctly-implemented exchange needs no
    other one -- the host ring is redundant to a net that is already there,
    and this refusal is the reason removing it is safe: it is the thing
    that makes "still wrong weights" impossible in the case a plain removal
    would have made cheap and silent.

    TWO INDEPENDENT WAYS THIS TAG CAN BE LEFT WITH NOTHING, both checked,
    either one enough to refuse:

    1. **The exchange is armed but not AUTHORITATIVE.**
       ``--weg2-xchg-inject shadow`` only COMPARES the exchange's assembled
       bytes against what already landed -- it never writes them (#1391's
       own lesson, generalised past that ticket). A tag whose ring backup
       is off under ``weights_cpu_backup_armed()`` because
       ``exchange_armed()`` is True, while ``inject_authoritative()`` is
       False, has NO writer at all: not the ring (off), not the exchange
       (shadow-only, an observer). This fires regardless of how complete
       the plan is -- a perfect plan under ``shadow`` still writes nothing.
    2. **The exchange is authoritative but THIS rank's own plan is empty
       for this tag.** Even with a real writer armed, if
       ``_weg2_shadow_plan``'s source-side descriptors for this exact tag
       are empty, nobody deposits these bytes for the peer to collect --
       #1394's own shape (``weights_draft``'s region is never selected by
       the single-runner plan resolution) generalised to any tag a future
       change could silently drop from the plan.

    ``weights_draft`` IS EXEMPTED from case 2 (but not from needing SOME
    source): #1394's own fix gives it a dedicated disk-reload wake path
    (:meth:`weight_updater.SchedulerWeightUpdaterManager._weg2_xchg_draft_reload_from_disk`)
    instead of the exchange, precisely because the exchange structurally
    never covers it.

    NOT a byte-count check (``Weg2XchgCoverageRefused``/W84's own
    restriction, kept here too): allocator overhang is real, so this is a
    POPULATION question -- does the tag have a plan/mechanism at all -- not
    an equality against ``tms_tag_bytes``.
    """


class Weg2XchgCoverageRefused(RuntimeError):
    """W84 -- a live tensor under an exchanged tag has no source.

    #1273 spec section 6/S2.  The weight-byte exchange fills the destination's
    weight pages from the SOURCE's VRAM, descriptor by descriptor, and restores
    every ``named_buffers()`` entry from ``_export_static_state``
    (weight_updater.py:1291 -> :1755).  A tensor that is neither -- a stray
    ``torch.Tensor`` attribute on a module, a Parameter the plan does not carry
    -- is a page the destination never receives.  It does not fault and it does
    not raise: it serves whatever the arena held, which is the worst class of
    wrongness this design can produce.

    Raised at the end of weight loading and re-checked in the wake RPC's
    preamble, BEFORE the first ``resume`` -- where an abandon costs nothing
    because neither side's VRAM has been mutated (spec section 3.6).

    NOT raised for SLACK.  "Cover every byte of the tag" is impossible:
    allocator overhang is real and measured at +0.08 to +0.58 GiB per rank, so
    the arithmetic against ``tms_tag_bytes`` is PRINTED with the slack named
    and never compared for equality.  What is asserted is the population --
    every live tensor is a plan Parameter or a registered buffer -- not the
    byte count.
    """


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


#: Task #47 Scheibe 6a (20.09.): the launcher's ``--flip-weights resident`` form.
#: Both groups keep their weights MAPPED for the whole boot and only kv_cache
#: (+ cuda_graph) flips, so no wake ever refills a weights tag -- the
#: backup-OFF lock's premise ("the wake reloads the weights from disk") does
#: not apply, and a release that names a weights-family tag is a defect, not a
#: sleep. Set by weg2/launcher.build_env, read here and in the release handler.
WEIGHTS_RESIDENT_ENV = "SGLANG_WEG2_WEIGHTS_RESIDENT"


def weights_resident_armed() -> bool:
    return os.environ.get(WEIGHTS_RESIDENT_ENV, "0").strip() == "1"


def assert_backup_off_wake_refill_is_defined(
    *, quantization: Optional[str], context: str,
    exchange_owns_wake_refill: bool = False,
) -> None:
    """W4: refuse a backup-OFF wake whose refill would re-run a repacking pass.

    ``exchange_owns_wake_refill`` (the launch arm's ONLY exemption, #1378
    Schritt B): under ``exchange`` + ``authoritative`` the wake refill of the
    exchange-owned weights is the exchange COLLECT, not
    ``update_weights_from_disk`` -- the premise of everything below does not
    hold, so the lock passes and the per-tag completeness guard (W106, sleep
    side, tag + byte count, BEFORE the pause) takes over.  The caller computes
    it through :func:`exchange_owns_wake_refill` -- ONE authority, never an
    inline env re-parse.  The DRAFT's disk-reload path does NOT get this
    exemption: that path really is ``update_weights_from_disk``, undefined on
    a quantized checkpoint, and its own call site keeps the lock unconditional.

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
    if exchange_owns_wake_refill:
        return
    if weights_resident_armed():
        # Scheibe 6a: the weights never sleep on this arm, so there is no
        # wake refill to define; the release handler refuses a weights tag.
        logger.info("WEG2-WEIGHTS-RESIDENT: backup-OFF wake lock not applicable "
                    "(%s) -- the weights family never sleeps on this boot", context)
        return
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


#: FIX 3 round 3, finding 1 -- THE VERSION IS A ROW, NEVER A PREFIX.  FIX 2 wrote
#: it as ``v2|<uuid>=...`` and that is worse than no version at all: a reader
#: that does not know the token does not drop everything, it mangles exactly the
#: FIRST card's uuid (on this rig the 5090, which carries the co-located pair and
#: the flip's critical path) and resolves the other two correctly -- so the card
#: that matters takes ONE key while the launcher prints SPLIT for it and arms.
#: That is boot weg2rg2's precondition verbatim, on the worst card, and it was a
#: REGRESSION: the un-versioned predecessor's own string resolved all three
#: cards correctly in that same reader.
#:
#: As a row (``format=v2,<uuid>=<ratio>:split,...``) every ``<k>=<v>,`` parser in
#: this family already tolerates it: an older reader takes ``format`` for a card
#: uuid it does not have, ``float("v2")`` raises, that ONE row is dropped, and
#: every real card still resolves.  A reader that DOES know the key requires the
#: row and refuses by name (W36) when it is absent or unknown -- so the degrade
#: is a named stop in the new reader and a no-op in the old one, which is what a
#: version is for.  ``format`` is safe as a key because an NVML card id is always
#: ``GPU-``/``MIG-`` prefixed and can never collide with it.
PCIE_DUPLEX_VERSION_KEY = "format"
PCIE_DUPLEX_FORMAT = "v2"


class Weg2DuplexDecisionRefused(RuntimeError):
    """W36 -- this rank cannot resolve the launcher's per-card key decision.

    Two shapes, one refusal, both of them "the launcher and the rank do not
    agree on what the key is", which is the precondition for boot weg2rg2's
    deadlock (S blocked in ``acquire`` holding the key W needs, W31 at the
    120 s budget, W29 on every rank, group-fatal W4 -- 120 s into a flip that
    has already mutated VRAM):

    * NO FORMAT ROW AT ALL -- a published string from a launcher older than
      :data:`PCIE_DUPLEX_VERSION_KEY` (including FIX 2's ``v2|`` prefix form,
      whose token is not a row and is therefore not one here either).  This
      reader will not guess which dialect it is holding;
    * an UNKNOWN FORMAT VERSION -- a newer launcher against an older tree on
      ``PYTHONPATH``, the partial-rebase case the launcher's own W34 comment
      names;
    * DECISIONS PUBLISHED BUT NONE PARSED -- a non-empty published string that
      yields no ``split``/``single`` row at all.  Falling back to R17's ratio
      gate there is precisely the silent degrade: the ratio gate answers a
      question about VALUE and would leave the x4 card on one key.

    There is no fallback and none is claimed.  The launcher's own launch check
    resolves the key IN THE TREE THE RANKS IMPORT -- one subprocess per boot
    with the ranks' ``PYTHONPATH`` (FIX 3 round 3, finding 2) -- so a tree that
    raises this is reported SERIALISED and refused by W34 BEFORE either group
    starts.  Reaching it mid-flip means the rank env and the launch check's env
    diverged after that check, which nothing in this tree does.
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
    """``format=v2,<uuid>=<ratio>[:split|:single],...`` -> (ratios, decisions).

    The ratio may be empty (``<uuid>=:split``): a card the step-0 probe never
    measured still needs a decision, and under gathered legs that decision is
    ``split`` -- see :func:`pcie_lock_separates_directions`.

    THE VERSION IS ONE ROW AMONG THE CARD ROWS (FIX 3 round 3, finding 1), not a
    prefix on the first card -- see :data:`PCIE_DUPLEX_VERSION_KEY` for why the
    prefix mangled exactly the first card's uuid instead of failing whole.  It is
    parsed here like any other row and never reaches ``ratios``/``splits``.

    Refuses (W36) on a MISSING version row, on an unknown version, and on a
    non-empty string that produced no decision at all.  The EMPTY string is not a
    refusal: it is the honest "nothing was published on this boot", and it
    resolves to no split, which the launcher's own check then reads as SERIALISED
    and refuses at launch (W34).
    """
    raw = os.environ.get(PCIE_DUPLEX_ENV, "") if published is None else published
    raw = (raw or "").strip()
    if not raw:
        return {}, {}
    ratios: Dict[str, float] = {}
    splits: Dict[str, bool] = {}
    version: Optional[str] = None
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        uuid, _, value = item.partition("=")
        uuid = uuid.strip()
        if uuid == PCIE_DUPLEX_VERSION_KEY:
            version = value.strip()
            continue
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
    if version is None:
        raise Weg2DuplexDecisionRefused(
            f"W36 Weg2DuplexDecisionRefused {PCIE_DUPLEX_ENV} is set "
            f"({raw[:200]!r}) but carries no {PCIE_DUPLEX_VERSION_KEY}="
            f"{PCIE_DUPLEX_FORMAT} row -- it was published by a launcher whose "
            "dialect this tree cannot name.  Reading it anyway is what mangles "
            "one card's key while the other cards resolve (FIX 2's 'v2|' prefix, "
            "on the FIRST card, which on this rig carries the flip's critical "
            "path), and one un-split key is the deadlock of boot weg2rg2 "
            "(W31 -> W29 -> W4).  No fallback: relaunch with one tree on "
            "PYTHONPATH."
        )
    if version != PCIE_DUPLEX_FORMAT:
        raise Weg2DuplexDecisionRefused(
            f"W36 Weg2DuplexDecisionRefused {PCIE_DUPLEX_ENV} carries format "
            f"{version!r}, and this tree reads {PCIE_DUPLEX_FORMAT!r} -- the "
            "launcher and this rank would build DIFFERENT PCIe keys, which is "
            "the single-key deadlock of boot weg2rg2 (W31 -> W29 -> W4).  No "
            "fallback: relaunch with one tree on PYTHONPATH."
        )
    if not splits:
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


class Weg2XchgLaneNeverDrainedRefused(RuntimeError):
    """W108 -- a bounce lane already carries undrained bands nobody will ever
    take, named THE MOMENT it is found instead of after a 120 s budget.

    RENUMBERED 2026-09-14 from W100 to W108: TRAIN2's census against the
    merged tree found W100 double-assigned -- ``host_ledger.py``'s
    ``Weg2SleepLegCushionDeficit`` (from ``8799c7f945``, #1361 fix6) held it
    FIRST. Per the "later arrival renumbers, the census keys on the
    exception NAME not the digit" rule, this class (picked tonight, #1391)
    is the later arrival and moves. W108 was verified free by manual grep
    against the merged train tip (``bb70894d76``) across
    weg2/managers/mem_ledger/planner/model_executor/scripts, the same
    2-digit wcode-census blind spot this docstring already named for W106.

    #1391 (DESK10), boot weg2xsn31 versuch 5 AND versuch 7, byte-identical:
    D's three ranks (diag on card 0, cross pairs ``1-0`` and ``2-0``) deposit
    weights_0 for P PP0 COMPLETELY and CORRECTLY -- ``sem_getvalue`` read
    ``full=16/8/8 empty=0``, and that is D's own TP-shard split (17:7:8,
    matched one poster per lane), not a double post and not a broken
    invariant.  D then tries weights_1 and blocks in ``wait_drained``
    (waiting for weights_0 to be collected) on all three lanes, for its full
    120 s, and P's own LATER ``weights_3`` credit wait blocks for its full
    120 s beside it -- two walls that read as unrelated
    (``Weg2XchgBouncePhaseUnordered``, raised as ``W68 Weg2XchgPlanDisagree``,
    on D; ``W35 Weg2VramCreditRefused`` on P) and cost two DEBUG_HOLD boots to
    trace to one fact: NOBODY EVER COLLECTED weights_0, so nobody ever posted
    ``drained``.

    THE MECHANISM, found by reading against the boot's own launch flags
    (``arm_xsn31.sh``: ``--weg2-xchg-inject shadow --weg2-seam-digest``): under
    shadow mode the only collect call is
    :meth:`weight_updater.SchedulerWeightUpdaterManager._weg2_xchg_shadow_compare`
    -> ``_weg2_xchg_inject_weights(mode=INJECT_SHADOW)``, called ONCE after
    the WHOLE weights family has resumed (``family_complete``), with NO
    ``tag`` argument -- it grades the assembled plan as one unit, which is
    correct for grading. But D's per-tag lockstep deposit (#1374) needs
    ``drained(t)`` posted PER TAG, DURING the resume, to let tag t+1's
    deposit proceed -- and posting per tag requires knowing which tag, which
    a whole-plan grading pass run once at the end structurally cannot supply
    in time. Whenever a co-located pair's traffic spans more than one
    ``weights_<k>`` chunk tag (true for P's rank 0 in this boot's PP-cut, not
    for ranks 1/2, which is why only card 0 wedges), the deposit and the
    only collect can never meet: this is an unconditional ordering deadlock
    under ``--weg2-xchg-inject shadow``, not a race and not a per-rank
    identity bug.

    THIS REFUSAL DOES NOT FIX THAT ORDERING GAP -- it only refuses to let a
    lane already showing the gap's fingerprint (real, undrained bytes beside
    a leg that has nothing to say about that lane) burn a full 120 s budget
    first. Checked in two places: from
    :meth:`weight_updater.SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg`
    the moment a COLLECT-phase leg's own lane set does not cover a lane that
    already holds undrained bands for this rank's card (regardless of
    whether ``tag`` is a string or ``None`` -- the frame both real callers,
    deposit and collect, reach every time); and from
    :class:`VramCredit`'s :meth:`~VramCredit.wait_for` poll loop itself (a
    callback, not a one-shot check at entry -- the metal measured a real
    race there: this rank entered clean and the peer's saturating post
    landed four seconds later, invisible to a predicate that only looks
    once).
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



class Weg2VramCreditAllocatableShort(Weg2VramCreditRefused):
    """W85 -- the peer funded it, the CARD cannot allocate it (#1331).

    A SUBCLASS of the credit refusal on purpose: every caller that already
    handles a refused credit handles this one unchanged, and the W-code names
    which of the two facts failed. `Weg2VramCreditRefused` (W35) means "the
    peer will never fund this"; W85 means "the peer's counter says it is
    funded and the device disagrees".

    THE CENSUS WAS RUN, not remembered: the assigned Weg2 codes around it are
    W80-W84 (the xchg band) and W86-W89 (the serve line); **W85 was the one
    free number between them**, and the two `W85` hits elsewhere in the tree
    are `W855-...` specimen DIRECTORY names, i.e. the bare-code-as-grep
    -pattern trap. `test_weg2_wcode_uniqueness_1263` is the authority.
    """

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

    TWO HALVES SINCE #1349, AND THE SECOND ONE IS WHY xsn21b DIED.  ``publish``
    is the credit half; :meth:`claim` is the DEBIT half, and until #1349 it did
    not exist -- nothing ever subtracted what a tag had already spent, so the
    same released bytes funded every tag of the leg in turn.  The available
    balance is therefore ``credit_bytes - consumed_bytes``, never
    ``credit_bytes``; see :meth:`claim` for the boot that measured it.

    Second bookkeeping?  No: these bytes are not recorded anywhere else, and the
    debit lives in THIS counter rather than beside it for the same reason.  The
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
                    # #1349: the debit half is reset by the SAME step that
                    # resets the credit half, so a leg can never open with a
                    # previous leg's spending against this leg's releases.
                    "consumed_bytes": 0,
                    "leg_complete": False,
                    "publisher_pid": int(pid if pid is not None else os.getpid()),
                    "tags": [],
                    "claims": [],
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

    def claim(
        self,
        tag: str,
        want_bytes: int,
        *,
        epoch: Optional[Any] = None,
        require_full: bool = False,
        gate=None,
    ) -> Dict[str, Any]:
        """W takes bytes OFF this leg's credit -- the debit half of `publish`.

        #1349, MEASURED ON BOOT weg2xsn21b (D rank 0, the 5090
        GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d, 2026-09-12 16:06:26Z):

            credit=2560 MiB requested=1906 MiB free_mib=3276   tag=weights_6 GRANTED
            credit=2560 MiB requested=1906 MiB free_mib=1370   tag=weights_7 W85

        The counter read the SAME 2560 MiB one tag later, because nothing
        subtracted what ``weights_6`` had just spent.  The card's arithmetic is
        exact and leaves no unnamed remainder: 3276 - 1906 = 1370, and the whole
        2560 was ONE peer release (P rank 0's ``weights_4``, P log 16:06:27Z).
        The 1190 MiB "gap" between credit and free was never held by anything --
        it is 1906 already-spent minus 716 MiB of card-free that was never
        credit.  With the debit, ``weights_7`` would have seen 654 MiB available,
        stayed in the wait, and been funded by the peer's NEXT release (2988 MiB,
        about a second later) instead of being refused by name.

        WHY THE GATE RUNS INSIDE THE LOCK.  ``gate(available)`` is called between
        the decision and the write: if it raises, nothing is debited, so a
        refusal never consumes the bytes it refused, and no second claimant can
        slip between the grant and the debit.  It is a metadata read (#1250's
        NVML free column) and the publisher's :meth:`publish` is blocked only for
        its duration -- that is the price of granting and debiting in ONE step
        instead of two.

        The write happens ONLY when something is actually taken, so a stale-epoch
        or empty counter is read and left alone rather than rewritten -- a
        consumer must not be able to CREATE a counter for a leg nobody opened.
        """
        want = max(0, int(want_bytes))
        with self._locked(True) as handle:
            state = self._load(handle)
            stale_epoch = None
            if epoch is not None and state and str(state.get("epoch")) != str(epoch):
                # Same rule, same reason as :meth:`wait_for`: another leg's
                # state is NO credit, and therefore nothing to spend either.
                stale_epoch = state.get("epoch")
                state = {}
            credit = int(state.get("credit_bytes", 0))
            consumed = int(state.get("consumed_bytes", 0))
            available = max(0, credit - consumed)
            covered = available >= want
            claimed = 0 if (require_full and not covered) else min(available, want)
            gate_value = None
            if gate is not None and (covered or not require_full):
                gate_value = gate(available)  # may raise; nothing is written
            if claimed > 0:
                state["credit_bytes"] = credit
                state["consumed_bytes"] = consumed + claimed
                claims = list(state.get("claims", []))
                claims.append(str(tag))
                state["claims"] = claims
                self._store(handle, state)
            return {
                "tag": str(tag),
                "credit_bytes": credit,
                "consumed_bytes": consumed + claimed,
                "available_before_bytes": available,
                "available_bytes": available - claimed,
                "claimed_bytes": claimed,
                "covered": covered,
                "leg_complete": bool(state.get("leg_complete")),
                "stale_epoch": stale_epoch,
                "gate": gate_value,
            }

    def peer_deposit_position(self) -> str:
        """#1374 F3: WHERE THE PEER IS IN ITS PER-TAG DEPOSIT, for a refusal.

        Boot weg2xsn30's W35 said `credit=0 published=0 consumed=0
        requested=2916 peer_leg_complete=False` -- every number about THIS
        rank and none about the peer it was waiting for, so the line could not
        distinguish "the peer has not started" from "the peer is stuck two tags
        back". The credit ledger already knows: the tags the peer has published
        ARE the tags it has finished depositing and pausing, in order.
        """
        try:
            with self._locked(False) as handle:
                published = list(self._load(handle).get("tags", []))
        except BaseException:  # noqa: BLE001 - a refusal may not depend on this
            return "unreadable"
        if not published:
            return "none-yet (the peer has published no tag's release)"
        return f"{len(published)} paused, last={published[-1]}"

    def wait_for(
        self,
        need_bytes: int,
        *,
        budget_s: float,
        tag: str,
        free_bytes_now: Optional[int] = None,
        free_reader=None,
        floor_bytes: Optional[int] = None,
        poll_s: float = 0.01,
        epoch: Optional[Any] = None,
        #: #1391 (DESK10) ROUND 3: an OPTIONAL callback, ``() -> [(lane, full,
        #: empty), ...]``, polled on its OWN cadence (below, not every
        #: ``poll_s`` -- each call is a handful of ctypes ``sem_getvalue``
        #: syscalls, and this loop already spins at 100 Hz) WHILE this method
        #: waits. A ONE-SHOT check at entry (the first version of this fix)
        #: cannot see a condition that develops mid-wait: boot weg2xsn31's
        #: instrument run measured exactly that race -- this rank entered
        #: clean, the peer's saturating deposit landed 4 s later, and the
        #: 120 s poll ran to its end without ever looking again. ``None``
        #: (default) keeps every existing caller's behaviour byte-identical.
        stuck_lane_reader=None,
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
        floor = max(0, int(floor_bytes or 0))
        if free_bytes_now is not None and int(free_bytes_now) >= need:
            # #1349: THE EARLY EXIT SPENDS CREDIT TOO, and not debiting it was
            # the second route to the same wall.  "The card already holds the
            # bytes" does not say WHOSE bytes they are: on a co-located pair the
            # free this exit is taken against is largely the peer's just-released
            # pages, i.e. exactly the bytes the counter is publishing.  Leaving
            # them undebited let a later tag be licensed by a balance earlier
            # tags had already eaten, which is the xsn21b grant one step removed.
            # PARTIAL, never `need`: this exit claims the CARD holds the bytes,
            # not that the peer funded them, so it takes only what the counter
            # actually has.  `epoch` keeps a stale or absent counter at zero, so
            # a single-group boot writes nothing and still costs nothing.
            spent = self.claim(tag, need, epoch=epoch)
            return {
                "waited_s": 0.0,
                "free_bytes": int(free_bytes_now),
                "credit_bytes": spent["credit_bytes"],
                "consumed_bytes": spent["consumed_bytes"],
                "available_bytes": spent["available_bytes"],
                "claimed_bytes": spent["claimed_bytes"],
                "reason": (
                    "the card already holds the bytes, so no peer release funds "
                    "this tag and none is waited for"
                ),
            }
        deadline = time.monotonic() + float(budget_s)
        t0 = time.perf_counter()
        stale_epoch = None
        # #1374 F3: A WAIT THAT SAYS NOTHING IS INDISTINGUISHABLE FROM A HANG.
        # Boot weg2xsn30's PP0 was SILENT for 121 s between its resume request
        # (13:48:12) and its W35 (13:50:13): no line said what it was waiting
        # for, so the log read as "PP0 did nothing" when it was blocked on a
        # credit the peer could not publish. One line every 10 s.
        _next_progress = time.monotonic() + 10.0
        # #1391 (DESK10) ROUND 3: A SEPARATE, SHORTER CADENCE for the lane
        # check -- 1 s, not the progress line's 10 s. The measured race
        # (peer's saturating post landed 4 s into the wait) would still slip
        # past a 10 s cadence more than half the time; 1 s bounds the miss to
        # a small fraction of the 120 s budget it replaces. Cheap: a handful
        # of ctypes reads, not an allocation or a syscall storm at 100 Hz.
        _next_lane_check = time.monotonic()
        while True:
            if stuck_lane_reader is not None and time.monotonic() >= _next_lane_check:
                _next_lane_check = time.monotonic() + 1.0
                try:
                    _stuck = stuck_lane_reader()
                except Exception:  # noqa: BLE001 -- a probe may not raise
                    _stuck = []
                if _stuck:
                    raise Weg2XchgLaneNeverDrainedRefused(
                        f"W108 Weg2XchgLaneNeverDrainedRefused: card={self.uuid} "
                        f"tag={tag}: waiting for a VRAM credit this card may "
                        f"never receive -- {len(_stuck)} lane(s) targeting "
                        f"this rank's card now hold undrained bands (found "
                        f"{time.perf_counter() - t0:.1f}s into the wait): "
                        + ", ".join(
                            f"lane={lane} full={full} empty={empty}"
                            for lane, full, empty in _stuck)
                        + ". Under the #1374 contract `full` is 0 between "
                        f"tags by construction, so a peer deposited real "
                        f"bytes for a DIFFERENT tag that this rank's "
                        f"collector never drained; the credit for THIS tag "
                        f"depends on that same peer's pause loop, which is "
                        f"itself stuck behind the undrained lane. Refusing "
                        f"now beats riding out the remaining "
                        f"{budget_s - (time.perf_counter() - t0):.0f}s of "
                        f"this wait into a W35 that would hide the real lane."
                    )
            device: Dict[str, Any] = {}

            def _grant_gate(available: int, _dev=device) -> None:
                # #1331: THE PEER'S ACCOUNTING IS NOT THE CARD'S ANSWER, and
                # granting on it alone is what killed boot weg2xsn7.
                #
                #   20:05:47 TP0 WEG2-VRAM-CREDIT card=GPU-31d7ef41 tag=weights_7
                #                credit=2560 MiB requested=1906 MiB
                #                (the peer's releases funded this tag)
                #   20:05:48     cu_mem_create -> CUresult 2 (out of memory)
                #
                # `free_bytes_now` was only ever an EARLY EXIT at the top of
                # this method ("the card already holds the bytes"). Once the
                # loop is entered -- i.e. exactly when the card WAS short --
                # the grant was decided purely on `credit_bytes`, a number in
                # a shared file written by the peer's release bookkeeping, with
                # no second look at the device. So the counter said 2560 MiB
                # were released and the card could not allocate 1906.
                #
                # ONE READER, NEVER A SECOND NVML LOOP -- see :meth:`_free_now`.
                #
                # THE PEER TERM SURVIVES as the first condition; it is simply
                # no longer the only one. #1349 sharpened WHICH peer number that
                # is: the UNSPENT balance, not the gross published total. This
                # gate runs INSIDE :meth:`claim`'s lock, between the decision and
                # the debit, so a tag this gate refuses consumes nothing.
                free_now = self._free_now(free_reader)
                allocatable_est = (
                    None if free_now is None else max(0, free_now - floor)
                )
                _dev["free_bytes"] = free_now
                _dev["allocatable_est_bytes"] = allocatable_est
                if allocatable_est is not None and allocatable_est < need:
                    # 2026-09-15 (wake overlap, weg2xsn108): the shortfall can
                    # be TRANSIENT -- the sleeper's on-card IPC staging for
                    # the tag just collected is freed once the wake worker's
                    # drain lands, and the waking main thread asks for the
                    # next tag's credit while that collect is still in
                    # flight. Poll the allocatable estimate (bounded) before
                    # refusing; a genuine shortfall still refuses by name.
                    import time as _time85
                    _t85 = _time85.perf_counter()
                    while (allocatable_est is not None and allocatable_est < need
                           and _time85.perf_counter() - _t85 < 30.0):
                        _time85.sleep(0.05)
                        free_now = self._free_now(free_reader)
                        allocatable_est = (
                            None if free_now is None else max(0, free_now - floor))
                    _dev["free_bytes"] = free_now
                    _dev["allocatable_est_bytes"] = allocatable_est
                    _dev["allocatable_waited_ms"] = int(
                        (_time85.perf_counter() - _t85) * 1000)
                if allocatable_est is not None and allocatable_est < need:
                    raise Weg2VramCreditAllocatableShort(
                        f"W85 Weg2VramCreditAllocatableShort card={self.uuid} "
                        f"tag={tag} credit={available // MIB} MiB "
                        f"requested={need // MIB} MiB "
                        f"free_mib={free_now // MIB} "
                        f"allocatable_est={allocatable_est // MIB} MiB "
                        f"corridor_floor_mib={floor // MIB} -- the peer's "
                        f"release counter funds this tag but the CARD does "
                        f"not: allocatable is below the request, so the "
                        f"resume would fail in cu_mem_create and leave every "
                        f"tensor of this tag unmapped (boot weg2xsn7: that "
                        f"OOM left the barlink control word unmapped and the "
                        f"process died in the driver). Refused by name rather "
                        f"than granted silently. #1349: `credit` here is the "
                        f"UNSPENT balance of this leg (published minus claimed), "
                        f"so this refusal can no longer be a tag re-licensed by "
                        f"bytes an earlier tag of the same leg already took."
                    )

            rec = self.claim(
                tag, need, epoch=epoch, require_full=True, gate=_grant_gate
            )
            if rec["stale_epoch"] is not None:
                # Not this flip's counter -- and "this flip" means THIS BOOT's
                # flip: the comparison is on the composed token of
                # :func:`credit_epoch`, so a previous boot's leftover file is
                # waited past for the same reason and by the same line as a
                # previous flip's.  Neither its bytes nor its leg_complete flag
                # say anything about the leg now in flight.
                stale_epoch = rec["stale_epoch"]
            credit = int(rec["credit_bytes"])
            available = int(rec["available_before_bytes"])
            consumed = int(rec["consumed_bytes"])
            if rec["claimed_bytes"] >= need:
                return {
                    "waited_s": time.perf_counter() - t0,
                    # #1349: the UNSPENT balance, which is what the grant was
                    # actually decided on -- the gross `credit_bytes` is beside
                    # it so a log line can show both halves of the book.
                    "credit_bytes": available,
                    "credit_published_bytes": credit,
                    "consumed_bytes": consumed,
                    "available_bytes": int(rec["available_bytes"]),
                    "claimed_bytes": int(rec["claimed_bytes"]),
                    "free_bytes": device.get("free_bytes"),
                    "allocatable_est_bytes": device.get("allocatable_est_bytes"),
                    "corridor_floor_bytes": floor,
                    "reason": "the peer's releases funded this tag",
                }
            if rec["leg_complete"]:
                # #1349: ASK THE CARD ONCE MORE BEFORE REFUSING.  The debit makes
                # the balance shrink within a leg, so "the peer will release no
                # more" must not be read as "these bytes are not there" -- the
                # early exit at the top of this method licenses a resume on the
                # card's own free, and a leg that ends while the card HAS the
                # bytes must reach the same answer late as it would have reached
                # early.  Without this the debit would have bought the W85 fix
                # with a new false W35, i.e. the same refusal MOVED rather than
                # removed, which is the danger direction this slice was named on.
                free_now = self._free_now(free_reader)
                if free_now is not None and free_now - floor >= need:
                    spent = self.claim(tag, need, epoch=epoch)
                    return {
                        "waited_s": time.perf_counter() - t0,
                        "credit_bytes": available,
                        "credit_published_bytes": credit,
                        "consumed_bytes": int(spent["consumed_bytes"]),
                        "available_bytes": int(spent["available_bytes"]),
                        "claimed_bytes": int(spent["claimed_bytes"]),
                        "free_bytes": free_now,
                        "allocatable_est_bytes": max(0, free_now - floor),
                        "corridor_floor_bytes": floor,
                        "reason": (
                            "the peer's leg is complete and unspent credit is "
                            "short, but the card itself holds the bytes now"
                        ),
                    }
                raise Weg2VramCreditRefused(
                    f"W35 Weg2VramCreditRefused (spec section 6 lists it as W30) "
                    f"card={self.uuid} tag={tag} credit={available // MIB} MiB "
                    f"published={credit // MIB} MiB "
                    f"consumed={consumed // MIB} MiB "
                    f"requested={need // MIB} MiB peer_leg_complete=True "
                    f"peer_deposit={self.peer_deposit_position()} "
                    f"free_bytes_now={free_bytes_now} free_mib_at_refusal="
                    f"{'n/a' if free_now is None else free_now // MIB} -- the "
                    f"sleeping rank has "
                    f"finished its whole leg and will release nothing further, so "
                    f"these bytes are never coming; refusing by name rather than "
                    f"waiting out the budget or walking into a CUDA OOM"
                )
            if time.monotonic() >= deadline:
                raise Weg2VramCreditRefused(
                    f"W35 Weg2VramCreditRefused (spec section 6 lists it as W30) "
                    f"card={self.uuid} tag={tag} credit={available // MIB} MiB "
                    f"published={credit // MIB} MiB "
                    f"consumed={consumed // MIB} MiB "
                    f"requested={need // MIB} MiB peer_leg_complete=False "
                    f"peer_deposit={self.peer_deposit_position()} "
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
            if time.monotonic() >= _next_progress:
                _next_progress = time.monotonic() + 10.0
                logger.info(
                    "WEG2-VRAM-CREDIT waiting card=%s tag=%s need_mib=%d "
                    "published_mib=%d consumed_mib=%d peer_leg_complete=%s "
                    "waited_s=%.0f of %.0f -- #1374 F3: blocked on the "
                    "co-located peer's per-tag release, which it publishes "
                    "right after pausing this tag; a growing wait here means "
                    "the peer has not reached this tag's pause yet",
                    self.uuid, tag, int(need) // MIB, int(credit) // MIB,
                    int(consumed) // MIB, bool(rec.get("leg_complete")),
                    time.perf_counter() - t0, float(budget_s))
            time.sleep(poll_s)

    @staticmethod
    def _free_now(free_reader) -> Optional[int]:
        """The card's free bytes through the CALLER's reader, or ``None``.

        ONE READER, NEVER A SECOND NVML LOOP (#1331): ``free_reader`` is injected
        by the caller and is the #1250 v2 free reader
        (``registry.nvml.memory_info_for_uuid`` / ``memory_snapshot()``); this
        module opens no NVML handle of its own.  ``None`` -- unreadable, or no
        reader passed -- keeps the pre-#1331 behaviour exactly: a blind reading
        may not manufacture a refusal any more than it may license a grant.
        """
        if free_reader is None:
            return None
        try:
            got = free_reader()
        except Exception:  # noqa: BLE001 -- a probe may not raise
            return None
        return None if got is None else int(got)


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
# ITEM `dormant`: THE CUDA-GRAPH TAG ON THE WEG-2 SLEEP PATH
# (record section [1y], 2026-09-08)
# ---------------------------------------------------------------------------
#
# The dormant group holds 1820 / 1368 / 1422 MiB per rank (rg6, NVML
# per-process bytes on the sleep-acceptance line).  Section [1y] attributes it;
# two of its rows are releasable and the mechanism for both already exists
# upstream, unwired:
#
#   R4  the CUDA-graph CAPTURE POOL (92 / 102 / 133 MiB).  Captures already
#       route through `memory_saver_adapter.cuda_graph(tag=cuda_graph)` when
#       `enable_memory_saver` AND `SGLANG_MEMORY_SAVER_CUDA_GRAPH`
#       (full_cuda_graph_backend.py:78-81, :132-139), and both RPC handlers
#       already have their `GPU_MEMORY_TYPE_CUDA_GRAPH in tags` branch.  What
#       is missing is that NOTHING SENDS THE TAG: the front's sleep RPC carries
#       [kv_cache] alone.
#   R3a the flashinfer FLOAT workspace (384 MiB/rank -- env default
#       384*1024*1024, and Qwen3_5ForConditionalGeneration is deliberately NOT
#       in HIGH_WORKSPACE_ARCHITECTURES, flashinfer_workspace.py:59-67).  It is
#       the one workspace on this path with a stated content contract, and the
#       contract is ZERO: `zero_flashinfer_workspaces()` wipes it after every
#       finished request, and its docstring carries the #50 GPU bisection
#       verbatim -- zeroing exactly the FLOAT workspace flattens the
#       request-ordinal output, "int workspace / kv_lens wipes do not".  So a
#       pause that unmaps it and a resume that maps fresh pages destroys
#       nothing, PROVIDED the resume restores the zero.
#
# WHAT IS DELIBERATELY LEFT RESIDENT, with its size, because it holds content
# a remap would destroy or content the wake needs:
#
#   * the flashinfer INT workspace (inside [1y] R3b, 87 MiB/rank together with
#     cuBLAS and the graph static buffers).  The decode backend is `full`, i.e.
#     "attention metadata is captured INSIDE the graph"
#     (full_cuda_graph_backend.py:54-57), so the plan the wrapper wrote there at
#     capture time is what the replay reads: a capture-time constant.  The #50
#     bisection is the positive evidence that this is NOT the zero-contract
#     buffer.  It stays tagged-out.
#   * the CUDA context ([1y] R1, 492 / 205 / 205 MiB) and the communicator
#     buffers ([1y] R2, 307 / 184 / 184 MiB, of which 120 MiB/rank is the
#     barlink BAR1 windows).  Untouchable by the item's own terms; BAR1
#     re-registration is a collective and its extension is a JIT build, which
#     RESTORE-NEVER-REBUILD forbids inside a cutover.
#   * the `_export_static_state` clones ([1y] R5, >= 100 MiB/rank).  They ARE
#     what the wake imports back.
#
# NO SECOND MECHANISM IS BUILT.  #102 ("Capture-Pools + IO-Buffer taggbar",
# htsglang:078feed5ea34 / :0195ffb3e1325) already owns the shape -- one private
# MemPool per tag plus `region_config`, with the size gate above -- but every
# one of its wrap sites is scoped to an ADAPTIVE DRAFT state build
# (`_ACTIVE_MANAGER`), and group P runs speculative_algorithm=None, so on this
# path they are all `nullcontext()`.  What follows reuses that shape keyed to
# the UPSTREAM `cuda_graph` tag, so there is exactly one tag and one ledger.


def weg2_group_name() -> str:
    """This rank's Weg-2 group name, or ``""`` on an engine that is not one.

    FIX 2, finding 1.  Cached for the process lifetime for the same reason the
    graph-tag answer is: a sleep and its wake must not be able to read
    different answers.
    """
    global _WEG2_GROUP_NAME
    if _WEG2_GROUP_NAME is None:
        _WEG2_GROUP_NAME = os.environ.get(WEG2_GROUP_ENV, "").strip()
    return _WEG2_GROUP_NAME


def weg2_env_present() -> bool:
    """True when SOME ``SGLANG_WEG2_*`` variable other than the group is set.

    FIX 3, finding 2.  The refusal below has to distinguish two things that
    look identical from inside one rank:

    * a STOCK engine, which never had :data:`WEG2_GROUP_ENV` and must stay
      byte-identical -- including in its log, so no new line may be printed
      there, and
    * a WEG-2 rank whose group name did not arrive, which is a silent
      capability loss (the #1246 shape) and must be loud.

    ``launcher.build_env`` publishes the group beside a family of other
    ``SGLANG_WEG2_*`` variables (``SGLANG_WEG2_WEIGHT_CHUNK_LAYERS``,
    ``SGLANG_WEG2_WEIGHT_CHUNKS``, ``SGLANG_WEG2_TMS_PRELOAD_SO``,
    ``SGLANG_WEG2_PCIE_DUPLEX``, ...), so the presence of any of them with the
    group ABSENT is the signature of the second case.  A stock engine has none
    of them and stays silent.

    HONEST LIMIT: the family is conditional (a boot with no chunking, no
    preload and no duplex publishes none of them), so this is a sufficient
    signal for the loud case, not a necessary one.  It never produces a FALSE
    loud line on a stock engine, which is the direction that would change
    upstream behaviour; it can miss a Weg-2 rank that had lost the whole
    family, and such a rank is not a Weg-2 rank in any other respect either.
    """
    for key in os.environ:
        if key.startswith("SGLANG_WEG2_") and key != WEG2_GROUP_ENV:
            return True
    return False


def _refuse_graph_tag(reason: str, detail: str) -> None:
    """Name a graph-tag refusal in the log, rate-limited, with its denominator.

    FIX 3, finding 2.  Before this, every conjunct of
    :func:`weg2_graph_tag_armed` refused SILENTLY: a rank that lost
    :data:`WEG2_GROUP_ENV` produced no line at the gate, no line at the region
    (``yield False`` with no logger call), no line at the sleep (the tag list
    came back unchanged) -- and therefore was byte-identical in the logs to a
    rank running the tree from before this item.  The boot arm looked for a
    degrade line that could only be emitted by ONE of the refusal paths (the
    adapter-build failure), so the one degrade it could not see was the one
    that silently disarmed the whole item.

    RATE LIMIT AND ITS DENOMINATOR: this is called once per sleep leg and once
    per attention-backend build, so it is not hot, but it must not scroll
    either.  The line fires on occurrence 1, 2, 4, 8, ... per reason and always
    prints ``occurrence=`` -- a reader can therefore tell "refused once" from
    "refused on every leg", which a plain once-per-process line cannot, and
    :data:`_GRAPH_TAG_REFUSALS` carries the exact count for a test.
    """
    n = _GRAPH_TAG_REFUSALS.get(reason, 0) + 1
    _GRAPH_TAG_REFUSALS[reason] = n
    if n & (n - 1):  # not a power of two -- suppressed, but counted above
        return
    logger.warning(
        "WEG2-SLEEP graph tag NOT armed: %s -- %s. occurrence=%d for this reason "
        "in this process (the line is rate-limited to occurrences 1,2,4,8,...; "
        "the count is the denominator, the line is not). The sleeping rank keeps "
        "the CUDA-graph capture pool and the flashinfer FLOAT workspace resident "
        "(pre-item behaviour, not a silent success)",
        reason,
        detail,
        n,
    )


def weg2_graph_tag_refusals() -> Dict[str, int]:
    """A copy of :data:`_GRAPH_TAG_REFUSALS` -- the rate limit's denominator."""
    return dict(_GRAPH_TAG_REFUSALS)


def weg2_graph_tag_armed(memory_saver_on: bool) -> bool:
    """True when this rank's sleep also releases ``GPU_MEMORY_TYPE_CUDA_GRAPH``.

    THREE conjuncts, and FIX 2 (findings 1 and 2) added two of them because the
    first version had only the third and was wrong in both directions:

    1. ``memory_saver_on`` -- the caller's own
       ``server_args.enable_memory_saver``.  It is the half of the capture
       site's pair (``full_cuda_graph_backend.py:78-81``,
       ``enable=enable_memory_saver and get_bool_env_var(...)``) that the first
       version's docstring claimed to read and did not.  Without it, an engine
       with the env set but the saver off armed the sleep tag while the capture
       built a Noop adapter -- the exact split ``flashinfer_backend.py`` calls
       out ("the two must be released together or not at all").
    2. :func:`weg2_group_name` -- THIS IS A WEG-2 MECHANISM AND MUST GATE ON
       WEG 2.  ``enable_memory_saver`` is an upstream flag on an upstream
       endpoint: gating on it alone silently widened a stock
       ``POST /release_memory_occupation {"tags":["kv_cache"]}`` into
       ``kv_cache + cuda_graph`` on any engine that ran the documented
       memory-saver + cuda-graph configuration.  The caller asked for one tag
       and got two; that is not this item's to change.  With this conjunct the
       stock path is byte-identical BY CONSTRUCTION, not by a test's opinion of
       what "stock" means.
    3. ``SGLANG_MEMORY_SAVER_CUDA_GRAPH`` -- read through
       :func:`get_bool_env_var`, which is the function the CAPTURE site calls,
       so the two sides cannot disagree on a value.

    On (3), why not ``envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()`` even though
    that is the canonical registry entry: MEASURED on this box, one process per
    value, the three readers disagree ::

        value    hand-rolled(v1)   get_bool_env_var   envs.EnvBool
        '1'      True              True               True
        'true'   True              True               True
        'yes'    True              False              True
        'on'     True              False              False
        'y'      False             False              True

    The invariant is agreement with the CAPTURE, not with the registry, and the
    launcher honours an operator override of this variable
    (``launcher.py:1383-1385``), so a non-canonical value is a reachable input
    rather than a hypothetical.  ``envs`` would fix ``'on'`` and newly break
    ``'y'`` and leave ``'yes'`` broken; ``get_bool_env_var`` is exact for all
    five because it is literally the other side's reader.  If the capture site
    is ever migrated to ``envs``, ``test_the_sleep_gate_reads_the_capture_sites
    _own_reader`` goes red rather than the boot.

    Only (3) is cached: it is the drifting term (an env), and the sleep ADDS
    the tag to ``offload_tags`` while the wake REMOVES it, so a resolver that
    answered differently between the legs would ``KeyError`` on a tag that was
    never paused.  (1) is a per-call argument because it is a per-caller fact,
    and both legs read it from the same ``server_args`` object.

    FIX 3, finding 2: every one of the three refuses BY NAME through
    :func:`_refuse_graph_tag` -- except on a stock engine, where the whole
    point is that nothing changes, including the log.  :func:`weg2_env_present`
    is what tells the two apart.
    """
    group = weg2_group_name()
    if not memory_saver_on:
        if group or weg2_env_present():
            _refuse_graph_tag(
                "no_memory_saver",
                "the caller's server_args.enable_memory_saver is False on a "
                "Weg-2 rank (group=%r); launcher.common_flags passes "
                "--enable-memory-saver, so this rank was launched by something "
                "else" % (group,),
            )
        return False
    if not group:
        # THE condition FIX 2 added and FIX 3 gave an instrument to.  A stock
        # engine legitimately has no group and must stay silent; a Weg-2 rank
        # that lost the variable is a silent capability loss, and this is the
        # only place in either tree that can say so.
        if weg2_env_present():
            _refuse_graph_tag(
                "no_group",
                "%s is empty or unset on a rank that carries other SGLANG_WEG2_* "
                "variables -- launcher.build_env publishes it at every launch "
                "site, so it was lost between the launcher and this process"
                % WEG2_GROUP_ENV,
            )
        return False
    global _GRAPH_TAG_ARMED
    if _GRAPH_TAG_ARMED is None:
        # #1366: this used to read get_bool_env_var deliberately, "the CAPTURE
        # site's own reader", to keep arming and capture from disagreeing. The
        # intent stands and the reader moved with it: every site now reads the
        # DECLARATION (environ.py:1978), which is where get_bool_env_var's own
        # first line says the variable belongs. Measured divergence before the
        # move: 'yes' and 'y' were true to EnvBool and false to the helper.
        # The import is module-level, so `envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH`
        # here and at the capture site are one OBJECT, not two equal names.
        _GRAPH_TAG_ARMED = bool(envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get())
    if not _GRAPH_TAG_ARMED:
        _refuse_graph_tag(
            "env_off",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH is not truthy to its declared "
            "reader envs.EnvBool (which since #1366 is ALSO the CAPTURE site's "
            "reader, so the capture did not route into "
            "the tag either -- releasing it alone would pause a workspace the "
            "graph still reads). An operator override of this variable is a "
            "legitimate input; this line says what it costs",
        )
    return _GRAPH_TAG_ARMED


@contextmanager
def weg2_graph_scratch_region(
    nbytes: int, memory_saver_on: bool, adapter: Any = None
) -> Iterator[bool]:
    """Route ONE large, content-free graph-side allocation into the graph tag.

    Yields True when the enclosed allocation is tagged, False when it is not --
    the caller may need to know, and a silent no-op is how a region that never
    ran gets reported as one that did.

    Three refusals to tag, each for a stated reason:

    * the tag is not armed (``weg2_graph_tag_armed(memory_saver_on)``) -- then
      the capture path did not route into the tag either, and a workspace alone
      in a paused tag would be released while the graph that reads it is not.
      ``memory_saver_on`` is the caller's own
      ``server_args.enable_memory_saver``: FIX 2, finding 2, the first version
      read the env alone here, so an engine WITHOUT the saver but WITH the env
      built a real ``TorchMemorySaverAdapter`` and routed 384 MiB into a region
      no sleep would ever pause;
    * ``nbytes`` is below :data:`WEG2_GRAPH_SCRATCH_MIN_BYTES` -- #102's
      correctness gate, see the constant;
    * ``torch.cuda.MemPool`` / ``use_mem_pool`` or the adapter's
      ``region_config`` is unavailable -- the pool is the thing that makes
      cross-tag free-list reuse impossible, so without it the allocation stays
      in the default pool rather than landing untracked in a shared segment.

    Two more are FAILURES rather than decisions, and both degrade loudly to the
    pre-change behaviour instead of raising: the adapter could not be BUILT
    (FIX 2, finding 4 -- ``TorchMemorySaverAdapter.create`` re-raises the
    missing-wheel import error), and the pool or the region could not be
    ENTERED.  Raising in either place would turn a VRAM optimisation into a
    boot killer at attention-backend build time.

    The caller is responsible for restoring the allocation's content contract
    after a resume; for the flashinfer float workspace that is
    ``zero_flashinfer_workspaces()``, which the wake path calls.
    """
    if (
        not weg2_graph_tag_armed(memory_saver_on)
        or int(nbytes) < WEG2_GRAPH_SCRATCH_MIN_BYTES
    ):
        yield False
        return
    import torch

    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH

    # FIX 2, finding 4.  ``TorchMemorySaverAdapter.create`` RE-RAISES the
    # import error when torch-memory-saver is missing
    # (``torch_memory_saver_adapter.py:36-45``), and in the first version this
    # call sat ABOVE the guarded block -- so the one exception this region can
    # produce escaped the very try written to stop it, straight into
    # ``FlashInferAttnBackend.__init__``, exactly the "boot killer at
    # attention-backend build time" the degrade path below names as the thing
    # it exists to prevent.  Caught HERE rather than inside that block because
    # the block also contains the ``yield``: wrapping the caller's body in an
    # ``except Exception`` would swallow the CALLER's exception, which is a
    # second defect and has its own test.
    #
    # ``enable=True`` is no longer a hardcoded claim either: the arming gate
    # above now requires ``memory_saver_on``, so this is the same value the
    # capture site passes (``full_cuda_graph_backend.py:78-81``) and the
    # upstream warning "enable_memory_saver is enabled, but torch-memory-saver
    # is not installed" can no longer be printed on an engine that never
    # enabled it.
    if adapter is None:
        try:
            from sglang.srt.utils.torch_memory_saver_adapter import (
                TorchMemorySaverAdapter,
            )

            adapter = TorchMemorySaverAdapter.create(enable=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "WEG2-SLEEP graph scratch NOT tagged (%d bytes): the memory-saver "
                "adapter could not be built (%s: %s) -- the allocation stays in "
                "the default pool and is RESIDENT across the sleep (pre-change "
                "behaviour, not a silent success)",
                int(nbytes),
                type(exc).__name__,
                exc,
            )
            yield False
            return

    region_config = getattr(adapter, "region_config", None)
    mempool_cls = getattr(torch.cuda, "MemPool", None)
    use_mem_pool = getattr(torch.cuda, "use_mem_pool", None)
    if region_config is None or mempool_cls is None or use_mem_pool is None:
        logger.warning(
            "WEG2-SLEEP graph scratch NOT tagged (%d bytes): "
            "region_config=%s MemPool=%s use_mem_pool=%s -- the allocation stays "
            "in the default pool, which is resident across the sleep",
            int(nbytes),
            region_config is not None,
            mempool_cls is not None,
            use_mem_pool is not None,
        )
        yield False
        return
    global _GRAPH_SCRATCH_POOL
    stack = ExitStack()
    try:
        if _GRAPH_SCRATCH_POOL is None:
            _GRAPH_SCRATCH_POOL = mempool_cls()
        stack.enter_context(use_mem_pool(_GRAPH_SCRATCH_POOL))
        stack.enter_context(region_config(tag=GPU_MEMORY_TYPE_CUDA_GRAPH))
    except Exception as exc:  # noqa: BLE001
        # The fourth refusal, and the only one that is a FAILURE rather than a
        # decision: the pool or the region could not be entered (no CUDA, an
        # uninitialised saver, a torch that refuses the pool).  Degrade to
        # untagged -- which is exactly the behaviour before this change, so the
        # allocation is correct and merely resident -- but say so with the
        # exception named.  Raising instead would turn a VRAM optimisation into
        # a boot killer at attention-backend build time.
        stack.close()
        logger.warning(
            "WEG2-SLEEP graph scratch NOT tagged (%d bytes): %s: %s -- the "
            "allocation stays in the default pool and is RESIDENT across the "
            "sleep (pre-change behaviour, not a silent success)",
            int(nbytes),
            type(exc).__name__,
            exc,
        )
        yield False
        return
    try:
        yield True
    finally:
        stack.close()


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


#: #134: wieviele Experten ein BAND fasst. Analog zu
#: WEIGHT_CHUNK_ENV_LAYERS, nur eine Ebene feiner: der 27B-Flip
#: verschiebt ganze Layer ueber Chunk-Tags, und ein Experte ist ein
#: Layer-Stueck -- dieselbe pause/resume-Mechanik traegt ihn, sobald er
#: einen eigenen Tag hat. 0 = Bandteilung AUS (byte-identisch zu vorher).
EXPERT_BAND_ENV_SIZE = "SGLANG_WEG2_EXPERT_BAND_SIZE"
EXPERT_BAND_ENV_COUNT = "SGLANG_WEG2_EXPERT_BANDS"


def expert_band_geometry() -> Tuple[int, int]:
    """(experts_per_band, band_count) aus der Env; (0, 0) = Bandteilung AUS.

    SYMMETRISCH zu :func:`weight_chunk_geometry`, und das ist kein Stil: die
    Tag-Bildung braucht die BREITE (welches Band eine Id trifft), die
    Familien-Aufzaehlung braucht die ANZAHL (welche Tags es gibt).  Eine Zahl
    allein kann nur eine der beiden Fragen beantworten -- die Chunk-Seite hat
    genau deshalb zwei, und ein Band, das nur die Breite kennt, waere ein
    Schreiber ohne Leser (Memory ``riegel-hinter-dem-was-er-sichert``).

    BEIDE muessen gesetzt sein: eine halbe Geometrie ist AUS, nicht geraten.
    """
    try:
        size = int(os.environ.get(EXPERT_BAND_ENV_SIZE, "0") or 0)
        count = int(os.environ.get(EXPERT_BAND_ENV_COUNT, "0") or 0)
    except ValueError:
        return 0, 0
    if size <= 0 or count <= 0:
        return 0, 0
    return size, count


def expert_band_size() -> int:
    """Experten je Band, oder 0 wenn die Bandteilung aus ist."""
    return expert_band_geometry()[0]


def expert_band_count() -> int:
    """Wieviele Baender ein Layer-Chunk hat, oder 0 wenn die Teilung aus ist."""
    return expert_band_geometry()[1]


def weight_band_tag(layer_id: int, band: int) -> Optional[str]:
    """Der Tag eines Bandes ueber seinen INDEX statt ueber eine Experten-Id.

    Der Austausch kennt aus dem Namen den Index, der Allokationspfad kennt die
    Id -- beide muessen denselben Tag bekommen, also rechnet nur EINE Stelle
    (``weight_chunk_tag``) und diese hier setzt bloss eine Id ihres Bandes ein.
    """
    size = expert_band_size()
    if size <= 0:
        return weight_chunk_tag(layer_id)
    return weight_chunk_tag(layer_id, int(band) * size)


def weight_chunk_tag(layer_id: int, expert_id: Optional[int] = None
                     ) -> Optional[str]:
    """Der Tag eines Layers -- oder eines EXPERTEN-BANDES darin (#134).

    Ohne `expert_id` unveraendert der Chunk-Tag des Layers, wie ihn der
    27B-Flip benutzt. Mit `expert_id` und eingeschalteter Bandteilung ein
    feinerer Tag `weights_<chunk>_e<band>`: dieselbe pause/resume-Familie,
    nur kleiner geschnitten, damit der Flip einzelne Experten verschieben
    kann statt immer den ganzen Layer.
    """
    layers, count = weight_chunk_geometry()
    if layers <= 0:
        return None
    _chunk = min(int(layer_id) // layers, count - 1)
    _base = f"{WEIGHT_CHUNK_PREFIX}{_chunk}"
    if expert_id is None:
        return _base
    _size, _count = expert_band_geometry()
    if _size <= 0:
        return _base
    # Geklemmt wie der Chunk oben: eine Id jenseits des letzten Bandes
    # gehoert ins letzte Band, statt einen Tag zu erfinden, den
    # ``weights_family_tags`` nie aufzaehlt (das waere W74 beim ersten Flip).
    return f"{_base}_e{min(int(expert_id) // _size, _count - 1)}"


def weights_family_tags(chunk_count: Optional[int] = None) -> list:
    """Every tag the sleep/wake path treats as 'the weights', chunks FIRST and
    the base tag LAST -- the order the front pauses them in, so the base tag
    (the remainder: embeddings, head, buffers) closes the sleep.

    #1273 B4k: under the exchange arm ``weights_draft`` sits BETWEEN the chunks
    and the base tag, and the position is load-bearing twice.  ``derive_waves``
    takes ``tags[-1]`` as the base tag that closes the last wave, and
    ``front.interleave_pause_order``'s stated contract is that the base tag
    closes the sleep -- appending the draft tag last would silently make IT the
    base of both.  It is not a chunk either (``is_weights_chunk_tag`` is the
    integer predicate and says so), because a chunk tag is a LAYER BAND of the
    target model and the draft head is not a band of anything.
    """
    # #134: mit eingeschalteter Bandteilung traegt JEDER Layer-Chunk seine
    #   Experten-Baender VOR sich selbst.  Zwei Gruende, beide tragend:
    # * Der Chunk-Tag behaelt den REST des Layers (Attention, Dense, Normen);
    #   die Experten wandern unter ihren eigenen Tags.  Die Summe ist
    #   unveraendert der ganze Layer -- kein Byte faellt zwischen zwei Tags.
    # * Die Baender zuerst, weil sie die grossen Bytes sind: die
    #   Pause-Reihenfolge ist zugleich die Freigabe-Reihenfolge, und der
    #   Basis-Tag muss LETZTER bleiben (``derive_waves`` liest ``tags[-1]``).
    # Ohne die Env ist die Liste byte-identisch zu vorher.
    if chunk_count is None:
        chunk_count = weight_chunk_geometry()[1]
    bands = expert_band_count()
    tags: list = []
    for k in range(int(chunk_count)):
        base = f"{WEIGHT_CHUNK_PREFIX}{k}"
        tags.extend(f"{base}_e{b}" for b in range(bands))
        tags.append(base)
    if draft_tag_in_family():
        tags.append(GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
    return tags + [GPU_MEMORY_TYPE_WEIGHTS]


def chunk_tag_cards(
    stage_layers: Sequence[int],
    layers_per_chunk: int,
    chunk_count: int,
    card_of_stage: Optional[Sequence[int]] = None,
) -> Dict[str, Tuple[int, ...]]:
    """Which CARDS hold bytes of each ``weights_<k>`` chunk tag.

    #1233 boot weg2dk4 (2026-09-07) died here, and the defect is a geometry
    mismatch the tag index HIDES.  A chunk tag is a LAYER band
    (``weight_chunk_tag``: ``layer_id // layers_per_chunk``).  Under PIPELINE
    parallelism the layer axis is split across cards, so a chunk tag names
    exactly the one or two stages whose layer range it overlaps; under TENSOR
    parallelism every card holds a shard of every layer, so the same tag index
    names ALL cards.  The one-backup interleave pauses the source's tag k and
    resumes the destination's tag k in the same step (front.flip), which is a
    fair trade only when both sides mean the same card by ``k``.

    MEASURED on weg2dk4 with a PP source (``--pp-stage-ratio 32,18,14`` over 64
    layers, 8 layers per chunk) and a TP destination: the last stage's card
    (nvml2) holds P's bytes only in ``weights_6``/``weights_7``, so the pauses
    of ``weights_0..5`` released NOTHING there while D's resumes took ~813 MiB
    each -- driver_free fell 4,511 -> 277.8 MiB in five steps with no
    recovery and ``cu_mem_create`` returned CUDA OOM on the sixth
    (BOOT_weg2dk4_0907.md).  This map is what lets the interleave pause the
    tight card's bands FIRST; see ``front.interleave_pause_order``.

    ``card_of_stage`` gives the card id (NVML index) of each PP stage in stage
    order; it defaults to the stage ordinals.  The base weights tag is NOT in
    the map: its bytes (embeddings on the first stage, head on the last,
    buffers everywhere) are not a layer band, and the family contract pauses it
    last regardless.  A TP group has no layer split and therefore no map -- the
    caller reads an EMPTY map as "uniform", never as "no bytes".
    """
    if layers_per_chunk <= 0 or chunk_count <= 0 or not stage_layers:
        return {}
    cards = list(card_of_stage) if card_of_stage is not None else list(range(len(stage_layers)))
    if len(cards) != len(stage_layers):
        # #1294: was a bare ValueError, reachable from launcher.py's main()
        # after the dry-return and caught by neither funnel (not in
        # REFUSALS, not named Weg2*) -- left the sglang groups alive on the
        # cards as an uncaught traceback.  Weg2ChunkCardMismatch subclasses
        # BOTH Weg2LaunchRefused (cli()'s funnel) and ValueError (the
        # pre-existing test_weg2_flip_order_1233.py assertion), so no
        # existing catcher of either type loses its match.  Deferred import:
        # this module is used by the live serving managers, not only by the
        # launcher, so the launcher's own (control-plane) import graph is
        # brought in only on this failure path, never at module load.
        from sglang.srt.weg2.launcher import Weg2ChunkCardMismatch

        raise Weg2ChunkCardMismatch(
            f"W59 Weg2ChunkCardMismatch: card_of_stage {cards} does not match "
            f"the {len(stage_layers)} PP stages {list(stage_layers)}"
        )
    acc: Dict[str, set] = {}
    layer = 0
    for stage, n_layers in enumerate(stage_layers):
        for _ in range(int(n_layers)):
            tag = f"{WEIGHT_CHUNK_PREFIX}{min(layer // int(layers_per_chunk), int(chunk_count) - 1)}"
            acc.setdefault(tag, set()).add(int(cards[stage]))
            layer += 1
    # #134: ein Experten-Band liegt auf der Karte SEINES Chunks -- es ist ja
    # ein Stueck derselben Layer.  Ohne diesen Eintrag faende
    # ``interleave_pause_order`` fuer das Band keine Karte, naehme den
    # ``missing``-Zweig und fiele fuer JEDEN Flip auf die Identitaets-Ordnung
    # zurueck (genau der Verlust, den der Kommentar dort an weg2dk4 festhaelt),
    # und ``derive_waves`` behandelte es als karten-uniform.
    bands = expert_band_count()
    if bands > 0:
        for tag in list(acc):
            for b in range(bands):
                acc[f"{tag}_e{b}"] = set(acc[tag])
    return {tag: tuple(sorted(v)) for tag, v in sorted(acc.items())}


#: A CHUNK TAG IS ``weights_<integer>``, NOT ``weights_<anything>``.
#:
#: #1273 S2.  ``WEIGHT_CHUNK_PREFIX`` is the literal ``"weights_"``, so a
#: ``startswith`` test also matches every FUTURE tag that merely shares the
#: prefix -- and the exchange introduces exactly such a tag,
#: ``weights_draft`` (constants.py), whose entire purpose is to be OUTSIDE the
#: family.  Under the prefix test it was inside it, which would have put the
#: drafter back into every leg, census and wave and left the exchange with a
#: destination range that has no VRAM source (W74).  Same defect one layer down
#: for the planned ``weights_vision`` (spec section 6/S8).
#:
#: The index is what ``weight_chunk_tag`` writes, so the index is what the
#: predicate reads.  Built from the prefix rather than a second literal.
_WEIGHT_CHUNK_TAG_RE = re.compile(
    r"^" + re.escape(WEIGHT_CHUNK_PREFIX) + r"(\d+)(?:_e(\d+))?$"
)


def is_weights_chunk_tag(tag: Any) -> bool:
    """``weights_<integer>`` -- one layer band of the weights family.

    #134: ``weights_<integer>_e<integer>`` joins, and it MUST.  An expert band
    is a layer band cut finer (Nutzer 22.09.: "wir verschieben nicht nur ganze
    layer sondern auch ganze experten -- aber das sind ja auch nur layer"), so
    it is the same kind of thing the exchange already transports.  Left out,
    ``is_weights_family_tag`` says no, ``build_plan`` raises W74
    Weg2XchgSourceMissing for every expert tensor, and
    ``interleave_pause_order`` drops the bands into ``rest`` where they have no
    card list -- the identical defect the integer predicate was written for
    (``weights_draft`` inside the family under a prefix test), only mirrored.

    The suffix stays INTEGER on both halves for the same reason it was integer
    before: the predicate reads exactly what ``weight_chunk_tag`` writes, and
    nothing else that merely shares the prefix.
    """
    return isinstance(tag, str) and _WEIGHT_CHUNK_TAG_RE.match(tag) is not None


def chunk_of_band_tag(tag: Any) -> Optional[str]:
    """The LAYER band a tag belongs to: itself for ``weights_<n>``, the parent
    ``weights_<n>`` for ``weights_<n>_e<b>``, None for anything else (#134).

    One authority, so the card map, the wave derivation and the pause order
    cannot each grow their own way of stripping the band suffix.
    """
    if not isinstance(tag, str):
        return None
    m = _WEIGHT_CHUNK_TAG_RE.match(tag)
    if m is None:
        return None
    return f"{WEIGHT_CHUNK_PREFIX}{m.group(1)}"


def expert_band_of_tag(tag: Any) -> Optional[int]:
    """The band index of ``weights_<n>_e<b>``, or None if the tag names a whole
    layer band (#134)."""
    if not isinstance(tag, str):
        return None
    m = _WEIGHT_CHUNK_TAG_RE.match(tag)
    if m is None or m.group(2) is None:
        return None
    return int(m.group(2))


def exchange_owns_wake_refill() -> bool:
    """Is the WAKE refill of the weights the exchange itself, on THIS arm?

    #1378 Posten "Schritt B" (user order 2026-09-14, ring-off boot weg2xsn34):
    the launch-time W4 lock (``assert_backup_off_wake_refill_is_defined``)
    refuses every backup-OFF arm because its premise is "the wake refills the
    weights with ``update_weights_from_disk``".  Under ``exchange`` +
    ``authoritative`` that premise is FALSE for the weights the exchange owns:
    the wake refill is the exchange COLLECT (``_weg2_xchg_inject_from_peer``),
    whose completeness is guarded per tag -- BEFORE the pause mutates any VRAM
    -- by the sleep-side gate (``weight_updater._weg2_xchg_wake_source_gap`` ->
    W106, tag + measured byte count), with the draft's own fallback refusing
    separately on a quantized checkpoint.  A blanket launch refusal there
    condemns a DEFINED arm, which is exactly the class of wall this lane keeps
    paying windows for.

    ONE AUTHORITY, the same conjunction everywhere: ``exchange_armed() AND
    inject_authoritative()`` -- the SAME pair ``weights_cpu_backup_armed``'s
    auto branch reads (ring absence is a property of the INJECT arm, boot
    weg2xsn13's lesson).  Shadow keeps the ring as the ground truth it grades
    against, so shadow+off stays REFUSED by the lock.  The import is LAZY
    because ``weight_exchange`` imports THIS module at line 84 -- a
    module-level import would be a cycle (same shape as
    ``draft_tag_in_family`` beside this function).
    """
    try:
        from sglang.srt.weg2.weight_exchange import (
            exchange_armed,
            inject_authoritative,
        )
    except Exception:  # noqa: BLE001 -- the saver must import without the lane
        return False
    return bool(exchange_armed()) and bool(inject_authoritative())


def draft_tag_in_family() -> bool:
    """Is ``weights_draft`` a member of the weights family on THIS arm?

    #1273 B4k, spec AMENDMENT 6 (user ruling 2026-09-11): *"warum sollte der mtp
    kopf nur auf d liegen? ... die draft layer bytes, also die draft gewichte,
    muessen genauso aus dem vram geflippt werden"*.  Section 4.1's premise --
    *"group P carries no ``--speculative-*`` in this form, so none of those
    bytes has a VRAM source on the other side"* -- is STALE on the shipped
    default: ``launcher.DRAFT_KV_ON_P_DEFAULT`` is ``"on"`` and
    ``model_runner.py:1548`` says the draft-KV group *"runs the checkpoint's MTP
    head after every target chunk"*.  Boot weg2xsn15 measured the head on both:
    ``WEG2-XCHG-RESIDENT tag=weights_draft`` 1440/1280/1280 MiB on D's three
    ranks and 1572 MiB on P's LAST STAGE ONLY, all ``in_family=no`` -- residency
    the exchange never moved and no flip ever paused.

    ONE GATE, ONE READER OF THE MODE.  It answers through
    ``weight_exchange.exchange_armed``, the same predicate the tag site
    (``weights_region_tag_for``) already reads, so the membership and the tag
    cannot disagree; a second env parse here would be exactly the two-readings
    defect this lane keeps paying for.  The import is LAZY because
    ``weight_exchange`` imports THIS module at line 84 -- a module-level import
    would be a cycle.

    THE RING ARM IS UNTOUCHED, and that is not a cosmetic exclusion (spec 1.1:
    ``ring`` stays today byte for byte).  Under ``ring`` the draft runner never
    receives its own tag at all -- ``weights_region_tag_for`` hands it the BASE
    tag -- so the drafter is already paused as part of the base tag and already
    travels the host ring.  Admitting the tag to the family there would add a
    family member with no bytes to the wave list and the pause order, on a path
    nobody asked to change.
    """
    try:
        from sglang.srt.weg2.weight_exchange import exchange_armed
    except Exception:  # noqa: BLE001 -- the saver must import without the lane
        return False
    return bool(exchange_armed())


def is_weights_family_tag(tag: Any) -> bool:
    """The ONE answer to "are these bytes exchanged".

    Read by the plan's inventory loop, the census, the coverage arm and the
    wave list, so a tag that is in the family is planned, censused, waved and
    covered with no special case anywhere -- which is what AMENDMENT 6 asks for
    and why the draft tag joins HERE rather than at four call sites.

    ``weights_draft`` is the only non-integer suffix that joins, and it joins
    because BOTH groups were MEASURED holding its bytes.  ``weights_vision``
    (S8) has had no such measurement and stays out: a naming rule is not a VRAM
    source.
    """
    return isinstance(tag, str) and (
        tag == GPU_MEMORY_TYPE_WEIGHTS
        or is_weights_chunk_tag(tag)
        or (tag == GPU_MEMORY_TYPE_WEIGHTS_DRAFT and draft_tag_in_family())
    )


#: #134: DER NAME, unter dem ein Experten-Band am MoE-Modul haengt.
#: Der Austausch leitet den Tag eines Tensors aus seinem NAMEN ab
#: (``weight_exchange.tag_of_parameter_name``), waehrend die Allokation ihn
#: aus ``weight_chunk_scope`` bekommt -- beide muessen dasselbe sagen.  Ein
#: Band traegt deshalb seinen Index IM NAMEN, und der Bildner unten ist die
#: einzige Stelle, die ihn schreibt, wie der Leser darunter die einzige ist,
#: die ihn liest.  Zwei Seiten einer Naht, EINE Funktion je Richtung
#: (Memory ``zwei-seiten-einer-naht-fragen-dasselbe``).
EXPERT_BAND_ATTR_PREFIX = "weg2_eband"
_EXPERT_BAND_IN_NAME = re.compile(
    r"(?:^|\.)" + re.escape(EXPERT_BAND_ATTR_PREFIX) + r"(\d+)_"
)


def expert_band_attr_name(band: int, attr: str) -> str:
    """Wie ein Experten-Band am Modul heisst (#134)."""
    return f"{EXPERT_BAND_ATTR_PREFIX}{int(band)}_{attr}"


def expert_band_from_param_name(name: str) -> Optional[int]:
    """Der Band-Index aus einem Tensornamen, oder None (#134)."""
    m = _EXPERT_BAND_IN_NAME.search(name or "")
    return int(m.group(1)) if m else None


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


#: THE TAG THE OPEN WEIGHTS REGION WAS CREATED WITH, process-local.
#:
#: #1273 S2.  ``tms_set_current_tag`` is write-only from Python (there is no
#: ``tms_get_current_tag`` in the vendored hook, entrypoint.cpp:101, and
#: ``current_tag_`` is thread_local C++), so the one caller that must restore a
#: tag -- ``weight_chunk_scope`` -- cannot ask the saver which tag the region
#: set.  It used to restore the literal ``GPU_MEMORY_TYPE_WEIGHTS``, which is
#: correct for exactly one region tag and silently wrong for any other.  With
#: the draft's own region tag that becomes a real defect: the drafter's block
#: is a one-layer ``Qwen3_5ForCausalLM`` (qwen3_5_mtp.py:277-283) built through
#: ``make_layers`` (utils/common.py:2017), so its layer would be tagged
#: ``weights_0`` -- back inside the family -- and everything built after that
#: scope would carry the BASE weights tag instead of the draft tag.
#:
#: NOT second bookkeeping beside an upstream truth: the upstream truth is
#: unreadable from here.  Single-threaded by construction (model construction
#: and the post-load pass), nested by context manager, restored on exit.
#:
#: ASYMMETRY, stated rather than assumed (#1273 S2 review F11): the C
#: ``current_tag_`` this shadows is ``thread_local``, this publisher is
#: PROCESS-global.  Sound for every load path in the tree today -- the only
#: ``threading.Thread`` on a load path (model_loader/loader.py:2686) is the
#: remote-instance loader, which no weights region reaches -- and WRONG the day
#: a parallel loader opens two weights regions in two threads, where one
#: thread's exit would restore its base tag over the other's.  A future
#: parallel loader must make this a ``threading.local``, not merely hope.
_WEIGHTS_REGION_TAG = GPU_MEMORY_TYPE_WEIGHTS

#: Every tag a WEIGHTS region may legally be opened with.  A weights region
#: carrying anything else is a naming accident, not a decision: the pause
#: population, the census and the family predicate all read these names.
_LEGAL_WEIGHTS_REGION_TAGS = frozenset(
    (GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
)


def current_weights_region_tag() -> str:
    """The tag ``model_runner``'s open weights region was created with."""
    return _WEIGHTS_REGION_TAG


@contextmanager
def weights_region_tag(tag: str) -> Iterator[str]:
    """Publish the weights region's tag for the duration of the region."""
    global _WEIGHTS_REGION_TAG
    previous = _WEIGHTS_REGION_TAG
    _WEIGHTS_REGION_TAG = tag
    try:
        yield tag
    finally:
        _WEIGHTS_REGION_TAG = previous


#: #1378 xsn69 -- A NATIVE BACKTRACE ON SIGSEGV, opt-in by env, loaded from
#: INSIDE the rank: the tms adapter REPLACES LD_PRELOAD for every rank, so an
#: operator's preload never arrives. faulthandler (enabled at process start)
#: stays the previous handler: this one prints the native frames first, then
#: re-raises into it, so the boot log carries both. Weg2xsn69 died in three
#: different pure-Python frames within two seconds of the first resume; the
#: Python frames alone could not name the corrupter.
_SEGVBT_HANDLE = None


def _load_native_segv_backtrace() -> None:
    global _SEGVBT_HANDLE
    so = os.environ.get("SGLANG_WEG2_SEGVBT_SO", "")
    if not so or _SEGVBT_HANDLE is not None:
        return
    try:
        import ctypes

        _SEGVBT_HANDLE = ctypes.CDLL(so)
        logger.info("WEG2-SEGVBT native SIGSEGV backtrace loaded from %s", so)
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a rank
        logger.warning("WEG2-SEGVBT NOT loaded (%s): %s", so, exc)


_load_native_segv_backtrace()


#: #1378 xsn66 -- ONE CACHING-ALLOCATOR POOL PER WEIGHTS TAG, process-local.
#:
#: torch_memory_saver tags at ``cudaMalloc`` granularity, i.e. per caching-
#: allocator SEGMENT, and the allocator packs every allocation under ~10 MiB
#: into shared segments (2 MiB small pool, 20 MiB medium blocks).  A segment
#: therefore carries the tag that was current when it was OPENED, and every
#: later small tensor that lands in it -- from any tag -- is paused and resumed
#: with THAT tag, not with the tag its name puts it under in the manifest.
#: MEASURED on weg2xsn66 (846bf739ec): P rank 0 opens the base region
#: (embed_tokens + its INT8 weight_scale) before layer 0, so layer 0's
#: ``input_layernorm.weight`` (10240 B) sits in a segment owned by ``weights``;
#: after ``resume(weights_0)`` the driver read it as type=0 (unmapped) and the
#: first copy-out of lane c0 died with SIGSEGV -- xsn55..xsn66, twelve boots.
#: P rank 2 has no embedding, its first small segment was opened under
#: ``weights_6``, and layer 52's ``dt_bias`` read type=2 after
#: ``resume(weights_6)``: the same code, pure by accident of load order.
#:
#: The pool is what makes a tag's segments its own -- the same mechanism the
#: graph scratch uses above (``_GRAPH_SCRATCH_POOL``): allocations inside
#: ``use_mem_pool(pool)`` only ever share segments with each other.  One pool
#: per tag, for the base region, the draft region and every layer band.  Cost:
#: each pool keeps its own partially filled small/medium segment, up to ~22 MiB
#: per tag per rank -- measured by the post-load VRAM census, never assumed.
_TAG_MEM_POOLS: Dict[str, Any] = {}
_TAG_POOL_UNAVAILABLE_SAID = False


def tag_mem_pool(tag: str) -> Any:
    """The pool of ``tag``, created on first use; None when torch has none."""
    import torch

    mempool_cls = getattr(torch.cuda, "MemPool", None)
    if mempool_cls is None:
        return None
    pool = _TAG_MEM_POOLS.get(tag)
    if pool is None:
        pool = mempool_cls()
        _TAG_MEM_POOLS[tag] = pool
        logger.info(
            "WEG2-TAG-POOL tag=%s pool=new pools=%d -- this tag's allocations "
            "share caching-allocator segments with nothing else (#1378 xsn66: "
            "a segment carries the tag it was OPENED under)",
            tag, len(_TAG_MEM_POOLS),
        )
    return pool


def release_active_tag_pools(reason: str = "") -> int:
    """Hand every tag pool's FREE blocks back to the driver.  Returns the count.

    fnFL2 v45-v49 (21.09.), and it is the load-time twin of what
    ``use_mem_pool`` already does on exit.  Under ``--flip-weights family``
    the weights load inside :func:`tag_pool_scope`, whose pool is CACHED in
    ``_TAG_MEM_POOLS`` and therefore not left until the whole load is over.
    ``torch.cuda.empty_cache()`` does not reach into a live private pool
    (measured, pool_probe.py: 1.8 GiB freed inside a MemPool stays reserved
    through empty_cache), so every layer's Marlin repack transient stayed
    pinned for the rest of the boot.

    MEASURED COST of not doing this (boots fnFL2v47 vs v48, same ranks, two
    very different expert bookings):

        rank   layers   model tensors      occupied        overhead
        pp2       8      6.71 / 4.78 GiB   14.29 / 12.36    7.58 GiB (both)
        pp1      11      7.72 / 5.04 GiB   17.81 / 15.12   10.10 GiB (both)

    The overhead does not move with the booking at all and rises 0.84 GiB per
    LAYER -- so group P's 29-layer stage 0 pays ~25 GiB of a 32.6 GiB card
    before a single expert is resident, and v45-v49 died in ``cu_mem_create``
    at layer 28 of 29 rather than at any gate.

    ``_cuda_releasePool`` releases a pool's CACHED blocks; blocks still held
    by live tensors stay exactly where they are, which is why this is safe to
    call in the middle of a load.  Absent in a torch without MemPool: returns
    0 and says so once, the same shape as tag_pool_scope's own degrade.
    """
    import torch

    try:
        from torch.cuda.memory import _cuda_releasePool
    except Exception:  # noqa: BLE001 -- a torch without private pools
        return 0
    if not _TAG_MEM_POOLS or not torch.cuda.is_available():
        return 0
    device_index = torch.cuda.current_device()
    before_reserved = torch.cuda.memory_reserved()
    n = 0
    in_use = 0
    for tag, pool in list(_TAG_MEM_POOLS.items()):
        # THE ACTIVE POOL MUST NOT BE RELEASED (fnFL2 v50, the first cut of
        # this function): ``_cuda_releasePool`` asserts ``use_count == 0``
        # internally and PP2 died on
        # ``it->second->use_count == 0 INTERNAL ASSERT FAILED``
        # (CUDACachingAllocator.cpp:3473) the moment it reached the pool of
        # the layer being loaded.  ``MemPool.use_count()`` is the question
        # the allocator itself asks, so ask it here rather than track which
        # scope is open: the CURRENT layer's transient is simply released one
        # layer later, when its own scope has closed.
        try:
            if int(pool.use_count()) != 0:
                in_use += 1
                continue
            _cuda_releasePool(device_index, pool.id)
            n += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "WEG2-TAG-POOL release FAILED tag=%s (%s) -- this tag's load "
                "transients stay pinned for the rest of the boot", tag, exc,
            )
    freed_gib = (before_reserved - torch.cuda.memory_reserved()) / (1024 ** 3)
    torch.cuda.empty_cache()
    # AT INFO, and that is the lesson of fnFL2v51: the first two cuts of this
    # function logged at DEBUG, the boot's overhead came back IDENTICAL to the
    # unfixed run (7.58 GiB on the 8-layer stage, 10.10 on the 11-layer one --
    # to the decimal), and the log could not say whether the release had run,
    # found nothing to free, or never been reached.  An instrument whose
    # verdict is invisible cannot be read as a null result.
    logger.info(
        "WEG2-TAG-POOL release pools=%d released=%d skipped_in_use=%d "
        "freed_gib=%.2f%s",
        len(_TAG_MEM_POOLS), n, in_use, freed_gib,
        f" reason={reason}" if reason else "",
    )
    return n


def tag_pool_occupancy(tag: str) -> Optional[dict]:
    """What is still LIVE in ``tag``'s private pool, from the allocator itself.

    A MEASUREMENT, not a fix, and it is the step that was missing before
    fnFL2v56 tried to drop a pool object and died in
    ``c10::AcceleratorError -- CUDA error: invalid argument``: whether the
    pool CAN be dropped is a question the allocator answers, and nobody
    asked it.

    Returns ``{"segments", "active_blocks", "active_gib", "inactive_gib"}``
    or None when torch has no private pools.  ``active_blocks == 0`` is the
    precondition for dropping the pool; anything else names how much is in
    the way.
    """
    import torch

    pool = _TAG_MEM_POOLS.get(tag)
    if pool is None or not torch.cuda.is_available():
        return None
    try:
        segs = pool.snapshot(include_traces=False)
    except TypeError:  # older signature
        segs = pool.snapshot()
    except Exception as exc:  # noqa: BLE001
        logger.warning("WEG2-TAG-POOL occupancy unreadable tag=%s (%s)", tag, exc)
        return None
    active = inactive = 0
    n_active = 0
    for seg in segs or ():
        for blk in seg.get("blocks", ()):
            size = int(blk.get("size", 0))
            if blk.get("state") == "active_allocated":
                active += size
                n_active += 1
            else:
                inactive += size
    return {
        "segments": len(segs or ()),
        "active_blocks": n_active,
        "active_gib": active / (1024 ** 3),
        "inactive_gib": inactive / (1024 ** 3),
    }


def log_tag_pool_occupancy(tag: str, when: str = "") -> Optional[dict]:
    """:func:`tag_pool_occupancy` on ONE line, at INFO."""
    occ = tag_pool_occupancy(tag)
    if occ is None:
        return None
    logger.info(
        "WEG2-TAG-POOL occupancy tag=%s%s segments=%d active_blocks=%d "
        "active_gib=%.2f inactive_gib=%.2f -- active_blocks=0 is what a pool "
        "drop requires; inactive_gib is what empty_cache cannot reach here",
        tag, f" when={when}" if when else "",
        occ["segments"], occ["active_blocks"], occ["active_gib"],
        occ["inactive_gib"],
    )
    return occ


@contextmanager
def tag_pool_scope(tag: str) -> Iterator[Any]:
    """Route every allocation inside into ``tag``'s own pool.

    Degrades LOUDLY, once, when torch has no ``MemPool``/``use_mem_pool``: the
    allocations then share segments across tags, which is the weg2xsn66 shape,
    and a reader of the boot log must be able to see that this is so.
    """
    global _TAG_POOL_UNAVAILABLE_SAID
    import torch

    use_mem_pool = getattr(torch.cuda, "use_mem_pool", None)
    stack = ExitStack()
    pool = None
    why = ""
    try:
        pool = tag_mem_pool(tag) if use_mem_pool is not None else None
        if pool is not None:
            stack.enter_context(use_mem_pool(pool))
    except Exception as exc:  # noqa: BLE001 -- no CUDA context, a torch that
        # refuses the pool: degrade to the shared pool, LOUDLY, same as the
        # graph scratch region above. Raising here would turn a segment-purity
        # fix into a load-time boot killer.
        stack.close()
        pool = None
        why = f"{type(exc).__name__}: {exc}"
    if pool is None:
        if not _TAG_POOL_UNAVAILABLE_SAID:
            _TAG_POOL_UNAVAILABLE_SAID = True
            logger.warning(
                "WEG2-TAG-POOL UNAVAILABLE (MemPool=%s use_mem_pool=%s%s): weights "
                "tags SHARE caching-allocator segments -- a small tensor of one "
                "tag can sit in a segment another tag pauses (weg2xsn66 shape)",
                getattr(torch.cuda, "MemPool", None) is not None,
                use_mem_pool is not None,
                f" {why}" if why else "",
            )
        yield None
        return
    global _ACTIVE_TAG_POOL
    _prev_active = _ACTIVE_TAG_POOL
    _ACTIVE_TAG_POOL = pool
    try:
        with stack:
            yield pool
    finally:
        _ACTIVE_TAG_POOL = _prev_active


#: The pool ``tag_pool_scope`` currently routes to, so a nested step can step
#: OUT of it for the duration of a transient (:func:`outside_tag_pool`).  A
#: module global rather than a parameter because the step that needs it -- the
#: sub-byte unpack inside ``process_weights_after_loading`` -- is several
#: frames below the scope, in a quantization scheme this tree does not own.
_ACTIVE_TAG_POOL: Any = None


@contextmanager
def outside_tag_pool(reason: str = "") -> Iterator[bool]:
    """Route allocations inside to the DEFAULT pool, then come back.

    A private ``MemPool`` NEVER hands a cached block back to the driver while
    it lives: ``emptyCache`` releases ``graph_pools_freeable`` only, and a tag
    pool is cached per tag for the whole boot.  So every transient born inside
    one leaves a segment there forever.

    MEASURED (fnFL2v62 allocator snapshot, PP0): the eleven chunk pools held
    22.0 GiB reserved for 2.6 GiB of live weights -- 82 of 302 segments were
    COMPLETELY EMPTY (18.1 GiB), and the alloc history named the source: the
    INT4 unpack allocated 161 GiB cumulative and the widening 43 GiB inside
    those pools, in 2147 short-lived tensors of changing size, against 2.0 GiB
    of marlin repack output that is what actually has to stay.

    Yields True when it really stepped out.  Everything allocated inside is in
    the DEFAULT pool, in segments that carry no tag, so a caller MUST copy
    anything that has to survive back in before leaving the block -- else the
    exchange would pause a segment holding another tag's bytes (#1378 xsn66).
    """
    import torch

    pool = _ACTIVE_TAG_POOL
    if pool is None:
        yield False
        return
    try:
        from torch.cuda.memory import (
            _cuda_beginAllocateCurrentThreadToPool,
            _cuda_endAllocateToPool,
        )
    except Exception:  # noqa: BLE001 -- a torch without private pools
        yield False
        return
    device_index = torch.cuda.current_device()
    _cuda_endAllocateToPool(device_index, pool.id)
    try:
        yield True
    finally:
        _cuda_beginAllocateCurrentThreadToPool(device_index, pool.id)


@contextmanager
def weights_region(adapter: Any, tag: str, *, enable_cpu_backup: bool) -> Iterator[str]:
    """Open a WEIGHTS region AND publish its tag -- the only way to do either.

    #1273 S2 (refuter F6).  Publishing the region tag and opening the region
    were two statements at two call sites (model_runner's boot load and
    weight_updater's W73 roll-forward), and one of them adopted the publisher
    while the other kept a hardcoded ``GPU_MEMORY_TYPE_WEIGHTS``.  That is
    split brain, not a TODO: the roll-forward's post-load repack runs
    ``weight_chunk_scope`` (model_loader/loader.py:941), which reads THIS
    publisher, so a region opened with one tag while the publisher says another
    tags the reloaded parameters into the wrong family -- silently, and only
    the NEXT flip fails, with a destination range that has no VRAM source.

    One opener makes the two impossible to disagree.  Nesting order is
    load-bearing: publish BEFORE the region opens (an allocation inside it must
    already see the tag) and restore AFTER it closes.
    """
    if tag not in _LEGAL_WEIGHTS_REGION_TAGS:
        raise ValueError(
            f"a weights region may be opened with {sorted(_LEGAL_WEIGHTS_REGION_TAGS)}, "
            f"not {tag!r}: the pause population, the per-tag census and "
            "is_weights_family_tag all read this name. A layer band "
            "(weights_<n>) is a SCOPE inside the base region, never a region "
            "of its own -- see weight_chunk_scope."
        )
    with weights_region_tag(tag):
        if weights_resident_armed():
            # Scheibe 6a / fnFL2 v4-v6 (20.09.): the weights never sleep on
            # this arm, so they need no saver pool -- and a saver pool is a
            # PRIVATE torch.cuda.MemPool whose freed blocks torch.cuda.
            # empty_cache() never returns while the pool object lives
            # (metal probe pool_probe.py: 1.8 GiB freed inside a MemPool stays
            # reserved through empty_cache; only use_count 0 releases it).
            # The load-time transients (presplit full [E] stacks, marlin
            # repack) therefore pinned 7.1 GiB on PP1 (13.05 reserved vs
            # 5.97 allocated) and the KV sizing, which charges the card's
            # occupancy, refused. Outside the pool the transients go back to
            # the driver at the sizing's empty_cache. The tag is still
            # published: the census, weight_chunk_scope (a no-op without a
            # TMS region) and the release refusal read the name, not the pool.
            logger.info(
                "WEG2-WEIGHTS-RESIDENT: %s loads OUTSIDE the memory-saver pool "
                "(no pause on this arm; load transients return to the driver)",
                tag,
            )
            yield tag
            return
        with adapter.region(tag, enable_cpu_backup=enable_cpu_backup):
            # #1378 xsn66: the base tag's own pool, INSIDE the region so its
            # segments are cudaMalloc'ed under this tag. A layer band nested
            # below (weight_chunk_scope) switches to its own pool for its
            # duration and returns here on exit.
            with tag_pool_scope(tag):
                yield tag


@contextmanager
def weight_chunk_scope(layer_id: Optional[int]) -> Iterator[Optional[str]]:
    """Tag every allocation inside as the chunk of ``layer_id``.

    Valid ONLY inside the BASE weights region (model construction and the
    post-load pass, both under model_runner's region(GPU_MEMORY_TYPE_WEIGHTS)):
    on exit the current tag is restored to that base tag, which is what the
    region set.  No-op when chunking is off, when no region is open, when
    ``layer_id`` is None (a module outside any layer), and -- #1273 S2 -- when
    the open region is NOT the base weights region: a layer band is a subdivision
    of the exchanged weights family and of nothing else, so the draft region's
    allocations keep the draft tag they were opened with.
    """
    base = current_weights_region_tag()
    tag = (
        None
        if (layer_id is None or base != GPU_MEMORY_TYPE_WEIGHTS)
        else weight_chunk_tag(layer_id)
    )
    cdll = None if tag is None else _tms_cdll_in_region()
    if cdll is None:
        yield None
        return
    cdll.tms_set_current_tag(tag.encode("utf-8"))
    try:
        # #1378 xsn66: the tag alone is not enough -- it names the segment the
        # allocator OPENS, not the one it REUSES. The band's own pool is what
        # keeps its small tensors out of another tag's segment.
        with tag_pool_scope(tag):
            yield tag
    finally:
        cdll.tms_set_current_tag(base.encode("utf-8"))


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
