"""DFlashWorkerV2 side of ``FLLIPER_ADAPTIVE_K`` (adaptive verify width).

The decision lives in :mod:`adaptive_verify_width` (pure Python). This module
is the glue the DFlash worker calls, kept out of the 4k-line worker:

* ``build_states`` -- at boot, on EVERY rank (solo shadows included: they run
  the target verify), one target attention backend + one target decode graph
  runner per candidate width ``k < B`` (the boot runner is the ``k = B`` one).
  Same construction as EAGLE's adaptive ladder
  (``eagle_worker_v2.build_adaptive_runtime_state``, target half only): the
  DFlash draft is unchanged, it still proposes ``B`` tokens.
* ``begin_round`` -- once per decode round, before the DCP verify prebuild
  writes into the active backend: folds in the observations of round ``N-2``,
  plans this round's width, and swaps ``model_runner.attn_backend`` /
  ``decode_cuda_graph_runner`` to the pre-captured state. Never a capture at
  runtime; a width without a captured state is never returned.
* ``record`` -- after the accept: a non-blocking D2H copy of the rank-synced
  ``commit_lens`` with an event.
* ``restore`` -- puts back exactly the pointers ``begin_round`` replaced, so
  every non-verify forward (prefill, flip, sleep) sees the boot state.

Learning lags two rounds on purpose: round ``N`` consumes exactly the rounds
``<= N - 2``, independent of event timing, so every rank folds the same
observations into the same state on the same round (rank-uniform k without a
broadcast). Under the overlap scheduler round ``N-2``'s result has already
been processed (synchronised) when round ``N`` launches, so the event wait
is free there. Every ``census_every`` rounds one "ADAPTIVE-K census" line is
logged and the TP ranks all-gather ``state_digest``; a mismatch is a
crash-stop (Nutzer-Gesetz 2026-08-29: Raenge nie uneins).
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import time

import msgspec
import torch

from flliper.srt.environ import envs
from flliper.srt.speculative.adaptive_verify_width import (
    AdaptiveKConfig,
    AdaptiveVerifyWidth,
    RowCostFit,
    VerifyCostCurve,
    fit_coef,
    fit_observe,
    load_curve_file,
    state_digest,
)

logger = logging.getLogger(__name__)

#: Bundled 27B curve (A1, 28.10.), used when FLLIPER_ADAPTIVE_K_CURVE is "".
DEFAULT_CURVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "adaptive_k_curves",
    "qwen38_27b_dflash2.json",
)
#: The 1-GB-per-card corridor (planning target, VRAM rule 24.10.).
CORRIDOR_MIB = 1024


class VerifyWidthState(msgspec.Struct, kw_only=True):
    """Target-side resources of one pre-captured verify width."""

    width: int
    attn_backend: object
    graph_runner: object


class PendingObservation(msgspec.Struct, kw_only=True):
    round_idx: int
    width: int
    commit_lens: torch.Tensor
    event: object


class VerifyViews(msgspec.Struct, frozen=True, kw_only=True):
    """The verify-side tensors of one round at width ``k``."""

    candidates: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_loc_2d: torch.Tensor


def adaptive_k_config_from_env(*, block_size: int) -> AdaptiveKConfig:
    return AdaptiveKConfig(
        block_size=int(block_size),
        min_width=int(envs.FLLIPER_ADAPTIVE_K_MIN_WIDTH.get()),
        ema_alpha=float(envs.FLLIPER_ADAPTIVE_K_EMA_ALPHA.get()),
        margin_pct=float(envs.FLLIPER_ADAPTIVE_K_MARGIN_PCT.get()),
        min_dwell=int(envs.FLLIPER_ADAPTIVE_K_MIN_DWELL.get()),
        warmup_rounds=int(envs.FLLIPER_ADAPTIVE_K_WARMUP.get()),
        probe_every=int(envs.FLLIPER_ADAPTIVE_K_PROBE_EVERY.get()),
        census_every=int(envs.FLLIPER_ADAPTIVE_K_CENSUS_EVERY.get()),
        force_width=int(envs.FLLIPER_ADAPTIVE_K_FORCE.get()),
    )


# --------------------------------------------------------------------------
# Pure tensor helpers (CPU-testable; width == B returns today's tensors)
# --------------------------------------------------------------------------


def verify_views(
    *,
    draft_tokens: torch.Tensor,
    positions_2d: torch.Tensor,
    cache_loc_2d: torch.Tensor,
    width: int,
) -> VerifyViews:
    """Slice the B-wide draft block to the verify width.

    At ``width == B`` every field is the exact tensor (same storage, same
    view) the pre-adaptive code used, so the verify is bit-identical. Below
    ``B`` the slices are made contiguous: the Triton accept kernel refuses a
    strided ``candidates`` and that refusal permanently disables it.
    """
    block = int(draft_tokens.shape[1])
    if int(width) >= block:
        return VerifyViews(
            candidates=draft_tokens,
            input_ids=draft_tokens.reshape(-1),
            positions=positions_2d.reshape(-1),
            cache_loc=cache_loc_2d.reshape(-1),
            cache_loc_2d=cache_loc_2d,
        )
    k = int(width)
    candidates = draft_tokens[:, :k].contiguous()
    loc_2d = cache_loc_2d[:, :k].contiguous()
    return VerifyViews(
        candidates=candidates,
        input_ids=candidates.reshape(-1),
        positions=positions_2d[:, :k].reshape(-1).contiguous(),
        cache_loc=loc_2d.reshape(-1),
        cache_loc_2d=loc_2d,
    )


def pad_out_tokens(*, out_tokens: torch.Tensor, draft_tokens: torch.Tensor) -> torch.Tensor:
    """Widen a ``(bs, k)`` committed block to the result stride ``B``.

    The scheduler reads ``next_token_ids[i*B : i*B + accept_lens[i]]`` and
    ``accept_lens <= k``, so the filler columns are never read; they are the
    unverified drafts, deterministic on every rank. ``k == B`` returns the
    tensor unchanged.
    """
    block = int(draft_tokens.shape[1])
    if int(out_tokens.shape[1]) >= block:
        return out_tokens
    filler = draft_tokens[:, out_tokens.shape[1] :].to(out_tokens.dtype)
    return torch.cat([out_tokens, filler], dim=1)


def _selftest_block(*, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic dummy round (same on every rank): rows that miss at
    once, in the middle, one before the end and never."""
    g = torch.Generator().manual_seed(20261028)
    draft = torch.randint(0, 1000, (4, block_size), generator=g)
    target = torch.randint(1000, 2000, (4, block_size), generator=g)
    for r, agree in enumerate((0, block_size // 2, block_size - 2, block_size - 1)):
        target[r, :agree] = draft[r, 1 : agree + 1]
    return draft, target


def _eager_accept(*, candidates: torch.Tensor, target_predict: torch.Tensor):
    from flliper.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus
    from flliper.srt.speculative.dflash_worker_v2 import _commit_accept

    accept, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates, target_predict=target_predict
    )
    out, commit = _commit_accept(candidates, accept, bonus)
    return commit.to(torch.int64), out


def _triton_accept(*, candidates: torch.Tensor, target_predict: torch.Tensor, block_size: int):
    from flliper.kernels.ops.speculative.dflash import (
        _compute_dflash_accept_bonus_triton_unchecked,
    )

    bs, dev = int(candidates.shape[0]), candidates.device
    i32 = dict(dtype=torch.int32, device=dev)
    commit = torch.zeros((bs,), **i32)
    out = torch.zeros((bs, block_size), dtype=torch.int64, device=dev)
    _compute_dflash_accept_bonus_triton_unchecked(
        candidates=candidates,
        target_top1=target_predict,
        accept_lens_out=torch.zeros((bs,), **i32),
        commit_lens_out=commit,
        bonus_ids_out=torch.zeros((bs,), **i32),
        out_tokens_out=out,
        prefix_lens=torch.zeros((bs,), dtype=torch.int64, device=dev),
        new_seq_lens_out=torch.zeros((bs,), dtype=torch.int64, device=dev),
    )
    return commit.to(torch.int64), out


def accept_selftest(*, device, block_size: int, widths, use_triton: bool) -> bool:
    """Boot self-test of the accept at every tier (NF: 'bei k=N bitgleich').

    Reference = eager accept on the CPU at the full width. On the device, for
    every width k (B included): eager and (when the worker uses it) Triton
    accept of the first k slots must commit min(commit_B, k) tokens, equal to
    the reference's first ones -- at k = B that is bit-identity. Exercises the
    Triton BLOCK_SIZE variant each tier compiles. Any failure -> False (the
    ladder stays off group-wide via ``plan_ladder``).
    """
    try:
        draft, target = _selftest_block(block_size=block_size)
        ref_commit, ref_out = _eager_accept(candidates=draft, target_predict=target)
        for k in sorted({int(w) for w in widths} | {int(block_size)}):
            cand = draft[:, :k].contiguous().to(device)
            pred = target[:, :k].contiguous().to(device)
            runs = [_eager_accept(candidates=cand, target_predict=pred)]
            if use_triton:
                runs.append(_triton_accept(candidates=cand, target_predict=pred, block_size=block_size))
            want = torch.clamp(ref_commit, max=k)
            for commit, out in runs:
                commit, out = commit.cpu(), out.cpu()
                if not torch.equal(commit, want):
                    logger.error("ADAPTIVE-K self-test k=%d commit %s != %s", k, commit.tolist(), want.tolist())
                    return False
                for r in range(int(want.shape[0])):
                    n = int(want[r])
                    if not torch.equal(out[r, :n], ref_out[r, :n]):
                        logger.error("ADAPTIVE-K self-test k=%d row %d tokens differ", k, r)
                        return False
        return True
    except Exception as e:  # noqa: BLE001 -- any failure keeps k = B, loudly
        logger.error("ADAPTIVE-K self-test raised %s: %s", type(e).__name__, e)
        return False


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


class DFlashVerifyWidth:
    """Per-worker adaptive verify width; inert (always ``B``) when disabled."""

    def __init__(self, *, block_size: int, tp_rank: int) -> None:
        self.block_size = int(block_size)
        self.tp_rank = int(tp_rank)
        self.enabled = bool(envs.FLLIPER_ADAPTIVE_K.get())
        self.cfg = adaptive_k_config_from_env(block_size=self.block_size)
        curve_path = envs.FLLIPER_ADAPTIVE_K_CURVE.get() or DEFAULT_CURVE_PATH
        curve = load_curve_file(path=curve_path) if self.enabled else None
        self.controller = AdaptiveVerifyWidth(cfg=self.cfg, curve=curve)
        self.states: dict[int, VerifyWidthState] = {}
        self.pending: collections.deque[PendingObservation] = collections.deque()
        self.round_idx = 0
        self.width = self.block_size
        self.boot_runner: object = None
        self.saved: tuple | None = None
        self.stale_ct = 0
        self.ineligible_ct = 0
        # Rank-LOCAL measurements; they reach the decision only through the
        # census gather (rank 0's fit, adopted by every rank the same round).
        self.local_fits: dict[int, RowCostFit] = {}
        self.period_sum: dict[int, list] = {}
        self.last_round: tuple[int, int, float] | None = None
        if self.enabled and self.tp_rank == 0:
            logger.info(
                "ADAPTIVE-K boot: B=%d curve=%s candidate_widths=%s",
                self.block_size,
                curve.source if curve is not None else "none (k stays B)",
                list(self.controller.widths),
            )

    # ---- boot ------------------------------------------------------------

    def plan_ladder(
        self, *, cpu_group, free_mib: float, user_reserve_mib: int, selftest_ok: bool
    ) -> bool:
        """Boot-time, group-wide: may the k<B states be captured at all?

        One gather of (curve bytes, free MiB) over the TP group; every rank
        evaluates the same gathered list, so the verdict is uniform:
        * ranks read different curve files (a rewrite between two reads) ->
          no ladder;
        * the group minimum of free VRAM must cover the user reserve (never
          touched), the 1-GB corridor (planning target) and
          ``FLLIPER_ADAPTIVE_K_STATE_MIB`` per extra state -- this is planning
          (admission sizing at boot), never a runtime refusal.
        """
        if not self.enabled or not self.controller.active:
            return False
        import torch.distributed as dist

        mine = (msgspec.msgpack.encode(self.controller.curve), float(free_mib), bool(selftest_ok))
        box = [mine]
        if cpu_group is not None and dist.is_initialized() and dist.get_world_size(cpu_group) > 1:
            box = [None] * dist.get_world_size(cpu_group)
            dist.all_gather_object(box, mine, group=cpu_group)
        extra = len(self.controller.widths) - 1
        need = float(user_reserve_mib) + CORRIDOR_MIB + extra * float(envs.FLLIPER_ADAPTIVE_K_STATE_MIB.get())
        min_free = min(b[1] for b in box)
        reason = None
        if any(b[0] != box[0][0] for b in box):
            reason = "ranks read different curves"
        elif not all(b[2] for b in box):
            reason = f"accept self-test failed on ranks {[i for i, b in enumerate(box) if not b[2]]}"
        elif min_free < need:
            reason = f"group min free {min_free:.0f} MiB < need {need:.0f} MiB"
        if self.tp_rank == 0:
            logger.info(
                "ADAPTIVE-K ladder %s: widths=%s min_free=%.0f MiB need=%.0f MiB "
                "(user_reserve %d + corridor %d + %d x %d)",
                "OFF (" + reason + ") -> k stays B" if reason else "ON",
                list(self.controller.widths),
                min_free,
                need,
                int(user_reserve_mib),
                CORRIDOR_MIB,
                extra,
                int(envs.FLLIPER_ADAPTIVE_K_STATE_MIB.get()),
            )
        if reason is not None:
            self.controller = AdaptiveVerifyWidth(cfg=self.cfg, curve=None)
            return False
        return True

    def build_states(self, *, target_model_runner, server_args, build_attn_backend, runner_cls) -> None:
        """Capture one target verify state per candidate width ``k < B``.

        Same order on every rank (ascending widths). Any failure drops the
        whole ladder on this rank; the rank check at the first census then
        crash-stops instead of letting ranks run different widths.
        """
        if not self.enabled or not self.controller.active or self.states:
            # Off, nothing to choose from, or already built (the cold-stack
            # build is idempotent; a second capture would double the memory).
            return
        if target_model_runner.decode_cuda_graph_runner is None:
            logger.warning("ADAPTIVE-K: target has no decode graph runner -> k stays B")
            self.controller = AdaptiveVerifyWidth(cfg=self.cfg, curve=None)
            return
        self.boot_runner = target_model_runner.decode_cuda_graph_runner
        for width in self.controller.widths:
            if width == self.block_size:
                continue
            self.states[width] = _capture_width_state(
                width=width,
                target_model_runner=target_model_runner,
                server_args=server_args,
                build_attn_backend=build_attn_backend,
                runner_cls=runner_cls,
                tp_rank=self.tp_rank,
            )

    # ---- per round ---------------------------------------------------------

    def begin_forward(self, *, target_model_runner, is_decode: bool) -> None:
        """Every forward: undo a width swap a raised round left behind, and
        break the round-period chain on a non-decode forward (prefill / idle
        rounds in between would be billed to the decode rows)."""
        self.restore(target_model_runner=target_model_runner)
        if not is_decode:
            self.last_round = None

    def begin_round(self, *, target_model_runner, batch_size: int, eligible: bool) -> int:
        """Plan this decode round's width and activate its state."""
        if not self.enabled:
            return self.block_size
        self._measure_period(batch_size=int(batch_size))
        self.round_idx += 1
        self._consume(upto=self.round_idx - 2)
        width = self.controller.next_width(batch_size=batch_size)
        if self.controller.census_due():
            self._census()
        if width != self.block_size and (not eligible or width not in self.states):
            # Not eligible (grammar / logprobs) or no state captured on this
            # path (producer form, graphs off): today's round. Both are
            # batch/boot properties, hence rank-uniform.
            self.ineligible_ct += 1
            width = self.block_size
        if width != self.block_size and target_model_runner.decode_cuda_graph_runner is not self.boot_runner:
            # A flip/sleep replaced the boot runner since the ladder was
            # captured: our graphs may hold stale pool addresses. Group-wide
            # event -> the fallback is the same on every rank.
            self.stale_ct += 1
            width = self.block_size
        self.width = width
        self.last_round = (int(batch_size), int(batch_size) * width, time.perf_counter())
        if width != self.block_size:
            state = self.states[width]
            self.saved = (
                target_model_runner.attn_backend,
                target_model_runner.decode_cuda_graph_runner,
            )
            target_model_runner.attn_backend = state.attn_backend
            target_model_runner.decode_cuda_graph_runner = state.graph_runner
        return width

    def record(self, *, commit_lens: torch.Tensor, width: int) -> None:
        if not self.enabled or not self.controller.active:
            return
        host = torch.empty(
            commit_lens.shape, dtype=commit_lens.dtype, pin_memory=commit_lens.is_cuda
        )
        host.copy_(commit_lens, non_blocking=commit_lens.is_cuda)
        event = None
        if commit_lens.is_cuda:
            event = torch.cuda.Event()
            event.record()
        self.pending.append(
            PendingObservation(round_idx=self.round_idx, width=int(width), commit_lens=host, event=event)
        )

    def restore(self, *, target_model_runner) -> None:
        if self.saved is None:
            return
        target_model_runner.attn_backend, target_model_runner.decode_cuda_graph_runner = self.saved
        self.saved = None

    # ---- internals -------------------------------------------------------

    def _measure_period(self, *, batch_size: int) -> None:
        """Round period (begin of the previous decode round -> now) for the
        previous round's rows, when it ran at the same bs with no other
        forward between (NF: window period, verify launch to verify launch)."""
        last = self.last_round
        if last is None or last[0] != batch_size or not self.controller.active:
            return
        ms = 1000.0 * (time.perf_counter() - last[2])
        fit = self.local_fits.get(batch_size, RowCostFit())
        self.local_fits[batch_size] = fit_observe(fit=fit, rows=last[1], ms=ms)
        acc = self.period_sum.setdefault(batch_size, [0.0, 0])
        acc[0] += ms
        acc[1] += 1

    def _local_coefs(self) -> list[tuple[int, float, float]]:
        out = []
        for bs, fit in sorted(self.local_fits.items()):
            coef = fit_coef(fit=fit, min_spread=bs)
            if coef is not None:
                out.append((bs, coef[0], coef[1]))
        return out

    def _consume(self, *, upto: int) -> None:
        while self.pending and self.pending[0].round_idx <= upto:
            obs = self.pending.popleft()
            if obs.event is not None:
                obs.event.synchronize()
            self.controller.observe(
                num_correct_drafts=[int(c) - 1 for c in obs.commit_lens.tolist()],
                width=obs.width,
            )

    def _census(self) -> None:
        """Rank check + group adoption of rank 0's cost fits, then the line.

        One all_gather of (round, digest, local fits): digests must match
        (else crash-stop), and every rank installs the SAME fits (rank 0's)
        in the same round, so the rank-local wall clock never splits k.
        """
        box = _gather_census(
            payload=(
                self.round_idx,
                state_digest(state=self.controller.state),
                self._local_coefs(),
            )
        )
        if any(b[:2] != box[0][:2] for b in box):
            raise RuntimeError(
                f"ADAPTIVE-K RANK DIVERGENCE at round {self.round_idx}: "
                f"{[b[:2] for b in box]} -- crash-stop (ranks would replay "
                "different verify graphs)"
            )
        self.controller.adopt(fits=box[0][2])
        if self.tp_rank == 0:
            periods = " ".join(
                f"bs{b}:{v[0] / v[1]:.1f}({v[1]})" for b, v in sorted(self.period_sum.items()) if v[1]
            ) or "-"
            logger.info(
                "%s round_ms=%s stale_fallback=%d ineligible=%d states=%s",
                self.controller.census(),
                periods,
                self.stale_ct,
                self.ineligible_ct,
                sorted(self.states),
            )
        self.period_sum = {}


def _gather_census(*, payload: tuple) -> list:
    """all_gather over the TP cpu group; ``[payload]`` without a group."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return [payload]
    from flliper.srt.distributed import get_tp_group

    group = get_tp_group().cpu_group
    if dist.get_world_size(group) == 1:
        return [payload]
    box: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(box, payload, group=group)
    return box


@contextlib.contextmanager
def _capture_override(*, server_args, width: int):
    backup = server_args.speculative_num_draft_tokens
    server_args.override("adaptive_k.capture_override", speculative_num_draft_tokens=width)
    try:
        yield
    finally:
        server_args.override("adaptive_k.capture_restore", speculative_num_draft_tokens=backup)


def _capture_width_state(
    *, width: int, target_model_runner, server_args, build_attn_backend, runner_cls, tp_rank: int
) -> VerifyWidthState:
    from flliper.srt.utils import get_available_gpu_memory

    tic = time.perf_counter()
    device, gpu_id = target_model_runner.device, target_model_runner.gpu_id
    before = get_available_gpu_memory(device, gpu_id, distributed=False)
    saved_backend = target_model_runner.attn_backend
    with _capture_override(server_args=server_args, width=width):
        backend = build_attn_backend(target_model_runner)
        runner = runner_cls(
            target_model_runner,
            attn_backend=backend,
            speculative_num_draft_tokens=width,
        )
    target_model_runner.attn_backend = saved_backend
    after = get_available_gpu_memory(device, gpu_id, distributed=False)
    if tp_rank == 0:
        logger.info(
            "ADAPTIVE-K captured verify width %d: %.2fs, %.0f MiB",
            width,
            time.perf_counter() - tic,
            (before - after) * 1024.0,
        )
    return VerifyWidthState(width=width, attn_backend=backend, graph_runner=runner)
