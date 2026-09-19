"""Graph-vs-eager decode check (19.09., Task #33 / #452 B2).

WHY. A captured decode graph on an expert-offload boot decodes text that is
coherent but context-free: the first token (from the eager prefill) is right,
every later one ignores the prompt (fn3m/fn3n/fn3o: needle MISS at 351 and
10k tokens alike; #452 B2 saw the same "divergence at character 5" on
DeepSeek-V4-Flash). Reading code did not find it; this tap measures it.

WHAT. ``SGLANG_GRAPH_EAGER_CHECK=N`` (default 0 = off): for the first N
decode steps served by a decode graph, run the SAME batch once more eagerly
right after the replay and compare (a) the next-token logits and (b) every
decoder layer's output (and the first layers' submodules), captured through
forward hooks: at capture time the hook clones the layer output INTO the
graph (a strong reference keeps the clone's pool block reserved, so the
replay refreshes it in place), at eager time it clones normally. The GDN
(conv/temporal) state and the PLE n-gram history of the request are saved
after the replay and restored after the eager pass, so the request continues
exactly as the graph left it and the comparison sees the same start state.

Every finding is one log line per layer plus ``FIRST-DIVERGENCE``; nothing
here runs unless the env is set. Diagnosis only, never a serving setting.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

ENV = "SGLANG_GRAPH_EAGER_CHECK"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)$")
_CHILD_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.([A-Za-z_][A-Za-z0-9_]*)$")
SUBMODULE_LAYERS = 4  # the first layers also record their direct children
REL_TOL = 0.05


def steps_from_env(env=None) -> int:
    env = os.environ if env is None else env
    raw = str(env.get(ENV, "0")).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def tensors_of(value: Any) -> List[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [v for v in value if isinstance(v, torch.Tensor)]
    logits = getattr(value, "next_token_logits", None)
    return [logits] if isinstance(logits, torch.Tensor) else []


def hook_names(model: torch.nn.Module) -> List[Tuple[str, str]]:
    """(role, module name) for every decoder layer and the first layers' children."""
    out = []
    for name, _m in model.named_modules():
        m = _LAYER_RE.search(name)
        if m:
            out.append((f"L{int(m.group(1))}", name))
            continue
        c = _CHILD_RE.search(name)
        if c and int(c.group(1)) < SUBMODULE_LAYERS:
            out.append((f"L{int(c.group(1))}.{c.group(2)}", name))
        elif name.endswith("embed_tokens"):
            out.append(("embed", name))
    return out


def compare(graph: Dict[str, List[torch.Tensor]], eager: Dict[str, List[torch.Tensor]],
            order: List[str], rel_tol: float = REL_TOL) -> Tuple[List[str], Optional[str]]:
    """One line per role; the first role whose max|g-e| exceeds rel_tol * max|e|."""
    lines, first = [], None
    for role in order:
        g, e = graph.get(role), eager.get(role)
        if not g or not e or len(g) != len(e):
            lines.append(f"{role}: n/a (graph={len(g or [])}, eager={len(e or [])})")
            continue
        parts, bad = [], False
        for j, (tg, te) in enumerate(zip(g, e)):
            if tg.shape != te.shape:
                parts.append(f"[{j}] shape {tuple(tg.shape)} vs {tuple(te.shape)}")
                bad = True
                continue
            tg32, te32 = tg.float(), te.float()
            diff = float((tg32 - te32).abs().max()) if tg.numel() else 0.0
            ref = float(te32.abs().max()) if te.numel() else 0.0
            parts.append(f"[{j}] maxdiff={diff:.4g} ref={ref:.4g}")
            if diff > rel_tol * max(ref, 1e-6):
                bad = True
        lines.append(f"{role}: " + " ".join(parts) + (" DIVERGES" if bad else ""))
        if bad and first is None:
            first = role
    return lines, first


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def state_deltas(pre: dict, post_g: dict, post_e: dict) -> List[str]:
    """How the graph and the eager pass each moved the recurrent state from
    the same start: a graph that leaves the state untouched (or writes another
    slot) shows graph-vs-pre=0 while eager-vs-pre is not."""
    lines = []
    for key in ("conv", "temporal", "ngram"):
        if key not in pre:
            continue
        pv, gv, ev = pre[key], post_g[key], post_e[key]
        if isinstance(pv, list):
            gp = max(_maxabs(g, p) for g, p in zip(gv, pv))
            ep = max(_maxabs(e, p) for e, p in zip(ev, pv))
            ge = max(_maxabs(g, e) for g, e in zip(gv, ev))
        else:
            gp, ep, ge = _maxabs(gv, pv), _maxabs(ev, pv), _maxabs(gv, ev)
        lines.append(f"{key}: graph-vs-pre={gp:.4g} eager-vs-pre={ep:.4g} graph-vs-eager={ge:.4g}")
    return lines


class GraphEagerCheck:
    def __init__(self, model_runner, steps: int):
        self.mr = model_runner
        self.left = int(steps)
        self.mode: Optional[str] = None  # None | "eager"
        self.cap: Dict[str, List[torch.Tensor]] = {}
        self.eag: Dict[str, List[torch.Tensor]] = {}
        self.order: List[str] = []
        self._handles = []
        self.step = 0

    def attach(self, model: torch.nn.Module) -> int:
        from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

        mods = dict(model.named_modules())
        for role, name in hook_names(model):
            self.order.append(role)

            def make(role_name: str):
                def hook(_m, _args, output):
                    ts = tensors_of(output)
                    if not ts:
                        return
                    if self.mode == "eager":
                        self.eag[role_name] = [t.detach().clone() for t in ts]
                    elif get_is_capture_mode():
                        self.cap[role_name] = [t.detach().clone() for t in ts]
                return hook

            self._handles.append(mods[name].register_forward_hook(make(role)))
        return len(self._handles)

    # ---- state save/restore around the eager pass -------------------------
    def _slots(self, forward_batch):
        pool = self.mr.req_to_token_pool
        logical = pool.get_mamba_indices(forward_batch.req_pool_indices)
        lin = getattr(self.mr.attn_backend, "linear_attn_backend", self.mr.attn_backend)
        tr = getattr(lin, "_translate_mamba_indices", None)
        phys = tr(logical) if tr is not None else logical
        return logical.to(torch.long), phys.to(torch.long)

    def _save_state(self, forward_batch):
        pool = self.mr.req_to_token_pool
        logical, phys = self._slots(forward_batch)
        mc = pool.mamba_pool.mamba_cache
        snap = {
            "conv": [c[:, phys].clone() for c in mc.conv],
            "temporal": mc.temporal[:, phys].clone(),
        }
        ng = getattr(pool, "ngram_pool", None)
        if ng is not None and getattr(ng, "context", None) is not None:
            snap["ngram"] = ng.get_context(logical).clone()
        return logical, phys, snap

    def _restore_state(self, logical, phys, snap):
        pool = self.mr.req_to_token_pool
        mc = pool.mamba_pool.mamba_cache
        for c, s in zip(mc.conv, snap["conv"]):
            c[:, phys] = s
        mc.temporal[:, phys] = snap["temporal"]
        if "ngram" in snap:
            pool.ngram_pool.set_context(logical, snap["ngram"])

    def _active(self, forward_batch) -> bool:
        return (
            self.left > 0
            and forward_batch.forward_mode.is_decode()
            and int(forward_batch.batch_size) == 1
        )

    # ---- the check ---------------------------------------------------------
    def before_graph(self, forward_batch) -> None:
        """Snapshot the request's recurrent state BEFORE the replay, so the
        eager pass can start from the same state the graph started from."""
        self._pre = None
        if not self._active(forward_batch):
            return
        try:
            torch.cuda.synchronize()
            self._pre = self._save_state(forward_batch)
        except Exception as exc:
            logger.warning("GRAPH-EAGER-CHECK pre-snapshot failed: %s: %s", type(exc).__name__, exc)

    def after_graph(self, forward_batch, ret) -> None:
        if not self._active(forward_batch) or getattr(self, "_pre", None) is None:
            return
        self.left -= 1
        self.step += 1
        try:
            self._run(forward_batch, ret)
        except Exception as exc:  # diagnosis must never kill the rank
            logger.warning("GRAPH-EAGER-CHECK step=%d failed: %s: %s", self.step, type(exc).__name__, exc)
            self.mode = None
        finally:
            self._pre = None

    def _run(self, forward_batch, ret) -> None:
        torch.cuda.synchronize()
        bs = int(forward_batch.batch_size)
        g_layers = {k: [t.clone() for t in v] for k, v in self.cap.items()}
        g_logits = ret.next_token_logits[:bs].detach().float().clone()
        buffers = getattr(self.mr.decode_cuda_graph_runner, "buffers", None)

        def _show(name):
            live = getattr(forward_batch, name, None)
            stat = getattr(buffers, name, None) if buffers is not None else None
            l = live[:bs].tolist() if isinstance(live, torch.Tensor) else live
            s = stat[:bs].tolist() if isinstance(stat, torch.Tensor) else None
            return f"{name}: live={l} static={s}"

        logical, phys, post_g = self._save_state(forward_batch)
        _l, _p, pre = self._pre
        logger.warning(
            "GRAPH-EAGER-CHECK step=%d inputs: %s | %s | %s | %s | %s | mamba slot logical=%s phys=%s | captured roles=%d",
            self.step, _show("input_ids"), _show("positions"), _show("req_pool_indices"),
            _show("seq_lens"), _show("out_cache_loc"), logical.tolist(), phys.tolist(), len(g_layers),
        )
        # eager pass on the very same batch, from the PRE-replay state
        self._restore_state(logical, phys, pre)
        torch.cuda.synchronize()
        self.eag = {}
        self.mode = "eager"
        post_e = None
        try:
            attn = self.mr.attn_backend
            attn.init_forward_metadata(forward_batch)
            out_e = self.mr.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            torch.cuda.synchronize()
            _l2, _p2, post_e = self._save_state(forward_batch)
        finally:
            self.mode = None
            self._restore_state(logical, phys, post_g)  # continue as the graph left it
            torch.cuda.synchronize()
        if post_e is not None:
            for ln in state_deltas(pre, post_g, post_e):
                logger.warning("GRAPH-EAGER-CHECK step=%d STATE %s", self.step, ln)
        e_logits = out_e.next_token_logits[:bs].detach().float()
        lines, first = compare(g_layers, self.eag, self.order)
        ag, ae = int(g_logits.argmax(-1)[0]), int(e_logits.argmax(-1)[0])
        ldiff = float((g_logits - e_logits).abs().max())
        for ln in lines:
            logger.warning("GRAPH-EAGER-CHECK step=%d %s", self.step, ln)
        logger.warning(
            "GRAPH-EAGER-CHECK step=%d logits: argmax graph=%d eager=%d %s maxdiff=%.4g | FIRST-DIVERGENCE=%s",
            self.step, ag, ae, "SAME" if ag == ae else "DIFFERENT", ldiff, first,
        )


def maybe_attach(model_runner) -> Optional[GraphEagerCheck]:
    steps = steps_from_env()
    if steps <= 0:
        return None
    chk = GraphEagerCheck(model_runner, steps)
    n = chk.attach(model_runner.model)
    logger.warning("GRAPH-EAGER-CHECK armed: %d decode steps, %d hooks", steps, n)
    return chk
