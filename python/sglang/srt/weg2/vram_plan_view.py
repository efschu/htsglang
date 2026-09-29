# SPDX-License-Identifier: Apache-2.0
"""VRAM-VERTRAG M1: the launcher's budget passes, collected into ``vram_plan``.

One :class:`PlanView` per launcher run. The passes hand it what their solvers
already computed -- the budget terms of ``budgets_from_dc``, the P card fits
(``PP-CUT P-KARTE``), the D FRACTION-SOLVE fits -- and :meth:`PlanView.build`
lays them into one ``weg2.vram_plan/1`` document per pass. Nothing here
decides: a number the solvers did not compute is named (``open``) instead of
guessed, a value the profile pins by hand is listed as OVERRIDE.
"""
from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import vram_plan as vp

#: VRAM-shaping flags and envs a profile may pin by hand (--extra-*/--env-*).
#: Each one found is a planner debt: listed as OVERRIDE in every plan.
PIN_FLAGS = ("--max-total-tokens", "--rank-moe-ratio", "--rank-moe-resident-fraction",
             "--rank-user-reserve-mib", "--rank-gpu-memory-mib", "--mem-fraction-static",
             "--max-running-requests", "--context-length", "--hicache-size",
             "--max-mamba-cache-size", "--cuda-graph-bs-decode", "--chunked-prefill-size")
PIN_ENVS = ("SGLANG_MOE_SCRATCH_SLOTS", "SGLANG_MOE_RESIDENT_EXPERT_FRACTION",
            "SGLANG_WEG2_D_KV_STAGE_TOKENS", "SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS",
            "SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES", "SGLANG_HICACHE_ARENA_GIB",
            "SGLANG_HICACHE_ARENA_MAMBA_SLOTS", "SGLANG_WEG2_TAIL_KEEP_MIB",
            "SGLANG_MOE_POOL_STAGING")
#: launcher flags that set VRAM directly (ns attribute, flag)
PIN_NS = (("pp_cut_expert_device_fraction", "--pp-cut-expert-device-fraction"),
          ("pp_cut_expert_lru_rows", "--pp-cut-expert-lru-rows"),
          ("pp_stage_ratio", "--pp-stage-ratio"),
          ("d_foreign_context_mib", "--d-foreign-context-mib"),
          ("d_nontorch_mib", "--d-nontorch-mib"),
          ("user_reserve_mib", "--user-reserve-mib"))

PASS_BY_LABEL = {"P": "p_budget", "D(Karte, Erwartung)": "map", "D(dry, expectation)": "dry",
                 "D(d-only, expectation)": "d_only", "D": "d"}


def _argv_pairs(text: str) -> Dict[str, str]:
    toks = str(text or "").split()
    out: Dict[str, str] = {}
    for i, t in enumerate(toks):
        if t.startswith("--") and "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
        elif t.startswith("--"):
            vals = []
            for nxt in toks[i + 1:]:  # a list flag keeps every value (--cuda-graph-bs-decode 1 2 3)
                if nxt.startswith("--"):
                    break
                vals.append(nxt)
            if vals:
                out[t] = " ".join(vals)
    return out


def _env_pairs(text: str) -> Dict[str, str]:
    out = {}
    for part in str(text or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def profile_pins(ns) -> List[dict]:
    """The profile's hand-set VRAM values, read off the launcher's own inputs
    BEFORE any solver publishes into them (call once, at the start of main)."""
    out = []
    for group, extra, env in (("P", "extra_p", "env_p"), ("D", "extra_d", "env_d")):
        a = _argv_pairs(getattr(ns, extra, ""))
        for k in PIN_FLAGS:
            if k in a:
                out.append({"group": group, "key": k, "value": a[k], "source": "--%s" % extra.replace("_", "-")})
        e = _env_pairs(getattr(ns, env, ""))
        for k in PIN_ENVS:
            if k in e:
                out.append({"group": group, "key": k, "value": e[k], "source": "--%s" % env.replace("_", "-")})
    for attr, flag in PIN_NS:
        v = getattr(ns, attr, None)
        if v not in (None, "", [], ()):
            out.append({"group": "launcher", "key": flag, "value": str(v), "source": "argv"})
    return out


def card_dict(c) -> dict:
    return {"uuid": str(c.uuid), "nvml": int(c.nvml_index), "class": str(getattr(c, "name", "")),
            "total_mib": int(c.total_mib), "driver_reserved_mib": int(getattr(c, "reserved_mib", 0) or 0),
            "user_reserve_mib": 0}


class PlanView:
    """What the passes computed, as they computed it."""

    def __init__(self):
        self.reset()

    def reset(self, overrides: Sequence[Mapping] = (), identity: Optional[Mapping] = None):
        self.budgets: Dict[str, dict] = {}
        self.p_card: Optional[dict] = None
        self.d: Dict[str, dict] = {}
        self.asleep: Dict[str, Dict[str, dict]] = {"P": {}, "D": {}}
        self.overrides = [dict(o) for o in overrides]
        self.identity = dict(identity or {})
        self.plans: Dict[str, dict] = {}
        self.last: Optional[dict] = None

    # --- notes from the passes -------------------------------------------------
    def note_budget(self, label: str, cards, prelim: Sequence[int], final: Sequence[int],
                    terms: Sequence[Mapping]) -> None:
        rows = {}
        for i, c in enumerate(cards):
            t = dict(terms[i]) if i < len(terms) else {}
            rows[str(c.uuid)] = {
                "total": int(c.total_mib), "carve": int(t.get("carve", 0)),
                "floor": int(t.get("floor", 0)), "dormant": int(t.get("dormant", 0)),
                "growth": int(t.get("growth", 0)),
                "awake": int(t.get("awake", 0)), "awake_source": str(t.get("awake_source", "")),
                "budget_prelim": int(prelim[i]), "budget": int(final[i]),
                "corridor_lowering": int(prelim[i]) - int(final[i])}
        self.budgets[label] = rows

    def note_asleep(self, group: str, cards, mib: Mapping[str, int], provenance: str) -> None:
        self.asleep[group] = {str(c.uuid): {"mib": int(mib[c.uuid]), "provenance": str(provenance)}
                              for c in cards if c.uuid in mib}

    def note_p_card(self, cards, fits: Sequence, fractions: Sequence[float],
                    lru_rows: Sequence[int], source: str, *, num_experts: int = 0) -> None:
        self.p_card = {"fits": list(fits), "cards": list(cards),
                       "fractions": [float(x) for x in fractions],
                       "lru_rows": [int(x) for x in lru_rows], "source": str(source),
                       "num_experts": int(num_experts or 0)}

    def note_d_solve(self, label: str, cards, fits: Sequence, *, objective: str,
                     kv_tokens_max: int, seats: Optional[int], stage_rows: Mapping[int, int],
                     fixed_src: str = "", act_src: str = "") -> None:
        self.d[label] = {"fits": list(fits), "cards": list(cards), "objective": str(objective or ""),
                         "kv_tokens_max": int(kv_tokens_max or 0), "seats": seats,
                         "stage_rows": {int(k): int(v) for k, v in dict(stage_rows or {}).items()},
                         "fixed_src": fixed_src, "act_src": act_src}

    # --- the plan ----------------------------------------------------------------
    @staticmethod
    def _budget_only_rank(uuid: str, bt: Mapping, why: str) -> dict:
        """A rank whose solver left no ledger: its budget, every post named open."""
        return {"card": str(uuid), "fixed": {}, "elastic": {}, "transient_by_state": {},
                "budget_mib": int(bt.get("budget", 0)), "torch_cache_cap_mib": 0,
                "provenance": {"budget_mib": "MODEL(budgets_from_dc)",
                               "posts": "UNMEASURED(%s)" % why},
                "cells": {}}

    def _p_group(self, open_items: List[str]) -> Tuple[dict, List[vp.Closure]]:
        ranks, closure = {}, []
        pc = self.p_card
        b = self.budgets.get("P", {})
        if pc is None:
            # no P card (27B, or the card bilanz fell away): the budgets alone
            for i, u in enumerate(b):
                ranks["pp%d" % i] = self._budget_only_rank(u, b[u], "PP-CUT P-KARTE entfiel")
            open_items.append("P.closure UNMEASURED (PP-CUT P-KARTE entfiel in diesem Pass)")
            return {"ranks": ranks}, closure
        for f in pc["fits"]:
            s = int(f.stage)
            if s >= len(pc["cards"]):
                continue
            c = pc["cards"][s]
            budget = int(b.get(str(c.uuid), {}).get("budget", 0))
            ranks["pp%d" % s] = {
                "card": str(c.uuid),
                "fixed": {"experts_resident": int(round(float(f.expert_mib))),
                          "draft": int(round(float(f.draft_mib)))},
                "elastic": {"kv": {"mib": int(round(float(f.kv_mib)))},
                            "experts_lru": {"rows": int(pc["lru_rows"][s]) if s < len(pc["lru_rows"]) else 0},
                            "fraction": round(float(f.fraction), 4),
                            "ceiling_fraction": (None if f.ceiling_fraction is None
                                                 else round(float(f.ceiling_fraction), 4))},
                "transient_by_state": {"chunk=last,prompt=%d" % int(f.prompt_tokens):
                                       int(round(float(f.transient_mib) + float(f.growth_mib)))},
                "budget_mib": budget, "torch_cache_cap_mib": 0,
                "provenance": {"closure": "RECORD(%s)" % pc["source"],
                               "activation": "RECORD(%s)" % str(f.transient_source)[:120],
                               "kv": "MODEL(cell_bytes x tokens)",
                               "experts_resident": "MODEL(rows x layers x row_mib)"},
                "cells": {}}
            total = int(c.total_mib)
            # every expert of the stage resident: the buffer holds all E rows (the
            # fraction alone rounds -- -frp PP2 0.9956 is 512/512 rows)
            n_exp = int(pc.get("num_experts") or 0)
            full = (int(f.buffer_rows) >= n_exp) if n_exp > 0 else float(f.fraction) >= 0.996
            closure.append(vp.Closure(
                card=str(c.uuid), phase="P awake", state="chunk=last,prompt=%d" % int(f.prompt_tokens),
                total_mib=total,
                terms=(("p_card_booked", total - int(round(float(f.headroom_mib)))),
                       ("near_oom", int(round(float(f.near_oom_mib))))),
                row_mib=int(round(float(f.row_card_mib) or float(f.layer_row_mib))),
                rest_to="experts_resident" if not full else "none",
                bound_by=vp.bound_by_of(experts_full=full,
                                        rest_to="experts_resident" if not full else "none")))
        return {"ranks": ranks}, closure

    def _d_group(self, label: str, open_items: List[str]) -> Tuple[dict, List[vp.Closure]]:
        ranks, closure = {}, []
        d = self.d.get(label)
        b = self.budgets.get(label, {})
        if d is None:
            for i, u in enumerate(b):
                ranks["tp%d" % i] = self._budget_only_rank(u, b[u], "kein D-FRACTION-SOLVE")
            open_items.append("D.closure UNMEASURED (kein D-FRACTION-SOLVE in %s)" % label)
            return {"ranks": ranks}, closure
        asleep_p = self.asleep.get("P", {})
        for f in d["fits"]:
            r = int(f.rank)
            if r >= len(d["cards"]):
                continue
            c = d["cards"][r]
            u = str(c.uuid)
            bt = b.get(u, {})
            row_mib = int(round(float(f.n_layers) * float(f.slot_mib)))
            lru = int(f.buffer_rows) - int(f.resident_rows)
            ranks["tp%d" % r] = {
                "card": u,
                "fixed": {"weights": int(round(float(f.fixed_mib) + float(f.draft_vocab_delta_mib))),
                          "state_pools": int(round(float(f.mamba_mib) + float(f.spec_mib))),
                          "experts_resident": int(f.resident_rows) * row_mib},
                "elastic": {"kv": {"tokens": int(f.kv_tokens), "mib": int(round(float(f.kv_mib)))},
                            "experts_lru": {"rows": lru, "mib": lru * row_mib},
                            "fraction": round(float(f.fraction), 4)},
                "transient_by_state": {"n=%s,S0" % (d["seats"] or "?"): int(round(float(f.activation_mib)))},
                "budget_mib": int(round(float(f.budget_mib))), "torch_cache_cap_mib": 0,
                "provenance": {"weights": ("RECORD(%s)" % d["fixed_src"]) if d["fixed_src"]
                               else "MODEL(reference logs)",
                               "activation": ("RECORD(%s)" % d["act_src"]) if d["act_src"]
                               else "BUILTIN(reference activation)",
                               "kv": "MODEL(cell_bytes x tokens)",
                               "experts_resident": "MODEL(rows x layers x slot_mib)"},
                "cells": ({"stage_rows": int(d["stage_rows"][r])} if r in d["stage_rows"] else {})}
            used = int(round(float(f.budget_mib) - float(f.kv_rest_mib)))
            asleep = asleep_p.get(u, {}).get("mib")
            # P's residue as measured (records or the reading after sleep(P)) plus
            # its measured served growth; the booked value is in budget_terms --
            # a booking above the measurement shows up here as rest
            terms = [("driver_carve", int(bt.get("carve", 0))), ("corridor_floor", int(bt.get("floor", 0))),
                     ("asleep_P", int(asleep) + int(bt.get("growth", 0)) if asleep is not None
                      else int(bt.get("dormant", 0))),
                     ("awake_rest", int(bt.get("awake", 0))),
                     ("corridor_lowering", int(bt.get("corridor_lowering", 0))),
                     ("rank_booked", used)]
            if not bt:
                open_items.append("D.%s budget terms UNMEASURED (%s)" % (u[:12], label))
            full = int(f.resident_rows) >= int(f.local_experts)
            ctx = int(f.kv_tokens) >= int(d["kv_tokens_max"]) > 0
            closure.append(vp.Closure(
                card=u, phase="D awake", state="n=%s,S0" % (d["seats"] or "?"), total_mib=int(c.total_mib),
                terms=tuple(terms), row_mib=row_mib,
                rest_to="experts_resident" if not full else "kv",
                bound_by=vp.bound_by_of(objective=d["objective"], ctx_at_max=ctx, experts_full=full)))
        return {"ranks": ranks}, closure

    def build(self, pass_name: str, cards, *, boot_id: str = "", rev: str = "",
              d_label: Optional[str] = None) -> dict:
        open_items: List[str] = []
        groups, closure = {}, []
        g, cl = self._p_group(open_items)
        groups["P"] = g
        closure += cl
        if d_label is not None:
            g, cl = self._d_group(d_label, open_items)
            groups["D"] = g
            closure += cl
        asleep = {grp: {u: v["mib"] for u, v in per.items()} for grp, per in self.asleep.items()}
        for grp, per in self.asleep.items():
            for u, v in per.items():
                if not vp.provenance_ok(v["provenance"]):
                    open_items.append("asleep.%s.%s provenance %r" % (grp, u[:12], v["provenance"]))
        plan = vp.make_plan(
            boot_id=boot_id, rev=rev, pass_name=pass_name, identity=self.identity,
            cards=[card_dict(c) for c in cards], groups=groups, asleep=asleep,
            budget_terms=self.budgets, closure=closure, open_items=open_items,
            overrides=self.overrides)
        self.plans[pass_name] = plan
        return plan
