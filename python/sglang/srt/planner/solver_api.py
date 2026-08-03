# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""solver_api — the JSON surface of the key solver (#272).

Three payload functions, all pure reads:

  * ``key_solver_payload``           — the distribution key for one goal, a
                                       goal under constraints, or a goal pair;
  * ``key_solver_aggregate_payload`` — several coexisting instances: the
                                       additive throughput and the #260
                                       bracket that decides whether it counts;
  * ``key_solver_model_payload``     — predicted vs measured for every
                                       regression anchor, so the model error
                                       is visible.

Neither boots, allocates nor measures anything: they read the card probe and
the split-probe store from disk and ``config.json`` for the geometry. Both
are therefore safe to call against a dashboard somebody else is using, the
same contract the Guide-tab endpoints carry.

WIRING (deliberately not done here). The dashboard's ``webui.py`` dispatches
by path prefix inside ``_Handler.do_POST``; binding these needs exactly two
lines there, next to the ``/api/lever_profiles`` block::

    if self.path.startswith("/api/key_solver/model"):
        self._json(200, key_solver_model_payload(payload))
        return
    if self.path.startswith("/api/key_solver/aggregate"):
        self._json(200, key_solver_aggregate_payload(payload))
        return
    if self.path.startswith("/api/key_solver"):
        self._json(200, key_solver_payload(payload))
        return

The longer prefixes must be tested FIRST or they are swallowed by the
shorter one — the ordering rule that file already documents at several other
pairs. The UI strand owns ``webui.py`` and does the binding; this module is
complete
and callable without it (``python -m`` is not offered, the planner CLI and
the tests call the functions directly).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

from sglang.srt.planner import key_solver
from sglang.srt.planner.bench_factors import Remeasure

__all__ = [
    "key_solver_payload",
    "key_solver_aggregate_payload",
    "key_solver_model_payload",
    "key_solver_spread_payload",
    "regime_switch_payload",
    "cached_card_probe",
    "cached_hardware_profile",
]


def cached_card_probe() -> Optional[dict]:
    """The newest ``card_probe-*.json`` artifact on disk, or ``None``.

    A torn or unreadable file is skipped rather than raised on: the next
    older artifact is a usable answer, and "no probe" already has a remedy
    path in the payload below.
    """
    pattern = os.path.join(
        os.path.expanduser("~"), ".cache", "sglang", "card_probe-*.json"
    )
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("cards"):
            return dict(data)
    return None


def cached_hardware_profile() -> Optional[dict]:
    """The cached rig profile, or ``None``. Never triggers a probe.

    The solver needs it for one thing (#359): the v3 ``gemm_lanes`` map, which
    holds the quantized GEMM lanes the card probe cannot measure. Without it an
    int8 / nvfp4 / W4A16 checkpoint is scored on the dense bf16 probe -- a
    measured number on a lane the serving path will not take.
    """
    try:
        from sglang.srt import uneven_perf

        profile, _inventory = uneven_perf.get_cached_hardware_profile()
    except Exception:
        return None
    return profile or None


def _card_probe_remedy() -> dict:
    return Remeasure(
        kind="job",
        label="measure the card rates and the pair matrix",
        path="/api/card_probe",
        status_path="/api/card_probe/status",
        body={"node_id": "local"},
        cost="about 5 s over three cards; no running server is interrupted",
    ).to_json()


def _energy_model_from_payload(anchors):
    """Build an ``EnergyModel`` from the request's power anchors, or ``None``.

    ``anchors`` is a per-rank list of ``{"idle_w":, "active_w":, "source":}``
    (source "measured" for NVML-calibrated cards, "estimate-tdp" otherwise --
    the same two-tier labelling ``roofline.CardEnergy`` uses). ``None`` or an
    empty list means "no anchors supplied", which the solver reports as an
    unscorable energy request rather than filling in.
    """
    if not anchors:
        return None
    from sglang.srt.planner.objective import EnergyModel, RankPower

    return EnergyModel(
        per_rank=tuple(
            RankPower(
                idle_w=float(a["idle_w"]),
                active_w=float(a["active_w"]),
                source=str(a.get("source") or "estimate-tdp"),
            )
            for a in anchors
        )
    )


def key_solver_payload(payload: Optional[dict] = None) -> dict:
    """``POST /api/key_solver`` — the distribution key, computed.

    Body::

        {"model_path": str,                    # required
         "tp_size": int = 3,
         "rank_gpu_id": [int] = [0..tp-1],     # --rank-gpu-id space
         "rank_gpu_memory_mib": [int],         # required: the per-rank budget
         "base_plan": [int] = rank_gpu_memory_mib,
         "goal": "maxkv"|"sessions"|"dec"|"enc" = "maxkv",
         "goal_b": same set                    # -> Pareto front
         "constraints": {goal: min_value}      # -> constrained optimum
         "target_context": int = 8192,
         "roles": ["shard"|"kv_donor"|"replica"],   # pin instead of search
         "kv_cache_dtype", "speculative_algorithm",
         "speculative_num_draft_tokens", "speculative_draft_model_path",
         "max_running_requests"}

    ``goal`` alone gives one candidate; adding ``constraints`` gives the
    constrained optimum; adding ``goal_b`` gives a 3-5 point front including
    the knee. Every candidate carries every goal's predicted value — the
    sacrificed ones too — plus its trade-off line and its remeasure hook.
    """
    from sglang.srt.uneven_perf import PlanInputs

    p = dict(payload or {})
    model_path = str(p.get("model_path") or "").strip()
    if not model_path:
        return {
            "ok": False,
            "reasons": ["no model_path given"],
            "remedy": "pass the checkpoint directory or .gguf the key is for",
        }

    try:
        tp_size = int(p.get("tp_size") or 3)
    except (TypeError, ValueError):
        return {"ok": False, "reasons": ["tp_size must be an integer"]}
    if tp_size < 1:
        return {"ok": False, "reasons": ["tp_size must be >= 1"]}

    raw_ids = p.get("rank_gpu_id") or list(range(tp_size))
    try:
        rank_gpu_id = [int(x) for x in raw_ids]
    except (TypeError, ValueError):
        return {"ok": False, "reasons": ["rank_gpu_id must be a list of ints"]}
    if len(rank_gpu_id) != tp_size:
        return {
            "ok": False,
            "reasons": [
                f"rank_gpu_id has {len(rank_gpu_id)} entries but tp_size is "
                f"{tp_size}; the flag names one physical GPU per rank"
            ],
        }

    budgets_in = p.get("rank_gpu_memory_mib")
    if not budgets_in:
        return {
            "ok": False,
            "reasons": [
                "no per-rank budget given; pass rank_gpu_memory_mib. The "
                "planner's derive_auto_plan produces it from the hardware "
                "spec and the reserves, which is where the dashboard gets it"
            ],
        }
    try:
        budgets = [int(x) for x in budgets_in]
        base_plan = [int(x) for x in (p.get("base_plan") or budgets)]
    except (TypeError, ValueError):
        return {
            "ok": False,
            "reasons": ["rank_gpu_memory_mib / base_plan must be lists of ints"],
        }
    if len(budgets) != tp_size or len(base_plan) != tp_size:
        return {
            "ok": False,
            "reasons": [
                "rank_gpu_memory_mib and base_plan need one entry per rank "
                f"({tp_size})"
            ],
        }

    probe = cached_card_probe()
    if probe is None:
        return {
            "ok": False,
            "reasons": [
                "no card probe on disk — the solver has no per-card rates and "
                "will not invent any"
            ],
            "remeasure": _card_probe_remedy(),
        }

    try:
        rates = key_solver.rates_from_probe(
            probe, rank_gpu_id, hardware_profile=cached_hardware_profile()
        )
    except ValueError as e:
        return {"ok": False, "reasons": [str(e)], "remeasure": _card_probe_remedy()}

    constraints_in = p.get("constraints") or {}
    try:
        constraints: Dict[str, float] = {
            str(k): float(v) for k, v in dict(constraints_in).items()
        }
    except (TypeError, ValueError):
        return {
            "ok": False,
            "reasons": ["constraints must be a {goal: minimum value} mapping"],
        }
    unknown = [k for k in constraints if k not in key_solver.GOALS]
    if unknown:
        return {
            "ok": False,
            "reasons": [
                f"unknown constraint goal(s) {unknown}; known: "
                f"{list(key_solver.GOALS)}"
            ],
        }

    inputs = PlanInputs(
        tp_size=tp_size,
        model_path=model_path,
        kv_cache_dtype=str(p.get("kv_cache_dtype") or "auto"),
        speculative_algorithm=p.get("speculative_algorithm"),
        speculative_num_draft_tokens=p.get("speculative_num_draft_tokens"),
        speculative_draft_model_path=p.get("speculative_draft_model_path"),
        max_running_requests=p.get("max_running_requests"),
        rank_gpu_id=list(rank_gpu_id),
        effective_vram_mib=list(budgets),
    )

    try:
        answer = key_solver.solve(
            inputs,
            base_plan,
            budgets,
            rates,
            goal=str(p.get("goal") or "maxkv"),
            goal_b=(str(p["goal_b"]) if p.get("goal_b") else None),
            constraints=constraints or None,
            target_context=int(p.get("target_context") or 8192),
            roles=(list(p["roles"]) if p.get("roles") else None),
            # #350 phase 3: the objective axis. Absent/"throughput" keeps the
            # call byte-identical; "energy" needs power anchors, which the
            # caller supplies (or the solve is refused with a named reason
            # rather than answered from the throughput axis).
            objective=str(p.get("objective") or "throughput"),
            energy_model=_energy_model_from_payload(p.get("power_anchors")),
        )
    except (ValueError, KeyError, OSError) as e:
        return {"ok": False, "reasons": [str(e)]}

    out: Dict[str, Any] = answer.to_json()
    out["model_path"] = model_path
    out["rank_gpu_id"] = list(rank_gpu_id)
    out["rank_gpu_memory_mib"] = list(budgets)
    return out


def key_solver_aggregate_payload(payload: Optional[dict] = None) -> dict:
    """``POST /api/key_solver/aggregate`` — several instances at once.

    Body::

        {"gpu_total_mib": {"0": 32607, "1": 20480, ...},   # required
         "reserve_mib": {"0": 0, ...},
         "prefill_tokens": int = 20000,
         "instances": [
           {"key": str, "model_path": str, "tp_size": int,
            "rank_gpu_id": [int], "rank_gpu_memory_mib": [int],
            "base_plan": [int], "mlp_vector": [int],
            "kv_cache_dtype": str, "speculative_algorithm": str,
            "speculative_num_draft_tokens": int, "max_running_requests": int,
            "kind": str, "local": bool, "min_kv_tokens": int,
            "share_group": str|null},
           ...]}

    One mechanic for every family that adds a serving source — an extra solo
    server, a PD lane, a replicated drafter, a full replica, a foreign-rig
    satellite. The answer is the sum over the instances IF the coexistence
    bracket closes, and the naming of the overflowing GPU if it does not.

    ``share_group`` is the dual-group runtime: instances that reuse one
    resident weight set (rank reuse, nested shards) name the same group and
    their weights are counted ONCE per card. Leaving it out prices naive
    duplication, which is a different — and usually infeasible — plan.
    """
    p = dict(payload or {})
    raw_instances = p.get("instances") or []
    if not raw_instances:
        return {"ok": False, "reasons": ["no instances given"]}
    totals_in = p.get("gpu_total_mib") or {}
    if not totals_in:
        return {
            "ok": False,
            "reasons": [
                "no gpu_total_mib given; the bracket needs each card's NVML "
                "total to be able to say 'does not fit' with a number"
            ],
        }
    probe = cached_card_probe()
    if probe is None:
        return {
            "ok": False,
            "reasons": ["no card probe on disk"],
            "remeasure": _card_probe_remedy(),
        }

    try:
        gpu_total = {int(k): int(v) for k, v in dict(totals_in).items()}
        reserve = {int(k): int(v) for k, v in dict(p.get("reserve_mib") or {}).items()}
        specs = []
        for i, raw in enumerate(raw_instances):
            d = dict(raw)
            rgid = [int(x) for x in d["rank_gpu_id"]]
            specs.append(
                key_solver.InstanceSpec(
                    key=str(d.get("key") or f"instance{i}"),
                    model_path=str(d["model_path"]),
                    tp_size=int(d.get("tp_size") or len(rgid)),
                    rank_gpu_id=rgid,
                    budgets_mib=[int(x) for x in d["rank_gpu_memory_mib"]],
                    base_plan=(
                        [int(x) for x in d["base_plan"]] if d.get("base_plan") else None
                    ),
                    mlp_vector=(
                        [int(x) for x in d["mlp_vector"]]
                        if d.get("mlp_vector")
                        else None
                    ),
                    kv_cache_dtype=str(d.get("kv_cache_dtype") or "auto"),
                    speculative_algorithm=d.get("speculative_algorithm"),
                    speculative_num_draft_tokens=d.get("speculative_num_draft_tokens"),
                    max_running_requests=d.get("max_running_requests"),
                    kind=str(d.get("kind") or "serving"),
                    local=bool(d.get("local", True)),
                    min_kv_tokens=int(d.get("min_kv_tokens") or 4096),
                    share_group=(
                        str(d["share_group"]) if d.get("share_group") else None
                    ),
                )
            )
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "reasons": [f"malformed instance list: {e}"]}

    try:
        out: Dict[str, Any] = key_solver.aggregate(
            specs,
            probe,
            gpu_total,
            reserve_mib=reserve or None,
            prefill_tokens=int(p.get("prefill_tokens") or 20000),
        )
    except (ValueError, KeyError, OSError) as e:
        return {"ok": False, "reasons": [str(e)]}
    return out


def key_solver_model_payload(payload: Optional[dict] = None) -> dict:
    """``POST /api/key_solver/model`` — predicted vs measured, per anchor.

    The validation loop in one call: every regression anchor is re-derived
    from the checkpoint and the probe on disk, and reported next to the
    measurement it was built from, with the tolerance and the reason that
    tolerance is what it is. A model nobody can check against a measurement
    is not a model, it is a preference.
    """
    p = dict(payload or {})
    model_path = str(p.get("model_path") or "").strip()
    if not model_path:
        return {
            "ok": False,
            "reasons": [
                "no model_path given; the anchors are Qwen3.6-27B-FP8 points, "
                "so pass that checkpoint's directory"
            ],
        }
    probe = cached_card_probe()
    if probe is None:
        return {
            "ok": False,
            "reasons": ["no card probe on disk"],
            "remeasure": _card_probe_remedy(),
        }
    try:
        rows: List[dict] = key_solver.check_regressions(model_path, probe)
    except (ValueError, KeyError, OSError) as e:
        return {"ok": False, "reasons": [str(e)]}

    additive: Optional[dict] = None
    q3 = str(p.get("q3_gguf_path") or "").strip()
    if q3:
        try:
            additive = key_solver.check_additive_regression(
                q3,
                model_path,
                probe,
                small_model_path=(str(p.get("small_model_path") or "") or None),
            )
        except (ValueError, KeyError, OSError) as e:
            additive = {"error": str(e)}

    return {
        "ok": True,
        "model_path": model_path,
        "noise_floor_pct": key_solver.NOISE_FLOOR_PCT,
        # The floor travels with its provenance: it was measured on the
        # reference rig, and a reader on other hardware has to be able to see
        # that from the payload alone (#434).
        "noise_floor_source": key_solver.NOISE_FLOOR_SOURCE,
        "collective_efficiency": key_solver.COLLECTIVE_EFFICIENCY,
        "anchors": rows,
        "additive": additive,
        "all_within_tolerance": all(r["within_tolerance"] for r in rows),
        "all_signs_match": all(r["signs_match"] for r in rows),
        "remeasure": Remeasure(
            kind="job",
            label="re-measure an anchor vector against its prediction",
            path="/api/split_probe",
            status_path="/api/split_probe/status",
            body={"candidate": "6,1,1", "tp_size": 3, "model_path": model_path},
            cost="one cold boot to READY plus a prefill and a decode window",
        ).to_json(),
    }


def regime_switch_payload(payload: Optional[dict] = None) -> dict:
    """``POST /api/regime_switch`` -- the #363 §20.1 worth-it autocheck.

    Pure computation: it reads the phase table the caller supplies and the
    ledger numbers the caller supplies, and touches no GPU, no probe and no
    server. Body::

        {"phase_table": {...},                     # required, see
                                                   # regime_switch.phase_table_from_json
         "prefill_layout": "prefill",
         "decode_layout": "decode",
         "workload": {"prefill_tokens": 20000, "decode_tokens": 512,
                      "switches_per_round": 2},
         "geometry": {"units": 12, "bytes_per_unit": 1717986918},
         "ledger":   {"card_total_bytes": [...], "committed_bytes": [...],
                      "graph_state_bytes": [...],   # optional, ESTIMATE
                      "corridor_bytes": 419430400,
                      "pre_captured": true},
         "pair_mode": {"tolerance_pct": 2.0,
                       "candidates_prefill": [{"layout": {...},
                                               "tok_s": ..., "provenance": ...,
                                               "source": ...}, ...],
                       "candidates_decode":  [...]}}

    ``geometry`` and ``ledger`` are optional; without them the verdict is
    priced at RUNG 1 and says so, rather than assuming the cheap rung.
    ``pair_mode`` is the §20.3 secondary objective and is reported alongside
    the verdict, never folded into it.

    WIRING: the same two-line ``webui.py`` binding the module docstring
    describes for the key-solver paths, tested before the shorter prefixes.

    Nothing this returns executes: #363 slice 1 is the decision layer.
    """
    from sglang.srt.planner import regime_switch as rs

    p = dict(payload or {})
    raw_table = p.get("phase_table")
    if not raw_table:
        return {
            "ok": False,
            "reasons": [
                "no phase_table given; §20.1's whole point is that the "
                "verdict is read off the measured table, so there is nothing "
                "to decide from without one"
            ],
        }
    try:
        table = rs.phase_table_from_json(raw_table)
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "reasons": [f"malformed phase_table: {e}"]}

    prefill_name = str(p.get("prefill_layout") or "prefill")
    decode_name = str(p.get("decode_layout") or "decode")
    try:
        table.layout(prefill_name)
        table.layout(decode_name)
    except KeyError as e:
        return {"ok": False, "reasons": [str(e)]}

    workload = None
    if p.get("workload"):
        try:
            w = dict(p["workload"])
            workload = rs.WorkloadShape(
                prefill_tokens=int(w.get("prefill_tokens", 20000)),
                decode_tokens=int(w.get("decode_tokens", 512)),
                switches_per_round=float(w.get("switches_per_round", 2.0)),
            )
        except (TypeError, ValueError) as e:
            return {"ok": False, "reasons": [f"malformed workload: {e}"]}

    overlap = None
    residency = None
    geom = p.get("geometry") or {}
    ledger = p.get("ledger") or {}
    if geom:
        try:
            overlap = rs.layout_overlap(
                table.layout(prefill_name),
                table.layout(decode_name),
                units=int(geom["units"]),
                bytes_per_unit=float(geom["bytes_per_unit"]),
                active=decode_name,
            )
        except (KeyError, TypeError, ValueError) as e:
            return {"ok": False, "reasons": [f"malformed geometry: {e}"]}
    if ledger:
        if overlap is None:
            return {
                "ok": False,
                "reasons": [
                    "a ledger was given without a geometry; the rung "
                    "arithmetic needs the shard overlap to know how many "
                    "extra bytes dual residency costs"
                ],
            }
        try:
            gsb = ledger.get("graph_state_bytes")
            residency = rs.residency_rung(
                overlap,
                card_total_bytes=[float(x) for x in ledger["card_total_bytes"]],
                committed_bytes=[float(x) for x in ledger["committed_bytes"]],
                graph_state_bytes=(
                    [float(x) for x in gsb] if gsb is not None else None
                ),
                pre_captured=bool(ledger.get("pre_captured", True)),
                corridor_bytes=float(ledger.get("corridor_bytes") or 0.0),
            )
        except (KeyError, TypeError, ValueError) as e:
            return {"ok": False, "reasons": [f"malformed ledger: {e}"]}

    result = rs.autocheck(
        table,
        prefill_layout=prefill_name,
        decode_layout=decode_name,
        workload=workload,
        overlap=overlap,
        residency=residency,
    )
    out: Dict[str, Any] = {"ok": True, "phase_table": table.to_json()}
    out.update(result.to_json())

    pair = p.get("pair_mode")
    if pair:
        if not geom:
            return {
                "ok": False,
                "reasons": [
                    "pair_mode needs a geometry: the secondary objective is "
                    "shard overlap, which is measured in units and bytes"
                ],
            }
        try:
            cands_a = _pair_candidates(pair.get("candidates_prefill"), "prefill")
            cands_b = _pair_candidates(pair.get("candidates_decode"), "decode")
            out["pair"] = rs.solve_layout_pair(
                cands_a,
                cands_b,
                units=int(geom["units"]),
                bytes_per_unit=float(geom["bytes_per_unit"]),
                tolerance_pct=float(
                    pair.get("tolerance_pct", rs.DEFAULT_PAIR_TOLERANCE_PCT)
                ),
                # No `active` override: the candidates carry their own names,
                # which need not match the phase table's rows, and the decode
                # candidate is the active layout by the same standing default.
            ).to_json()
        except (KeyError, TypeError, ValueError) as e:
            return {"ok": False, "reasons": [f"malformed pair_mode: {e}"]}
    return out


def _pair_candidates(rows, phase: str):
    """Decode one phase's near-optimal candidate list for the pair objective."""
    from sglang.srt.planner import regime_switch as rs
    from sglang.srt.planner.cost_model import Provenance, Rate

    if not rows:
        raise ValueError(f"pair_mode: no candidates given for phase {phase!r}")
    out = []
    for raw in rows:
        d = dict(raw)
        ly = dict(d["layout"])
        kv = ly.get("kv_tokens")
        source = str(d.get("source") or "").strip()
        if not source:
            raise ValueError(
                f"pair_mode candidate for {phase!r} has no source; an "
                "unsourced score cannot be ranked against a measured one"
            )
        prov = Provenance(str(d.get("provenance") or "estimate"))
        if prov is Provenance.ABSENT:
            score = Rate.absent(source, unit="tok/s")
        else:
            score = Rate(float(d["tok_s"]), prov, source, "tok/s", "")
        out.append(
            rs.PhaseCandidate(
                phase=phase,
                layout=rs.LayoutVector(
                    name=str(ly["name"]),
                    weights=tuple(int(x) for x in ly["weights"]),
                    kv_tokens=(tuple(int(x) for x in kv) if kv else None),
                ),
                score=score,
            )
        )
    return out


def key_solver_spread_payload(payload: Optional[dict] = None) -> dict:
    """``POST /api/key_solver/spread`` -- the spreading decider (#274 slice D).

    WHICH CARDS each lane runs on, as opposed to what the key solver answers
    (what the KEY on a given card set should be).  Pure read: it consults the
    cached card probe for the pair matrix and computes nothing on a GPU.

    Body::

        {"cards": [0, 1, 2],                       # required
         "max_lanes_per_card": int|null,
         "lanes": [
           {"key": str,                            # required
            "label": "sm"|"bw"|"unknown",          # or give intensity+balance
            "intensity_flop_per_byte": float,      # optional, labels the lane
            "balance_flop_per_byte": float,        #   together with the above
            "cards": [int],                        # pin it, or leave out
            "card_count": int = 1,
            "allowed_cards": [int],
            "priority_class": int = 0},
           ...]}

    A lane may be labelled directly or by its two numbers; giving neither is
    allowed and yields ``unknown``, which is ranked pessimistically and
    reports no expected E rather than a guess.
    """
    from sglang.srt.planner import spread as _spread

    p = dict(payload or {})
    cards = [int(c) for c in (p.get("cards") or [])]
    if not cards:
        return {"ok": False, "reasons": ["no cards given"]}
    rows = p.get("lanes") or []
    if not rows:
        return {"ok": False, "reasons": ["no lanes given"]}

    loads: List[Any] = []
    try:
        for row in rows:
            label = str(row.get("label") or "").strip()
            intensity = row.get("intensity_flop_per_byte")
            balance = row.get("balance_flop_per_byte")
            if not label:
                label = _spread.label_from_intensity(
                    None if intensity is None else float(intensity),
                    None if balance is None else float(balance),
                )
            loads.append(
                _spread.LaneLoad(
                    key=str(row["key"]),
                    label=label,
                    cards=tuple(int(c) for c in (row.get("cards") or [])),
                    card_count=int(row.get("card_count") or 1),
                    allowed_cards=tuple(
                        int(c) for c in (row.get("allowed_cards") or [])
                    ),
                    intensity=None if intensity is None else float(intensity),
                    balance=None if balance is None else float(balance),
                    priority_class=int(row.get("priority_class") or 0),
                )
            )
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "reasons": [f"malformed lane list: {e}"]}

    probe = cached_card_probe()
    pairs = (
        _spread.pair_bandwidth_from_probe(probe, cards) if probe is not None else None
    )
    mlpc = p.get("max_lanes_per_card")
    answer = _spread.spread_plan(
        loads,
        cards,
        max_lanes_per_card=None if mlpc is None else int(mlpc),
        pair_bandwidth_gbs=pairs,
    )
    out: Dict[str, Any] = answer.to_json()
    out["objective"] = _spread.describe_objective()
    out["lanes"] = [ld.to_json() for ld in loads]
    out["pair_matrix"] = "card probe" if pairs else "absent"
    if probe is None:
        out.setdefault("reasons", []).append(
            "no card probe on disk: ties between equally-scored placements "
            "were broken arbitrarily instead of by link bandwidth"
        )
        out["remeasure"] = _card_probe_remedy()
    return out
