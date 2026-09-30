#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Messprogramm fuer MoE-Dispatch/Combine: bar1ep gegen DeepEP gegen torch.

NICHT AUSGEFUEHRT
-----------------
Dieses Programm ist geschrieben und **uebersetzt** (Syntax, Importe), aber
nicht gelaufen: die Karten hielt waehrend der Entstehung ein anderer Lauf.
Jede Zahl, die es ausgibt, ist damit noch keine gemessene Zahl.

WAS ES VERGLEICHT -- und was der Vergleich wirklich sagt
--------------------------------------------------------
Drei Varianten, **verschraenkt in derselben Schleife** (A,B,C,A,B,C,...),
damit Takt, Temperatur und Nachbarlast alle gleich treffen:

``bar1ep``
    ``Bar1EPDispatcher``. Zaehlwerte ueber die CPU-Gruppe, Nutzlast und
    Metadaten ueber ``bar1_all_to_all`` direkt in die BAR1-Apertur des Ziels.

``torch``
    **Dieselbe Buchhaltung**, andere Leitung: eine Unterklasse, die nur
    ``_a2a_zeilen`` durch ``torch.distributed.all_to_all_single`` ersetzt.
    Das ist Absicht. Ein Vergleich gegen einen fremd gebauten
    Referenz-Dispatcher haette Sortierung, Metadatenpackung und Reduktion
    mitgemessen; hier unterscheidet sich genau eine Sache -- der Weg der
    Bytes. Wer den Unterschied "Dispatcher gegen Dispatcher" will, nimmt die
    DeepEP-Variante.

``deepep``
    ``DeepEPDispatcher``, also die Bibliothek. **Nachgesehen, nicht
    angenommen**: im Abbild ``htsglang-qwen35-gguf:cu130-3e76cbbf1`` ist
    ``deep_ep`` nicht installiert (``importlib.util.find_spec('deep_ep')``
    liefert ``None``; ebensowenig ``mooncake``, ``nixl``, ``mori``). Diese
    Variante wird dort also **tot** gemeldet, mit Grund -- und es wird nichts
    an ihrer Stelle gemessen. Eine ersatzweise gemessene Variante waere eine
    Zahl unter falschem Namen.

REGELN, DIE HIER GELTEN
-----------------------
* **Korrektheit vor Zeit, und in JEDER Runde.** Gemessen wird nur, was in
  derselben Runde die Probe besteht. Die Probe ist geschlossen und
  ordnungsunabhaengig: mit einem Identitaets-"Experten" muss
  ``combine(dispatch(x))`` gleich ``x' * (Zahl der Raenge, die fuer dieses
  Token zustaendig sind)`` sein, wobei ``x'`` die lokal quantisierte Form von
  ``x`` ist (bei bf16 ist ``x' == x``). Sie vergleicht nicht die
  Empfangsreihenfolge -- die darf zwischen Bibliotheken verschieden sein --
  sondern das Ergebnis.
* **Vorlauf mindestens drei Sekunden** je Variante und je Form, nicht eine
  feste Rundenzahl: die JIT-Uebersetzung von Triton/DeepGEMM faellt sonst in
  die Messung.
* **Tote Varianten mit Grund melden.** Nie ersatzweise messen, nie
  stillschweigend weglassen.
* Der Zaehlwerteabgleich vor dem Datenpfad ist ein Host-Kollektiv (bei
  DeepEP steckt er im Kernel, hier in ``dist.all_gather``). Er ist **in der
  Messung enthalten**, weil er zum Dispatch gehoert; die Aufteilung
  Vorbereitung/Datenpfad steht in der Ausgabe getrennt.

AUFRUF
------
::

    FLLIPER_BARLINK=1 FLLIPER_BARLINK_TRANSPORT=bar1 \\
    torchrun --nproc_per_node=3 benchmark/bench_moe_dispatch.py \\
        --hidden 4096 --experts 24 --topk 8 \\
        --tokens 128,512,2048 --distribution gleich,skewed

``FLLIPER_BARLINK`` und ``FLLIPER_BARLINK_TRANSPORT`` muessen **vor** dem Aufbau
der Prozessgruppe stehen -- der ``GroupCoordinator`` liest sie dort. Das
Programm prueft das und bricht mit Grund ab, statt eine gloo-Ebene zu messen
und sie ``bar1ep`` zu nennen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Aufbau
# ---------------------------------------------------------------------------


def _env_int(name: str, fallback: int) -> int:
    return int(os.environ.get(name, str(fallback)))


def build_env(tp_size: int, rank: int, local_rank: int):
    """Verteilte Umgebung und TP-Gruppe, wie der Scheduler sie baut."""
    from flliper.srt.distributed import parallel_state

    torch.cuda.set_device(local_rank)
    parallel_state.init_distributed_environment(
        world_size=tp_size,
        rank=rank,
        distributed_init_method=os.environ.get(
            "FLLIPER_BENCH_INIT", "env://"
        ),
        local_rank=local_rank,
        backend="nccl",
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        # EP spannt die TP-Gruppe -- dieselbe Festlegung, die
        # arg_groups/overrides.py:_a2a_ep_size fuer jede a2a-Backend trifft.
        expert_model_parallel_size=tp_size,
    )
    return parallel_state.get_tp_group()


def set_moe_flags(a2a: str) -> None:
    """Genau die Felder, die ``initialize_moe_config`` setzt -- ohne ServerArgs.

    ``initialize_moe_config`` braucht ein vollstaendiges ``ServerArgs`` (also
    einen Modellpfad); hier wird nur der MoE-Teil gebraucht. Die Felder sind
    aus ``layers/moe/utils.py:initialize_moe_config`` uebernommen.
    """
    from flliper.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend, MoeRunnerBackend
    from flliper.srt.runtime_context import get_flags

    moe = get_flags().moe
    moe.a2a_backend = MoeA2ABackend(a2a)
    moe.runner_backend = MoeRunnerBackend("auto")
    moe.speculative_runner_backend = MoeRunnerBackend("auto")
    moe.speculative_a2a_backend = MoeA2ABackend(a2a)
    moe.deepep_mode = DeepEPMode("normal")
    moe.deepep_config = ""
    moe.tbo_enabled = False
    moe.sbo_enabled = False
    moe.tbo_token_distribution_threshold = 0.0
    moe.disable_fp4_allgather = False
    moe.quantization = None


# ---------------------------------------------------------------------------
# Varianten
# ---------------------------------------------------------------------------


def _torch_reference_class():
    """``Bar1EPDispatcher`` mit ``torch.distributed`` statt BAR1.

    Nur ``_a2a_zeilen`` ist ersetzt, und der Konstruktor laesst die
    BAR1-Pruefungen aus (er braucht keinen Direktpfad). Alles andere -- die
    Zerlegung, die Sortierordnung, die Metadatenpackung, die Reduktion des
    Rueckwegs -- ist Zeile fuer Zeile dieselbe. Der Unterschied in der
    Messung ist damit genau der Weg der Bytes.
    """
    from flliper.srt.layers.moe.token_dispatcher.bar1ep import Bar1EPDispatcher

    class TorchRefDispatcher(Bar1EPDispatcher):
        def __init__(self, proc_group, **kw):
            # Absichtlich NICHT super().__init__: der Elternkonstruktor
            # verlangt den BAR1-Transport und laeuft seinen Byte-Beleg. Beides
            # gehoert nicht zu einer torch-Referenz.
            from flliper.srt.layers.moe.token_dispatcher.base import BaseDispatcher
            from flliper.srt.layers.moe.token_dispatcher.deepep import (
                DeepEPPDispatchHooks,
            )

            BaseDispatcher.__init__(self)
            self.proc_group = proc_group
            self.comm = None
            self.transport = None
            self.device_group = proc_group.device_group
            self.cpu_group = proc_group.cpu_group
            self.welt = dist.get_world_size(proc_group.cpu_group)
            self.rank = dist.get_rank(proc_group.cpu_group)
            self.router_topk = int(kw["router_topk"])
            self.num_experts = int(kw["num_experts"])
            self.num_local_experts = int(kw["num_local_experts"])
            self.hidden_size = int(kw["hidden_size"])
            self.params_dtype = kw["params_dtype"]
            self.device = torch.device("cuda", torch.cuda.current_device())
            self.deepep_mode = kw.get("deepep_mode")
            self.expert_mask_gpu = None
            self.quant_config = None
            self.use_fp8 = False
            # Unbegrenzter "Schlitz": torch teilt selbst nicht auf, also gibt
            # es hier auch keine Runden. Das ist kein Vorteil, den man
            # wegrechnen muesste -- es ist der Unterschied zwischen einem
            # Fenster und keinem.
            self._slot = 1 << 62
            self._send_rows = []
            self._recv_rows = []
            self._send_index = None
            self._token_number = 0
            self._max_rows = 0
            self._output_dtype = self.params_dtype
            self._dispatch_interim = None
            self._combine_interim = None
            self._bar1_dispatch_hooks = DeepEPPDispatchHooks()
            self._set_ausgabetyp()

        def _check_window(self) -> None:
            return

        def _selftest_if_needed(self) -> None:
            return

        def _a2a_rows(self, aus, ein, send_rows, recv_rows, zeilenbytes,
                        max_rows, rows_per_round=None):
            dist.all_to_all_single(
                aus,
                ein,
                output_split_sizes=[int(n) for n in recv_rows],
                input_split_sizes=[int(n) for n in send_rows],
                group=self.device_group,
            )
            return aus

    return TorchRefDispatcher


def build_variants(args, proc_group, welt: int) -> Tuple[Dict[str, object], List[str]]:
    """Die lebenden Varianten und die Gruende der toten."""
    alive: Dict[str, object] = {}
    tot: List[str] = []

    kw = dict(
        router_topk=args.topk,
        permute_fusion=True,
        num_experts=args.experts,
        num_local_experts=args.experts // welt,
        hidden_size=args.hidden,
        params_dtype=torch.bfloat16,
        async_finish=False,
        return_recv_hook=False,
    )

    # -- bar1ep
    if "bar1ep" in args.variants:
        try:
            from flliper.srt.layers.moe.token_dispatcher.bar1ep import (
                Bar1EPDispatcher,
                bar1ep_available,
            )
            from flliper.srt.layers.moe.utils import DeepEPMode

            ok, refuse_reason = bar1ep_available(proc_group)
            if not ok:
                tot.append(f"bar1ep: {refuse_reason}")
            else:
                alive["bar1ep"] = Bar1EPDispatcher(
                    group=proc_group.device_group,
                    deepep_mode=DeepEPMode.NORMAL,
                    **kw,
                )
        except Exception as e:  # noqa: BLE001
            tot.append(f"bar1ep: {type(e).__name__}: {e}")

    # -- torch
    if "torch" in args.variants:
        try:
            alive["torch"] = _torch_reference_class()(proc_group, **kw)
        except Exception as e:  # noqa: BLE001
            tot.append(f"torch: {type(e).__name__}: {e}")

    # -- deepep. Erst nachsehen, ob es die Bibliothek ueberhaupt gibt.
    if "deepep" in args.variants:
        if importlib.util.find_spec("deep_ep") is None:
            tot.append(
                "deepep: die Bibliothek `deep_ep` ist in dieser Umgebung nicht "
                "installiert (importlib.util.find_spec liefert None). Es wird "
                "NICHTS an ihrer Stelle gemessen."
            )
        else:
            try:
                from flliper.srt.layers.moe.token_dispatcher.deepep import (
                    DeepEPDispatcher,
                )
                from flliper.srt.layers.moe.utils import DeepEPMode

                alive["deepep"] = DeepEPDispatcher(
                    group=proc_group.device_group,
                    deepep_mode=DeepEPMode.NORMAL,
                    **kw,
                )
            except Exception as e:  # noqa: BLE001
                tot.append(f"deepep: {type(e).__name__}: {e}")

    return alive, tot


# ---------------------------------------------------------------------------
# Last
# ---------------------------------------------------------------------------


class _TopK:
    """Was ein Dispatcher von ``TopKOutput`` wirklich anfasst.

    Nachgesehen: ``deepep.py:508`` und ``:673`` lesen genau
    ``topk_output.topk_weights`` und ``topk_output.topk_ids``. Mehr braucht
    weder DeepEP noch bar1ep im Normalpfad.
    """

    def __init__(self, topk_ids, topk_weights):
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights


def build_last(tokens: int, hidden: int, experts: int, topk: int,
              distribution: str, rank: int, dev, seed_value: int):
    """Eine Last, die aussieht wie eine echte.

    ``gleich``: jeder Experte gleich wahrscheinlich -- der freundliche Fall,
    in dem alle Bloecke aehnlich gross sind.

    ``skewed``: eine Zipf-aehnliche Gewichtung, die einen kleinen Teil der
    Experten den Grossteil der Token bekommen laesst. Das ist der Fall, der
    MoE-Dispatch wirklich weh tut: die Bloecke werden ungleich, ein Ziel
    bekommt ein Vielfaches der anderen, und die Rundenzerlegung greift. Ohne
    diesen Fall misst man den Sonderfall und nennt ihn Normalfall.
    """
    g = torch.Generator(device="cpu").manual_seed(seed_value + 1000 * rank)
    if distribution == "gleich":
        weight = torch.ones(experts, dtype=torch.float32)
    elif distribution == "skewed":
        # Zipf(1.0) auf einer je Rang verschobenen Expertenreihenfolge --
        # sonst waeren die heissen Experten auf allen Raengen dieselben und
        # die Schieflage traefe genau einen Zielrang statt einer Verteilung.
        rang_order = torch.randperm(experts, generator=g)
        w = 1.0 / (torch.arange(experts, dtype=torch.float32) + 1.0)
        weight = torch.empty(experts, dtype=torch.float32)
        weight[rang_order] = w
    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    if tokens == 0:
        ids = torch.zeros((0, topk), dtype=torch.int64, device=dev)
        gew = torch.zeros((0, topk), dtype=torch.float32, device=dev)
    else:
        ids = torch.multinomial(
            weight.expand(tokens, experts).contiguous(),
            topk,
            replacement=False,
            generator=g,
        ).to(dev)
        gew = torch.rand((tokens, topk), generator=g).to(dev, torch.float32)
        gew = gew / gew.sum(dim=1, keepdim=True)

    x = (
        torch.randn((tokens, hidden), generator=g, dtype=torch.float32)
        .to(dev)
        .to(torch.bfloat16)
    )
    return x, _TopK(ids, gew)


def expected_result(x, topk_ids, num_local_experts, use_fp8):
    """Die geschlossene Sollform von ``combine(dispatch(x))`` mit Identitaet.

    Ordnungsunabhaengig: sie sagt nichts darueber, in welcher Reihenfolge
    Zeilen ankommen, nur was am Ende herauskommen muss. Damit ist sie fuer
    jede Bibliothek dieselbe Probe.
    """
    if use_fp8:
        from flliper.srt.layers import deep_gemm_wrapper
        from flliper.srt.layers.quantization.fp8_kernel import (
            flliper_per_token_group_quant_fp8,
        )

        xq, _ = flliper_per_token_group_quant_fp8(
            x.contiguous(),
            128,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        basis = xq.to(torch.bfloat16)
    else:
        basis = x
    target = torch.div(topk_ids, num_local_experts, rounding_mode="floor")
    target = torch.where(topk_ids >= 0, target, torch.full_like(target, -1))
    n = torch.zeros(
        (topk_ids.shape[0], 1), dtype=torch.float32, device=x.device
    )
    for k in range(target.shape[1]):
        neu = target[:, k : k + 1]
        already = torch.zeros_like(neu, dtype=torch.bool)
        for j in range(k):
            already |= target[:, j : j + 1] == neu
        n += ((neu >= 0) & ~already).to(torch.float32)
    return (basis.to(torch.float32) * n).to(torch.bfloat16)


def one_round(d, x, topk, events) -> Tuple[torch.Tensor, float, float]:
    """Ein Dispatch, ein Identitaets-"Experte", ein Combine. Getrennt gestoppt."""
    e0, e1, e2 = events
    e0.record()
    aus = d.dispatch(hidden_states=x, topk_output=topk)
    e1.record()
    y = aus.hidden_states
    if y.dtype == torch.float8_e4m3fn:
        y = y.to(torch.bfloat16)
    from flliper.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPNormalCombineInput,
    )

    z = d.combine(
        combine_input=DeepEPNormalCombineInput(y, aus.topk_ids, aus.topk_weights)
    )
    e2.record()
    torch.cuda.synchronize()
    return z, e0.elapsed_time(e1), e1.elapsed_time(e2)


# ---------------------------------------------------------------------------
# Messung
# ---------------------------------------------------------------------------


def messe(args, alive, proc_group, welt, rank, dev):
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
    results = []
    nle = args.experts // welt

    for distribution in args.distribution:
        for tokens in args.tokens:
            x, topk = build_last(
                tokens, args.hidden, args.experts, args.topk, distribution,
                rank, dev, args.seed_value,
            )
            names = list(alive.keys())
            want = {}
            for name, d in alive.items():
                want[name] = expected_result(
                    x, topk.topk_ids, nle, getattr(d, "use_fp8", False)
                )

            # -- Vorlauf: mindestens drei Sekunden JE VARIANTE, nicht eine
            #    feste Rundenzahl. Die JIT-Uebersetzung faellt sonst in die
            #    Messung.
            for name, d in alive.items():
                t0 = time.perf_counter()
                round_list = 0
                while time.perf_counter() - t0 < args.warmup or round_list < 3:
                    one_round(d, x, topk, events)
                    round_list += 1
                # Ueber die CPU-Gruppe, nicht ueber die Vorgabegruppe: bei
                # aktivem barlink ist die Vorgabegruppe NCCL, und auf einer
                # Gruppe ueber zwei Hersteller ist das kein langsamerer Weg,
                # sondern ein Haenger.
                dist.barrier(group=proc_group.cpu_group)

            # -- Messung, verschraenkt.
            times = {n: [] for n in names}
            errors = {n: 0 for n in names}
            deviation = {n: 0.0 for n in names}
            for _ in range(args.round_list):
                for name in names:
                    z, td, tc = one_round(alive[name], x, topk, events)
                    # KORREKTHEIT IN JEDER RUNDE -- nicht einmal am Anfang.
                    d_abs = (
                        (z.to(torch.float32) - want[name].to(torch.float32))
                        .abs()
                        .max()
                        .item()
                        if z.numel()
                        else 0.0
                    )
                    deviation[name] = max(deviation[name], d_abs)
                    if d_abs > args.bound_value:
                        errors[name] += 1
                    times[name].append((td, tc))
                dist.barrier(group=proc_group.cpu_group)

            for name in names:
                pairs = times[name]
                ds = sorted(t[0] for t in pairs)
                cs = sorted(t[1] for t in pairs)
                m = len(ds) // 2
                results.append(
                    dict(
                        variant=name,
                        distribution=distribution,
                        tokens=tokens,
                        dispatch_ms_median=ds[m],
                        combine_ms_median=cs[m],
                        sum_ms_median=ds[m] + cs[m],
                        dispatch_ms_min=ds[0],
                        combine_ms_min=cs[0],
                        round_list=len(ds),
                        fehlrunden=errors[name],
                        max_deviation=deviation[name],
                    )
                )
    return results


def reports(results, tot, args, rank):
    if rank != 0:
        return
    print()
    print("=" * 78)
    print("MoE-Dispatch/Combine -- interleaved in the same run")
    print(
        f"hidden={args.hidden} experts={args.experts} topk={args.topk} "
        f"warmup>={args.warmup}s round_list={args.round_list} bound_value={args.bound_value}"
    )
    print("=" * 78)
    if tot:
        print("DEAD VARIANTS (not measured, not replaced):")
        for refuse_reason in tot:
            print(f"  - {refuse_reason}")
        print()
    if not results:
        print("No alive variant. Nothing to report.")
        return
    header = (
        f"{'Variante':<10}{'Verteilung':<12}{'Token':>7}"
        f"{'Dispatch ms':>13}{'Combine ms':>12}{'Summe ms':>11}"
        f"{'Fehlrunden':>12}{'max|d|':>10}"
    )
    print(header)
    print("-" * len(header))
    for e in results:
        print(
            f"{e['variant']:<10}{e['distribution']:<12}{e['tokens']:>7}"
            f"{e['dispatch_ms_median']:>13.4f}{e['combine_ms_median']:>12.4f}"
            f"{e['sum_ms_median']:>11.4f}{e['fehlrunden']:>12d}"
            f"{e['max_deviation']:>10.4g}"
        )
    bad = [e for e in results if e["fehlrunden"]]
    if bad:
        print()
        print(
            "ACHTUNG: Rows with error rounds are KEINE measured values -- in these "
            "Runden stimmte das Ergebnis nicht, und eine Zeit ohne richtiges "
            "Ergebnis sagt nichts."
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ergebnisse": results, "tot": tot}, f, indent=2)
        print(f"\nJSON: {args.json}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--experts", type=int, default=24)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--tokens", default="128,512,2048")
    p.add_argument("--distribution", default="gleich,skewed")
    p.add_argument(
        "--variants", default="bar1ep,torch,deepep",
        help="Comma-separated. A variant not mentioned will neither be built "
             "noch als tot gemeldet -- sie wurde nicht gefragt.",
    )
    p.add_argument("--warmup", type=float, default=3.0,
                   help="Seconds of warmup per variant and form. Lower bound 3.")
    p.add_argument("--round_list", type=int, default=30)
    p.add_argument("--seed_value", type=int, default=1234)
    p.add_argument(
        "--bound_value", type=float, default=0.0,
        help="Maximum allowed deviation from the expected result. 0 means "
             "bitgenau -- das ist der richtige Wert fuer bar1ep und torch, "
             "weil beide in float32 summieren und danach dieselbe Rundung "
             "machen. Fuer DeepEP kann eine kleine Schranke noetig sein; wer "
             "sie setzt, sagt damit, wieviel er zu glauben bereit ist.",
    )
    p.add_argument("--json", default=None)
    args = p.parse_args()

    args.tokens = [int(t) for t in args.tokens.split(",") if t]
    args.distribution = [v for v in args.distribution.split(",") if v]
    args.variants = [v for v in args.variants.split(",") if v]
    if args.warmup < 3.0:
        print("Warmup below 3 s is not intended -- raised to 3 s.")
        args.warmup = 3.0

    welt = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if welt < 2:
        print(
            "This program measures a collective; with one rank there is "
            "keines. Aufruf ueber torchrun --nproc_per_node=N (N >= 2)."
        )
        return 2
    if args.experts % welt:
        print(
            f"--experts {args.experts} is not divisible by {welt}. Both "
            f"Dispatcher bilden Experte e auf Rang e // num_local_experts ab; "
            f"ohne gleiche Teilung gibt es diese Abbildung nicht."
        )
        return 2

    # Diese beiden liest der GroupCoordinator beim Aufbau. Sie hier zu setzen
    # waere zu spaet -- also wird nur geprueft und mit Grund abgebrochen.
    if "bar1ep" in args.variants:
        if os.environ.get("FLLIPER_BARLINK", "0") in ("0", "false", ""):
            print(
                "FLLIPER_BARLINK is not set. Without barlink there is no "
                "BAR1-Transport, und was dann liefe, waere die gloo-Ebene "
                "unter dem Namen bar1ep. Abbruch."
            )
            return 2
        if os.environ.get("FLLIPER_BARLINK_TRANSPORT", "device") not in (
            "bar1", "matrix"
        ):
            print(
                f"FLLIPER_BARLINK_TRANSPORT="
                f"{os.environ.get('FLLIPER_BARLINK_TRANSPORT')!r} ist kein "
                f"Direktpfad. bar1ep braucht 'bar1' oder 'matrix'. Abbruch."
            )
            return 2

    proc_group = build_env(welt, rank, local_rank)
    set_moe_flags("bar1ep" if "bar1ep" in args.variants else "deepep")
    dev = torch.device("cuda", local_rank)

    alive, tot = build_variants(args, proc_group, welt)
    # Tote Varianten muessen auf ALLEN Raengen dieselben sein -- sonst misst
    # ein Rang etwas, in das der andere nicht hineinlaeuft, und das Ergebnis
    # ist ein Haenger. Also einmal abgleichen, bevor irgendetwas laeuft.
    carrier: list = [None] * welt
    dist.all_gather_object(carrier, sorted(alive.keys()), group=proc_group.cpu_group)
    common_ids = set(carrier[0])
    for t in carrier[1:]:
        common_ids &= set(t)
    for name in list(alive.keys()):
        if name not in common_ids:
            tot.append(
                f"{name}: nicht auf allen Raengen verfuegbar "
                f"({[i for i, t in enumerate(carrier) if name not in t]}). "
                f"Eine Variante, die nur ein Teil der Gruppe fahren kann, ist "
                f"keine Variante, sondern ein Haenger."
            )
            del alive[name]

    results = messe(args, alive, proc_group, welt, rank, dev) if alive else []
    reports(results, tot, args, rank)
    dist.barrier(group=proc_group.cpu_group)
    return 0


if __name__ == "__main__":
    sys.exit(main())
