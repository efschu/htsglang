"""27B-Port von formhp-qguard-1028 (NF 052281902): jede Variante des fusionierten q-guard-Kernels, die der Wrapper
``_selector_q_guard_scatter_fused`` starten kann, KOMPILIERT fuer sm_120 -- ohne GPU.

Regel (Memory triton-kernel-compile-test-sm120-1028): ein TRITON_INTERPRET-Test fuehrt den Kernel als Python aus und kennt die
Compilerregeln nicht (DFAST scheiterte so auf der Karte, obwohl der Interpreter gruen war). Hier wird der echte Wrapper mit
aufgezeichnetem Start (``flliper.test.triton_compile_check``) ueber die Matrix gefahren, die er ANNIMMT, und jeder verschiedene Start geht
durch ``triton.compile`` (cc=120, Frontend bis ptxas, triton 3.6.0 wie im Image). Matrix: mit/ohne Zaehler (HAS_STATS), K = 1/2/5/8/16
(BK 2..16), Kandidaten-IDs int32/int64, q fp32/bf16/fp16, draft_tokens als Sicht candidates[:, 1:] (nicht zusammenhaengend) und als eigener
Puffer, B = 1/6, G = 3/7 (Teilbarkeit der Strides durch 16 ist ein Compilerschluessel), Vokabular 248320 (27B) und 61 (Schluessel ungerade),
ausgerichtete und um ein Element versetzte Zeiger.
Rot-Beweis: Kernel mit einer im Compiler unbekannten Namensbindung (siehe Mutation im Bericht) liefert die Compilermeldung woertlich."""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=300, suite="base-a-test-cpu")

import ast
import os
import subprocess
import sys
import textwrap

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python"))

_CODE = textwrap.dedent(
    """
    import itertools, torch
    from flliper.srt.speculative import dflash_utils as du
    from flliper.test import triton_compile_check as tcc
    du._qguard_kernel()                     # builds and caches the jit function; capture() swaps the cached object
    g = torch.Generator().manual_seed(0)
    def case(B, G, K, V, idt, qdt, view, shift):
        ids = torch.stack([torch.randperm(V, generator=g)[:K] for _ in range(B * G)]).view(B, G, K).to(idt)
        q = torch.rand(B, G, K, generator=g)
        q = (q / q.sum(-1, keepdim=True)).to(qdt)
        cand = torch.zeros(B, G + 1, dtype=torch.int64)
        cand[:, 1:] = ids[..., 0].to(torch.int64)
        x = cand[:, 1:] if view else cand[:, 1:].contiguous()
        buf = torch.zeros(B * G * V + 8)
        dp = buf[shift: shift + B * G * V].view(B, G, V)
        return dp, ids, q, x
    n = 0
    with tcc.capture(du, "_QGUARD_KERNEL") as rec:
        for (B, G) in ((1, 3), (6, 7)):
            for V in (248320, 61):
                for K in (1, 2, 5, 8, 16):
                    for idt, qdt in ((torch.int32, torch.float32), (torch.int64, torch.bfloat16), (torch.int32, torch.float16)):
                        for view, shift, stats in ((True, 0, True), (False, 0, False), (True, 1, True)):
                            dp, ids, q, x = case(B, G, K, V, idt, qdt, view, shift)
                            st = torch.zeros(len(du.SELECTOR_Q_GUARD_STATS), dtype=torch.int64) if stats else None
                            du._selector_q_guard_scatter_fused(draft_probs=dp, candidate_ids=ids, q_rows=q, draft_tokens=x, stats=st)
                            n += 1
    res = tcc.compile_all(rec, cc=120)
    bad = [(k, e) for k, e in res if e is not None]
    print("RES", [n, len(rec), len(res), [k + " -> " + e for k, e in bad][:4]])
    """
)


def test_every_qguard_variant_compiles_for_sm120():
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=_PY)
    p = subprocess.run([sys.executable, "-c", _CODE], env=env, capture_output=True, text=True, timeout=1800)
    out = [l for l in p.stdout.splitlines() if l.startswith("RES ")]
    assert out, p.stdout[-2000:] + p.stderr[-3000:]
    n_calls, n_launch, n_variants, bad = ast.literal_eval(out[0][4:])
    assert n_calls > 0 and n_launch == n_calls and n_variants >= 5, (n_calls, n_launch, n_variants)
    assert bad == [], bad


_BROKEN = textwrap.dedent(
    '''
    import sys
    import torch, triton, triton.language as tl
    from flliper.test import triton_compile_check as tcc

    @triton.jit
    def _k(x_ptr, FLAG: tl.constexpr, BK: tl.constexpr):
        k = tl.arange(0, BK)
        if FLAG:
            nn = 1                      # bound in a static if block, used outside: the DFAST / NGK defect class
        tl.store(x_ptr + k, (k + nn).to(tl.float32))

    with tcc.capture(sys.modules[__name__], "_k") as rec:
        _k[(1,)](torch.zeros(8), FLAG=False, BK=8)
    res = tcc.compile_all(rec, cc=120)
    print("RES", [[k, e] for k, e in res])
    '''
)


def test_harness_reports_the_compiler_error_of_a_scoping_defect(tmp_path):
    """The harness is only worth its name if a kernel the interpreter runs happily but the compiler rejects (a constexpr bound inside a
    static if and used outside it -- formb164's NGK) comes back as an error."""
    src = tmp_path / "broken_kernel_mod.py"
    src.write_text(_BROKEN)
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=_PY)
    p = subprocess.run([sys.executable, str(src)], env=env, capture_output=True, text=True, timeout=600)
    out = [l for l in p.stdout.splitlines() if l.startswith("RES ")]
    assert out, p.stdout[-1500:] + p.stderr[-2500:]
    res = ast.literal_eval(out[0][4:])
    assert len(res) == 1 and res[0][1] is not None and "nn" in res[0][1], res
