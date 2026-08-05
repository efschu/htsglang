# NOTE_515 — the prefill CUDA graph control arm, and what it settles

Status: control arm executed on GPU 2026-08-05. One control still missing
(boot-to-boot content floor, §6). Nothing un-gated, no production change.

Artifacts:
`/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w1/` (small vehicle)
and `.../2026-08-05_prefill_graphs_w2/` (production recipe).
Harness: `tests/prefill_graphs/`.

---

## 1. The premise was a conflation

The standing story was that the captured prefill route is gated because #452
found a content divergence there. It is not. #452's B2 (the "systematic from
character 5" divergence) and B4 (the 6.6x regression) both live in the MoE
expert-offload **capturable decode** arm — see commit dcdaeff884 and
`NOTE_452_desync_boot_refutation.md`. The production model,
Qwen3.6-27B-INT8-W8A8, is dense (`num_experts` absent from its config), so
that path is structurally absent from the production geometry. #452 never
gated the prefill route.

## 2. What actually makes production prefill eager

An upstream config-time rule, not ours:

    python/sglang/srt/server_args.py:8511
    ("multimodal model", lambda: self.get_model_config().is_multimodal)

inside `_disable_breakable_cudagraph_if_incompatible`, added by upstream
sglang PR #29458 ("Enable Breakable Cuda Graph as Default", commit
0543246184). Breakable (BCG) is the CUDA default for prefill
(`cuda_graph_config.py:95 default_prefill_backend`), and this rule knocks it
straight to `disabled`. It fires on every production boot:

    [2026-08-05 09:13:33] Breakable CUDA graph is incompatible with
    multimodal model; disabling prefill CUDA graph.

The rule is *correct*: the served checkpoint really is multimodal (333
`model.visual.*` tensors, `deepstack_visual_indexes` in `vision_config`). It
is merely coarse — a whole-server switch for a per-batch property, while
production traffic is text-only. tc_piecewise is no escape hatch: it is
independently disabled for this boot by both `multimodal model` and
`CPU offload / hierarchical cache` (server_args.py:8437-8452).

Note the consequence for anyone who later softens the multimodal rule:
production prefill would silently switch to captured, with the content
behaviour in §4. That is why §4 is recorded here rather than left implicit.

## 3. The experiment needs no source change

`--cuda-graph-backend-prefill <backend>` *locks* the prefill phase
(`server_args.py:8343`), and `_apply_cuda_graph_compatibility` returns early
for locked phases (`server_args.py:8376`). So the graph arm is the unmodified
tree plus one flag. Every result below was produced this way.

## 4. Content: graphs and eager are not bit-equivalent

Oracle: `tests/prefill_graphs/content_gate.py` — 8 greedy prompts, all short
enough to stay under the ~109-token GDN prefill reproducibility ceiling, text
plus full-precision token logprobs. **A-vs-A floor passed in every arm**
(8/8 byte-identical), so the oracle is valid.

Small vehicle — Qwen3.5-2B, same architecture, TP=1, no spec, no DCP:

| comparison | text | logprob-only |
|---|---|---|
| eager vs graphs | 0/8 | 1/8 |

The single mover was the prompt with the largest bucket pad (39 tokens padded
into the 48 bucket, ~1e-3 relative on the logprobs). The 12-token prompt,
which lands exactly on a bucket and pads by zero, was clean. Padding
magnitude, not the graph as such, is what moves the arithmetic.

Production recipe — TP=3 uneven DCP, INT8-W8A8, NEXTN 3, chunked prefill 2048:

| comparison | text | logprob-only |
|---|---|---|
| eager vs graphs | 4/8 (chars 45, 10, 13, 21) | 3/8 |

Both arms are fluent and on-topic; the divergences are near-tie flips
(`**Paris**` vs `Paris`, `" dolphin"` vs `"dolphin"`). This is the epsilon
signature `NOTE_452_desync_boot_refutation.md` §2 predicted, not corruption.

The jump from 1/8 logprob-only at TP=1 to 4/8 text at TP=3 is the finding:
the epsilon is amplified by the collective path. Padded rows change each
rank's partial sums, uneven DCP makes the shards asymmetric, and NEXTN
verification then resolves near-ties differently.

**This closes NOTE_452 §2 experiment 1.** The arms still diverge with no
offload anywhere in the picture, so B2 was never about the offload. It is
upstream graphs-vs-eager non-bit-equivalence, and the honest reading of B2
remains what that note already said: the two arms are not bit-equivalent.

## 5. Throughput: no measurable gain

`tests/prefill_graphs/prefill_perf.py`, unique ~1970-token prompts,
`max_tokens=1`, 3 warmups discarded, `cached_tokens_total == 0` verified so no
prefix hit contaminates the number:

| arm | median | mean | stdev | min | max |
|---|---|---|---|---|---|
| eager | 1725.9 tok/s | 1717.9 | 53.4 | 1645.8 | 1784.1 |
| graphs | 1692.3 tok/s | 1690.1 | 25.8 | 1638.5 | 1721.9 |

Median delta -1.9 %, against a within-arm spread of 8.0 % (eager) and 4.9 %
(graphs). **The delta is inside the noise. This is "no measurable gain", not
a regression** — claiming a 1.9 % regression from these samples would be
reading noise. Chunked prefill at 2048 is GEMM-dominated, which is the
expected reason there is nothing to win.

Capture is not free either: 31.1 s and 1.03 GB on the production recipe,
leaving 1.74 GB avail where eager had 4.00 GB.

## 6. What is still missing

The **boot-to-boot content floor**. §4 compares an eager boot against a graph
boot; the A-vs-A floors were taken *within* each boot. If two eager boots
already disagree, the 4/8 is boot noise and §4 must be withdrawn.
`NOTE_452_desync_boot_refutation.md` §2 experiment 2 flags this same gap.
`tests/prefill_graphs/window3_boot_floor.sh` runs it (eager, eager, graphs;
pinned `--random-seed`) and also takes the perf probe at 256/900/1900 tokens,
since the short-prompt regime — where launch-train tightening could plausibly
pay against the 68-75 % collective share from #252 — is unmeasured.

Weak supporting evidence meanwhile: the restored production server (a
different boot, radix + hicache on) reproduced the eager arm's prompt#0
continuation past character 45, exactly where the graph arm diverged. Two
differently-configured eager boots agreeing there is suggestive, not a
control.

## 7. Recommendation

Do not enable the prefill graph backend on the production recipe. It cannot
pass a content-identity gate, and it buys nothing measurable in the regime
production actually runs. The current `disabled` state is right for the wrong
reason — worth knowing, because the reason is one upstream rule away from
flipping.
