# NOTE_515 — the prefill CUDA graph control arm, and what it settles

Status: control arm executed on GPU 2026-08-05. One control still missing
(boot-to-boot content floor, §6). Nothing un-gated, no production change.

Artifacts:
`/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w1/` (small vehicle)
and `.../2026-08-05_prefill_graphs_w2/` (production recipe).
Harness: `tests/prefill_graphs/`.

**Measurement conditions.** GPU power caps on this rig were recently reduced:
both RTX 3080s from 320 W to **200 W**, the RTX 5090 from 525 W to **400 W**
(verified at the hardware — `power.limit` reads 200/400/200 against
`power.default_limit` 320/575/320). Every number in this note is a same-rig,
same-cap A/B: both arms of every comparison ran under these caps, so the
deltas and the eager-vs-eager floors are unaffected. **Comparisons against
archive numbers taken before the change — the #320 Messbuendel tables in
particular — are confounded and must not be read as like-for-like.** The
absolute tok/s in §5 are therefore valid against each other and against
nothing else. `window3_boot_floor.sh` prints the measured caps in its report
header for the same reason.

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

### 1a. The recorded artifacts settle it — the prefill backend was never on trial

`/spinning/gpu-battery-results/2026-08-03_452_arms/` is the #452 B2 follow-up,
and its own results file says so outright (`B2_RESULTS.md:90`): the fold of the
treatment was the **decode** backend (`disabled` -> `full`), while

> `prefill=PhaseConfig(backend='disabled', …)` is **byte-identical in both**

Both #452 arms ran with the prefill graph backend OFF, on
`Qwen3.6-35B-A3B-AWQ-4bit` (an MoE model), not the dense production
checkpoint. No #452 arm ever captured a prefill graph.

Worse for the original B2 claim, that window's oracle was invalid.
`B2_RESULTS.md:254` §9 declares the greedy-text half of B2 **UNANSWERABLE**:
three identical greedy requests per arm produced **3/3 distinct** hashes in
*both* arms, so neither arm was internally deterministic. Recomputing from the
raw files confirms it — and shows how badly:

| comparison | first differing char |
|---|---|
| eager_1 vs eager_2 (same boot) | 568 |
| eager_1 vs eager_3 (same boot) | 582 |
| graphs_1 vs graphs_2 (same boot) | 478 |
| eager_1 vs graphs_1 (across arms) | **748** |

The cross-arm divergence is *later* than the within-arm divergence. That run
could not have demonstrated a graph-vs-eager effect of any size. Its setup —
one long prompt, 192 new tokens — sat well above the ~109-token GDN prefill
reproducibility ceiling, which is precisely why §4's harness keeps every probe
underneath it. §9 prescribed the fix (`--enable-deterministic-inference` plus a
pinned `random_seed`), and that is what `window3_boot_floor.sh` now runs.

So the "systematic from character 5" headline rests on the earlier
2026-08-02 window, which carried **no** A-vs-A floor at all — only
`decode_eager1.json` and `decode_graphs.json`, with no eager repeat.

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

## 5a. Window 3 — the floor closes, and the throughput claim is WITHDRAWN

Artifacts: `/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w3/`.
Arms E1 (eager), E2 (eager, identical flags), G (graphs), pinned
`--random-seed 12345`. Perf points rebuilt to the sustained standard: warmup
discarded, ~5 s measured window of back-to-back draws, scored on aggregate
tok/s, with SM clock and P-state sampled *during* each measured window.

**Content — the missing control, now taken, and it holds.**

| comparison | result |
|---|---|
| **E1 vs E2 (boot-to-boot floor)** | **PASS — 8/8 byte-identical** |
| E1 vs G | FAIL — 4/8 text, 3/8 logprob-only |
| E2 vs G | FAIL — 4/8 text, 3/8 logprob-only, *same prompts, same char offsets* |

Two independent eager boots are byte-identical, and the graph boot diverges
from **both** of them at identical positions. §4's divergence is real,
reproducible and attributable to the prefill graph backend — not boot noise.
This is the control `NOTE_452` §2 experiment 2 asked for, and it passes.

**Throughput — withdrawn as confounded.** The raw numbers looked like a clean
negative (256: -4.3 %, 900: -1.8 %, 1900: -3.0 % against eager floors of
0.6 %, 0.8 %, 0.4 %). The clock telemetry that the same directive required
then falsified the reading. The 3080 median SM clock decays **monotonically
with arm order**, and G ran last every time:

| point | E1 | E2 | G |
|---|---|---|---|
| 256 | 1886 MHz | 1838 (-2.6 %) | 1774 (-6.0 %) |
| 900 | 1898 | 1830 (-3.6 %) | 1762 (-7.1 %) |
| 1900 | 1875 | 1822 (-2.8 %) | 1740 (-7.2 %) |
| 256c4 | 1860 | 1804 (-3.0 %) | 1740 (-6.5 %) |

The decay is aligned with run order, and it is larger than every throughput
delta measured. The experiment cannot support a directional claim in either
direction, and the §5 sentence "no measurable gain" now rests on window 2, not
on this one.

**Retracted (2026-08-05).** An earlier revision of this section argued that
because G's throughput deficit was smaller than its clock deficit, "per-MHz the
graph arm is not behind at all". That inference is invalid and is withdrawn.
At a *fixed* power limit -- and the caps were identical across every run -- a
lower clock usually means HIGHER load, not a disadvantage: a power-limited card
downclocks precisely when it is doing more work per cycle. Low clock with high
power draw is a busy card; low clock with low power is an idle one. Clock alone
therefore proves nothing in either direction, and normalising throughput by
clock is not a legitimate operation. The withdrawal of the throughput claim
stands, but for the plain reason that the numbers are confounded -- not because
of any per-MHz argument.

Standing rule adopted from this (applies to all later windows): score on **ms
per fixed unit of work**, sample **clock and power together**, read them
**jointly and only as diagnostic annotation of load character**, and never use
either as a validity criterion or a normaliser. Validity is carried by the
interleaved pairing and the A-vs-A floor.

The 256c4 point is separately unusable: its eager floor is +9.1 % (E1 1511.8
vs E2 1649.6), swamping any effect.

**Fix for the follow-up window**, built and smoke-tested at desk as
`tests/prefill_graphs/window4_interleaved.sh` with the ticket in
`TICKET_prefill_graphs_window4.md`:

* interleave the arms (E, G, E, G, E, G) so slow drift loads on both
  treatments instead of being aliased onto one -- drift elimination is
  interleaving's only job here;
* score **ms per fixed unit of work** (byte-identical prompt set, same count,
  same order, verified equal across arms), paired G_i against E_i;
* two workload points only, each sized into the 5-20 s band, because the
  spread is saturated there and more points only inflate the window;
* clock and power sampled together, printed as annotation, never a gate.

The 256c4 point's +9.1 % "floor" was also diagnosed and was **my harness's
fault, not the server's**: E1 and E2 had done byte-identical work (32 draws,
8523 tokens) and differed only by 0.471 s of drain tail inside a
tokens-over-wall denominator. Scored on fixed work both give exactly
1704.6 tok/s -- a 0.0 % floor. The point is kept, with the denominator fixed.

**ED / GD did not run.** Both determinism arms failed to boot: OOM during
capture, `boot_ED.log` showing `avail mem=0.05 GB` on TP1 followed by
`multimem all-gather disabled (CUDA driver error: out of memory)`.
The cause is now derived rather than guessed:
`--enable-deterministic-inference` rewrites the flashinfer workspace from the
384 MiB default (`environ.py:1270`) to 2048 MiB
(`flashinfer_backend.py:1007`), a delta of **1664 MiB per rank process**. The
served arch `Qwen3_5ForConditionalGeneration` is not on the 512 MiB Qwen list
at `flashinfer_backend.py:987`, so the baseline really is 384 MiB. Since
`TERM_ATTN_WORKSPACE` is not budget-funded (`engine.py:105`), that demand comes
out of reserve headroom. The failed boot confirms it: E1 consumed 2.49 -> 1.91
GB between pool end and capture begin, ED 2.48 -> 0.30 GB, a 1649 MiB extra
demand against 1664 MiB derived, with identical KV pools.

The follow-up window must therefore boot with
**`RESERVE="7164,5464,5464"`** (= 5500+1664, 3800+1664, 3800+1664). No code
change is needed. See `TICKET_prefill_graphs_window4.md` Part B, which also
records the one thing the VRAM ledger cannot express yet: `engine.py:293` reads
`SGLANG_FLASHINFER_WORKSPACE_SIZE` at ledger-build time, but the backend
rewrites that variable afterwards, so `TERM_ATTN_WORKSPACE` does not move when
`enable_deterministic_inference` moves. The missing piece is a resolver
replicating the backend's own rule, plus that flag in the term's declared
inputs.

## 6. What is still missing

~~The boot-to-boot content floor.~~ **CLOSED by §5a**: E1 vs E2 passed 8/8
byte-identical and the graph arm diverges from both eager boots identically.
The content result stands, and it is clock-independent — byte-identity does
not care what frequency the cards ran at, which is precisely why it survived
the confound that took down the throughput number.

Still open:

* **A throughput answer at all.** §5a withdrew window 3's; window 2's survives
  only as "no measurable gain", and both need the interleaved-arm redesign.
* **The determinism pair**, blocked on capture OOM (§5a), needs a larger 3080
  reserve.
* **The barlink row** (§6a), blocked on `BARLINK_VERDICT` — barlink-583's
  first soak was a null result (thousands of collectives carried, deadlines
  never approached, abort path unexercised), so the row stays UNTESTED.

Weak supporting evidence meanwhile: the restored production server (a
different boot, radix + hicache on) reproduced the eager arm's prompt#0
continuation past character 45, exactly where the graph arm diverged. Two
differently-configured eager boots agreeing there is suggestive, not a
control.

The same script also runs an `--enable-deterministic-inference` pair, which is
a sharper test of §4's mechanism than it first appears. Deterministic
inference is legal under BCG — it appears in the tc_piecewise rule list
(server_args.py:8455) but *not* in the breakable one (:8487) — and it sets
`enforce_disable_flashinfer_allreduce_fusion = True` (:11515), i.e. it pins
the collective reduction order. §4 claims the padding epsilon is *amplified by
the collective path*. So:

* if graphs and eager agree under determinism, the amplification story is
  confirmed and a byte-strict content gate on this recipe is reachable;
* if they still diverge, the epsilon is entering before the collectives
  (padded shapes selecting different GEMM tiles), the gate can only ever be
  distribution-level, and §7 hardens from a recommendation into a conclusion.

## 8. Untested, and deliberately not shipped

Everything above exercises **text-only** batches. `can_run_graph`
(`runner/prefill_cuda_graph_runner.py:615`) has no multimodal per-batch guard,
so forcing `--cuda-graph-backend-prefill` on a multimodal model would also
send image/video batches through the captured path — which is exactly what the
upstream rule calls a fault. A `contains_mm_inputs()` guard there
(`forward_batch_info.py:1019` already provides the predicate) would turn that
footgun into a safe text-only fast path. It is *not* implemented here: with no
measurable throughput to win (§5), shipping a guard for a path nobody should
turn on would be untested code defending an unused door.

## 6a. The barlink hypothesis — the one live reason §5 might not be the end

User directive, 2026-08-05: the claim to prove or refute is that a captured
prefill only pays **in combination with barlink**. The mechanism is specific
and falsifiable. §5 measured graphs against NCCL, whose collectives are
host-driven: the launch train a graph would tighten is not the part that
costs, so a captured prefill has nothing to recover. Barlink's device-side
collectives can sit *inside* the captured graph, so the 68-75 % collective
share of the prefill window (#252) becomes reachable in a way it is not under
NCCL. If that is right, §5's flat result is a property of the transport, not
of prefill graphs.

`window3_boot_floor.sh` therefore runs a 2x2 of transport x prefill-backend,
staged via `STAGE=nccl|barlink|all`:

|  | prefill eager | prefill graphs |
|---|---|---|
| **NCCL** | E1, E2 | G |
| **barlink** | BE1, BE2 | BG |

Each transport carries **its own** eager-vs-eager floor (E1/E2 and BE1/BE2);
a graph delta in either row counts only against the floor of its own row, and
the reported swing counts only if it clears both. Points are the long-prompt
1900 and the bs>1 short-prompt concurrency mix (256 tokens, 4 in flight,
scored on `aggregate_tok_s` because per-request rates understate a concurrent
regime).

Two preconditions are **enforced in the script, not assumed**: the barlink
stage refuses to run unless `b001d102fa` (the #583 tripped-spin-kernel /
CUDA-context fix) is an ancestor of HEAD, and unless the operator passes
`BARLINK_VERDICT=confirmed` from barlink-583's live-load repro window. If that
window finds a second killer, the NCCL stage stands alone and the barlink row
becomes a follow-up — the report degrades to "hypothesis UNTESTED at this
point" rather than guessing.

Note the content axis does not get a free pass here either: the barlink row
has its own content gate, because a transport change is exactly the kind of
thing that moves reduction order.

## 7. Recommendation

Do not enable the prefill graph backend on the production recipe **under
NCCL**. It cannot pass a content-identity gate, and it buys nothing measurable
in the regime production actually runs. The current `disabled` state is right
for the wrong reason — worth knowing, because the reason is one upstream rule
away from flipping.

This recommendation is scoped to the transport it was measured on. §6a is the
open question that could reopen it, and it is a real one: if the barlink row
shows a graph delta that clears its own floor, the correct conclusion becomes
"prefill graphs are a barlink feature", not "prefill graphs are worthless" —
and the content gate then becomes the thing worth spending effort on rather
than a reason to stop.
