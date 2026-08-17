# ANALYSE #335 — compatibility surfaces: inventory, verdict, and the one delta built

Desk only, 2026-08-17. No boot, no model load, no server started (the running
30030 was not probed). Branch cut from `integration/r2`, not `main` — `main`
is the upstream mirror and not a fork target.

## 1. Inventory against the #335 list

Mount is proven by route registration in `entrypoints/http_server.py`, not by
a directory existing — the #421 lesson is that an adapter directory with no
route registration is an advertised-but-unwired feature.

| surface | verdict | evidence |
|---|---|---|
| **OpenAI (core)** | **exists, wired** | `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` (`:2149`) |
| **OpenAI images** | **exists, wired** | `/v1/images/generations`, `/edits`, `/variations` |
| **OpenAI audio** | **exists, wired** | `/v1/audio/speech`, `/v1/audio/transcriptions` |
| **OpenAI files / fine-tuning** | exists, wired | `/v1/files*`, `/v1/fine_tuning/jobs*` |
| **Ollama emulation** | **exists, wired, PARTIAL — and was silently dropping fields** | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`; `OllamaServing` mounted at `:327`. See §2. |
| **ComfyUI node pack** | **ABSENT** | one prose mention in `planner/webui.py:6808`; no node pack, no routes |
| **A1111 `sdapi`** | **BUILT 2026-08-17, see §7** | `entrypoints/sdapi/`, routes in `http_server.py` |
| **KoboldCpp** | **ABSENT** | no hit anywhere under `python/sglang/srt/` |

So the OpenAI front is materially complete, and the #335 gap is narrower than
the list suggests: one partial surface (Ollama) and three absent ones.

**Provenance worth recording:** the Ollama adapter is **upstream** code
(`31d48d7f6f`, "Add Ollama-compatible API endpoints + Smart Router (#14376)"),
extended by fork work (`fbded87cd3`, universal client liveness). It is not
something this fork built and abandoned.

## 2. The Ollama surface: mounted, but it was overruling callers silently

This is the highest-value delta and the only thing built this cut.

`OllamaChatRequest` / `OllamaGenerateRequest` declared four fields the adapter
**never read**:

- **`format`** — structured output. A caller asking for JSON got free-form
  text, then `json.loads()` it and broke. The worst of the four, because the
  failure surfaces far from its cause.
- **`think`** — never read.
- **`keep_alive`** — never read.
- and every **`options`** key outside the eight it maps —
  `repeat_penalty`, `num_ctx`, `min_p`, `mirostat`, `tfs_z`, … all dropped,
  so the model sampled differently than asked and said nothing.

That is the #710 tool-arg-loss family exactly: a value the caller supplied
that vanishes between the request and the sampler.

**Fixed by named refusal before generation**, never a silent drop and never a
plausible wrong answer. `keep_alive` is the one exception and is declared
**inert** rather than refused: SGLang keeps the model resident for the process
lifetime, so the caller's intent is already satisfied — but it is now stated
instead of looking honoured by accident.

Refusals route rather than merely block: the `format` refusal names
`/v1/chat/completions` + `response_format` as the working path.

17 pins, 10 subtests. Mutation (restoring the drops) fails 11.

### The structural finding underneath, NOT fixed here

**The Ollama front is a parallel serving path, not a composition.** It drives
`tokenizer_manager` directly and applies the chat template itself
(`serving.py`, `handle_chat`), so it never reaches the OpenAI front's
`response_format` machinery. That is *why* `format` could not simply be
wired.

Making it compose the OpenAI path is the right end state — one sampling path,
one place for structured output, one place for reasoning control — but it is a
structural change to an upstream-owned adapter, not a bug fix, and it wants
its own red-first pass. Recorded, not smuggled in.

## 3. The three absent surfaces: scope, honestly

None built this cut, and none should be until the media lanes exist as serving
surfaces (#333). Effort is *implementation* effort only; each also needs a
contract-test pass of roughly the size of §2's.

| surface | effort | what it actually needs | blocking dependency |
|---|---|---|---|
| **KoboldCpp** | **S** | `/api/v1/generate` + `/api/extra/*`; a thin text-only translation over the existing OpenAI path. The closest thing to trivially thin on this list. | none — this is the one that could be done next |
| **A1111 `sdapi`** | **M** | `/sdapi/v1/txt2img`, `/img2img`, `/options`, `/progress`; base64 image I/O and a progress model SGLang has no equivalent of | image generation as a *serving* surface (#333) |
| **ComfyUI node pack** | **L**, and it is not an HTTP surface at all | a Python node package shipped into ComfyUI's tree, plus a client against our API. Different artifact, different release path, different repo hygiene. | #333, plus a decision about shipping a second installable |

**KoboldCpp is the honest "trivially thin" candidate** the brief asked me to
flag: text-only, no media dependency, and the same translate-over-OpenAI shape
the Ollama front already demonstrates — except it should compose the OpenAI
serving path rather than repeat the Ollama front's parallel-path mistake.

## 4. What this note does not claim

No surface was exercised against a running server. The refusal surface is a
pure function of the request and is pinned as such; whether a real Ollama
client is *satisfied* by the refusals — as opposed to merely receiving them —
is a live question and belongs to a window, together with the four-endpoint
happy path.

The three absent surfaces were verified absent by grep across
`python/sglang/srt/`; that is an absence of implementation, and I did not look
for a gate refusing them, because none is claimed to exist.

---

## 5. KoboldCpp — BUILT (2026-08-17), and it demonstrates the right shape

`entrypoints/kobold/` + 30 pins. Prior-art gate first: no `kobold` hit in
FEATURE_CATALOG, `docs/`, or `git log --all --grep` outside this note.

Mounted and proven at the source, not assumed (§1's own standard):
`http_server.py` registers `/api/v1/generate`, `/api/v1/model`,
`/api/extra/version`, `/api/extra/generate/stream`, `/api/extra/abort`, and
constructs `state.kobold_serving` beside the Ollama one.

**It composes, and the pins hold that.** `handle_generate` translates into a
`CompletionRequest` and calls `openai_serving_completion.handle_request` —
there is no sampling, no template application and no tokenizer manager in this
adapter. A source pin forbids `apply_chat_template`,
`tokenizer_manager.generate_request` and `sampling_params` from appearing in
the module at all, so "thin translation" is enforced rather than intended.

**Two endpoints are refused rather than approximated**, which is the honest
half of a compatibility surface:

- **streaming** — Kobold streams via a polled `/api/extra/generate/check`
  protocol. Emitting an OpenAI-shaped stream under a Kobold URL would parse
  until it did not, mid-generation, with no error to read. A half-faithful
  stream is worse than a 501, and the refusal names `/api/v1/generate` and
  `/v1/completions`.
- **abort** — Kobold aborts *the* current generation, which presumes one user.
  This server is multi-tenant: there is no "the" generation and cancelling a
  guessed one would stop a stranger's request.

**`rep_pen` is the worked example of refuse-over-map.** Kobold's is a
multiplicative repetition penalty; the OpenAI path offers `frequency_penalty`,
an additive logit adjustment. No constant converts one into the other, so
mapping them would sample differently than asked with nothing saying so.

**Three findings came from my own pins**, which is what they are for: two
refusals (`sampler_order`, `memory`) blocked without routing or explaining and
were rewritten; and `CompletionRequest.max_tokens` defaults to **16**, so a
Kobold client omitting `max_length` would have received a 16-token fragment.
That default is now an explicit `DEFAULT_MAX_TOKENS` constant — choosing a
number is an adapter's responsibility, hiding that it was chosen is not.

## 6. What an Ollama compose-refactor would take (recorded, not built)

So the structural finding in §2 does not evaporate.

The Ollama front would have to stop calling `tokenizer_manager` and stop
applying the chat template itself, and instead build a `ChatCompletionRequest`
and hand it to `openai_serving_chat.handle_request` — the shape §5
demonstrates. Concretely: `handle_chat` maps messages straight across (they
are already OpenAI-shaped), `handle_generate` maps `prompt` onto a
`CompletionRequest` exactly as the Kobold adapter does, and
`_convert_options_to_sampling_params` disappears in favour of the OpenAI
request's own fields.

What that buys immediately: `format` becomes wirable (it is
`response_format`), `think` becomes wirable (it is the reasoning toggle the
Anthropic front already reaches via `chat_template_kwargs`), and both stop
being refusals. What it costs is the risk of the refactor itself: the Ollama
streaming response shape is Ollama's own NDJSON, so the streaming path must be
re-translated from the OpenAI SSE stream rather than produced directly — the
same faithfulness question §5 refused to guess at for Kobold, but with a shape
that IS determinable because the adapter already emits it.

Two things make it more than a rename: it is upstream-owned code, so the
change wants to be defensible to upstream rather than a fork divergence; and
its current behaviour is what existing Ollama clients have been served, so the
refactor needs the contract tests from `test_ollama_no_silent_drops_335.py`
extended to the happy path first, or it is a rewrite without a net.


---

## 7. A1111 `sdapi` — BUILT, and §3's blocker was stale

Prior-art gate first: no `sdapi` / `a1111` HTTP surface anywhere in the tree or
on any branch (`git log --all -i --grep`), and no module-name duplicate. The one
adjacent hit is `multimodal_gen/.../lora_format_adapter.py`, which handles
A1111's LoRA **file format** and not its API.

**§3 said this was blocked on "image generation as a serving surface (#333)".
That is no longer true, and the evidence is in this tree:** the OpenAI images
front is wired (`http_server.py:2216`) to a real diffusion lane through
`OpenAIServingImages` (`entrypoints/openai/serving_images.py`), which resolves
a lane URL and — this is the part that matters — **already refuses by name when
no lane is configured**, carrying the registry's own numbers
(`_reject_no_lane`, `:62`). So the dependency the blocker named exists, and the
"capability not served by the backend" case the surface must handle is already
handled one layer down.

`entrypoints/sdapi/` therefore composes rather than parallels, exactly as §5's
Kobold adapter does: it builds an OpenAI images request and hands it to
`OpenAIServingImages`. It never resolves a lane URL, never forwards HTTP, and a
source pin forbids `tokenizer_manager` / `sampling_params` / `httpx` from
appearing in the module at all. **What composing buys is not tidiness:** the
lane-absent refusal and the #344 client-disconnect cancellation in `_guard` are
inherited rather than re-derived, and a parallel path would have got the second
one wrong.

### What is refused, and why each is a refusal rather than a mapping

A1111 carries diffusion controls the OpenAI images protocol has no field for —
`steps`, `cfg_scale`, `sampler_name`, `seed`, `subseed`, `denoising_strength`,
`restore_faces`, `tiling`. There is no lossy mapping available; there is no
mapping. Passing them through would drop them at the boundary and render an
image the caller did not ask for with nothing saying so — **the #710
tool-arg-loss family, which is the same defect §2 of this very note was written
to fix on the Ollama front.** So a non-default value is refused by name in
A1111's own error envelope, and the refusal routes to
`POST /v1/images/generations`.

Three more judgements worth recording:

* **`negative_prompt` is refused, not concatenated.** Folding it into the
  prompt would send it as a POSITIVE instruction — wrong in the direction
  hardest to notice in the output.
* **An unexpressible canvas is refused, not rounded.** A1111 accepts any
  multiple of 8; the OpenAI protocol expresses five sizes. Rounding someone's
  768x768 to a neighbour silently is the same class of defect as dropping
  `steps`.
* **`img2img` is 501, not routed to `/v1/images/edits`.** img2img re-noises the
  whole image to `denoising_strength` and denoises back; `edits` is MASK
  INPAINTING. A client asking for a 0.3-strength restyle would get its image
  back with a hole filled in.

**`/sdapi/v1/progress` returns honest zeros** in the protocol's shape with the
reason in `textinfo`, because `serving_images.py`'s own `_guard` records that
"an image generation writes once, at the end, so there are no frames for a
progress-based watchdog to count". 404 would look like a broken server to a
client polling on a timer; a number would be invented.

Inert fields (`save_images`, `script_name`, `styles`, `override_settings`, …)
are **declared inert with reasons**, following §2's `keep_alive` precedent.

22 pins, 26 subtests. Can-fail proven on all three load-bearing properties:
dropping the unmappable fields fails 12; emitting `info` as an object instead
of a JSON string fails 1; introducing engine vocabulary into the module fails 2.

Routes registered in the #510 state-changing ratchet with their reasoning —
`txt2img` because it composes the same front `/v1/images/generations` does, and
`img2img` because it is an unconditional 501 that mutates nothing, with the
condition attached that a real implementation must be re-judged rather than
inherit the line.

**NOT claimed:** no stock A1111 client has been pointed at this, and no image
has been generated. Whether a real client is SATISFIED by these refusals — as
opposed to receiving them — is the same live question §4 raised for Ollama, and
it belongs to the same window.
