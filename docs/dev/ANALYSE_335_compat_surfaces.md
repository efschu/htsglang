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
| **Ollama emulation** | **exists, wired, COMPOSED (2026-08-17, §7)** | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`; now translates onto the OpenAI fronts. See §2 for the history and §7 for the rewrite. |
| **ComfyUI node pack** | **ABSENT** | one prose mention in `planner/webui.py:6808`; no node pack, no routes |
| **A1111 `sdapi`** | **ABSENT** | no hit anywhere under `python/sglang/srt/` |
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

## 7. The Ollama compose-refactor — BUILT

§6 scoped this and did not build it. It is built now, and §6's own plan is what
it followed: `handle_chat` maps the (already OpenAI-shaped) messages across and
calls `openai_serving_chat.handle_request`; `handle_generate` maps `prompt` onto
a `CompletionRequest` exactly as the Kobold adapter does;
`_convert_options_to_sampling_params` is gone in favour of the OpenAI request's
own fields.

**The win §6 predicted, delivered:** `format` is `response_format` and is now
HONOURED rather than refused -- `"json"` becomes `{"type": "json_object"}`, and
a schema is wrapped as a named `json_schema` with `strict` set, because a caller
who supplied a schema wants it obeyed rather than approximated. It is out of
`UNSUPPORTED_FIELDS` and pinned reaching the front on BOTH the chat and generate
paths.

**`think` stays refused.** §6 said it "becomes wirable" via
`chat_template_kwargs`. That is a claim about a mechanism I did not verify at
code, so wiring it on that basis would be exactly the kind of plausible guess
this surface family refuses elsewhere. The refusal now names the route
(`/v1/chat/completions` and its `chat_template_kwargs`) instead of just
blocking, and the wiring is left as its own cut with its own evidence.

**The net came first, deliberately.** §6's warning was that the refactor "needs
the contract tests extended to the happy path first, or it is a rewrite without
a net". `test_ollama_golden_shapes_335.py` was written against the PARALLEL
path and committed BEFORE the rewrite (`b0baecf94f`), so it could not be edited
to fit the outcome. After the swap it passed with exactly TWO changes, both the
intended one: `format` is no longer refused. Every other client-visible
behaviour -- key sets, NDJSON delta semantics, `done` sequencing, the
empty-prompt short circuit, the named refusals, the 2048-token default -- passed
untouched.

**The streaming question §6 raised, answered.** The Ollama NDJSON shape is now
re-translated from the OpenAI SSE stream, and it turned out to be simpler than
feared in one respect and needed care in another:

* simpler: the OpenAI stream already carries DELTAS, so the running-text
  subtraction the parallel path did (it received cumulative text) disappears;
* care: `guard_generate_stream` installs the #344 watchdog on the response's
  `body_iterator` (`liveness/stream.py:182`). This adapter consumes THAT
  iterator, so the watchdog sits upstream of the translation rather than being
  bypassed -- when the client stops reading, Starlette stops pulling this
  generator, which stops pulling the guarded iterator, and the KV blocks are
  released as before. Stated in the module, because "we wrapped a guarded
  stream" is exactly the kind of thing that silently stops working.

**Metadata was nearly a regression.** `/api/show` reports `context_len`, which
the parallel path read off the tokenizer manager. A stub would have been a quiet
capability loss, so it is passed in as a VALUE at construction: metadata is not
a reason to hold a serving handle.

**No dual path exists.** The module was replaced in place rather than added
beside, so there is no window in which two implementations answer the same
routes differently. Importer check: `http_server.py`, the two #335 test files,
and `openai_sdk_harness.py` were the only constructors, and all four were moved
to the new signature in this commit. `smart_router.py` never touched the
serving internals.

30 pins across the two files, 12 subtests. Can-fail proven on all three
load-bearing properties: reintroducing a serving handle fails 1, dropping the
`response_format` mapping fails 3, routing `/api/generate` through the CHAT
front fails 7.

---

## 8. `think`: the decline, revisited at code — and reversed for one half

§7 declined to wire `think` because §6's "it becomes wirable via
`chat_template_kwargs`" was a claim about a mechanism nobody had checked. It is
checked now, and the mechanism is real:

* `ChatCompletionRequest` carries `chat_template_kwargs`
  (`entrypoints/openai/protocol.py:844`); `CompletionRequest` does **not**;
* the chat front consumes it -- `merge_chat_template_kwargs`
  (`serving_chat.py:160`), applied in `_convert_to_internal_request`
  (`:566-582`), which also lifts `reasoning_effort` out of it onto the request;
* #557 (`eef0e7734c`) built the per-request path and the Anthropic front
  already drives it through `OpenAIServingChat.apply_reasoning_enabled`
  (`serving_chat.py:1761`, used at `anthropic/serving.py:701,716`).

So the mechanism reaches the front this surface composes, and the decline is
reversed **for the boolean**. `think: true` / `think: false` now go through
`apply_reasoning_enabled` — the front's own method, which knows the model's
reasoning family (hunyuan, mistral, always-on) and RAISES for a model with no
reasoning parser at all. That raise IS the per-template refusal the brief asked
for, so it is delegated rather than re-derived: a second authority for a
capability question is how two answers start disagreeing.

**Two cases stay refused, and neither is squeamishness.**

* **An effort level** (`"low"`/`"medium"`/`"high"`) would have to become
  `reasoning_effort`. The served checkpoint takes its effort BY OMISSION, and
  explicit high/max has been observed to fail at the server. That is live-model
  behaviour this desk cannot verify, and sending a value the backend rejects
  turns the caller's request into an error they did not cause. Refused with the
  route named, so a caller who wants a level owns the choice on a path where
  the error is theirs to see.
* **Any `think` on `/api/generate`.** That path composes `CompletionRequest`,
  which carries no `chat_template_kwargs` at all — there is no template being
  applied, so there is no toggle. The asymmetry is a property of the protocols,
  not an omission.

The golden corpus took its second — and, by construction, last — intended
delta: `think: true` is honoured, an effort level is still refused, and every
other client-visible behaviour passed untouched.

---

## 9. ComfyUI — SCOPE ONLY, no build. Recommendation: PARK.

**It is not a server-side surface, and that is the whole finding.** Every other
entry in §1's matrix is an HTTP surface this server mounts. A ComfyUI
integration is a *node pack*: a Python package the **user** installs into
*their* ComfyUI tree, which then talks to us over HTTP as a client. Different
artifact, different install path, different release cadence, different repo.

**Do not confuse it with two things already in the tree**, both of which make
"ComfyUI" grep-positive and neither of which is this:

* `multimodal_gen/registry.py:233` handles **ComfyUI-format pipelines** — we
  can LOAD what ComfyUI loads. That is a model-format concern.
* `IA_342_frontend_v2.md:69-81` studies ComfyUI's **UI layout** as a source of
  patterns for our own webui. That is design research.

### What a minimal node pack would contain

| node | talks to | notes |
|---|---|---|
| `SGLangChat` | `POST /v1/chat/completions` | text in / text out; the honest MVP, and it needs nothing this server does not already serve |
| `SGLangImage` | `POST /v1/images/generations` | only useful when a diffusion lane is configured; `OpenAIServingImages` already refuses by name when none is (`serving_images.py:62`), so the node inherits a good error instead of inventing one |
| `SGLangEndpoint` | — | a config node holding base URL + API key, so the other two do not each carry connection state |

Plus the packaging ComfyUI requires: `__init__.py` exporting
`NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`, a `pyproject.toml` for
the Comfy Registry, and a README. Perhaps 300-400 lines including the manifest.

### Where it would live

**Its own repository**, not a subdirectory here. Three reasons, in order of
force: ComfyUI's registry installs from a repo root, so a subdirectory needs a
publishing shim that will rot; its dependency set is ComfyUI's, not ours, and
vendoring that into this tree's requirements would be a real cost for a client
artifact; and its release cadence follows ComfyUI's breaking changes, which
have nothing to do with this server's.

### Effort against yield, honestly

Implementation is **S-to-M** — the nodes are thin HTTP calls. The cost is not
the code:

* a second installable to version, publish and support, on a registry with its
  own review process;
* it cannot be tested in this repo's CI without a ComfyUI checkout, so its
  falsifiers live somewhere we do not currently run anything;
* the image node's value is gated on the diffusion lane being up, which on this
  rig is not the default state.

**Yield is real but indirect**: it reaches ComfyUI's user base, which is large
and is exactly the audience the sdapi surface was also aimed at — and that
audience can already reach us through `/sdapi/v1` using existing ComfyUI nodes
that speak A1111. **That overlap is the strongest argument for parking**: the
surface built this week may already serve most of this population.

### Recommendation

**PARK**, and revisit only if a concrete request for it appears, or after the
sdapi window item shows real A1111-speaking clients working (which would both
prove the reach and tell us what the node pack would need to do differently).
The decision is the user's; this section exists so it can be made on the shape
of the work rather than on the name.

### The adjacent gap the brief asked about

**OpenAI embeddings on a generative-only checkpoint: NOT ESTABLISHED.** The
route is mounted (`http_server.py:2168`) and forwards to
`openai_serving_embedding.handle_request`. I did not find a named refusal for
"this checkpoint cannot embed" in `serving_embedding.py`, but I only grepped it
— I did not trace the path to its failure mode, and an absence claim needs the
file:line of the gate that refuses. So this is a CANDIDATE gap, not a finding,
and it is cheap to settle: one hermetic test asking a generation-only server
for an embedding and reading what comes back.
