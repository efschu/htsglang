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
