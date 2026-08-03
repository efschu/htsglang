# DESIGN #466 — Live speech-to-speech translator, voice-preserving

Status: **Phase 3: audio out works.** Rung B synthesises cross-lingual cloned speech end to end (talker, code predictor, codec, Opus) on the desk; see §13. Routing v2 (unordered pairs, fan-out, constrained ASR whitelist) is in with its UI. Open: GPU latency, an ASR intelligibility round trip, the preset clips (§14). Phase 1 was survey + architecture +
selection + MVP skeleton; Phase 2 added the real TTS serving path, the Opus
transport, the preset pool descriptors, the ops scripts, and intra-segment
speaker splitting. No GPU was taken; every python invocation ran under
`CUDA_VISIBLE_DEVICES=99`. Hard deadline ~2026-08-10.

The system: a phone streams microphone audio over a WireGuard tunnel to the
rig. Each closed utterance is recognized, attributed to a speaker, translated
by our own htsglang server over its OpenAI-compatible API, and re-synthesized
so that **everyone keeps their voice and speaks a different language**.

---

## 0. Dated user decisions

These are decisions, not findings. They override anything below that predates
them.

| Date | Decision |
|---|---|
| 2026-08-03 | **License gate lifted for runtime backends.** This is a private, non-commercial deployment. Restrictive licenses (CPML, CC-BY-NC, gated weights) do NOT disqualify a backend. The constraint is repo cleanliness: no restrictively-licensed source vendored into this Apache-2.0 repository, no weights redistributed. Candidates are ranked on fitness; license is annotated factually so a future public/commercial decision can swap backends through the interface. |
| 2026-08-03 | **Accent carry-over is wanted.** The speaker's original-language accent surviving into the cloned translated output is explicitly fine and "even cool", provided intelligibility holds. Quality ordering for TTS selection: **intelligibility first (hard gate), speaker similarity second (ranked), accent character neutral-to-positive (never penalised)**. OmniVoice's documented cross-lingual accent carry is therefore a feature here, not a defect. |
| 2026-08-03 | **Voice mode is a per-session option.** `clone` (each speaker in their own voice) and `preset` (each speaker gets a distinct artificial voice, class-matched to man/woman/boy/girl) are both first-class, switchable at runtime. `preset` is also the automatic degradation path when a reference buffer is too short or too noisy, and the downgrade is visible per speaker. |
| 2026-08-03 | **Preset pool sizing.** Realistic worst case 6-8 participants, usually 2-4. Pool shape **6 man / 6 woman / 3 boy / 3 girl = 18**: eight participants can plausibly skew hard to one adult class, children rarely exceed three. On exhaustion: never crash, never silently reuse an identical voice — share a base voice with a deterministic pitch offset and raise a named notice. |
| 2026-08-03 | **Crash-safety backup.** Every restrictive or irreplaceable component (weights, pinned wheels, patched libs) is backed up to a PRIVATE GitHub repo. Private forever — CPML and NC terms permit private copies, not redistribution. The public fork never points at it. |
| 2026-08-03 | **Target phone:** OnePlus 10 Pro NE2213_11, Android 16, unlocked bootloader, **not rooted**. Nothing may require root. |
| 2026-08-03 | **TLS comes from the existing reverse proxy.** The rig already runs an nginx reverse proxy in an LXC container with Certbot-managed Let's Encrypt certificates and a public hostname. The translator is published as one more `location` there. The `chrome://flags` workaround is **demoted to fallback**; it is no longer the primary path. Consequence: risk #1 collapses from "design a secure-context story" to "verify once through the proxy from mobile data". |
| 2026-08-03 | **REVOKED: the vLLM-Omni TTS sidecar. No external serving engines at all.** Everything runs inside htsglang. Reason, and it is architectural rather than aesthetic: the whole memory-hierarchy program — spill/offload/eviction across ALL assets, one ledger, the #286 register, the #305 residency ladder — can only arbitrate between assets that live under one runtime. A second serving engine owns VRAM the ledger cannot see, so every cross-asset decision silently becomes wrong. Phase 2's `serve_tts.sh` / `setup_tts_venv.sh` are dead and the running server was killed. A dependency-pin conflict is now a named engineering item, never a reason to add an engine. The pluggable TTS backend interface is what makes this cheap: the backend swaps, the pipeline, the session, the journal and the client all stand. |
| 2026-08-03 | **Scheduler unblock deferred to #488, not built in rung B.** The `self.input_embeds = None` clear in `prepare_for_decode` is a rung-A prerequisite: rung B runs its own generation loop and never enters the scheduler, so a gated patch here would have no caller and no falsifier — unvalidated ballast of exactly the kind the desk-written-never-executed rule forbids. Recorded as a named #488 prerequisite (§11.2) so it is not lost. |
| 2026-08-03 | **Path routing, not a subdomain.** The translator is mounted as `/translate/` under an existing `server_name` with the existing certificate. No DNS record, no `certbot --expand`, nothing touched in the user's certificate setup — zero user action and zero risk to a cert that fronts a dozen other services. A dedicated subdomain is a post-vacation item. |
| 2026-08-03 | **Routing v2: a rule is an unordered PAIR, and fan-out is the intent.** `de <-> es` routes both directions from one row. Two rows sharing a language (`{de<->es, de<->fr}`) render one German utterance in BOTH targets, played sequentially with a language tag. This inverts two v1 decisions that were the wrong reading of the requirement: "one source routes to exactly one target" made the three-language case unexpressible, and "a duplicate source is refused" refused the very thing the user wants. A repeated pair is deduplicated, not rejected. Capability refusal moves to the pair level, named and greyed out per direction. The ASR keeps identifying the source language (the direction stays observed, never configured) but from a candidate set narrowed to the table's languages; a low-confidence decision resolves to the best in-set language and is TAGGED, never discarded. |
| 2026-08-03 | **Bad mobile internet is a first-class requirement, not a robustness nicety.** The translator must handle interruptions, high latency, low bandwidth and reconnects as well as it can. Concretely: (1) **no audio is ever lost** — the client buffers microphone segments locally during a disconnect (bounded ring buffer; dropping the oldest entry is permitted ONLY with a visible notice) and uploads them on reconnect, while the server replays missed translations from the session journal per client cursor; (2) **adaptive bitrate** — an Opus ladder 24→16→12 kbps driven by measured RTT/loss in both directions, every switch logged, never silent; (3) **latency tolerance** — per-segment end-to-end timestamps, late results DELIVERED and marked late rather than silently dropped, and no head-of-line blocking (a new utterance never waits on an old delivery); (4) **UX** — visible connection state (connected / reconnecting / buffered N segments) and push-to-talk that works offline, queued and shown as queued; (5) **WS resume** — fast reconnect with exponential backoff plus jitter and session-token resume, with every state proven to survive (routing rules, preset assignments, journal cursor). Test duty as always: hermetic network-chaos tests with injected drops, artificial 1–5 s delays and a bandwidth throttle in the transport layer, one can-fail test per behaviour. A real mobile-network test (aeroplane-mode toggle) is a manual acceptance point in the travel-readiness report. **Priority: after the travel-readiness milestone, before any other polish.** |
| 2026-08-03 | **Five conversation-surface orders (silence mode, always-on transcript, speaker names, uncertainty marking, speaker buttons).** Specified in full in §17. Recorded here because they arrived in one working thread and were, for one session, carried only in that thread — which is exactly the state the analysis-in-a-file rule exists to prevent. The load-bearing constraints: the transcript is held SERVER-side and survives reconnect; name suggestions are never auto-applied; uncertainty marking is "very important" to the user and shows the top-3 candidates by real similarity with `speaker-N (new)` as a full candidate; a manual speaker button makes the assignment ground truth. |
| 2026-08-03 | **No enrollment before the trip.** The user has no time to record voice samples. The user's voice therefore runs the *same* path as every other speaker: a rolling reference buffer accumulated from live speech, with preset mode covering the cold start. No special user-enrollment flow in the MVP. The enrollment endpoint stays (it costs nothing and already works) but is a post-vacation upgrade slot, and the GPU A/B must score cloning from **10-30 s of accumulated live speech**, not from clean curated audio. |

---

## 1. Architecture: cascade, not end-to-end

```
 phone ──WireGuard──► WS /api/translator/stream
                       │
                       ├─ VAD + turn segmenter        (hangover closes the turn)
                       ├─ ASR + language ID           (what was said, in which language)
                       ├─ speaker embedding + cluster (who said it; reference buffer)
                       ├─ route: detected language → target(s) by elimination
                       ├─ MT  ── HTTP ──► htsglang /v1/chat/completions   ← the dogfood hop
                       └─ TTS conditioned on THAT speaker's voice (or their preset)
                       │
 phone ◄──audio frames─┘
```

**End-to-end S2ST was surveyed and rejected on evidence, not preference.** No
open-weight end-to-end model does DE↔ES with speaker preservation:

- Hibiki / Hibiki-Zero (Apache code, CC-BY-4.0 weights — permissive, and the
  best-architected streaming S2ST found): **French→English only**, stated on
  the model card.
- StreamSpeech (MIT): **X→English only**, unit HiFi-GAN with a fixed voice.
- SeamlessM4T v2 / SeamlessStreaming: cover de↔es, but synthesize with a
  **fixed synthetic vocoder voice** — no speaker preservation. Secondary
  sources claiming otherwise conflate them with SeamlessExpressive's PRETSSEL.
- SeamlessExpressive: preserves vocal style, but `gated: manual`.
- Qwen3-Omni: **three preset voices only**, no arbitrary reference cloning.
  Independently, ANALYSE_334 §3a already recorded that its Talker/Code2Wav
  audio-out is a three-stage tenant-composition problem gated behind #333, not
  a model-loading task.

Zero end-to-end models accept a reference clip. True cloning exists only on the
TTS side. The cascade is therefore the only architecture that meets
requirement 1, and it also keeps our 27B multilingual LLM — by far the
strongest translator in the stack — on the translation hop.

### 1.1 One runtime — the sidecar is revoked (2026-08-03)

**Superseded.** Phase 2 argued for a separate process on two grounds: that
DESIGN_333 §2.3's Class-3 scheduler does not exist yet, and that every
candidate TTS package pins a `transformers` version conflicting with sglang's.
The first is still true. The second turned out to be **largely false** (§1.1.2).
Neither survives the user order: cross-asset arbitration requires one runtime,
so an external engine is excluded regardless of convenience.

What replaces it:

* **TTS talker** — a native htsglang model on a #305/#274 lane. The talker is a
  Qwen3-architecture AR model that emits audio-codec tokens; #333/#334 named
  exactly this as the audio-out route.
* **12 Hz codec / vocoder** — an in-process module in the translator's process
  tree, its VRAM a ledgered asset class in the #286 register, parkable and
  evictable like everything else.
* **ASR and speaker embeddings** — in-process modules under the same ledger.
  faster-whisper (CTranslate2, int8) is now the primary and runs in-process
  cleanly; **the NeMo-venv spike is cancelled**, because its only delivery
  vehicle was a second environment.

The translator process still owns its own CUDA context and card pin. The
difference is that nothing it loads is invisible to the ledger.

#### 1.1.1 What the checkpoint actually contains

Measured from the downloaded weights, not from the model card:

| Module | Params | bf16 | Role |
|---|---|---|---|
| `talker.model` | 754.8 M | ~1.5 GB | Qwen3-arch AR backbone, 28 layers, hidden 1024, 16 heads / 8 KV, head_dim 128, **M-RoPE** (`mrope_section [24,20,20]`, interleaved) |
| `talker.code_predictor` | 141.6 M | ~283 MB | 5-layer depth transformer, `num_code_groups 16`, vocab 2048 |
| `talker.codec_head` | 3.1 M | ~6 MB | first-codebook head, 1024 → 3072 |
| `talker.text_projection` | 6.3 M | ~13 MB | text hidden 2048 → talker 1024 |
| `speaker_encoder` | 8.9 M | ~18 MB | ECAPA-class x-vector extractor — **this is the cloning conditioner** |
| codec **decoder** | 114.3 M | **229 MB** | `Qwen3TTSTokenizerV2Decoder`: causal convnets, ConvNeXt blocks, a small rotary transformer, SnakeBeta, split RVQ |
| codec **encoder** | 56.3 M | ~113 MB | subclasses transformers' `MimiModel`; only needed for in-context-learning mode |

**The generation shape is the crux.** One autoregressive step is *not* one
token. The backbone produces a hidden state, `codec_head` samples codebook 0,
and then `code_predictor` — its own 5-layer transformer — produces the
remaining 15 residual codes for that same audio frame. All 16 are embedded and
summed to form the next step's input. At 12.5 Hz, one second of audio is 12.5
backbone steps plus 12.5 depth-transformer invocations, i.e. 200 codes.

The upside of that frame rate is real: 10 s of speech is 125 decode steps with
a tiny KV cache. The 0.6 B backbone is not the cost; the per-step nested
transformer is the architectural problem.

**In `x_vector_only_mode` the codec ENCODER is not needed** — cloning
conditions on the speaker encoder's x-vector, not on tokenized reference audio.
That drops 56 M params and the `MimiModel` dependency from the serving path.

#### 1.1.2 The transformers pin is conservative, not real — measured

`qwen-tts` 0.1.1 pins `transformers==4.57.3`; this venv carries 5.12.1 for
sglang. Phase 2 treated that as unresolvable and routed around it. It is not.

Executed on the desk, no GPU (`CUDA_VISIBLE_DEVICES=99`), with the wheel merely
unpacked on `PYTHONPATH` and nothing installed into the venv:

* All **20** `transformers` symbols the modeling files import resolve in 5.12.1.
* Both modeling modules import and construct after **three** small shims:
  1. `check_model_inputs` became a plain decorator (was a decorator factory) —
     **one** call site;
  2. `ROPE_INIT_FUNCTIONS` no longer has a `"default"` key (now `dynamic`,
     `linear`, `llama3`, `longrope`, `proportional`, `yarn`);
  3. the mask helpers renamed `input_embeds` → `inputs_embeds` and dropped
     `cache_position`.
* **The codec decoder then loaded all 496 tensors and produced audio.** On CPU,
  fp32: 25 frames → 2.00 s of audio in 0.73 s (**RTF 0.36**); 125 frames →
  10.00 s in 3.18 s (**RTF 0.32**). Output finite, peak 0.96.

So the vocoder half of the native path is not a plan, it is a thing that has
run. The port is roughly fifteen lines of compatibility, not a version war —
and on a GPU the codec cost is negligible against a 2-3 s budget.

### 1.2 Transport: WebSocket over WireGuard, not WebRTC

WebRTC would give congestion control, NACK, jitter buffering and DTLS-SRTP —
genuinely better for lossy mobile audio. It also drags in ICE/STUN/TURN, SDP
negotiation and a media server. Over WireGuard the peers share one private
subnet, so ICE has nothing to discover and the tunnel already encrypts. What
WebRTC would actually buy here is uplink loss recovery, and half-duplex
turn-taking is tolerant of the jitter its machinery hides — the audio is
seconds behind the speaker by design. Against a one-week deadline, WebSocket is
one connection, one reconnect path, and a wire format that can be printed.

Codec: Opus ~24 kbps both directions when PyAV is present, PCM16 negotiated
fallback (always available, ~256 kbps at 16 kHz — fine on the LAN, too
expensive for Spanish mobile data). The client advertises, the server picks,
and a deployment without a decoder simply does not offer Opus rather than
failing on the first frame from Spain.

---

## 2. Component survey and verdicts

Full evidence and URLs live in the two survey agent reports; this is the
decision record. **Note on provenance:** the surveying agents' knowledge
cutoffs precede several of the strongest candidates (Nemotron-3.5-ASR,
Qwen3-TTS, VoxCPM2, Voxtral Realtime), which is why the picks below are not
the Whisper/XTTS pair one would have guessed six months ago.

### 2.1 ASR + language identification

| Candidate | License (code / weights) | Streaming | Languages (de/es) | VRAM | Verdict |
|---|---|---|---|---|---|
| **nvidia/nemotron-3.5-asr-streaming-0.6b** | Apache-2.0 / **OpenMDW-1.1** (permissive) | true cache-aware, 80 ms–1.12 s | 40 locales, both in the transcription-ready tier; built-in LID via `target_lang=auto` | ~2–2.5 GB | **PRIMARY** |
| **faster-whisper large-v3-turbo** | MIT / MIT | chunked only | 99, published list | 1545 MB int8 | **FALLBACK (certainty)** |
| **nvidia/parakeet-tdt-0.6b-v3** | Apache-2.0 / CC-BY-4.0 | cache-aware capable | 25 European, auto-LID | ~2–2.5 GB | second fallback |
| Voxtral-Mini-4B-Realtime | Apache-2.0 both | true streaming, 80 ms grid | 13 incl. de/es | **≥16 GB** | reject on budget |
| Qwen3-ASR-0.6B/1.7B | Apache-2.0 | via vLLM | 52 | 1.5–4 GB | fallback; streaming is vLLM-coupled and we are an sglang fork |
| Kyutai STT | MIT/Apache / CC-BY-4.0 | true streaming | **EN/FR only** | — | reject, language set fatal |
| Moonshine | MIT | true streaming, no 30 s pad | **no German** | <1 GB | reject |
| distil-large-v3.5 | MIT | chunked | **English only** | 756 MB | reject |
| Qwen3-Omni-30B-A3B | Apache-2.0 | real-time | 19 | 79–145 GB | reject on budget |

**The decisive technical argument, and it is architectural rather than about
speed:** Whisper pads every input to a fixed 30 s mel window, so a 4 s
conversational turn costs the same encoder pass as a 30 s one. Our entire
workload is short turns. Cache-aware streaming CTC/TDT models scale with the
actual utterance. Whisper remains the fallback because it is MIT, mature, one
dependency (CTranslate2) and certain to work by 08-10; NeMo is a heavy,
torch-version-sensitive dependency that must be **proven to coexist with this
venv in the GPU window** before it can be the default.

Measured anchor (RTX 2080 Ti, 13 min audio, beam 5): large-v3-turbo 19.155 s
fp16 / 19.591 s int8, 2537 / 1545 MB. A 3080 is ~1.5–1.8× that card. **Do not
divide this by length for a 5 s turn** — the 30 s pad makes RTFx actively
misleading at our utterance length. The 250–450 ms figure in §3 is an
extrapolation from architecture, explicitly labelled, and is the first thing
the GPU window measures.

**Do not build a separate language-ID stage.** Every viable candidate does LID
in-pass for free. **Do not collapse translation** into the ASR: every open
ASR+AST model is English-pivoted (Whisper's `translate` task is verified
X→English only; Canary is EN↔24; Granite is hub-and-spoke through English), and
routing DE→EN→ES through a 0.6–2B speech decoder compounds error worse than
DE→text→27B LLM→ES.

**VAD:** silero-vad (MIT, <1 ms per 30 ms chunk, CPU, zero VRAM). Fallback
ten-vad if turn-end latency binds — its license says "Apache 2.0 with
additional conditions" and that wording must be read before vendoring.
`EnergyVad` is the dependency-free default that makes the hermetic suite
runnable; it is honest about being crude.

### 2.2 Diarization — deliberately none

**Online diarization does not earn its complexity here, and the reasoning is
worth recording because it looks like a corner being cut and is not.** The goal
is a retrospective label on a *completed* segment so the right reference audio
is reused. Online diarization's entire value is emitting labels *mid-utterance*
under bounded latency — value we cannot spend in a turn-based UI.

- The segment boundary is free: the pause that finalizes ASR *is* the
  diarization boundary.
- Online clustering is measurably worse (documented DER drop vs offline
  spectral/VBx; offline can revise early guesses, online cannot peek ahead).
- diart's rolling buffer imposes 500 ms – 5 s *before translation can start*,
  to produce a label computable in ~15 ms.
- We get retroactive correction for free: every segment embedding is retained,
  so the whole conversation can be re-clustered to repair an early mislabel.
  For a voice-cloning reference buffer this matters — one mislabel poisons a
  speaker's voice for every later turn.
- Overlapped speech, online diarization's other selling point, produces
  unusable ASR under half-duplex anyway.

**Chosen:** per-segment embedding + incremental cosine clustering behind a
pluggable interface. `pyannote/wespeaker-voxceleb-resnet34-LM` (CC-BY-4.0,
**ungated**) via ONNX Runtime, which the venv already has — ~26 M params,
~100 MB, no new heavy dependency. Fallbacks: SpeechBrain ECAPA (Apache-2.0,
EER 0.80%), ReDimNet (MIT, 1–15 M).

The one real failure mode of per-segment diarization is handled cheaply: two
people speaking back-to-back inside one VAD segment. `split_points_by_dispersion`
embeds ~1.5 s windows and flags an adjacent pair below a cosine threshold. A
few dozen lines instead of a streaming diarizer.

`nvidia/diar_streaming_sortformer_4spk-v2` (CC-BY-4.0, 0.32 s, 4 speakers) is
the drop-in alternate **if** the product ever goes full-duplex. Note the
counter-intuitive licensing: the *streaming* v2 is CC-BY-4.0 while the offline
v1 is CC-BY-**NC**-4.0. Easy to get backwards.

Gated pyannote weights are legally fine for us: the gate collects contact info,
the licenses behind it (MIT / CC-BY-4.0) impose no use restriction. One-time
click, operational not legal.

### 2.3 TTS — zero-shot cross-lingual cloning

| Candidate | License (code / weights) | Streaming | Languages (de/es) | Min ref | VRAM | Verdict |
|---|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-0.6B-Base** | Apache-2.0 / Apache-2.0 | via vLLM-Omni: 97 ms first packet, PCM + WebSocket | 10, both ✓ | **3 s** | 2.5 GB weights, ~3–4 GB runtime | **PRIMARY** |
| **VoxCPM2** (2B) | Apache-2.0 / Apache-2.0 | true in-process `generate_streaming()` | 30, both ✓ | short | **~8 GB on a 4090** | **FALLBACK (quality when VRAM allows)** |
| **Fun-CosyVoice3-0.5B-2512** | Apache-2.0 / Apache-2.0 | best-in-class, 150 ms bi-streaming, named `inference_cross_lingual()` | 9, both ✓ | 3 s | 9.75 GB repo | second fallback if latency binds; no pip package, offline-only in vLLM-Omni |
| **XTTS-v2** | MPL-2.0 / **CPML non-commercial** | real `inference_stream()` | 17, both ✓ | 6 s | — | **permitted here** (private use), but 2023 quality; keep as legacy comparison arm |
| Chatterbox Multilingual V3 | MIT / MIT | none public | 23, both ✓ | ~5 s | — | **reject on this rig**: pins `torch==2.6.0`, predates sm_120 — will not run on the 5090 |
| OmniVoice | Apache-2.0 | — | 646 listed | 3–10 s | — | candidate; its documented cross-lingual accent carry is a **feature** here per the dated decision |
| F5-TTS / E2-TTS | MIT / CC-BY-NC | — | **zh/en base**; German and Spanish are *separate community checkpoints* | — | — | structural reject: de→es in one model is impossible |
| CosyVoice **2** | Apache-2.0 | yes | **zh/en/ja/ko only** | — | — | **trap**: the HF card was overwritten with the unified CosyVoice README advertising 9 languages. No German, no Spanish. Pick CosyVoice**3**. |
| IndexTTS-2/2.5 | bilibili Model Use License (not Apache, widely misreported) | — | **no German** | — | — | reject |
| Higgs Audio v2/v3 | Apache code / `license: other` (relicensed 2026-06-25) | — | en/zh/de/ko — **no Spanish** | — | — | reject |
| MegaTTS3 | Apache-2.0 | — | — | — | — | reject: WaveVAE encoder weights withheld, cloning needs a manual approval round-trip |
| Kokoro-82M | Apache-2.0 | — | — | — | — | reject: cannot clone, no German |
| Fish-Speech / OpenAudio S1 | Fish Audio Research License, gated | — | 13 | — | — | permitted but gated weights block unattended setup |

**The honest caveat, verified across every paper read: nobody publishes de↔es
cross-lingual speaker-similarity numbers.** CosyVoice3's cross-lingual eval set
covers only `to_en/to_zh/to_ja/to_ko`; Qwen3-TTS Table 7 covers the same CJK+en
square; VoxCPM2 has no cross-lingual table. Every de/es number in circulation
is **in-language**. The ranking below is mechanism + in-language quality +
deployment fit, and §7 specifies the local A/B that actually settles it.

The convergent evidence that *is* available: all four leaders implement
cross-lingual cloning the same way — **by keeping the reference transcript out
of the LM context**. CosyVoice `inference_cross_lingual()` · Qwen3-TTS
`x_vector_only_mode=True` · VoxCPM2 `reference_wav_path` alone · Chatterbox
`cfg_weight=0`. Four spellings of one mechanism.

**Why Qwen3-TTS-Base primary:** Apache-2.0 on both halves; 2.5 GB (the only top
candidate that comfortably fits beside the 27B); 3-second cloning; the best
published per-language similarity of the field (de 0.737 / es 0.731, highest of
its 10 languages against MiniMax and ElevenLabs; WER de 0.96 / es 1.13); a real
runtime language contract (`get_supported_languages()` plus
`config.json → talker_config.codec_language_id`, with `_validate_languages()`
raising on unknowns); and 2.5 M downloads/month.

**Why VoxCPM2 fallback:** beats CosyVoice3 on CV3-Eval in *both* our languages
(de 4.77 vs 6.43, es 3.80 vs 4.47) and has the cleanest reference decoupling.
It costs ~8 GB, so it is the quality option when VRAM allows, not the default.
It also has **no language parameter at all** — synthesis language is inferred
from the input text — which for a translator that *knows* its target language
is a point against it.

**Serving.** vLLM-Omni (Apache-2.0) serves TTS behind stock
`POST /v1/audio/speech` via `vllm serve <model> --omni`, and solves the schema
problem (OpenAI's `voice` field takes a preset string, not an audio upload)
with a `/v1/audio/voices` registry: upload a reference clip once, then call
`voice="alice"`. One server, swap the model, identical client code — no
per-model HTTP wrapper. That registry maps directly onto our preset pool.

### 2.4 Machine translation

Our own **Qwen3.6-27B-INT8** over `/v1/chat/completions`, called as an ordinary
HTTP client (client-compatibility principle: we do not reach into the engine).
Requirement 6 is satisfied structurally rather than by gesture. `temperature=0`,
a system prompt that forbids conversational output, and a bounded rolling
history of previous turns — context measurably helps pronoun and formality
choice (tu/usted, German gender agreement on a referent named two turns ago),
which is exactly where a context-free sentence translator embarrasses itself
live.

---

## 3. Latency budget

Per turn, measured from **the moment the speaker stops talking** (which is what
the listener experiences), for a ~4 s utterance.

| Stage | Streams? | Fallback stack (Whisper) | Target stack (Nemotron + Qwen3-TTS) |
|---|---|---|---|
| VAD hangover — turn declared closed | — | **550 ms** | **550 ms** |
| ASR | no (Whisper) / partially (Nemotron) | 250–450 ms *(extrapolated, unmeasured)* | 50–150 ms |
| Speaker embedding | no | 15–40 ms | 15–40 ms |
| MT first token | **yes** | 150–400 ms | 150–400 ms |
| MT first clause complete | yes | +200–500 ms | +200–500 ms |
| TTS first audio | **yes** | 150–300 ms | ~100–200 ms |
| Network (tunnel, mobile) | — | 50–150 ms | 50–150 ms |
| **First translated audio after the pause** | | **~1.4–2.4 s** | **~1.1–2.0 s** |

Both stacks land inside the 2–3 s target with margin, which is why the safe
fallback is genuinely acceptable and the target stack is an improvement rather
than a requirement.

**What overlaps.** MT streams tokens; `SentenceAccumulator` regroups them into
clauses; TTS starts on the first complete clause while the tail is still
generating; audio frames go to the phone as they are produced. For a short
conversational turn this saves little. For a 20-word turn it removes most of
the MT and TTS stages from the critical path.

**What does not overlap, and cannot.** The hangover. It is the single largest
*fixed* term and the one the user feels. 550 ms is the default: below ~400 ms a
German speaker's clause-internal pauses start cutting turns; above ~800 ms the
exchange feels laggy. **This is the first thing to tune on real recordings** —
and if turn-end detection becomes the bottleneck, the shape of the fix is
already known (`parakeet_realtime_eou_120m-v1` collapses ASR + end-of-utterance
into one model, 160 ms p50 — English-only today, but it validates the pattern).

Every turn records its own `Stopwatch` unconditionally, not behind a benchmark
flag: the end-to-end number is the acceptance criterion, and a number only
available when someone remembered to enable it is a number nobody has when it
matters.

---

## 4. The skeleton

`python/sglang/srt/translator/`, following the `video_enhance` precedent
exactly (own package, own process, `test/registered/<name>/`).

| Module | Responsibility |
|---|---|
| `languages.py` | `LanguageMatrix` (runtime ASR × MT × TTS intersection), `ConversationLanguages` (routing by elimination), ISO canonicalisation |
| `backends.py` | `AudioChunk`, the four backend Protocols, and the hermetic fakes the whole suite runs against |
| `segmenter.py` | VAD protocol, `EnergyVad`, the turn state machine (onset / hangover / pre-roll / forced cut / floor) |
| `speakers.py` | Speaker registry, incremental clustering, the two-slot reference buffer, intra-segment speaker-change detection |
| `voices.py` | `VoiceMode`, preset pool, F0 voice classifier, sticky class-matched assignment, exhaustion offsets |
| `mt.py` | `OpenAiMt` against our own endpoint, direction prompt, `SentenceAccumulator` |
| `session.py` | The turn pipeline, the event journal, reconnect resume, the session manager |
| `audio.py` | Codec negotiation, PCM16/Opus, rational resampling |
| `server.py` | FastAPI: WS stream, `/api/translator/languages`, health, sessions, enrollment, voice mode |
| `asr_backends.py` | `FasterWhisperAsr`, `NemoStreamingAsr`, `SileroVad`, `OnnxSpeakerEmbedder` — heavy imports inside constructors only |
| `launch.py` | The tenant entry point; pins the card by NVML UUID before any CUDA context |
| `client/index.html` | The PWA: one file, no build, no external asset |

### 4.1 The language set is derived, never declared (requirement 5)

```
speakable(system) = ASR.transcribes  ×  MT.translates  ×  TTS.synthesizes
```

Recomputed from the live backends per request, so swapping a TTS checkpoint
changes the advertised set without a code edit. `GET /api/translator/languages`
returns the intersection **and the per-stage sets**, so a missing language is
attributable ("the TTS backend does not speak es") rather than merely absent.
An MT backend that claims universal coverage declares `None` explicitly, which
is a claim rather than a silent empty set.

Routing is elimination over a conversation's *participant set*, not a
source/target pair. Two participants reduces to "the other one"; three fans
out. **No language pair appears anywhere in the deciding modules** — the
falsifier asserts this two ways: behaviourally (the same code drives `ja↔fr`)
and by AST inspection (`session.py`, `segmenter.py`, `speakers.py`, `mt.py`
contain no `de`/`es`/`german`/`spanish` string literals).

### 4.2 Reference buffer: two slots

Following X-Translator's session-level speaker-prompt manager (arXiv
2607.17544) — **reimplemented from the paper's description; their code is
CC-BY-NC-SA and was neither linked nor vendored**:

- **Fixed enrollment prompt** (6 s, trimmed from the middle of curated audio):
  anchors identity, never evicted.
- **Rolling prompt** (6 s of the best recent field audio): tracks the current
  channel — the phone moved, the room changed.

The two sources disagree on retention and the disagreement is real:
speaker-verification practice says keep the best-K segments, the prompt-manager
design says keep the most recent window. Both are right about different failure
modes — a great old slice beats a poor new one, but a slice from a different
room is worse than a mediocre one from this one. The synthesis is quality
scored with an exponential recency decay (`recency_half_life_s`, default 180 s).

Admission thresholds, from the survey: **3 s hard minimum** before cloning is
enabled at all, **10–15 s target**, one contiguous 6–12 s slice preferred over
a concatenation of short ones (splices produce prosodic discontinuities that
zero-shot cloners reproduce audibly). An *ambiguous* segment (cosine between
0.70 and 0.80) is translated but barred from the reference buffer: a voice
split degrades gracefully, a voice merge corrupts both voices audibly.

### 4.3 Voice modes

`clone` uses the speaker's reference; `preset` assigns a distinct artificial
voice, sticky per session, matched to the speaker's F0-derived class. The
classifier is labelled heuristic everywhere because it is: F0 distributions
overlap, and **boy versus girl is not recoverable from F0 before puberty**. So
the classifier returns `CHILD` and pool entries tagged `BOY` or `GIRL` both
match it; anything finer would be a guess dressed as a measurement, and the API
offers a per-speaker override instead.

A preset is expressed as **reference audio per language**, not an opaque backend
voice name — so preset mode is clone mode pointed at a curated clip, works with
any zero-shot backend without an interface change, and a preset recorded in the
target language has no accent to carry. `backend_voice_id` is the escape hatch
for vLLM-Omni's voice registry.

Allocation: unused voice of the speaker's class → unused voice of any class
(distinctness beats class match: a listener who cannot tell two speakers apart
has lost more than one whose preset is the wrong gender) → shared base voice at
the next variant index, with a deterministic ±1.5/±3.0 semitone offset applied
to the reference and a named notice surfaced in the UI.

**Downgrade preference.** An unclonable speaker gets a *preset*, not another
participant's borrowed voice: a preset is honestly artificial, while a borrowed
voice attributes words to the wrong person. Borrowing survives only as the last
resort when no pool is loaded at all, and both are marked on the turn event.

### 4.4 Reconnect

The stated top operational risk is the mobile link dropping while roaming, so
reconnect is a first-class path. The journal is append-only with monotonic
sequence numbers; the client tracks its cursor; a reconnect replays from there.
Audio payloads are evicted under a byte budget **while their events remain**, so
a replay after a long outage yields a complete transcript with samples marked
absent, and a cursor below the floor produces an explicit `resume.gap` rather
than a silently shortened conversation.

The session outlives the socket. A reconnect resets the segmenter (audio in
flight is gone) and touches nothing else — the speaker registry and reference
buffers, which are the expensive thing to rebuild, survive untouched. That
asymmetry is the reason the two are separate objects.

---

## 5. Test results (desk, hermetic)

```
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=/spinning/wt-466-translator/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest test/registered/translator/ -q
```

**165 passed**, ~12 s, no GPU, no model weights loaded.

Per-file:

```
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_languages.py -v      # 17
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_segmenter.py -v      # 13
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_speakers.py -v       # 24
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_session.py -v        # 22
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_audio_and_http.py -v # 25
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_voices.py -v         # 23
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_voice_presets.py -v  # 11
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_opus.py -v           # 12
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_tts_backend.py -v    # 18
```

`test_tts_backend.py` is hermetic but **not mocked**: a real uvicorn server
implementing vLLM-Omni's two audio endpoints runs on a loopback port and the
adapter talks to it over real HTTP. The interesting behaviour -- streaming PCM
reassembly across odd chunk boundaries, voice-registry caching, error
surfacing -- only exists on the wire, so a mocked client would test the
adapter's shape and none of its behaviour.

Live boot smoke (fake backends, real HTTP, no GPU):

```
CUDA_VISIBLE_DEVICES=99 python -m sglang.srt.translator.launch --port 30841 --participants de,es
curl -s http://127.0.0.1:30841/api/translator/health      # status ok, budgets 7500 MiB
curl -s http://127.0.0.1:30841/api/translator/languages   # pair_count 2, default supported
curl -s http://127.0.0.1:30841/api/translator/voices      # clone default, no pool loaded
curl -s http://127.0.0.1:30841/                           # 21491 bytes of PWA
```

### 5.1 Six real bugs the suite caught before any GPU time

Recorded because each is a family this project has seen before.

1. **`journal or Journal()`** — a fresh journal has zero events and is
   therefore *falsy*, so every caller's configured bounds were silently
   replaced by defaults. Fixed to an `is not None` test.
2. **Dataclass `__eq__` over ndarray fields** — `item in list_of_chunks` raised
   "truth value of an array is ambiguous". `AudioChunk`, `SpeakerEmbedding` and
   `_ReferenceItem` are now `eq=False`; identity is the only meaningful
   equality here.
3. **`EnergyVad` deafness** — the noise floor was seeded from the first frame,
   so a stream that *opens* with speech (which push-to-talk always does) set
   the floor at speech level and no later frame could clear it. The detector
   went permanently deaf, silently: the turn simply never appeared. Found
   through the WebSocket test, because the unit tests used a scripted VAD.
   Falsifier added.
4. **Dead resampler guard** — `Fraction.limit_denominator(1000)` never fails,
   it *approximates*, so checking its output against the bound it was given was
   dead code and an unsupported rate pair passed through as a wrong pitch. Now
   compared against the exact ratio.
5. **Reconnect replayed the whole conversation** — a fresh connection's
   delivery cursor started at zero, and if the resume window happened to be
   empty nothing advanced it. Seeded from the resume point before replay.
6. **Unbounded wait in the test helper** — `ws.receive()` blocks forever;
   the suite wedged for real. Replaced with a ping/pong protocol fence plus a
   wall-clock budget, per the robustness canon.

Bug 3 is the instructive one: the unit tests injected a scripted VAD to isolate
the state machine, which was right, and that isolation is exactly what hid a
defect only the integrated path could show. **Only the integrated boot found
it** — the same lesson `integration/r2` recorded.

---

## 6. Deployment

### 6.1 Co-residency plan

| Tenant | Card | Budget | Notes |
|---|---|---|---|
| htsglang LLM (Qwen3.6-27B-INT8) | all three, uneven TP | as today | unchanged; the translator is a client |
| ASR (Nemotron-3.5 0.6B or faster-whisper) | 5090 | 3000 MiB | own process, own venv, pinned by NVML UUID |
| Speaker embedder (ResNet34-LM ONNX) | CPU or 5090 | ~500 MiB | ~26 M params; CPU is viable |
| VAD (silero) | CPU | 0 | <1 ms per chunk, one thread |
| TTS talker (Qwen3-TTS-0.6B) | 5090 | ~2200 MiB | native htsglang model on a #305/#274 lane |
| TTS codec decoder (12 Hz) | 5090 | ~300 MiB | in-process module, ledgered asset class (#286); 229 MB bf16 measured |
| Speaker encoder (x-vector) | 5090 or CPU | ~30 MiB | 8.9 M params, ships inside the TTS checkpoint |

Every row is now inside one runtime and visible to the ledger — which is the
point of the 2026-08-03 order, not a detail of it. Total additional ~6 GB,
taken from the 5090's allocation. **This directly
competes with the LLM's budget** and must be entered in the reserve per the
VRAM corridor rule (free ≥400 MiB absolute, waste ≤1.5 GiB net) before the GPU
window — the LLM boot line needs its per-rank budget reduced accordingly. That
arithmetic is a GPU-window task, not a desk one.

### 6.2 WireGuard

No real addresses in this repository; the `/root/rig-env.sh` convention
supplies them (`${VAR:-<PLACEHOLDER>}`). Server side, on the rig:

```bash
sudo apt install wireguard
wg genkey | tee /etc/wireguard/server.key | wg pubkey > /etc/wireguard/server.pub
wg genkey | tee /etc/wireguard/phone.key  | wg pubkey > /etc/wireguard/phone.pub
```

`/etc/wireguard/wg0.conf`:

```ini
[Interface]
Address    = ${WG_SERVER_ADDR:-<10.x.y.1/24>}
ListenPort = ${WG_PORT:-<51820>}
PrivateKey = <contents of server.key>

[Peer]
PublicKey  = <contents of phone.pub>
AllowedIPs = ${WG_PHONE_ADDR:-<10.x.y.2/32>}
```

```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo systemctl enable --now wg-quick@wg0
sudo wg show
```

Forward `${WG_PORT}/udp` on the router to the rig. Bind the translator to the
tunnel address, never to `0.0.0.0`:

```bash
python -m sglang.srt.translator.launch --host ${WG_SERVER_ADDR%%/*} --port 30800
```

Phone side (OnePlus 10 Pro, Android 16, **no root needed** — the official app
is a userspace VPN):

1. Install **WireGuard** from Play Store or F-Droid.
2. `+` → *Create from scratch*. Name `rig`.
3. Private key: *Generate*. Send the shown **public** key to the rig's
   `[Peer] PublicKey`.
4. Addresses `${WG_PHONE_ADDR}`, DNS blank.
5. Add peer: server public key, endpoint `<home-ddns>:${WG_PORT}`,
   AllowedIPs `${WG_SERVER_ADDR%%/*}/32` (split tunnel — only rig traffic goes
   through, so Spanish mobile data still routes normally), persistent
   keepalive **25**.
6. Toggle on, then open `http://${WG_SERVER_ADDR%%/*}:30800/` in Chrome.
7. Chrome menu → *Add to Home screen* for standalone PWA mode.

### 6.2.1 Secure context — solved by a path mount on the existing proxy

`getUserMedia` needs a secure context, and `http://` over the tunnel is not one
on Android Chrome unless the origin is a loopback. The rig already runs an nginx
reverse proxy in an LXC container with Certbot-managed certificates, so the
translator is published there — **as a path under an existing hostname, not as
a new subdomain**. That choice removes the last two user actions entirely: no
DNS record to create and no `certbot --expand` to run against a certificate
that fronts a dozen other services.

The host already path-mounts several services (`/go2rtc/`, `/llama/`,
`/plex/`), so `/translate/` is idiomatic there and collision-free. The trailing
slash on `proxy_pass` strips the prefix, so the backend sees its own paths
unchanged; only the client has to know where it lives.

**What the client had to change**, and it is small: every URL is now built
relative to `location.pathname` rather than from `/`, and the manifest's
`start_url`/`scope` are `"./"` so the installed PWA scope follows the mount
point without the server being told where it is. The same file therefore works
at the site root and under any prefix.

Three settings in the location are load-bearing rather than cosmetic:

* **`proxy_read_timeout 3600s`** — a conversation idles between turns and the
  client's own ping proves liveness; the 60 s default reaps the socket
  mid-conversation.
* **`proxy_buffering off`** — buffered synthesized audio arrives in bursts and
  destroys the streaming playback the entire latency budget rests on.
* **`client_max_body_size 32m`** — enrollment posts a base64 PCM clip and the
  1 MB default truncates it.

Procedure used, and to be repeated for any future change: back up
(`/root/nginx-backup/`), edit, `nginx -t`, **reload only** (never restart),
then verify. Verified on 2026-08-03 through the real front door: the health
endpoint and the PWA serve correctly under the prefix, the manifest carries the
relative scope, and the WebSocket handshake returns **`HTTP/1.1 101 Switching
Protocols`**. A diff of the live config against the pre-change backup shows
**29 added lines and zero deletions or modifications** — neighbouring locations
are provably untouched.

`scripts/translator/nginx-translator.conf.template` remains in the repository
as the standalone-subdomain variant, for a deployment that wants its own
hostname.

**Fallback only**, if a proxy is unavailable: `chrome://flags` → *Insecure
origins treated as secure*. Documented so it exists; not needed here.

### 6.3 The PWA wall, evaluated (not silently switched)

Per the architecture directive, the native-app decision hinges on a named wall.
The wall is real and it is **screen-off / background operation**: Android
suspends a backgrounded page's `AudioContext`, and no web API overrides that.
`navigator.wakeLock` keeps the screen on while the app is in use and is wired
in; it does **not** enable background audio.

**Recommendation: stay with the PWA for the trip, screen-on as a stated MVP
constraint.** In the actual use case — a phone held between two people having a
conversation — the screen is on and in hand. A native Kotlin client buys
lock-screen operation and a foreground-service notification, which is days of
work for a constraint that does not bind in the intended use. Everything the
native client would need is already on the server side of the WebSocket
protocol, so the port stays cheap if the constraint ever does bind.

Android 16 Chrome supports `getUserMedia` with `echoCancellation` /
`noiseSuppression` / `autoGainControl` and the Wake Lock API. Those three
constraints are the reason the client is a browser at all: they dissolve the
loudspeaker-into-microphone feedback loop for half-duplex playback, for free.

---

### 6.4 Crash-safety backup (dated decision, 2026-08-03)

Components that are restrictively licensed, gated, or published by a source
that can disappear are backed up to a **private** GitHub repository. A local
copy on one machine is not a backup, and Coqui — publisher of XTTS-v2 — is
defunct, so its HF mirror is exactly the kind of source that vanishes.

`scripts/translator/vendor_backup.sh` — small files (configs, licences,
vocabularies, install recipes) go into git; large weights go into GitHub
release assets, split at 1900 MiB because a single asset caps at 2 GiB. Every
entry lands in `MANIFEST.md` with its sha256, license and source URL, plus the
`cat <name>.part-* > <name>` reassembly line, so a restore is *verified* rather
than hoped at.

`init` will not push anything until the GitHub API itself answers
`private: true`. Trusting the create request's intent instead of the API's
answer is how a non-commercial checkpoint ends up world-readable.

The repository is **private forever**: CPML and CC-BY-NC permit private copies,
not redistribution. The public fork never references it — the fork only knows
the neutral backend interface.

Backed up in this phase: **XTTS-v2** (`coqui/XTTS-v2`, Coqui Public Model
License 1.0.0, non-commercial) — `LICENSE.txt`, `config.json`, `vocab.json`,
`README.md` in git; `model.pth`, `dvae.pth`, `speakers_xtts.pth`,
`mel_stats.pth` as assets on release tag `xtts-v2`. Qwen3-TTS and VoxCPM2 are
Apache-2.0 and ungated, so they are re-downloadable and do not need this; they
get backed up anyway once pinned to a specific revision, because Higgs Audio
silently moved from `apache-2.0` to `license: other` on 2026-06-25 and that is
the failure mode this guards against. **Pin revisions, not repo names.**

## 7. Test plan before the flight

**(a) Desk, done.** 119 hermetic tests + the live boot smoke above.

**(b) GPU window — the ticket.** Ordered so a failure at an early step does not
waste the window:

0. **Render the preset pool** (`scripts/translator/render_preset_voices.py`,
   ~36 clips). First, because preset mode is the fallback for every other
   thing on this list, and a window that runs out of time with no pool leaves
   the cloning downgrade path with nothing to degrade *to*.
1. **NeMo coexistence spike (30 min, decides the ASR).** Install
   `nemo_toolkit[asr]` into a *separate* venv; confirm it does not disturb
   `/spinning/htsglang-gpu/.venv`. If it fights the torch/CUDA stack, fall
   straight back to faster-whisper and stop spending time here.
2. **Measure ASR latency for a 5 s utterance** — faster-whisper
   `large-v3-turbo` `int8_float16` `beam_size=1` on a 3080, and the streaming
   model if step 1 passed. This replaces the extrapolated 250–450 ms in §3 with
   a number. A-vs-A floor first, per the measurement canon.
3. **TTS head-to-head, DE↔ES cross-lingual, from REALISTIC references.**
   Per the no-enrollment decision the reference is 10–30 s of accumulated
   live speech, so the A/B must use exactly that: a rolling-buffer-shaped
   reference (several field-quality slices, room noise, phone mic), not a
   clean studio clip. Run a clean clip as a *control* arm to separate "the
   model cannot clone cross-lingually" from "the model cannot clone from
   degraded short audio" — those have different fixes. Arms: Qwen3-TTS-Base
   (`x_vector_only_mode=True` **vs** ICL — the highest-value single
   comparison, it isolates the mechanism all four leaders rely on), VoxCPM2
   (`reference_wav_path` alone), XTTS-v2 as the legacy arm. Both directions.
   **Scoring per the dated accent decision:**
   ASR-round-trip WER as a **hard intelligibility gate** (transcribe the output
   with Whisper in the target language; above a WER threshold the candidate is
   out regardless of how good it sounds), then rank surviving candidates by
   ECAPA/ReDimNet speaker similarity against the source clip. **Accent is not
   scored and never penalised.**
4. **End-to-end latency** with real ASR + MT + TTS, `Stopwatch` per turn,
   against the §3 budget.
5. **VRAM corridor check at peak** (`forward_peak.py`), with the LLM's
   per-rank budget already reduced by the tenant total.

**(c) Real phone over mobile data.** User in the loop. WireGuard from a
non-home network (mobile data, not home WiFi), the secure-context question
settled, a full conversation, and a **deliberate connectivity kill** — aeroplane
mode for 20 s mid-conversation — to exercise reconnect + journal resume on the
real link rather than in a test.

**(d) Multi-speaker, two real voices.** Both voice modes. Clone mode: does each
speaker keep their own voice across turns, and does the second speaker's first
turn downgrade visibly? Preset mode: are the two presets distinct and
class-appropriate, and does the mapping survive a reconnect?

---

## 8. Open risks, ranked

The operator's proposed top three were roaming connectivity, diarization in
noise, and cross-lingual cloning quality. **Re-ranked on the evidence:**

1. **Roaming connectivity** (back to first, after the secure-context risk
   collapsed). Mitigated by design —
   reconnect with backoff, journal resume, explicit gap reporting, split-tunnel
   AllowedIPs, keepalive 25 — but a CGNAT-ed Spanish carrier plus a dynamic
   home IP can still defeat it. Mitigation to arrange before departure: a
   dynamic-DNS name and a verified inbound UDP path, tested from a foreign
   network, not from home WiFi.
2. **Cross-lingual cloning quality from SHORT, LIVE reference audio**
   (promoted, and the shape of the risk changed on 2026-08-03). The survey
   found **no published de↔es evidence for any model**, so the selection rests
   on mechanism and in-language numbers. The no-enrollment decision then made
   it harder: the reference is now 10–30 s of accumulated *live, in-the-field*
   speech rather than clean curated audio, for the user as much as for
   everyone else. The dated accent decision de-risks it in the other
   direction — with accent removed as a failure mode the bar is
   intelligibility, the property most likely to hold — and preset mode is a
   complete fallback if cloning from field audio proves too weak. Step (b)(3)
   must therefore score cloning from *degraded* references, not clean ones.
3. **Diarization in noise**. Kept mid-table for two reasons: the
   design already reduces the hard case (per-segment embedding on a completed
   turn, not frame-level tracking under overlap), and both failure modes
   degrade gracefully — an ambiguous segment is translated but barred from the
   reference buffer, and preset mode sidesteps voice identity entirely. Still
   real in a Spanish street or bar, and the mitigation lever is silero-vad
   (never `EnergyVad`) plus preset mode as the noisy-environment default.
4. **Talker bring-up under one runtime** (new, and now the schedule risk).
   The blocker is not the 0.6 B backbone, which is ordinary Qwen3 geometry; it
   is that one decode step must emit 16 codes through a nested 5-layer
   transformer, and a standard decode loop assumes one token per step. The
   fallback ladder in §10 exists precisely because this is the item most likely
   to miss 08-10. Note it is a *schedule* risk, not a feasibility one: rung B
   is already partly executed.

5. **`transformers` pin drift** (was routed around, now a named engineering
   item per the user order). Measured as three shims and one call site
   (§1.1.2), so it is small — but "imports and constructs" is not "numerically
   identical". The talker's attention/RoPE path under 5.12.1 is unverified, and
   M-RoPE with `interleaved: true` is exactly where a silent numerical
   difference would hide. Gate: byte-compare a short generation against the
   pinned-4.57.3 reference before trusting it.

6. **NeMo dependency risk — CLOSED.** Cancelled by the user order: its only
   delivery vehicle was a second environment. faster-whisper in-process is the
   primary. This removes step 1 from the GPU window.
7. **VRAM contention with the 27B.** ~6 GB from the 5090's allocation is real
   and must go through the corridor rule before the window — and now genuinely
   *can*, because every asset is ledger-visible.
8. **Secure context** (was #1, now last). Collapsed by the 2026-08-03 decision
   to publish through the existing nginx proxy: the mechanism already exists
   and is in daily use for a dozen other services. What remains is a single
   verification — load the PWA over HTTPS from mobile data once — rather than
   a design question.
9. **Preset pool renders late.** The 18 clips are produced in the GPU window by
   a second model (VoiceDesign). If that step is skipped or fails, preset mode
   and the cloning downgrade path both fall back to borrowing another
   participant's voice, which is the outcome the pool exists to avoid. Cheap
   insurance: render the pool FIRST in the window, before the latency work.

---

## 9. What is NOT built (Phase 2 boundary)

Stated so the next phase does not rediscover it. Struck-through items were open
at the end of Phase 1 and are now done.

Closed in Phase 2:

- ~~The real TTS backend adapter~~ — `tts_backends.OpenAiSpeechTts` speaks
  vLLM-Omni's `/v1/audio/speech` + `/v1/audio/voices`, with reference
  registration cached by content hash so an unchanged clip costs no round
  trip. 18 tests against a real loopback server.
- ~~Opus is wired but unexercised~~ — PyAV 18.0.0 installed, libopus present,
  round trip verified, measured 23 773 bps against a 24 000 target on
  speech-like audio. `available_codecs()` now reports `("opus", "pcm16")` on
  this deployment and the live health endpoint confirms it.
- ~~The preset voice descriptors~~ — 18 as data, with VoiceDesign prompts and
  pinned per-voice seeds; `render_preset_voices.py --dry-run` prints the
  36-clip plan.
- ~~Intra-segment speaker splitting is detected but not acted on~~ — segments
  are now re-cut at speaker changes *before* recognition, with a `turn.split`
  event and a falsifier proving neither speaker's reference buffer is poisoned.
- ~~WireGuard server side~~ — setup script with placeholders, plus
  `check_tunnel.sh`, whose pass *and* fail paths were both exercised.

Still open:

- **The preset clips themselves.** Descriptors, seeds, render plan and pool
  loader are done and tested against synthetic tones; the 36 wav files need a
  GPU (step 0 of the ticket).
- **DEAD as of 2026-08-03: the vLLM-Omni serving path.** `setup_tts_venv.sh`
  and `serve_tts.sh` are revoked and should be deleted in Phase 3, along with
  the `OpenAiSpeechTts` adapter's HTTP transport. Phase 2 flagged this stack as
  its largest unvalidated assumption; it is now moot, and the flag was
  well-placed — the assumption was never tested and now never will be. What
  survives and is worth keeping: the **backend interface** it was written
  against, the voice-registry *concept* (which maps onto preset voice ids
  in-process), and the 18 tests, which move to whatever backend replaces it.
- **The talker has never been run.** Only the codec decoder has (§1.1.2). The
  M-RoPE attention path under transformers 5.12.1 is unverified, and that is
  where a silent numerical difference would hide.
- **No real audio has ever gone through a real model.** Every clip in every
  test is a synthetic tone.
- **Full-duplex / barge-in.** Explicitly a stretch goal, not MVP.
- **Enrollment from curated samples.** Implemented and tested, but per the
  2026-08-03 decision it is deliberately unused before the trip; it becomes a
  per-speaker upgrade slot afterwards.

---

## 10. First contact, and the serving-runtime reversal (2026-08-03)

The GPU window ran step (b)(0)'s prerequisite — standing the TTS stage up — and
the run was cut short by a standing user order partway through: **no vLLM
anywhere, ever.** Everything serves inside htsglang, because cross-asset
spill/offload/eviction arbitration requires one runtime with one ledger, and a
foreign serving process is invisible to that register. vLLM-Omni is therefore
struck as the serving choice of §2.3, and a native bring-up (task #488)
replaces it.

The measurements survive the reversal and are the reason this section exists.
They are a *contract*, observed rather than read, and #488 has to implement it.

**The §9 warning was correct, and cost nine defects.** Every one would have
passed the desk suite, because the stub was built from the same documentation
as the adapter. Full list with evidence in
`/spinning/gpu-battery-results/2026-08-03_w3_t0_translator/RESULTS.md`. The
load-bearing ones:

* `--omni` is a `sys.argv` sentinel, not a flag — omitting it does not fail, it
  silently serves a *plain LLM* whose only symptom is a missing `/v1/audio/*`.
  Silent-wrong-mode beats loud-failure as a time sink, and it is the shape of
  bug to expect again in a native port.
* The voice registry takes `audio_sample` + a required `consent`, not `file` +
  `model`, and answers `{"success":…,"voice":{"name":…}}`.
* `language` is an English name (`German`), not an ISO code — the translator
  speaks ISO codes everywhere, so a mapping layer is structural, not cosmetic.
* `reference_text` is `ref_text`, and unknown keys are **silently dropped** —
  the reference transcript would have disappeared without an error.
* `x_vector_only_mode` is real but rejected on the registry path. This matters
  for §2.3: registering a clip **without** `ref_text` already keeps the
  reference transcript out of the LM context, which is the whole property the
  flag was selected for. The mechanism the four leaders converge on survives
  the reversal; only its spelling was wrong.

**Latency: the stage is not the risk.** Warm streaming first-audio on a 3080
measured **71 ms** against a 150–300 ms budget, RTF 0.16. Cold start is 27–29 s
of warmup and must never be quoted against the budget. A native implementation
inherits 71 ms as the bar, not as a target.

**Still owed, and now owed to #488:** the 36 preset clips. The render mechanism
was proven on a single clip (`task_type="VoiceDesign"` + `instructions` +
`seed`) immediately before the reversal; the batch never ran. §8 risk 7 stands
unchanged and is now the cheapest high-value step on the list — the pool is
what every other path degrades to, and nothing degrades to an empty pool.

---

## 11. Feasibility cut — native talker bring-up (#488)

Desk investigation, 2026-08-03. Numbers are measured or cited to file:line, not
estimated from the model card.

### 11.1 What the runtime already gives us

* **Model registration is a five-line contract.** Auto-discovery scans
  `python/sglang/srt/models/`; a module needs `EntryClass`, and the class name
  must equal the `architectures` string verbatim
  (`models/registry.py:92-131`). `qwen3_asr.py` is a 199-line worked example.
  One trap: unresolved architectures silently fall back to
  `TransformersForCausalLM` and import errors are swallowed with a warning
  (`registry.py:63,107`), so a typo degrades instead of failing.
* **M-RoPE is free and not entangled with vision.** `MRotaryEmbedding`
  (`layers/rotary_embedding/mrope.py:139`) is instantiated straight from
  config (`factory.py:164-177`), and the no-multimodal decode path is explicit
  (`forward_batch_info.py:1105-1114`). `mrope_section [24,20,20]` sums to 64 =
  `rotary_dim/2` for `head_dim 128`, so the auto-correction block does not fire.
* **The Qwen3 decoder stack is reusable verbatim.** `Qwen3Attention` already
  takes `head_dim` explicitly (`models/qwen3.py:141`) and carries q/k RMSNorm
  over `head_dim` — an exact match for the checkpoint's `[128]` norms.
* **RVQ machinery already exists in-tree**: `ResidualVectorQuantization` in
  `models/mimo_audio.py:260`. Encoder-side, unregistered, but it is not new ground.
* **Non-generation models are first class** (embedding/rerank/classify/score),
  so the vocoder has a home; `multimodal_gen/` even has a vocoder-loader tier.

### 11.2 The actual blocker, stated precisely

Not "the decode loop assumes one token per step" — that framing is wrong in
both directions and it matters:

* the **append** side is already N-tokens-per-step (spec decoding put it there:
  `batch_result_processor.py:752-758`);
* the **KV/position** side is strictly one-per-step
  (`schedule_batch.py:3045-3062`) — and that is exactly what we want, because
  16 codebook entries are **one audio frame at one sequence position**, not 16
  positions. The residual codes must never enter the KV sequence.

**#488 PREREQUISITE, not built in rung B** (operator ruling 2026-08-03).
Rung B runs its own generation loop and never enters the scheduler, so this
patch would have no caller and no falsifier here. It is recorded as a named
prerequisite of the native lane rather than shipped as unexercised gated code.

The real blocker is one line:

```python
# schedule_batch.py:3006-3008
self.input_embeds = None
```

The talker's next input is `codec_embedding[c0] + Σ codec_embedding_q[c_q] +
text_step` — a **vector**, not a token id — and decode force-clears the only
channel that carries one. The CUDA-graph side already supports embeds
(`decode_cuda_graph_runner.py:1710-1713`), so this is a scheduler unblock, not
a graph rewrite.

**The speculative stack is NOT the vehicle.** A draft head is the right
*shape* (small transformer per step) and the wrong *contract*: there is nothing
to verify, no target distribution, no accept length, no tree, and spec exists
precisely to add sequence positions — the opposite of the requirement.
Registering a `CustomSpecAlgo` would mean satisfying a verification contract we
do not have and would poison the spec metrics, the adaptive-draft ladder and
the cross-algo bandit permanently. Rejected.

**The precedent is the dLLM lane** (`srt/dllm/`, 935 lines + wiring at
`tp_worker.py:552`, `scheduler.py:4585`): a non-speculative custom decode
regime that runs extra model work per step and emits a block. Its
`LogitsProcessorOutput.full_logits` is the precedent for a model returning a
non-standard logits shape, and `customized_info`
(`logits_processor.py:217`) is a wired-but-unused transport for the extra codes
— no model in the fork sets it, so we would be the first user.

**No host exists.** #274 is a dual lane *within one model's world*, not a
tenant host; #305 is design-only ("no implementation",
`DESIGN_305_multi_model_serving.md:4-5`); DESIGN_333 M5 is explicitly
unformulated. ANALYSE_334 §3a already priced this family **HARD/L**. Whatever
we build lands as a registered model plus a decode-regime hook, in the slot
dLLM occupies.

### 11.3 Effort, against the reference

The only complete implementation — by the people with the weights and the
paper — is **5 192 lines across 9 files plus a 2 102-line bespoke model
runner**, and even they could not host it in a stock decode loop: their
talker entry point takes `(input_ids, input_embeds, last_talker_hidden,
text_step)` and returns `(inputs_embeds_out, audio_codes[B,Q])`, a shape no
stock runner speaks. Discounting for what this fork gives free, ~2 000-2 500
new lines.

| Phase | Days | Confidence |
|---|---|---|
| Config class + registry/AutoConfig wiring + boot | 0.5 | high |
| Talker trunk, embeddings, `text_projection`, `codec_head`, `load_weights` (3 prefix families) | 1.5 | high |
| Speaker encoder (ECAPA-TDNN), portable from reference | 0.5 | high |
| Code predictor: 5 layers, 16-slot scratch KV, 15 heads, sampling loop | 1.5 | medium |
| Offline parity harness vs reference, on desk | 1.5 | medium |
| Decode-regime hook (dLLM-shaped) | 2.5 | **low** |
| CUDA-graph capture over the nested predictor | 1.5 | **low** |
| Code2Wav vocoder (784-line reference) + streaming boundaries | 2.5 | **low** |
| Prompt/embeds builder (**1 596-line reference**) | 2.0 | **low** |
| End-to-end streaming + GPU windows | 2.0 | low |
| **Total** | **~16 days** | |

**Seven days does not buy the path.** It buys: the config and model file
loading all 478 tensors clean and boot-proven, the trunk parity-checked at
desk, the code predictor producing correct frames under a test harness, and a
written decode-hook design. It does **not** buy in-runtime streaming decode,
graph coverage, the vocoder, or `/v1/audio/speech` on the native path. **The
translator cannot move to the native lane before 08-10.**

### 11.4 Three traps worth pre-registering

1. **`interleaved` vs `mrope_interleaved`.** The checkpoint writes
   `"interleaved": true`; the factory reads
   `rope_scaling.get("mrope_interleaved", False)` (`factory.py:174`).
   Pass-through gives non-interleaved M-RoPE — a model that loads, runs, emits
   plausible codec tokens, and **sounds wrong**. Cheapest bug on the list, most
   expensive to diagnose from audio. Assert at config construction.
2. **The prompt/embeds builder is invisible from the config and is 1 596 lines
   in the reference** — text conditioning, `codec_bos 2149` / `eos 2150` /
   `pad 2148` / `think 2154` / `think_bos 2156` / `think_eos 2157`,
   `position_id_per_seconds 13`, speaker-prompt injection. None derivable. Do
   not estimate it at zero.
3. **`text_projection` uses `linear_fc1`/`linear_fc2` naming and carries
   biases**, so it will not match the standard stacked mapping.

### 11.5 Reference preserved

The vLLM-Omni 0.24.0 implementation lived only inside the venv now being
deleted. Copied to
`/spinning/llm_stuff/translator-models/vllm-omni-reference/` (20 files, with a
provenance note) — Apache-2.0, and the source of truth for the prompt builder,
the predictor loop and the vocoder. **Losing it would have cost more than any
other artifact in this project.**

### 11.6 Latency is not the risk

Window measurement, same model class: warm streaming first-audio **71 ms** on a
3080 against a 150-300 ms budget, RTF 0.16. The native lane inherits that as
the bar. The risk on #488 is integration surface, not performance.

---

## 12. Audio-out fallback ladder (Phase 3, task #488)

Every rung is **architecture-compliant**: one process tree, all VRAM visible to
the ledger. No rung is an external serving engine — that option no longer
exists, so the ladder is about *how much of the runtime the talker uses*, not
about whether it escapes it.

Descend only on evidence, and record which rung shipped.

**Rung A — native htsglang model on a #305/#274 lane.** The talker is
registered like any other model; the codec decoder is an in-process ledgered
module. Full benefit: lane scheduling, the residency ladder, spill/park, CUDA
graphs, batching across concurrent turns.
*Blocker*: one decode step must emit 16 codes through a nested 5-layer
transformer. This is not a normal LM head, and the decode loop is built around
one-token-per-step. The nearest precedent in the fork is the per-step draft
head in the speculative stack — same *shape* (a small transformer invoked
inside a decode step), different *purpose* (draft-and-verify, not residual
expansion), so it is a pattern to learn from rather than machinery to reuse
unchanged.
*Honest read against 08-10*: this is the right destination and an unlikely
one-week delivery, because it lands in the decode loop — the most
correctness-sensitive code in the runtime — days before a flight.

**Rung B — in-process torch module inside the translator process.** The talker
and codec run as plain `nn.Module`s in the translator's own process, weights
and activations registered as ledgered asset classes, driven by our own small
generation loop (the model is 0.6 B and the sequences are ~125 steps, so a
hand-written loop is not a performance problem at this scale).
*Compliant*: one process, one ledger, no second engine.
*Status*: **partly executed already.** The codec decoder loaded all 496 tensors
and produced 10 s of audio at RTF 0.32 on CPU under transformers 5.12.1 with
three shims (§1.1.2).
*Cost*: no lane scheduling, no CUDA-graph capture, no cross-request batching —
acceptable for one conversation at a time, which is the MVP.
*Honest read*: **this is the rung that ships before the flight.**

**Rung B− — rung B with the reference implementation vendored instead of
ported.** Same process and ledger, but the ~2 300-line talker modeling file is
carried as third-party source under `3rdparty/` with the three shims applied,
rather than reimplemented against our layers. Ugly, reviewable, and fast.
Only if rung B's port slips.

**Rung C — preset-only, no cloning.** Ship with the 18 rendered preset voices
and drop voice cloning for the trip. Requirement 1 is not met, but the
translator works and everyone is comprehensible. This rung needs *no* talker
work at all beyond rendering the pool once.
*Trigger*: audio-out not trustworthy by 08-08.

**Rung D — no audio out.** Text-only translation on the phone screen. The
segmenter, ASR, routing, diarization, journal, reconnect and PWA are all done
and tested; this rung is reachable today and is the floor.

**What the ladder protects.** Nothing above the TTS backend interface changes
between rungs: the session, the journal, reconnect, the voice-mode machinery,
the preset pool and the client are rung-independent. That is the concrete
payoff of having built the interface first, and it is why revoking the sidecar
costs a backend and not a pipeline.

---

## 13. Rung B status, 2026-08-03 — CLOSED, audio out works

**The decode seam and the weight load are both fixed, and the audio-out path
runs end to end.** Reference German speech in, Spanish out in that speaker's
voice, through talker -> code predictor -> codec -> Opus. Superseded blocker
narrative preserved in §13.3 because the wrong turns are the useful part.

### 13.1 What the fixes were

**Drift 6 — the decode seam. One missing input, not the position arithmetic.**
The talker branches on `cache_position` to tell prefill from decode
(`modeling_qwen3_tts.py:1693-1711`). transformers 4.57 created it in
`prepare_inputs_for_generation` for every model; 5.x removed it and now
creates it only for **remote-code** models (`generation/utils.py:596-604`,
behind `is_remote_code()`). `qwen-tts` is an ordinary installed package, so it
never qualifies: the talker saw `None` on every step and took the PREFILL
branch forever, rebuilding whole-sequence M-RoPE positions from an attention
mask that had already grown to the cache length while the query was one token.

Both reported symptoms collapse into this one cause — the `o_proj` failure
(`1x22528` = 11 cached positions x 2048) and the `text_projection` failure
(`1x36864` = 18 x 2048). **The prompt-builder suspicion of §13.3 is retired:
the builder was never the problem.**

The earlier narrowing was also wrong in a way worth recording. The talker's
`arange(seq_length)` reads `input_ids.shape`, but 5.x **does** still slice
`input_ids` to the query width for cached generation
(`next_sequence_length = 1`, `generation/utils.py:2793`). That line was never
defective, only unreachable — which is exactly why slicing `input_ids` in a
wrapper changed nothing.

The fix restores the input the talker's own position math was written against
rather than reimplementing that math, so its `rope_deltas` correction for left
padding keeps working unchanged.

**Drift 7 — the weights were never loaded, and this one nearly won.**
`from_pretrained` printed `Loading weights: 478/478` and loaded none of them.
Every talker and speaker-encoder tensor was still at transformers' random
initialisation while the checkpoint held trained values:

```
FILE  talker.text_projection.linear_fc2.bias  [0.00227, -0.00110, 0.00616, ...]
MODEL talker.text_projection.linear_fc2.bias  [0.0, 0.0, 0.0, ...]
```

An explicit `load_state_dict` resolves every key (0 missing, 0 unexpected,
0 mismatched), so the fault is 5.x's key remapping for this multi-prefix
checkpoint, not the checkpoint.

**Why it survived three rounds of investigation is the transferable part.** A
randomly initialised talker in front of a correctly loaded vocoder does not
fail — the vocoder turns any code sequence into speech, so the output was
fluent, finite, speech-shaped babble at a plausible level. The only external
symptom was that it never emitted the end-of-utterance code and ran to
`max_new_tokens`: 40.9 s of audio for a nine-word sentence. That looks exactly
like a sampling or conditioning bug, and the investigation went there first.
Greedy decoding and both streaming modes ran away identically, which correctly
ruled out sampling and pointed at something structural — the structural thing
being that there was no trained model.

It also produced a **spuriously excellent measurement**: 0.986 cosine
similarity between the reference clip's x-vector and the output's, both
extracted by the same randomly initialised speaker encoder. Two garbage
vectors from one garbage encoder agree perfectly. Fourth instance of the
reference-twin family in this project, and the reason `verify_and_load_weights`
compares against the checkpoint **bytes** and never against another copy of
the model. It runs on every load rather than behind a flag, because silent
non-loading is invisible to every cheap signal, and it raises rather than
continuing: a talker that cannot be loaded must fail loudly instead of
sounding plausible.

### 13.2 Measured, CPU, `CUDA_VISIBLE_DEVICES=99`

Reference: `xtts-v2/samples/de_sample.wav`, 3.11 s of real German speech.
Target text: one Spanish sentence, nine words.

| | before the weight fix | after |
|---|---|---|
| audio produced | 40.88 s (hit the 512-token cap) | **3.76 s** (terminated on `codec_eos`) |
| peak | 1.000 | 0.399 |
| Opus round trip | 24.6 kbps, decoded in full | **20.8 kbps**, decoded in full |
| speaker similarity | 0.986 *(spurious — unloaded encoder)* | 0.980 *(loaded — but see §15: this encoder has 0.04 of dynamic range and the number carries no information either way)* |

3.76 s against the reference implementation's own **3.85 s** for the same
checkpoint and direction (`base_de2es_xvec.wav`, §10) — the independent
cross-check that the port now behaves like the thing it ports.

RTF 5.19 on CPU is not a latency number and must not be quoted as one; the
GPU window replaces it. §11.6's 71 ms warm first-audio remains the bar.

Scripts, both re-runnable and both used to produce the numbers above:
`scripts/translator/probe_decode_seam.py` (the seam, with `--no-shim` to
reproduce the failure) and `scripts/translator/audio_out_smoke.py` (the
audible run, with structure and speaker-similarity reporting).

### 13.3 Superseded blocker narrative (kept for the wrong turns)

**Working, executed, not merely written:**

* the M-RoPE assert, with its can-fail proof;
* the ledger: four audio modules registered as `audio_modules`, park frees for
  real (parameters become meta tensors, the module stops working), restore
  returns bit-identical weights;
* the codec decoder: 114.3 M params, 10 s of audio in 3.18 s on **CPU**
  (RTF 0.32);
* the transport, end to end through the real front door: PWA, manifest, health
  and a **101** WebSocket handshake under `/translate/`;
* the compat layer: five `transformers` 4.57 -> 5.12 drifts found and fixed,
  the fifth (NaN rotary `inv_freq` from non-persistent buffers under
  meta-device construction) being the only silent one;
* `mel_filters.py`, validated element-wise against real librosa 0.11.0.

**Blocker, now localised to one frame (2026-08-03).** It is NOT the prompt
builder. Instrumenting the attention entry showed prefill succeeding with
`inputs_embeds (1, 10, 1024)` and the failure landing on the first DECODE step
with `(1, 1, 1024)`:

* `o_proj` is `Linear(2048 -> 1024)` and receives 22528 = **11 x 2048**, where
  11 is the CACHED sequence length and the query length is 1;
* an attention output always has the QUERY length, so the expansion happens
  before `o_proj` -- the strong candidate is the M-RoPE apply broadcasting a
  1-token query against full-length `cos`/`sin`, i.e. decode-step position
  bookkeeping under 5.x cache semantics rather than anything in the assembly;
* the `check_model_inputs` shim was **exonerated** by bypassing it entirely
  (identical failure), and promoting `cache_position` to `position_ids` in the
  mask helper did not change it either -- correct on its own terms, so it was
  kept, but it is not this bug.

**Narrowed further, 2026-08-03.** The talker builds its M-RoPE positions as
`batch_size, seq_length = input_ids.shape` and then `torch.arange(seq_length)`.
On 4.57 `prepare_inputs_for_generation` handed the decode step a *sliced*
`input_ids` of width 1; 5.x passes the full tensor and expresses the query
width through `cache_position` instead, which is exactly how `seq_length`
becomes the cache length. Slicing `input_ids` in a wrapper around the talker's
forward did NOT fire, so the positions are assembled on a path that does not
receive them as keyword arguments there -- that call path is the next thing to
read, and it is a short read now that the mechanism is known. (A signature
lesson worth keeping: `generate()` validates `model_kwargs` by inspecting the
forward signature, so any wrapper needs `functools.wraps` or every legitimate
kwarg is reported as unused.)

**The invariant is now pinned regardless of where the fix lands**:
`assert_position_contract` / `assert_rotary_contract` in `talker_config.py`,
with `test_shape_contracts.py` proving each fires on the exact decode-shaped
input that hid the bug, and one test showing that the identical construction
PASSES at prefill shape -- which is why prefill-shaped tests could never have
caught it.

Next step is that seam specifically: how the talker computes decode-step
position ids and what it hands the rotary, read against the preserved
reference. The prompt-builder text below stands as the pre-registered risk it
was, but it is no longer the leading suspect.

**Superseded suspicion: the prompt/embeds builder.** Generation runs through mel,
speaker-embedding extraction, prompt assembly and into sampling, then fails in
`text_projection` with a flattened input:

    mat1 and mat2 shapes cannot be multiplied (1x36864 and 2048x1024)

36864 = 18 text tokens x 2048. The projection is a plain two-Linear stack that
reshapes nothing, so the flattening happens upstream in the prompt assembly:
somewhere a `(1, T, 2048)` embedding is arriving as `(1, T*2048)`. Both the
simulated-streaming and `non_streaming_mode=True` text paths fail the same way
with only the token count changing, which points at the assembly rather than at
either path.

This is exactly the component §11.4 pre-registered as unestimatable: the
reference's builder is **1 596 lines**, none of it derivable from the config
(control ids, `position_id_per_seconds`, speaker-prompt injection). The
preserved reference at
`/spinning/llm_stuff/translator-models/vllm-omni-reference/` is the source of
truth for it, and reading it against the qwen-tts path is the next step.

**Ladder position: rung B ships.** Audio out works, so rungs C (preset-only)
and D (text-only) are now the fallbacks they were meant to be rather than the
likely outcome. Everything above the TTS backend interface -- session,
journal, reconnect, voice modes, preset pool, client, transport -- is done and
green, and unchanged by any of this.

---

## 14. What is open after Phase 3

* **No GPU measurement yet.** Everything in §13.2 is CPU. The latency budget
  in §3, the VRAM corridor entry in §6.1 and the ASR numbers all still need an
  arbitrated window. The audio-out path is correct; how fast it is on a 3080
  or the 5090 is unmeasured here.
* ~~**No ASR round trip.**~~ **DONE, and it passes.** §7(b)(3)'s gate is
  implemented (`scripts/translator/asr_roundtrip_gate.py`, threshold 0.15
  argued in the script) and executed on CPU with faster-whisper
  large-v3-turbo int8:

  | arm | heard | language | WER |
  |---|---|---|---|
  | synthesized output | "Hola buenos días, me alegro mucho de verte otra vez." | es @ 0.997 | **0.100 → PASS** |
  | control (pre-fix babble) | "I'm curious to see how you are in a small situation…" | en @ 0.515 | **19.200 → rejected** |

  The one substitution in the passing arm is `dias` → `días`: the recognizer
  produced the correctly accented form and the request text omitted it.
  Content error is zero. Left uncorrected because it also demonstrates that
  accent-preserving normalisation works — stripping accents would hide exactly
  the vowel-error class a cross-lingual cloner produces.

  **The control is the point.** It is the babble from the randomly initialised
  talker: finite, speech-shaped, correctly pitched, 0.986 speaker similarity.
  It transcribes as 193 words of English gibberish. The gate can fail on the
  one input that fooled every cheaper signal, and it re-proves that on every
  run rather than asserting it once. This also vindicates leaving WER
  uncapped — 19.2 instead of saturating at "1.0, same as silence" keeps the
  signal that says *running away*, not merely *wrong*.

  Constrained detection was exercised on the real model, not reasoned about:
  `('de','es') → 'es' @ 0.997`, and `('de','fr') → 'fr' @ 0.000` — the
  out-of-set fallback taking its best-in-set branch (faster-whisper populates
  `all_language_probs` whenever no language is passed,
  `transcribe.py:469-479`, so `fr` genuinely outranked `de` at a probability
  rounding to zero). Nothing discarded; the turn resolves and is tagged
  uncertain.

  faster-whisper lives in its own tree appended to `sys.path`; the shared
  serving venv is byte-unchanged (transformers 5.12.1, tokenizers 0.22.2,
  numpy 2.3.5 before and after), which mattered because two long-running
  services map it.
* **The 36 preset clips — rendering, with two real defects found and one open.**
  The renderer was ported from a desk-written API that never existed to the
  real `generate_voice_design(text, instruct, language)`. **The VoiceDesign
  checkpoint carries the same silent weight fault (`404 checked, 404
  repaired`)**, so without §13.1's guard the batch would have produced 36
  plausible, useless clips — and the pool is what every other path degrades
  TO. Three loads now confirm the fault is a property of this transformers
  version rather than a one-off.

  **Defect 1 — the measuring instrument (fixed).** See §15.

  **Defect 2 — presets had no identity across languages (fix written, not yet
  verified).** Rendering each preset independently per language gives the same
  descriptor and the same pinned seed a different sentence, and a model that
  designs a voice from a natural-language description constrains the voice
  CLASS, not the timbre. Measured on the validated encoder, the German and
  Spanish renders of the same preset scored **0.044–0.505** against each
  other, against a different-speaker median of 0.627. Every preset was two
  different people wearing one name — the exact failure §4.3 exists to
  prevent. `derive_preset_languages.py` fixes it by rendering ONE anchor
  language with VoiceDesign and producing every other language by CLONING that
  anchor with the serving checkpoint: the same cross-lingual zero-shot path a
  real turn takes, measured at WER 0.100. Accent carry-over from the anchor is
  explicitly fine per the 2026-08-03 dated decision. **Not yet re-measured** —
  the German anchors are still rendering.

  Within-class distinctness, on the other hand, is *mostly fine* on the
  validated encoder: man de median 0.458, man es 0.400, woman es 0.312,
  against the registry's 0.70 merge line. One pair sits above it
  (woman-02.de / woman-04.de at 0.734) and needs a re-render or a descriptor
  edit.

* **True incremental streaming** remains a #488 item; rung B chunks a finished
  utterance (module docstring in `inprocess_tts.py` states the gap).

---

## 15. The instrument-validation rule (2026-08-03, learned the hard way)

**Before a similarity number is used as evidence, measure its SPREAD on inputs
whose answer is already known.**

This project has now hit the reference-twin family five times. The fifth was a
new shape and worth naming separately, because the previous four were all "the
thing that validates agrees with the thing it validates" and this one is not.

Diarization was briefly moved onto the TTS checkpoint's own speaker encoder,
on a genuinely good argument: it is loaded anyway, it is what the cloning
conditions on, and using one encoder for both clustering and cloning makes
"the registry merges two speakers the cloner considers different" impossible
by construction. The argument was sound. The encoder cannot tell anyone apart:

| | TTS speaker encoder | wespeaker resnet34-LM |
|---|---|---|
| eight unrelated speakers | 0.949 – 0.990 | −0.048 – 0.784 |
| a voice vs its own clone | 0.981 (*inside* the above) | 0.608 |
| dynamic range | **0.04** | **0.83** |

Its mean vector has norm 9.83 against a median sample norm of 10.06 — the
shared component is ~98 % of every vector, so cosine is ~0.98 between any two
clips. Mean-centering restores range but not separation. The encoder is
trained to *condition synthesis*, not to *discriminate identity*, and those
are different jobs.

Had it shipped, `speakers.py` clusters at 0.70, so every participant would
have merged into one speaker, their reference buffers would have blended, and
the cloner would have synthesized an average of everyone — the voice-merge
failure §4.2 calls out, reached by a change whose own docstring claimed to
make it unrepresentable.

It was caught because `check_preset_pool.py` ran and declared the preset pool
collapsed at 0.94–0.99. The pool was fine. **The verdict was the instrument.**
That script now refuses to report anything until the encoder clears a spread
check on clips that are not the same voice, and the same discipline belongs on
every similarity claim in this design — including the 0.980 speaker-similarity
figure in §13.2, which on this evidence carries **no information** and should
not be quoted as a cloning-quality result. §7(b)(3)'s ranking step must use
the verification encoder, never the synthesis one.

---

## 15b. The preset pool, measured — and the trade-off the fix exposed

All 36 clips exist: 18 German anchors from VoiceDesign, 18 Spanish derived by
cloning those anchors. Measured with `wespeaker_en_voxceleb_resnet34_LM`
(instrument spread 0.832 over the control clips, so the numbers mean
something).

**The derive fix worked on its target.** Cross-language identity went from
0.044–0.505 to only three presets below the 0.60 floor, and those three sit at
0.586 / 0.592 / 0.594 — marginally under an arbitrary line rather than being
different people. That defect is closed.

**It also cost within-class distinctness, exactly as the script warned it
might.** Cloning every Spanish clip from one reference compresses the voice
space: all clones come from one model and sound more alike than the
independently designed originals.

| class | before derive (es) | after derive (es) |
|---|---|---|
| man, median | 0.400 | 0.556 |
| man, closest pair | 0.599 | **0.743** |
| woman, closest pair | 0.453 | **0.777** |

Five pairs now sit at or above the registry's 0.70 same-speaker line:
`girl-01/girl-03` (0.738 de, 0.873 es), `man-03/man-06` (0.743 es),
`woman-02/woman-04` (0.734 de, 0.777 es). Two participants handed those two
presets would be indistinguishable to the listener — and to the registry.

**ROOT FIX APPLIED 2026-08-03 (approved).** Preset mode no longer uses
pre-rendered per-language clips at all. The pool is the **18 German anchors**;
the target language is produced by cloning the anchor at REQUEST time — the
same path every real turn already takes, so the change costs nothing extra: it
only swaps which clip is handed to the cloner as the reference. The derived
per-language clips are demoted to an optional cache
(`preset-voices-derived-cache/`).

This removes the compression instead of pruning around it, and all four
retired voices come back. `PresetVoice.reference_for` already fell back to
"any clip" when the requested language is absent, so no new machinery was
needed — the change is policy, plus `speaks_natively()` now correctly
reporting `False` for the derived language, which the turn event already
records.

**Why it helps, measured.** The old path cloned twice: the pre-rendered
Spanish clip was itself a clone, and the turn cloned again from it. The new
path clones once, from the anchor.

| class | old path (double clone, es) | new path (anchor, de) |
|---|---|---|
| man, closest pair | 0.743 | **0.669** |
| woman, closest pair | 0.777 | **0.734** |
| boy, closest pair | 0.437 | **0.451** |

**PRESET MODE IS NOW USABLE: 17 voices, every class distinct.**

| class | closest pair (anchors) |
|---|---|
| boy | 0.451 |
| girl | 0.588 |
| man | 0.669 |
| woman | 0.690 |

All under the registry's 0.70 line; the checker reports *"pool is usable:
every class distinct, every preset stable"*.

Getting there took two further steps, both of a different kind than the
earlier pruning:

* `woman-02`/`woman-04` collided at 0.734 in the anchors themselves — a
  descriptor/seed problem, not compression, so re-rendering was the CORRECT
  fix here where it would have been symptom-chasing before. Seed
  466014→466114: **0.734 → 0.690**, fixed.
* `girl-01`/`girl-03` did not yield: 0.738 original, **0.777** after a
  re-seed, **0.769** after a description rewritten on three independent axes
  (timbre, dynamics, pace). A seed samples *within* a description and cannot
  escape it, and the description had already been moved as far as it goes — so
  the remaining explanation is that VoiceDesign has little separable range
  left in the small-girl corner. `girl-03` is retired. **Pruning is the right
  response to a model limit; it was the wrong response to the derive
  compression, which is why the two look alike and are not.** Cost is small:
  §4.3 notes boy/girl are indistinguishable from F0 before puberty and both
  match CHILD, so two girls plus three boys still gives five child voices
  against a stated worst case of three.

**Superseded — the earlier two-collision note:** `girl-01`/`girl-03`
at 0.738 and `woman-02`/`woman-04` at 0.734 collide *in the anchors
themselves* — two natural-language descriptions produced the same voice. That
is a descriptor/seed problem, not systemic compression, so re-rendering with a
new seed is now the CORRECT fix where before it would have been chasing a
symptom. Seeds changed 466033→466133 (`girl-03`) and 466014→466114
(`woman-04`); re-rendered and re-measured.

**Superseded:** the four-voice retirement below. It pruned around the cause
and is fully reverted; the reasoning is kept because the stopping rule it
produced still applies.

**A caveat on the absolute numbers, stated rather than buried.** The control
set is the XTTS demo clips, whose speaker identities are not independently
verified; two of them score 0.784, which is above the same-speaker line. So
the control proves the encoder HAS dynamic range — which is what it is used
for — but it does not calibrate an absolute threshold. The 0.70 line comes
from the registry, which is the operationally meaningful place, and the
relative before/after comparisons above are unaffected either way.

Nothing here blocks `clone` mode, which is the primary path; the pool is the
fallback.

---

## 16. Handover, 2026-08-03 (end of Phase 3 working session)

Branch `feat/live-translator-466`, pushed, tree clean, 286 hermetic tests
green under `CUDA_VISIBLE_DEVICES=99`.

### 16.1 What a phone test can do TODAY

Two processes, in this order — the tenant is an HTTP client of the first, so
it must be healthy before the second starts. Both need the GPU arbitration
holder. Verified together in window 5.

```bash
# 1. MT backend. Reserve 8000 on rank 0 (the 5090 in CUDA order) instead of the
# runbook's 3000: the translator tenant shares that card and needs ~5 GB.
VENV=/spinning/htsglang-gpu/.venv
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=<repo>/python
export SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
setsid $VENV/bin/python -m sglang.launch_server \
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8 \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 8000,2700,2700 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics --host 127.0.0.1 --port 30030 > <log> 2>&1 &

# 2. The tenant. --card-uuid pins it to the 5090 before any CUDA context.
setsid $VENV/bin/python -m sglang.srt.translator.launch \
  --host <wg-addr> --port 30800 --participants de,es \
  --card-uuid GPU-<5090-uuid-from-nvidia-smi> \
  --asr faster-whisper --asr-device cuda --asr-compute-type int8_float16 \
  --tts inprocess --tts-device cuda:0 --tts-dtype bfloat16 \
  --embedder onnx \
  --embedder-model /spinning/llm_stuff/translator-models/embedder/wespeaker_resnet34_LM.onnx \
  --preset-voice-dir /spinning/llm_stuff/translator-models/preset-voices \
  --mt-base-url http://127.0.0.1:30030/v1 --mt-model default > <log> 2>&1 &
```

`--preset-voice-dir` is not optional in practice: without it a first
utterance has no voice at all and the turn returns no audio (measured — the
clone reference buffer is empty on a speaker's first turn BY DESIGN).

| stage | state |
|---|---|
| VAD, segmenter, journal, reconnect, PWA, `/translate/` mount | real, tested |
| ASR | **real** — faster-whisper large-v3-turbo, 98 languages, 96.9 ms/utterance |
| speaker embedding | **real** — wespeaker resnet34-LM via ONNX |
| TTS | **real** — Qwen3-TTS in process on the GPU, RTF 1.15 |
| routing (pairs, fan-out, constrained detection) | real, tested |
| MT | **real** — our own 27B on TP=3, end-to-end WER 0.000 through the front door |
| preset voices | **real and in use** — 17 voices, sticky per session |
| **own-voice cloning** | engages (buffer fills, one identity) but the audio is silent — §16.6 |

So: a full DE↔ES conversation runs today, end to end, with nothing stubbed.
What it does NOT yet do is the headline requirement — every speaker currently
gets a preset voice rather than their own.

### 16.2 Immediate next steps, in order

1. ~~**Finish the pool.**~~ **Rendered and measured — see §15b.** All 36 clips
   exist. Cross-language identity is fixed; within-class distinctness
   regressed and five pairs now collide. Next action is the §15b step 1 drop
   of `girl-03` / `man-06` / `woman-04`, which needs no GPU.
2. **GPU latency — FIRST WINDOW TAKEN, blocked on a device bug.** Two faults
   found on the CUDA path, which had never been executed before (everything to
   date ran on CPU):

   * `device_map` routes through `accelerate` in transformers 5.x
     (`integrations/accelerate.py:134` raises without it), and accelerate is
     deliberately not in this venv. **Fixed** — the model now loads plainly and
     is moved afterwards, which is the same end state for 0.6 B and puts CPU
     and CUDA on ONE code path so the desk and the window cannot diverge.
     `_sample_elements` also had to land on CPU float32, since the model is on
     CUDA while the checkpoint sample is read on CPU and `allclose` across
     devices raises.
   * **STILL OPEN:** the first warm-up synthesis dies with
     `Expected all tensors to be on the same device, but got index is on cpu,
     different from other tensors on cuda:0 (wrapper_CUDA__index_select)`.
     Something in the reference wrapper's prompt assembly builds an index
     tensor without a device — it works on CPU because everything is CPU
     there. Next step is to instrument the assembly the way
     `probe_decode_seam.py` instrumented the decode seam and give the
     offending tensor the model's device. This is a small, well-localised bug,
     but it needs a card to reproduce.

   The card was released cleanly rather than held while blocked (heartbeat
   stopped BEFORE holder removal; cards verified at 0/0/0). Re-grab the holder
   before the next attempt.

   `latency_window.py` itself is unexercised beyond load; window 4 held
   all three cards for this entire session. It prints an A-vs-A floor before
   any number and is time-boxed. The MT row stays empty until the 27B is up.
3. **MT end to end.** Start the 27B, point `--mt-base-url` at it, run one real
   turn through the WebSocket.
4. **Then the bad-network package** (dated decision, 2026-08-03) — travel
   critical, and ahead of every other polish item. The reconnect/journal/sticky
   foundation already carries it; what is missing is the client-side buffer,
   the Opus ladder, late-delivery marking, the connection-state UI, and the
   hermetic chaos tests.

### 16.3 Window 5, 2026-08-03 — the CUDA path closed, MT live

Everything in §16.2 items 2 and 3 is done. The tenant now answers a real
WebSocket turn with real Spanish audio in a real voice, with the 27B doing the
translation.

**The device fault was a snapshot, not a missing device.** Every tensor in the
reference's prompt assembly is built with an explicit `device=self.talker.device`
read live off the module. The one tensor arriving from outside is `input_id`,
and it carries a device the wrapper decided once in `__init__`
(`inference/qwen3_tts_model.py:75-80`) and never revisited; `_tokenize_texts`
(line 282) is its only consumer. The reference always passed `device_map=`, so
that snapshot was right by construction; we cannot, so it said `cpu` while the
weights sat on `cuda:0`.

**And the same shape, one level deeper, cost 90 % of every call.** The codec
hangs off `model.speech_tokenizer`, and `Qwen3TTSTokenizer` is a PLAIN class
(`inference/qwen3_tts_tokenizer.py:44`), not an `nn.Module`. It is not in
`model._modules`, `model.to("cuda:0")` never reached it, and the whole 12 Hz
codec ran on the CPU in bfloat16. Nothing failed. `probe_stage_timing.py`
(new) opens the call:

| | before | after |
|---|---|---|
| prompt (codec encode + speaker encoder) | 5.3 s | 0.02 s |
| talker (autoregressive loop) | 4.3 s | 5.2 s |
| codec decode | **85.2 s** | 0.04 s |
| RTF | 24.18 | **1.15** |

`retarget_wrapper_device` now finds such holders structurally rather than by
name, and verifies the placement by walking every parameter and buffer
afterwards. Found in passing: `codec_decoder` was registered against
`inner.code2wav`, which does not exist on this checkpoint — the largest audio
asset was never in the #286 register at all. A registration with reach zero.

**Numbers, warm, 5090, `latency_window.py` n=10, A-vs-A floor 245 ms:**

| stage | measured | §3 budget |
|---|---|---|
| ASR (faster-whisper large-v3-turbo, int8_float16) | 96.9 ms for a 3.11 s utterance | 200-400 ms |
| TTS whole utterance | 4216 ms for 3.68 s of audio (RTF 1.146) | 150-300 ms *first audio* |
| MT | measured end to end only (see below) | 150-400 ms first token |

The TTS row is not comparable to the budget line and must not be quoted as if
it were: the budget is FIRST audio, and rung B has no incremental streaming,
so first audio is whole-utterance. `mt.py` splits translations into clauses, so
the practical unit is a clause. The talker is now 99 % of the call at ~90 ms
per codec frame — the rung-B limitation §12 already names, and what #488 exists
to remove.

**Front door, end to end** (`front_door_test.py`, the only harness that touches
nothing but the WebSocket):

```
recognized : 'Als ich sechs war, sah ich'    faster-whisper, de
translated : 'Cuando tenía seis años, vi'    our own 27B on port 30030
heard back : 'Cuando tenía seis años, vi'    es @ 0.994
WER 0.000 <= 0.15 PASS; control (German audio scored against the Spanish
text) 2.000, so the instrument discriminates.
```

Turn wall time 3.7 s for a 1.5 s utterance, whole chain.

### 16.4 The two findings this opened, in priority order

**1. The clone path is unreachable in a real conversation.** Every turn of
every front-door run reported `admitted=False reference=0s` and downgraded to
a preset voice. It is a threshold that does not bind at the served geometry:
the segmenter delivers ~1.5 s turns (from a 3.1 s clip AND from a 9.3 s one),
while `SpeakerConfig.min_slice_s` is 2.0, so no slice is ever admitted and the
3.0 s reference budget can never fill. Own-voice cloning is the headline
requirement, and today nothing in a conversation made of short utterances can
reach it. Read the segmenter and that threshold together; do not simply lower
the constant, since it exists to keep unrepresentative snippets out of the
reference buffer.

**2. One speaker became two.** The intra-segment re-cut split a single German
voice into `speaker-1` and `speaker-2`, each at similarity 1.0 against its own
fresh centroid. Independent of finding 1, and on its own it would give one
person two different preset voices mid-conversation.

Neither is a regression from this window; both were invisible until the chain
ran end to end from outside, which is the argument for the front-door harness.

### 16.5 Window 6 — the two findings were ONE defect, plus a third behind them

**Check the accused component first.** The segmenter was blamed for the ~1.5 s
turns and is innocent: fed the real clips it emits ONE segment
(`de_sample.wav` 3.11 s → 1 × 3.10 s; `de_long.wav` 9.34 s → 1 × 9.34 s). The
pieces came from `_split_at_speaker_changes`.

`probe_speaker_change.py --pool` (new) reads the numbers that re-cut actually
decides on, over all 17 pool voices — adjacent windows of ONE voice against
windows of DIFFERENT voices:

| window | within-speaker min | between-speaker max | 0.62 cuts same-speaker |
|---|---|---|---|
| **1.5 s (shipped)** | 0.392 | 0.679 | **27 of 60 (45 %)** |
| 2.0 s | 0.515 | 0.694 | 7 of 40 |
| 2.5 s | 0.624 | 0.734 | 0 of 26 |
| 3.0 s | 0.695 | 0.738 | 0 of 18 |

At 1.5 s the embedder's own within-speaker similarity falls to 0.392 — below
the 0.62 meant to separate two *people*. **The threshold was never the root
cause; the window was too short for the embedder to be stable on.** That one
defect produced both §16.4 findings: each half fell under `min_slice_s`, so
nothing was admitted (finding 1), and the halves did not match each other
(0.533 < the 0.70 match threshold), so one person became two (finding 2).
Lowering `min_slice_s` — the obvious move — would have papered over a
corrupted identity.

Window 1.5 → 2.5 s, `speaker_change_min_segment_s` 3.0 → 5.0 (it must stay
≥ 2 × window or `count < 2` returns early and the re-cut is *silently* dead —
now checked at construction, not asserted in a comment). Cost, stated: two
people alternating inside a segment shorter than 5 s are no longer split.

**Then a third bug, behind those two.** With the buffer finally filling
(`admitted=True`, 3.1 s → 6.22 s, one identity), every turn died with
`tts: reference is 0.00s, need >= 3.0s`. `SpeakerProfile.reference_audio()`
did not resample a slice at a different rate — it `continue`d past it. The
registry stores at 16 kHz, Qwen3-TTS asks for 24 kHz, so every slice was
dropped and an empty clip was returned *silently*, because the caller's
`reference_seconds()` guard had already passed on the stored durations. Guard
and returned buffer disagreed and nothing in between noticed.

**Three bugs of one family in one session** — desk fake and real backend
differing in a single attribute, invisible until the real one runs: the
wrapper's device snapshot, the codec's detached holder, and now the fake TTS
sharing the pipeline's sample rate. Any new desk fake should differ from its
real counterpart on purpose.

### 16.6 The one thing still between here and own-voice output

The clone path now engages and returns **4.24 s of exact zeros**. Located:

* the reference reaching the TTS is good — the 24k→16k→24k round trip on the
  real clip preserves the peak (0.591 → 0.579 → 0.594), concatenation
  included;
* the same TTS, same checkpoint, same 6.23 s doubled German reference, driven
  **directly** by `audio_out_smoke.py`, returns good audio (peak 0.398,
  speaker similarity 0.985, RTF 1.34);
* therefore the defect is in how `session.speak()` drives the synthesizer in
  clone mode — not in the synthesizer, the checkpoint or the reference.

Next reader: diff that call against the smoke path. `voice.backend_voice_id`
and `reference_text` are the two arguments the smoke path does not set.

Also recorded, deliberately not built: cumulative admission of sub-2 s slices.
`min_slice_s` exists because splices are audible, so counting six separate
1.5 s slices toward one budget would trade a silent identity bug for a silent
quality one. The right shape is to MERGE slices that are genuinely contiguous
(adjacent VAD segments across a 550 ms pause are one breath, not a splice),
and it needs an audio counter-arm before it ships.

### 16.7 Two things not to re-litigate

* **Do not diarize on the TTS speaker encoder.** Measured, reverted, §15. It
  has 0.04 of dynamic range and would merge every participant into one
  speaker.
* **Do not render preset languages independently.** Measured, §14 defect 2.
  Same descriptor and seed do not give the same voice; derive from an anchor.

### 16.8 Window 7 — the front door is public, and a phone can use it

The one step between the working chain and a real phone test was exposure.
It is done, and it was smaller than expected because §6.2.1's nginx location
was already live from the earlier verification; what was missing was that the
tenant behind it had been left running with **fake backends**. A health
endpoint answering `200` through the proxy therefore proved routing and
nothing about the pipeline — the exact shape of the success-claims rule.

Boot, both processes, cards held under `/spinning/gpu-arb/holder` with a
bounded heartbeat:

* MT: `Qwen3.6-27B-FP8`, TP=3 uneven, `--rank-auto-reserve-mib 8000,2700,2700`,
  `127.0.0.1:30030`, `--enable-metrics` (verified by reading `/metrics`, not by
  the flag being present on the line).
* Tenant: pinned to the 5090 by NVML UUID, **bound to `192.168.0.101:30800`**
  — the LAN address the CT208 location proxies to. §16.1 says `--host <wg-addr>`
  and that is the wrong instruction for this deployment: WireGuard does not
  terminate in this container, the phone reaches nginx over the public
  hostname, and nginx reaches the tenant over the LAN. Binding to a WG address
  here would have produced a 502 from the front door.

**Measured through the public URL**, `front_door_test.py --url
wss://efeu.ddnss.de/translate --repeats 2` (artefacts and full numbers in
`/spinning/gpu-battery-results/2026-08-03_466_frontdoor_public/`):

```
voice mode  clone            WER 0.000  (threshold 0.15)
control     1.000            peak 0.513 over 4.64 s (non-silence gate)
turn wall   14.55 s / 2 turns
```

Free VRAM with both tenants up: **647 / 3419 / 839 MiB**, all above the
corridor's 400 MiB floor, with no headroom to spare on the two 3080s.

What a phone test can and cannot do, stated honestly:

| | |
|---|---|
| works | full DE↔ES conversation, own voice from the second utterance on |
| first utterance | preset voice **by design** — the clone reference is built from a speaker's own completed turns |
| first audio | ~4 s after release; rung B has no incremental streaming (§12), so first audio is whole-utterance |
| capacity | one conversation at a time in practice; `max_sessions` is 8 but the 5090 budget is not |
| network | the public hostname works from anywhere; WireGuard is the private path and is not required for the front door |
| screen | must stay on (§6.3) |
| `--mt-thinking` | stays **off**; on, the 27B narrates its reasoning and it would be read aloud |
| `--preset-voice-dir` | not optional — without it a first utterance has no voice at all |

---

## 17. The conversation surface — five ordered features

Provenance: five user orders from the 2026-08-03 working thread, written down
here before any of them is built. They had lived only in the thread for one
session; a feature order that exists only in a transcript is lost at the next
context boundary, which is what the analysis-in-a-file rule is about. Nothing
in this section was implemented at the time of writing — every subsection
states its own falsifier so "built" and "documented" cannot drift apart.

These five share one theme, and it is worth naming because it decides several
design details below: **the phone screen is no longer only a talk button.** It
is the conversation's record and its correction surface. Everything the machine
GUESSED — who spoke, what their name is — must be visible as a guess, and must
be correctable by one tap, without ever rewriting history silently.

### 17.0 Cross-cutting rules

1. **Server-side truth.** Transcript lines, speaker names, manual assignments
   and confirmed suggestions live in the session on the server. The client
   renders them; it never owns them. A reconnect that restores a conversation
   from client memory would lose everything the moment the phone's browser
   tab is evicted, which on Android happens routinely.
2. **Never silent.** Every state change a user could disagree with is an event
   with a reason: a badge changes, a line updates, a chip appears. No
   overwrite happens without a corresponding event on the wire.
3. **Never auto-apply a guess.** Name suggestions and uncertain assignments are
   proposals until a human confirms. The one exception is the ORIGINAL
   diarization assignment, which must be automatic to work at all — hence
   §17.4, which makes it visibly provisional instead.
4. **Manual beats measured.** A human decision is ground truth. It is never
   overridden by a later similarity computation; it is only ever overridden by
   another human decision (which must be undoable).

### 17.1 (a) Silence mode — the reading mode

A session-level output mode, `voice` (default) or `silent`. In `silent` the
pipeline is **identical** up to and excluding synthesis: VAD, segmentation,
ASR, language identification, speaker assignment, routing and MT all run and
all emit their events. Only the TTS call is skipped, and with it
`EventKind.TURN_AUDIO`.

* Switchable at runtime through the same control frame that switches voice
  mode; sticky per session; survives reconnect (it is part of `state()`).
* Silent mode does NOT disable clone-reference accumulation. A conversation may
  start silent and continue with voice, and the reference buffer built while
  reading is exactly the buffer wanted at the switch.
* It is not a downgrade path and never chosen automatically. A TTS failure
  stays a per-turn failure with a reason; it does not silently become a mode.

Falsifier: one turn through the session in each mode with a counting TTS spy.
`silent` must show `tts.calls == 0`, zero `TURN_AUDIO` events, and a
`TURN_TRANSLATION` payload byte-identical to the `voice` run on the same input.
A test that only asserts "no audio" would pass against a broken pipeline that
produced nothing at all, so the identity assertion is the load-bearing half.

### 17.2 (b) The transcript is always there

**Every mode keeps a full, scrollable transcript.** It ends only at an explicit
user clear — never at a reconnect, a mode switch, a routing change or a journal
eviction.

This is not the journal. The journal (§4.4) is a bounded replay buffer sized in
events and audio bytes; it is allowed to evict, and it must stay allowed to.
The transcript is a separate, text-only, append-mostly log:

```
TranscriptLine:
  line_id, turn_id, at
  speaker_id, speaker_label          # label = confirmed name or "speaker-N"
  source_language, source_text
  translations: {lang: text}
  confidence: exact | uncertain      # see 17.4
  candidates: [...]                  # populated only when uncertain
  origin: auto | manual              # see 17.5
```

* Per line the client toggles **source text** and **translation** — both are
  carried, the toggle is pure rendering, and it works per line rather than
  globally so one line can be inspected without changing the view.
* Held server-side, addressed by session id. Restored on reconnect via
  `GET /api/translator/sessions/{id}/transcript?since=<line_id>` and delivered
  as part of the resume path, so a phone that lost its tab comes back whole.
* Bounded by an explicit, visible cap (10 000 lines; a full day of dense
  conversation is far below that). At the cap the oldest lines drop **with a
  marker line in the transcript**, because a transcript that silently loses its
  beginning is worse than one that says it did.
* `DELETE .../transcript` is the only thing that empties it, and the client
  confirms before calling it.
* A retroactive rename (§17.3) or a resolved uncertainty (§17.4) UPDATES the
  affected lines server-side and emits a line-update event. The client patches
  in place; it never re-fetches and never diverges.

Falsifier: drive N turns, drop and resume the WebSocket, assert the restored
transcript equals the pre-drop transcript line for line — and, separately,
overflow the JOURNAL's audio budget and assert the transcript is untouched.
The second half is the one that catches the tempting implementation where the
transcript is derived from journal events.

### 17.3 (c) Speaker names — manual, plus three-valued suggestions

**Manual naming is the primary path and always available.** Tap a speaker chip
or a transcript line, type a name, and it applies **retroactively to every line
of that speaker id** and forward to all new ones. It is stored on the profile,
survives reconnect, and is undoable.

**LLM suggestions are the assist.** The 27B that already does the translation
also runs an extraction prompt over recognized text and returns zero or more
name candidates, each classified into exactly one of three kinds. The
classification is the whole point: the same surface form means three different
things depending on who the name refers to.

| kind | example (verbatim test case) | meaning | effect |
|---|---|---|---|
| `self` | "ich bin Matthias Ehrenfeuchter" | the speaker names themselves | suggest this name FOR THE CURRENT SPEAKER |
| `third_party` | "darf ich vorstellen: Larisa Ehrenfeuchter" | a name is introduced, but it is NOT the speaker | suggestion FLOATS — attached to no speaker id, offered later if evidence appears |
| `addressed` | "sag hallo, Moritz" | the speaker addresses someone else by name | suggestion for whoever ANSWERS — resolved by adjacency, not by the LLM |

**The adjacency logic lives in the session, not in the LLM.** The model sees
one utterance's text and answers a classification question about that text; it
is never asked who was in the room. An `addressed` candidate becomes a
suggestion for speaker X when the next utterance within a bounded window
(default 15 s, at most 2 turns) comes from a **different** diarization id than
the addressing speaker, and X is that id. If the next utterance is from the
same speaker, or nothing follows inside the window, **the candidate expires**
and no chip is ever shown. Reason for the split: the LLM has no reliable access
to turn structure and would confabulate it, while the session knows the
diarization ids and timestamps exactly.

* **Never auto-apply.** A suggestion is a chip with confirm and discard. A
  discarded name is not offered again for the same speaker in the same session.
* **A cheap pre-filter runs first**, so most utterances never reach the LLM: an
  utterance is a candidate only if it contains a capitalised token that is not
  sentence-initial-only, or a per-language introduction cue ("ich bin", "ich
  heiße", "soy", "me llamo", "das ist", "darf ich vorstellen", "sag hallo",
  "di hola", vocative comma patterns). The filter is allowed false positives
  (they cost one small LLM call) and its misses are covered by manual naming.
* **Uncertain lines (§17.4) produce no suggestions at all.** Naming a speaker
  we are not sure spoke would attach a real name to a wrong identity, which is
  the one error this whole section exists to prevent.

Falsifier: the three verbatim cases above must each produce exactly their kind,
plus counter-cases that must produce NOTHING — a `third_party` line must never
name the speaker; "sag hallo, Moritz" followed by silence must expire; "sag
hallo, Moritz" followed by the SAME speaker must expire; a sentence containing
a place name or a capitalised sentence opener must not yield a chip. The
adjacency cases are hermetic (fake ASR text + synthetic diarization ids), so
they do not need a card.

### 17.4 (d) Uncertainty marking — "very important"

A speaker assignment the machine is not sure of is **marked as such on the
line**, and the mark carries the alternatives.

**The band is measured, not chosen.** `probe_speaker_change.py --pool` over the
17-voice pool at the shipped 2.5 s window, re-run 2026-08-03 for this section:

```
WITHIN  n=26   min 0.624  p05 0.637  median 0.731   worst: boy-02.de
BETWEEN n=136  max 0.734  p95 0.583  median 0.228   closest: boy-03.de vs girl-01.de
```

Within-speaker p05 **0.637** against between-speaker p95 **0.583** is the
operating band:

| similarity to nearest profile | verdict |
|---|---|
| `>= 0.70` (`match_threshold`) | confident same speaker — assign, fold, no badge |
| `0.583 .. 0.70` | **uncertain** — the assignment is made as usual, the LINE is badged, candidates offered |
| `< 0.583` | confident new speaker — mint `speaker-N`, no badge |

**Corrected against the code** (2026-08-03): an earlier draft of this table
put the upper edge at the within-speaker p05 of 0.637. The band's upper edge
is `match_threshold` (0.70), because that — not 0.637 — is where the
assignment actually changes its answer, and a badge that disagreed with the
decision it describes would mark lines the code was never in doubt about.
0.637 keeps its own job: it is the bar for auto-resolution below.

The verdict never changes WHO gets assigned. It only decides whether the line
carries a badge, which is what keeps this feature free of the risk that a
display concern silently re-tunes diarization.

**The populations overlap and no threshold closes the gap.** The pool's own
worst collision is `boy-03` against `girl-01` at **0.734** — two genuinely
different people scoring above the confident threshold, and above the
within-speaker minimum (0.624) as well. (An earlier note put this case at
0.679; that was the 1.5 s figure from §16.5. At the shipped 2.5 s window it is
worse, not better, because longer windows raise BOTH populations.) This is the
standing proof that no threshold makes the problem go away, and therefore the
reason the user-facing correction path is the actual fix rather than a nicety:
the machine cannot be made right, so it must be made **correctable**.

* The badge shows the **top-3 candidates ranked by real similarity** — the same
  cosine the assignment used, not a re-derived score.
* **`speaker-N (new)` is a full candidate** in that list, ranked by the
  new-speaker threshold, not appended as an afterthought. The mandatory test
  scenario is the children's case: two similar young voices (Moritz and Ben)
  where the correct answer is frequently "neither of the two known ones".
* **Tap resolves.** The tap re-labels the line, is undoable, and only on
  confirmation does the embedding fold into that speaker's centroid. An
  unconfirmed uncertain line must never move a centroid — that is how one
  ambiguous slice corrupts an identity permanently.
* **Later auto-resolution is visible.** When more audio makes an earlier
  uncertain line unambiguous, the badge CHANGES (uncertain → resolved, with the
  new label) through a line-update event. It is never rewritten silently, and a
  user-confirmed line is never re-decided at all (rule 17.0.4).
  The ranking for this excludes the line's OWN current speaker. That profile
  was seeded by this very embedding, so it matches itself at ~1.0 for ever;
  counting it would clear every badge on the following turn while learning
  nothing. The open question is whether the utterance belonged to somebody
  else, so only somebody else can answer it — concretely, a known speaker
  whose centroid has moved far enough to claim the stored embedding above
  `match_threshold`.

Falsifier: a synthetic embedding sequence placed deliberately in each of the
three bands must produce exactly the three verdicts; a centroid snapshot before
and after an unconfirmed uncertain line must be identical (this is the test
that fails against the obvious implementation); and the Moritz/Ben scenario
must place `speaker-N (new)` in the top 3. The candidate list must also be
proven to come from the assignment's own similarities — a spy that returns
distinct, recognisable numbers is enough to prove it is not recomputed.

### 17.5 (e) Speaker buttons — ground truth by one tap

A row of buttons sits **above** the record button, which stays large and
unchanged; it is a row, not a menu, so it can be hit without looking.

* **One button per known speaker**, showing the confirmed name or `speaker-N`.
* **Pressed BEFORE speaking**, it assigns the coming utterance to that speaker
  directly and **skips identification entirely** — no embedding comparison, no
  uncertainty badge, no new-profile decision.
* It applies to **exactly one utterance**. After that utterance completes the
  selection clears itself. The active button is visibly marked while armed, and
  tapping it again disarms it.
* **`+` mints a new speaker** for the unambiguous case where a new person joins:
  it allocates `speaker-N+1` immediately, and that speaker's first audio seeds
  the centroid rather than being matched against existing profiles.
* **A manual assignment is ground truth**: `origin: manual`, never badged
  uncertain, and never re-decided by a later similarity computation. Its audio
  MAY enter the reference buffer — subject to the ordinary quality criteria
  (`min_slice_s`, `min_reference_rms`, `max_slice_s`) but bypassing
  `reference_threshold`, since that threshold exists to keep MISIDENTIFIED
  audio out and identification is precisely what was skipped.

The quality criteria staying in force is deliberate: a human can vouch for WHO
spoke, but not for whether the microphone clipped or the segment is 0.4 s long.

Falsifier: an armed button must route the next utterance to its speaker even
when the embedder is a spy that would have returned a contradictory match (this
proves identification was actually skipped, not merely overruled afterwards);
the arming must clear after exactly one utterance, so the utterance AFTER it
goes through normal identification; `+` must seed rather than match; and a
manual line with too-short audio must still be refused for the reference
buffer.

### 17.6 Build order and what each step needs

Order is by dependency, not by user priority — (b) carries the surface every
other feature renders into, so it goes first even though (d) is the one the
user called "very important".

| step | feature | depends on | needs a card? | state |
|---|---|---|---|---|
| 1 | (b) server-side transcript + line-update events | — | no | **done**, 2026-08-03 |
| 2 | (a) silence mode | 1 (transcript is the whole output in silent mode) | no | **done**, 2026-08-03 |
| 3 | (e) speaker buttons + manual ground truth | 1 | no | **done**, 2026-08-03 |
| 4 | (d) uncertainty band, candidates, resolution | 1, 3 (manual overrides) | no | **done**, 2026-08-03 |
| 5 | (c) name suggestions, three-valued + adjacency | 1, 4 (uncertain lines get no suggestions) | LLM only, at the end | **server side done**, 2026-08-03 |

Steps 1-4 are entirely desk-testable under `CUDA_VISIBLE_DEVICES=99` with the
existing fake backends. Step 5's classifier needs the 27B for its end proof,
but its adjacency logic — the part with the interesting failure modes — does
not, and is hermetic.

### 17.7 Status, 2026-08-03 — server side complete, client surface open

All five are implemented and tested on the **server**: 380 hermetic tests
(from 286), `ruff` and `codespell` clean. The protocol carries everything the
five need, over both the WebSocket and REST:

| | control frames | REST |
|---|---|---|
| (b) | `transcript`, `transcript.clear`, `speaker.name` | `GET`/`DELETE .../transcript`, `POST .../speakers/{id}/name` |
| (a) | `output.mode` | `POST .../voice {"output_mode": ...}` |
| (e) | `speaker.arm`, `speaker.add` | `POST .../arm`, `POST .../speakers` |
| (d) | `line.resolve`, `line.undo` | `POST`/`DELETE .../lines/{id}/speaker` |
| (c) | `suggestion.confirm`, `suggestion.discard` | — (chips arrive as `speaker.suggestion` events) |

Plus: the handshake now delivers the written record before the journal
replay, so a reconnect from cursor zero restores the conversation whole.

**What is NOT built: the PWA surface.** `client/index.html` still shows the
pre-§17 screen — no transcript panel, no reading-mode toggle, no speaker
button row, no uncertainty badges, no name chips. Until that lands, the five
features are reachable only by a client that speaks the protocol (the test
suite does; a phone does not). That is the next step, it needs no card, and
it is the whole remaining distance between "implemented" and "usable on the
phone".

One end proof also remains open and does need the card: (c)'s classifier
against the real 27B. Its adjacency — where the interesting failure modes
live — is hermetic and green; what is unproven is how well the 27B itself
separates `self` from `third_party` from `addressed` on real speech.
