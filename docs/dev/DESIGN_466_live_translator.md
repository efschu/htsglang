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
| `>= 0.637` (`match_threshold`) | confident same speaker — assign, fold, no badge |
| `0.583 .. 0.637` | **uncertain** — the assignment is made as usual, the LINE is badged, candidates offered |
| `< 0.583` | confident new speaker — mint `speaker-N`, no badge |

**`match_threshold` moved 0.70 → 0.637, and the first real conversation is
why** (2026-08-03). 0.70 was the conventional ECAPA operating point taken from
the literature; our own sweep says the same-speaker population on THIS
embedder reaches down to 0.624, p05 0.637. A match bar above the floor of the
same-speaker population is not a match bar — it splits one person into a new
profile every few utterances, which is exactly what the first working phone
conversation did: one speaker, a different identity every turn. Between-speaker
p95 is 0.583, so 0.637 still clears the other population.

An interim draft of this table put the upper edge at 0.70 to match the code as
it then stood. The measurement won instead, and the two now agree.

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

### 17.5b (f) Manual speakers carry their voice class, and everyone is renameable

Two further orders, 2026-08-03, after the first working conversation.

**The "+" button asks for a class.** Man / woman / boy / girl — exactly the
four the preset pool is built from. The user knows it and the F0 heuristic
only guesses, and crucially it is known BEFORE the first utterance, which is
precisely when a preset voice has to be chosen. A manually added speaker
therefore: gets their own hold-to-speak button immediately; in `preset` mode
gets a distinct pool voice of that class at once (sticky per session, existing
exhaustion rules and their named notice unchanged); in `clone` mode uses the
class for the preset start until enough own reference exists (the existing
downgrade path); and may be named on creation.

Falsifier, and it must be about the VOICE rather than the label: two speakers
added with different classes must receive presets from different classes, and
the same speaker added as `boy` versus `woman` must get a different pool
voice. A test that only checks the stored class would pass against an
implementation where the class never reaches the pool.

**Every speaker is renameable, automatic or manual.** This was in the original
five ("with the possibility of giving the speakers names") but had lived only
in the protocol. Long-press a speaker button — the short press is
hold-to-speak — and type a name. A typed name outranks every suggestion and is
never overwritten by the automatic path.

### 17.3b Self-introductions apply, they do not queue

Revised 2026-08-03, against the first real conversation: the user said "hallo
ich bin matthias", stayed `speaker-1`, and reported the naming as simply not
working.

"Nothing is ever auto-applied" was a design call of mine, not a user order.
The user's order was that names come out of such statements. A `self`
classification is the one case with no ambiguity to resolve — the speaker is
talking about themselves, and there is no other party the name could belong
to. So `self` now applies immediately, is marked as automatic, and carries an
undo chip.

The two ambiguous kinds keep queueing, and the asymmetry is the point:
`third_party` belongs to nobody in the room yet, and `addressed` is inferred
from turn adjacency — exactly where a wrong guess is expensive.

Measured on the real 27B through the live endpoint, using the user's own
sentence as the first case:

| utterance | gate | classification |
|---|---|---|
| "hallo ich bin matthias" | passes (cue) | `matthias` / **self** |
| "darf ich vorstellen: Larisa Ehrenfeuchter" | passes (cue) | `Larisa Ehrenfeuchter` / **third_party** |
| "sag hallo, Moritz" | passes (cue) | **nothing** — the model declined |
| "Als ich sechs war, sah ich einmal ein wunderbares Bild." | blocked | never reaches the model |

So the pipeline did produce the right answer for the user's sentence. It was
invisible for two independent reasons, and both are now fixed: there was no
chip UI to show it, and `match_threshold` was minting a new speaker every
utterance so a name could never stick to anybody. The `addressed` row is an
honest gap — the model returned no candidate for it, and the adjacency logic
it feeds is therefore unexercised on real speech.

### 17.8 The acceptance gate: run the real client, not our own protocol client

Standing rule, 2026-08-03, after four consecutive defects reached the user.

Every fix in this project was proved with `front_door_test.py` — our own
WebSocket client. It speaks the protocol correctly and it never executes a
line of `client/index.html`. It was therefore structurally incapable of
catching any of: the suspended capture context (R1/R2), the frame accumulator
that never emitted (R2), the pitch shift from an announced sample rate (R3),
and a live push that does not reach the DOM (R5). Four rounds, one blind spot,
and the user was the test device for all four.

**`scripts/translator/client_gate.py` is now the gate.** Headless Chromium
with a fake microphone fed from a real speech clip, against the PUBLIC URL,
tapping the button with real DOM events. It asserts what a person sees: a
transcript line appears without a reload, audio reaches the playback path in
the right quantity for its announced rate, the console is clean, and the
socket is still OPEN after every turn.

It has two halves, and the second is the one that matters:

* **cold** — one or two turns from a fresh page;
* **soak** — twelve minutes with a turn every ~90 s. Idle sockets dying and
  sessions accumulating are both invisible in a cold run and are exactly what
  the user hit.

Chromium is a test vehicle and never part of the serving topology, so the
one-runtime law is untouched.

**The gate runs green twice before the user is asked to test anything.** The
user is final acceptance, never the test device.

#### 17.8.1 Handover state, 2026-08-03 (context exhausted mid-investigation)

**The gate exists, runs, and passes cold.** Measured on the live service
through the public URL with the real client:

| run | result |
|---|---|
| cold, 1 turn | line 6.3 s, audio 13.4 s, 268 frames @16 kHz, console clean, socket OPEN |
| 3 turns, 22 s apart | lines 1/3/5, audio frames 216/472/760, socket OPEN throughout |
| soak #1, 85 s gaps | turn 1 passed, **turn 2 produced nothing** |
| soak #2, 85 s gaps | turn 1 line 6.0 s / audio 13.1 s; **turn 2 line 4.0 s / audio 4.7 s** |

**Correction, and it matters for whoever continues.** After soak #1 I wrote
that the R5 fault is "idle-related". **Soak #2 refutes that**: identical 85 s
gaps, turn 2 passes in 4.0 s. The difference between the two runs was not
idle time — during soak #1 the user was testing on the phone concurrently
(`health` showed their session with 4 turns, idle 5.3 s, alongside the
gate's). Two conversations on one 5090.

So the leading hypothesis is now **contention, not an idle socket**: one
in-process TTS on the card serialising against the 27B and against a second
conversation. That also fits the user's "ultra langsam" better than any
transport fault, and it is consistent with the 52 s measured earlier when six
abandoned sessions were draining. **It is a hypothesis, not a finding** — the
evidence is two runs differing in load, not an instrumented measurement.

Excluded by evidence, not argument:

* **nginx idle timeout** — the live CT208 location carries
  `proxy_read_timeout 3600s` / `proxy_send_timeout 3600s`, read from
  `/etc/nginx/sites-enabled/efeu.ddnss.de.conf`, not from the repo template.
* **session pile-up at the time of R5** — `health` showed 2 of 8 sessions,
  both attached.
* **socket death** — `connection.ws.readyState` was 1 after every turn of
  every run, including across the 85 s gaps.
* **capture** — `microphone.frames` ~197 per turn in every turn, context
  `running` throughout.

**Exact next steps, in order:**

1. Read the soak #2 verdict:
   `tail -24 /root/.claude/jobs/1481bb40/tmp/gate_soak2.log` (9 turns, 85 s
   gaps; it prints a PASS/FAIL summary with per-turn client counters).
2. If it is green, run it a second time — the standing rule is **twice** —
   and only then tell the user to test:
   `CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python
   /spinning/htsglang-gpu/.venv/bin/python -u
   scripts/translator/client_gate.py --turns 9 --gap-s 85`
   (launch with `setsid`, or the shell exiting kills it — that cost one run).
3. Test the contention hypothesis deliberately rather than by accident: run
   the gate ALONE, then again while a second WebSocket client drives turns,
   and compare per-turn `line_s`/`audio_s`. If it holds, the answer is
   scheduling/admission between tenants on the 5090, not the transport.
4. Then §18: decompose the ~7 s first-audio figure per stage before
   rebuilding anything, per §18.4.

**Do not** re-litigate nginx, socket death or the capture chain without new
evidence — each is excluded above with its probe.

#### 17.8.2 The gate is green twice, and contention is now measured

**The gate passed twice in a row**, 9 turns at 85 s gaps each, console clean,
socket OPEN after every turn — soak #2 (inherited) and soak #3. §17.8.1's
"idle-related" reading stays refuted, and the standing rule is satisfied.

**The contention hypothesis was tested deliberately and it holds.** Two arms
through the public URL, same clip, same client, one with a second conversation
driven alongside by `scripts/translator/contention_probe.py` (new):

| stage, median | gate alone | gate + one second conversation |
|---|---|---|
| asr | 0.09 s | 0.10 s |
| embed | 0.02 s | 0.07 s |
| mt first token | 0.30 s | 0.19 s |
| tts first audio | 3.36 s | 8.21 s |
| **first audio (segment close → sound)** | **3.95 s** | **8.77 s** |
| worst turn | 7.00 s | 11.94 s |

**The contention is entirely in the synthesizer.** ASR is unchanged and MT is
unchanged — the 27B on TP=3 absorbs a second stream without cost, which also
means the MT hop is not the thing to protect. `inprocess_tts.py` serialises
every synthesis in the tenant behind one `asyncio.Lock`, so a second
conversation's turn does not run slowly, it does not run at all until the
first one finishes. The measured penalty is one whole synthesis: a median
+4.8 s, worst +8.4 s.

**Contention is real but it does NOT explain R5.** Soak #1's turn 2 produced
*nothing* within a 45 s budget; a 2.2× slowdown reaches ~9 s. So contention is
confirmed as a latency effect and is not sufficient as the R5 root cause. The
session-collection fix landed between those runs and is the other candidate.
Stated as an open end rather than closed, because two runs that differ in load
remain two runs.

**Decision: one conversation at a time is the honest capacity, and the queue
is correct.** The talker's measured RTF is 1.23 (4.88 s of audio from 6.00 s
of synthesis), so a single conversation already consumes more than real time
and no scheduling policy can make two fit. Queueing is therefore the right
behaviour and was already what happened. What was wrong is that it was
invisible — the second conversation simply appeared to have become slow, which
is exactly the shape the user reported. So:

* `InProcessQwen3Tts.busy` exposes the lock;
* a turn that finds it held emits **`turn.queued` before it waits**, and the
  client writes "waiting - another turn is being spoken" into that turn's
  translation slot, where the arriving translation clears it;
* `Stopwatch.tts_wait_ms` separates queueing from compute on every turn, via a
  `ContextVar` (`backends.TTS_QUEUE_WAIT_S`) rather than a backend attribute —
  an attribute would carry whichever turn finished last, i.e. be wrong exactly
  when two conversations run and the number finally matters.

Verified on the live service, not at the desk: over four turns under load,
`tts_wait` was 3.13 / 4.96 / 0.00 / 5.73 s and the page announced the queue on
exactly the three non-zero ones. Compute stayed ~3.3-3.7 s in both arms, which
is what makes the split trustworthy.

#### 17.8.3 The phone, read from its own data — sent is not played

The user reported no sound and no fluid updating on a real Android phone while
the gate was passing. Read from HIS session (`a6806f9f626c`) rather than from
rig runs, three things are now established and one is not.

**1. His client is NOT stale.** Every diagnostics frame from his device
reports the current build. The PWA/HTTP cache hypothesis — the only structural
difference between his device and the gate — is refuted by his own frames, not
argued away. His sessions are identifiable in the log because the gate's
report `track_label: "Fake Default Audio Input"` and his reports `"Standard"`.

**2. The server sent him audio, and the pipeline worked.** His five turns
carry correct German recognition and correct Spanish translations
("Hallo, ich bin Matthias. Wer bist du?" → "Hola, soy Matthias. ¿Quién eres
tú?"), the clone voice engaged on several of them (`mode: clone`,
`downgraded: false`), and the journal holds **53 `turn.audio` events**; a
read-only replay pulled **1024 binary frames, 655 kB**. The non-silence gate
never fired. So this is not "the server produced nothing" and not "the server
produced silence" — the audio existed and left the building.

**3. Therefore the break is between the socket and the speaker**, on the
device. That is a different fix from everything attempted in the previous
rounds, all of which were server- or transport-shaped.

**4. WHY it does not play is NOT yet proven, and must not be claimed.** There
was no output telemetry at all — the diagnostics frame described the capture
context and said nothing about the playback context. Two candidates survive
and they are distinguishable by one number:

* the output `AudioContext` never started (mobile autoplay policy), or
* it started and played into a muted or elsewhere-routed sink (media volume,
  Bluetooth), which no code change can fix.

**The defect found in the code, which is real regardless.** `Playback.ensure()`
created the output context lazily, and `beginTalking()` called it *after*
`await microphone.start()`. Awaiting `getUserMedia` ends the gesture's
transient activation, so the context was created outside a user gesture; a
mobile browser then leaves it `suspended`, ignores `resume()`, schedules every
buffer against a clock that never advances, and throws nothing. Fixed: the
unlock now runs as the first statement of `beginTalking()`, before any await,
and starts a one-sample buffer to make the browser commit.

**The gate could not have caught this, and now says so.** `client_gate.py`
counted `playback.push` calls — "the frames reached the playback path" — which
is true on a phone that makes no sound. Worse, it launched Chromium with
`--autoplay-policy=no-user-gesture-required`, waiving the exact policy at
issue. The override is removed and the gate now asserts the OUTPUT: context
state `running`, `currentTime` advancing, every pushed frame scheduled, no
push errors. **But that is still not equivalent to a phone**, and this was
tested rather than assumed: `--prove-can-fail` disables the page's unlock at
runtime, and headless Chromium *still* reports `running` — it starts its
context regardless of gesture, because it has no real audio device. So:

> The headless gate is necessary and it cannot speak for Android's autoplay
> policy. Any future "the gate is green" about audibility is worth exactly
> nothing on its own. The device telemetry below is the instrument for this
> class.

**What was added so the next real press settles it.** The diagnostics frame
now carries a `playback` half — context state, `currentTime`, sample rate,
frames pushed vs scheduled, the outcome of `resume()` (resolved/rejected), and
the last errors — logged server-side on one line next to the capture half. The
page also stops failing silently: when audio arrives while the output is not
running, it says "the browser is blocking sound output — tap 'tap to speak'
once to allow it" and sets the status to `sound blocked`.

**Decision procedure for the next press**, so nobody has to guess again:

| his `playback` telemetry | reading |
|---|---|
| `state: running`, `current_time` advancing, `pushed == scheduled` | the browser played it — the fault is the device's volume or output routing, not the code |
| `state: suspended`, `last_resume: rejected...` | the autoplay block; the unlock fix targets exactly this |
| `pushed > scheduled`, `errors` non-empty | a decode/scheduling fault, localised by the error text |

**The second complaint, "it does not update fluidly", is a separate and
measured thing**: his own turns took 3.3-6.1 s to first audio and up to 10.2 s
in total, which is the §18.4 decomposition (85 % non-incremental clause
synthesis), not a transport defect.

#### 17.8.4 The noise, and the one-root-four-symptoms theory — two real roots

User, escalating over one session: sound arrives as noise the instant "tap to
speak" is pressed, before anything is said; and "everything is ultra-noisy",
the cloned voices included. The working theory offered was a single root: the
microphone leaking to the speaker, contaminating capture, and therefore
poisoning recognition, language ID, embeddings and clone references at once.

**The proposed leak does not exist, and that was checked before anything was
changed.** Both capture paths already terminate in a zero-gain sink --
`index.html:397` (`this.sink.gain.value = 0`, worklet) and `index.html:449`
(`mute.gain.value = 0`, ScriptProcessor). The unlock buffer is one sample from
`createBuffer`, which is zero-filled by specification. Neither can put the
microphone on the speaker. Refuted by code, not by argument.

**Root 1, and the actual source of the noise: history was being spoken.** A
fresh page load resumes from cursor zero (`index.html:560`, `:579`), and the
server replayed the whole journal *including every `turn.audio` payload*,
emitted identically to live audio (`server.py:381`). His session held 53 such
events / 655 kB -- measured directly. Worse, in combination with the output
unlock: while the context is suspended `currentTime` is frozen, so every
replayed buffer was booked against the very start of the timeline; the moment
he tapped, the context resumed and **the entire conversation fired at once**,
overlapping. That is the "noise on tap, before I said anything", and it is
also why voices sounded noisy -- they were several turns playing over
each other. Fixed on both sides: the server marks replayed events, the client
refuses to speak history, refuses to schedule into a stopped clock, and resets
the queue on every new socket.

**Root 2, latent and genuinely of the shape that was suspected: two capture
paths at once.** The word `disconnect` appeared NOWHERE in the client.
`startFallback()` switched to the ScriptProcessor without retiring the
worklet, so a worklet that was merely SLOW to start -- more than the 400 ms
watchdog, ordinary on a cold phone -- kept delivering afterwards and *both*
paths called `handle()`: two interleaved copies of the microphone in one
uplink stream. That is one defect with exactly the four downstream symptoms
predicted -- garbled recognition, a flipping language decision, an embedding
that lands somewhere new every utterance, and a clone reference built from the
mess. Fixed by disconnecting the worklet and its sink when switching.

**The good news, and it is load-bearing:** noise coming OUT of the speaker
proves the output path on his device is no longer blocked. The unlock fix of
§17.8.3 works; what it unblocked was the wrong content.

**The synthesizer is not noisy, measured rather than assumed.** Through the
public front door, one full turn: WER **0.000** on the returned Spanish audio
re-recognized, peak 0.397 over 4.16 s, `clone` voice, speaker similarity
**0.971** against a 6.22 s reference, language `es` at 0.999. Server-side
audio quality is clean end to end, so the noise was client-side playback and
the embedder separates a speaker cleanly on clean input.

**Consequently the tuning order stands as instructed and for a measured
reason:** do not swap ASR models or recalibrate speaker thresholds against
contaminated input. Capture correctness first, then re-measure WER and speaker
similarity, then decide whether a model swap or recalibration is still needed.

#### 17.8.6 Constrained detection had reach almost zero

User: the identifier answered "English" for German speech, with English chosen
nowhere in the pair selection. Requirement 5 (addendum 5 v2) asked for
detection constrained to the table's languages, and it WAS implemented, wired
and tested -- `set_restrict_languages`, `constrained_language_choice`, the
lot.

**It was pushed only from `set_routing_pairs`.** Auto mode is the default, and
in auto mode the whitelist was explicitly CLEARED -- a test even asserted that
("auto mode clears the whitelist"). So a session opened with
`--participants de,es` and no manual table ran with an EMPTY restriction,
which `constrained_language_choice` correctly reads as "no restriction at
all". The identifier was free to answer any of 98 languages, and on a short
ambiguous German opening it answered `en`. A whitelist pushed only on a path
most sessions never take does not exist.

Fixed: the bound in auto mode is the PARTICIPANT set -- the same set auto
routing eliminates over -- and it is pushed before every recognition rather
than once, because one recognizer serves every session and a whitelist
installed at session start would be whichever session opened last. The old
test was rewritten to the corrected contract with the reason recorded in it.

Falsifier, as ordered: a recognizer fake that reports `en` at 0.55 with German
a close second, driven through the shipped decision function. Unfixed it
routes the turn as English; fixed it can only answer `de` or `es`. Verified by
reverting the fix and watching the assertion fail.

#### 17.8.8 The whitelist binds the LABEL, never the DECODE

User, after §17.8.6 had fixed constrained detection: his wife spoke German,
the identifier answered English, and the turn after that "something else
again". If detection were genuinely bound to `{de, es}` both answers would be
impossible, so the fix looked as if it had reach zero in production.

**It does not. Read from his session's journal rather than from the raw
recognizer log, the constraint demonstrably ACTED:**

| stage | value |
|---|---|
| faster-whisper raw identification | `is` (Icelandic), p = 0.99 |
| `turn.transcript` language | **`de`** |
| `turn.done` source / targets | **`de`** / `['es']` |
| `turn.transcript` text | **`Það er ló. Ég finn.`** |

The label was narrowed to the participant set exactly as designed. What was
never narrowed is the **decoding**: `asr_backends.py:222-228` calls
`self._model.transcribe(...)` with no `language=`, deliberately -- the comment
above it (`:214-219`) explains that passing one would disable Whisper's own
identification and pin the direction, which requirement 5 forbids.
`constrained_language_choice` then runs on the RESULT.

So Whisper decoded freely across 98 languages, produced Icelandic text, and
the label was corrected to German afterwards. Relabelling Icelandic text as
German does not make it German -- and the user sees the text, not the label.
The earlier turn is the same defect in the other direction: text `Thank you.`
carrying language `es`.

**The reach of the whitelist is one field.** This is the parameter-reach
lesson again in its purest form: the mechanism exists, it is wired, it is
tested, it demonstrably fires -- and it acts on a value that is not the one
the failure runs through.

The fix is a second pass, and it does NOT violate requirement 5: identify with
the model's own posterior, narrow that posterior to the participant set, then
DECODE with the language the constrained identification actually chose. The
direction is still discovered rather than declared; it is discovered from a
restricted candidate set instead of an unrestricted one. What must not happen
is decoding with a language nobody identified.

Gate arm to add with the fix: foreign or ambiguous audio must come out as the
best WHITELIST language *in the text as well as the label*, across a session
resume and with a second client driving turns concurrently.

**FIXED, 2026-08-03** (`asr_backends.py`, `FasterWhisperAsr.transcribe`).
The remedy is the one stated above and nothing more: when a restriction is in
force, `detect_language()` runs first, `constrained_language_choice` narrows
that posterior to the participant set, and `transcribe()` is then called with
`language=` set to what constrained identification chose. The direction is
still discovered from the audio -- discovered from a restricted candidate set
instead of an unrestricted one -- so requirement 5 is intact.

Three properties were deliberately built in, because each is a way this could
have shipped looking correct:

* **The unrestricted path is untouched.** No whitelist means no
  `detect_language` call, one encoder pass, byte-identical behaviour. Pinned
  by a test that counts the passes, not by inspection.
* **The cost is named.** One extra encoder pass, paid only when a restriction
  exists. ASR is 2 % of first audio (§18.4), so this is ~0.1 s bought for
  correctness -- it is not free and it is not significant, and both halves of
  that sentence are load-bearing.
* **A model without `detect_language` degrades** to the old single-pass path
  rather than raising. A version difference must not take a turn down.

Falsifier: `test/registered/translator/test_decode_language.py`, driving the
SHIPPED adapter over a fake whose free decode returns the journal's actual
Icelandic line and whose directed decode returns German. **Can-fail proof
run** (the §17.8.7 lesson): with the two-pass branch disabled, three
assertions fail with `AssertionError: 'Það er ló. Ég finn.' != 'Das ist alles,
ich glaube.'`; with it enabled, 6 pass. Suite 438 (from 424), ruff and
codespell clean.

**The NeMo backend still has this hole** and says so in a comment at the
place it would be closed. It is not the shipped recognizer and NeMo is not
installed in this venv, so a fix there could not be executed or tested --
writing it would be a desk fix, which is the thing this project keeps paying
for. Recorded rather than silently half-done.

Still open: the gate arm above (foreign audio through the real client, across
a resume, with a second client driving turns).

**Shipped to the live service and gated twice, 2026-08-03.** Build
`daac6753db`, tenant restarted from `boot_tenant.sh`, log
`tenant16_decodefix.log`. Cold run PASS; soak 9 turns / 85 s gaps PASS twice
in a row, console clean, socket OPEN throughout both.

This is also the two-pass path's first execution against the REAL
faster-whisper -- until then it had only ever run against a fake, which is the
desk-code trap this project keeps paying for. **The measured cost of the
second pass is smaller than estimated**: ASR medians 0.11 s and 0.12 s across
the two soaks, against the 0.09 s baseline of §18.4. Roughly +0.02 s, not the
~0.1 s budgeted -- `detect_language` is cheap next to a full decode.

One honest observation from the soak, recorded so it is not mistaken later for
a regression of this fix: the gate's fake microphone LOOPS a 3.9 s clip, so
some turns are fragments, and Whisper hallucinates on fragments (one turn
returned `espelho.` under a `de` label). That is a property of a looped test
clip, not of the constrained decode, and it is not evidence either way about
his real speech.

#### 17.8.5 Speaker identity: the cascade, and the continuity guard

User: "I am recognized as somebody different every time", and then the
consequence: "lots of different voices came out of what I said -- it cloned a
new voice every time."

**The cascade was one branch.** In the uncertainty band
`[uncertain_floor 0.583, match_threshold 0.637)` `SpeakerRegistry.assign()`
fell through to MINTING a speaker. A new speaker has no reference buffer, so
it receives a fresh preset or a fresh clone -- and one person changes voice
mid-conversation. The badge machinery was already there; what was missing was
that the borderline case defaulted to *discontinuity*.

**Continuity guard, with the asymmetry stated:** in that band the turn is now
attributed to the nearest EXISTING speaker and the line keeps its uncertain
badge. A wrong continuity is a badge the user can tap; a wrong new speaker is
a different voice that has already been heard. A genuinely distant voice
(below the floor) still mints, so two real people are never welded together.

Two things the tests forced, both worth keeping:

* **A guess must not move the centroid.** The pre-existing invariant "an
  unconfirmed uncertain line never moves a centroid" survives the guard: the
  guard changes who a line is attributed to, not what the system believes
  afterwards. Confirmation -- manual, or a later confident turn -- moves
  centroids.
* **A guess must not become the voice.** `_maybe_admit_reference` waives its
  identity gate for a profile's FIRST reference (the enrollment anchor), and
  since a guarded assignment does not fold, `observations` never grows past 1
  -- so every borderline turn would have slipped through that exemption and
  become the clone prompt. Guarded assignments are now barred from the
  reference buffer explicitly. This was caught by a failing assertion, not by
  reading.

`_auto_resolve` also had to stop excluding the line's own speaker from its
ranking. That exclusion was correct while an uncertain turn minted its own
self-matching profile; under the guard it meant a continuity guess could only
ever be overturned, never confirmed, and the badge would sit there forever.

**Still open on this front** (unbuilt, deliberately, because it must be
measured on CLEAN capture first per §17.8.4): threshold calibration against
his real device audio, per-decision candidate-cosine logging persisted
server-side, rolling reference growth, and speaker MERGE with reference-buffer
union and history relabelling.

#### 17.8.9 Handover, 2026-08-03 (third session, context exhausted)

**Live build `78865baf74`**, tenant restarted from
`/root/.claude/jobs/1481bb40/tmp/boot_tenant.sh`, log `tenant15_fixed.log`.
Branch pushed to `origin/feat/live-translator-466` at `2b51657d6d`. GPU
holder heartbeat is `hb466.sh` (PID reparented to init, survives the session).

**Done this session**

* the live-delivery regression: root-caused from his session, fixed on both
  sides, falsifier green in both directions (§17.8.7). CONFIRMED BY THE USER
  on his device afterwards -- "quality almost great";
* the gate's replay-then-live arm, with the can-fail proof that took two
  false greens to earn (§17.8.7);
* the language whitelist reach hole, root-caused with evidence and recorded
  but NOT fixed (§17.8.8). This is the top build item;
* two new standing orders recorded: reset button (§19.3b), temporal
  continuity prior (§19.7).

**Open, in the order the user's reports set**

1. **§17.8.8, the decode-side language fix.** Root cause is proven and the
   remedy is stated; only the build and its gate arm remain.
2. **The speaker complex.** Nothing of §19.2 is built yet, and the first step
   is forced: `speakers.py` contains NO logging at all, so every assignment
   decision is currently unreadable after the fact. Persist the per-decision
   candidate cosines FIRST or the calibration is blind again. Thresholds in
   force are `match_threshold 0.637` / `uncertain_floor 0.583`
   (`speakers.py:182,230`). His turn-3 mis-mint therefore sat BELOW 0.583
   against his own profile, which on clean audio is the number that wants
   explaining -- and the contaminated pre-fix references are gone now (the
   tenant restarts purged every profile), so the re-measurement is finally on
   clean input. Then the §19.7 prior, then merge.
3. **§18.4 item 1, the first-clause split.** Checked this session and worth
   recording so nobody re-checks it: MT-to-TTS streaming is ALREADY built --
   `session.py:1566-1578` runs `async for delta in mt.translate_stream(...)`,
   pushes each delta through `SentenceAccumulator`, and calls `speak(unit)`
   per completed unit. So the "does TTS wait for the whole translation"
   intermediate win does not exist; it is already streamed. What remains is
   exactly §18.4's item 1, shortening the FIRST unit: the accumulator emits at
   SENTENCE boundaries, so first audio still costs a whole sentence of
   synthesis. Note also that `speak()` is AWAITED inside the MT loop, so the
   `mt_total` figure in gate output includes synthesis and is not pure MT.
4. §19.3 UI redesign (unstarted), §19.5 quality ladder, §19.6.

**Second speaker exists now.** His wife's turns are the first real
second-speaker data in the project and the practical test case for §19.7's
language-change counter-signal -- once §17.8.8 makes her turns decode
correctly at all.

#### 17.8.7 A cursor outliving its session silenced every live event

User, on the build that had just fixed the replayed-history noise: "nothing
happens when I press tap to speak", and then the precision that made the
diagnosis: "if I reload the page, the text I spoke before is there. If I speak
again now, nothing updates again, no sound comes out. Reload again, and the
NEW translation is there too."

That last sentence excludes almost everything. The uplink, the recognizer, the
translator and the synthesizer all worked -- the translation existed
server-side and appeared on the next reload. Only LIVE delivery was broken.

**Read from his session before anything was changed**, in this order:

* two turns completed server-side, each with a full ASR/MT/TTS run in the
  tenant log (`faster_whisper`, the MT request, and `qwen_tts` synthesis lines
  per clause);
* the journal held every `turn.audio` payload intact -- `audio_follows` true,
  `audio_evicted` false -- read back over a read-only replay client, so the
  audio existed and the 24 MB eviction budget was nowhere near;
* his own `playback` telemetry reported the output context `running` with
  `currentTime` advancing, while `pushed`, `dropped` and `skipped_replay` were
  all **zero** a full minute after the first turn's audio existed.

`Playback.push()` increments `pushed` as its FIRST statement, so zero pushes
proves `onBinary` never ran; `skipped_replay` zero proves the fresh
replay-refusal was not eating them either. No binary frame ever reached the
device. That is a fourth case §17.8.3's decision table did not have, and the
table now needs it: **nothing arrived to play.**

**One cursor, both symptoms.** The client persists `translator.cursor` beside
`translator.session` and resets neither when the server hands back a session
it had to MINT. Once the idle collector has taken the old session, the server
re-creates it under the same id with a journal starting at zero, while the
page still carries the old conversation's high-water cursor. `_handshake`
seeded the delivery cursor from that number without validating it, and
`Journal.since()` answers a cursor past the end with an empty list and no gap
-- so `_journal_pump` had nothing to send until the new journal grew past a
high-water mark belonging to a conversation that no longer existed.

The reload kept working because the written record travels on the handshake's
`transcript` frame, keyed on a SEPARATE cursor the client does not persist.
That is precisely what hid this: the one path that still worked was the one
the user could see.

**The session-collection fix is what made a latent cursor bug reachable.**
Before it, his session survived and the cursor stayed valid. Neither change
was wrong on its own; the reach of the first was what the second changed.

Fixed on both sides, the server half deliberately, because it repairs clients
already on a phone: the handshake clamps a resume cursor to the journal's
`next_seq` and logs when it does; `resumed` reports whether the session
actually existed instead of `bool(session_id)`, which was true whenever the
client named any id; and the client drops its cursor when the id it gets back
is not the one it asked for, or when the server did not resume.

**The gate could not have caught this, and now has an arm that can.** Every
previous run loaded the page once and spoke into it, so the gate only ever
exercised a client with NO history -- a defect that needs a reload was
structurally invisible to it, the same blind-spot shape as §17.8's original
lesson. `--reload-after N` now drops the session server-side (the collector,
emulated, in a second instead of the idle timeout), reloads the page so
`sessionStorage` carries the stale cursor exactly as a phone does, and
requires the next turn to arrive live.

**Two things this arm taught, both worth keeping.** It PASSED twice against a
build with the fix deliberately disabled before it was pinned down: the client
reconnects on a backoff after the delete, that reconnect re-creates the
session, and if enough events land before the reload the new journal overtakes
the stale cursor and the defect hides. The arm now suppresses that reconnect
(`connection.closedByUser`) and PRINTS the carried cursor, because the number
is the difference between a gate and a decoration. Only after that did the
can-fail proof succeed -- carried cursor 84 against `journal_next_seq` 1,
FAIL unfixed, PASS fixed. An arm whose can-fail proof has not been run is not
evidence, and this one would have shipped green and blind.

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

---

## 18. Streaming, not turn batching (architecture order, 2026-08-03)

User order, verbatim in substance: *"that has to be streamed into each other,
the latency is ultra high like this — the translated text only arrives long
after I stop speaking. Our compute should be far more than enough to translate
this faster than real time."*

The order is correct about the shape of the system. It is recorded here before
any of it is built, and it is the top item after the R5 transport work — a
streaming pipeline on a socket that dies silently would be effort spent behind
a broken door.

### 18.1 What is actually streamed today — the reach check

To be filled with `file:line` per stage BEFORE any rebuild, because the
original architecture called for "streaming ASR" and the code may or may not
deliver it. Current expectation, to be verified rather than assumed:

| stage | expected today | evidence |
|---|---|---|
| segmentation | whole turn closes first | `segmenter.py`, VAD/release closes a segment |
| ASR | whole segment at once | `_recognize` takes a closed `Segment` |
| MT | streamed internally, regrouped into clauses | `mt.py` `translate_stream` + `SentenceAccumulator` |
| TTS | per clause, but only after MT starts | `_translate_and_speak`'s `speak(unit)` |
| audio out | per synthesized chunk over the binary channel | `_emit` sends frames as they land |

If that table holds, the big serial block is **turn-close → ASR → first MT
token**, and the win is in overlapping recognition with speech rather than in
rebuilding the whole chain.

**Measured, and the expectation above was wrong about where the time goes.**
The table's *shape* holds — ASR takes a closed segment, MT streams into
clauses, TTS runs per clause — but "turn-close → ASR → first MT token" is not
the serial block. It is 0.4 s of a 3.95 s wait. See §18.4.

### 18.2 Target

1. **ASR partials live** — text appears in the transcript WHILE the person is
   speaking, marked provisional, then finalized.
2. **MT starts on stable prefixes** — sentence or clause boundaries inside the
   utterance, not at turn end.
3. **TTS per clause**, already the shape; what changes is that it may begin
   before the utterance is finished.
4. **Audio out chunked**, already the shape.

### 18.3 Consistency rules, which are the hard part

* A partial must never silently rewrite a committed translation. Corrections
  are visible — the same rule as §17.4's badge changes.
* Speaker attribution keeps working on the FULL turn: the 2.5 s window floor
  (§16.5) exists because identity decisions on shorter audio are noise. So the
  streaming display may run ahead of attribution and be attributed
  retroactively — never the reverse.
* Reference-buffer admission stays on complete segments for the same reason.

### 18.4 Measure before rebuilding

The 7 s first-audio figure must be decomposed per stage — waiting time versus
compute time, per the ms-per-round doctrine — before anything is rebuilt, so
the largest term is cut first rather than everything being rewritten blindly.
The decomposition also answers whether faster-than-real-time is reachable on
rung B (in-process talker, measured RTF 1.15 with the codec on the GPU) or
where the hard floor is. **State the floor honestly; never promise it.**

**Done, 2026-08-03.** The server already measured every stage (`Stopwatch`)
and shipped the numbers on `turn.done`; nothing was instrumented that did not
already exist — the gate simply grew a second listener on the client's own
socket and now prints the decomposition per turn. Five turns through the
public URL with the real client, an idle tenant, medians:

| stage | median | share of first audio |
|---|---|---|
| ASR (faster-whisper large-v3-turbo) | 0.09 s | 2 % |
| speaker embedding | 0.02 s | 1 % |
| MT to first token (27B, TP=3) | 0.30 s | 8 % |
| MT to the first clause boundary | ~0.45 s | 11 % |
| **TTS, first clause** | **3.36 s** | **85 %** |
| **first audio, segment close → sound** | **3.95 s** | |

Steady state is 3.57 s; the first turn of a session costs 7.0 s (cold path,
empty reference buffer) and drags nothing but the maximum.

**The recognition side is already fast enough to be uninteresting.** ASR,
embedding and MT together are 0.4 s. Streaming ASR partials would therefore
buy roughly nothing in *audio* latency — it would improve the moment
**text** appears, which is a real and separate benefit for the reading mode
(§17.1), but it is not the answer to "the translation arrives long after I
stop speaking".

**The whole term is one non-incremental clause synthesis.** `inprocess_tts`
generates a complete waveform and only then chunks it (§12, stated there
before it was measured), so first audio for a clause equals full synthesis of
that clause. Measured RTF **1.23** — 4.88 s of audio from 6.00 s of work.

**The floor, honestly.** With RTF > 1 the talker cannot outrun speech, so
"faster than real time" is not reachable on rung B *at all*, and no amount of
pipelining changes that: a stream whose producer is slower than its consumer
starves. What IS reachable on rung B, in falling order of value:

1. **Shorten the first unit.** First audio is proportional to the length of
   the first clause, not the utterance. Splitting the first clause at the
   earliest defensible boundary — the accumulator already owns clause
   splitting — converts a 3.4 s wait into roughly RTF × (first phrase), and
   is the only change here that needs no new capability.
2. **Overlap synthesis with playback.** Clause N+1 can be synthesized while
   clause N plays. At RTF 1.23 this does not keep up indefinitely, but it
   hides the *second* clause onwards behind the first one's playback, which
   is where a long turn currently stutters.
3. **A faster talker** — #488's native lane. This is the only lever that
   moves the floor itself, and it is the one this rung was always a stand-in
   for.

Items 1 and 2 are the §18.2 targets worth building; §18.2's item 1 (ASR
partials) is reclassified as a reading-mode feature, not a latency fix. That
reordering is a consequence of the measurement and is why §18.4 exists.

**A constraint the first-unit work must honour** (from the parallel §19.3 UI
build, 2026-08-03): the new surface's per-line replay is built on the
`turn_id` attribution carried by `turn.audio` frames. Shortening the first
synthesis unit changes how audio is chunked, so the rule is stated before the
work rather than discovered by breaking it: **every audio chunk keeps its
`turn_id`**, whatever the clause boundaries become. A chunk that arrives
without one, or with a different one per clause, silently breaks replay in a
client this document does not own -- and any change to that attribution is
reported so the UI side can follow.

### 18.5 Acceptance

Through the standing headless-client gate (§17.8), extended with latency
assertions per turn: time to first partial text, time to first audio chunk,
both logged per turn so a regression is visible as a number rather than as a
user complaint.

**Logging is in place** — every gate turn now prints the full server-side
decomposition and `--json-out` keeps it, so two runs are comparable stage by
stage. The **assertion** exists as `--max-first-audio-s` and is deliberately
**off by default**: with the floor now measured at 3.57 s steady state and
3.95 s median on an idle tenant, a budget belongs in the gate command once
§18.4's item 1 lands and the number it should defend is the *new* one. Turning
it on against today's floor would only pin the latency this work exists to
remove.

---

## 19. Standing user orders, recorded before they are built (2026-08-03)

Written down here because a feature order that lives only in a transcript is
lost at the next handover (§17 exists for the same reason). Priority is the
one the user set, and it is a measured priority, not a taste one: contaminated
input must not be tuned against.

### 19.1 STT quality

(a) **The whitelist bug is FIXED** -- see §17.8.6. (b) Beyond it, unbuilt and
to be decided on numbers taken with CLEAN capture (§17.8.4): decoding
parameters (beam, VAD filter, `initial_prompt` carrying session context,
`condition_on_previous`), and a candidate bake-off measured as WER on a German
test set -- `large-v3` against `large-v3-turbo`, and the Qwen3-ASR adapter the
runtime already carries -- with the VRAM budget respected and the cards held.
Session bias: the recent turns' language as a prior. **Measure, do not
prefer.**

### 19.2 Speaker identity

Continuity guard shipped (§17.8.5). **Per-decision logging shipped,
2026-08-03** -- it was the forced first step, because `speakers.py` contained
no logging at all and every assignment was therefore unreadable after the
fact. What is recorded per decision, in `SpeakerRegistry.decisions` (bounded
ring, 200) and on one log line:

* the full candidate ranking, taken BEFORE any centroid folds -- a confident
  match moves the centroid it matched, so ranking afterwards would record
  numbers the decision never saw;
* which branch fired (`match` / `guard` / `capacity` / `mint` / `manual`), so
  a continuity guess is distinguishable from a confident match without
  re-deriving it from the score;
* both thresholds in force at the time, so a series stays interpretable
  across a retune;
* a `prior` slot, present and neutral, so the §19.7 prior does not change the
  record's shape when it lands and split the calibration series in two;
* manual attributions together with what the automatic path WOULD have said
  -- the only decisions where the correct answer is known.

Readable live at `GET /api/translator/sessions/{id}/speaker-decisions`, which
does not require grepping the tenant log and survives what a log rotation does
not. Pinned by `test/registered/translator/test_speaker_decision_log.py`.

Open: the calibration itself, on the clean profiles the restarts left behind
(§17.8.9) and now with a genuine second speaker; the §19.10 voice-class pick,
which comes first because it is audible on first contact; rolling reference
growth; and speaker merge (union the reference buffers -- more reference is a
better clone -- keep ONE voice, relabel the history, which is a correction and
not a rewrite of content).

### 19.3 UI redesign (unbuilt)

User: "the current design is absolutely ugly and unfunctional... look at how
others build live translators on mobile, take your lead from them", and about
the language chooser: "the selection at the top, really?"

Research the established patterns first (Google Translate conversation mode,
DeepL mobile, iTranslate) and rebuild to them: a clear two-direction view with
large language buttons, an unmistakable speaking-state indicator, large
readable conversation bubbles coloured per speaker, and a completely reworked
language-pair chooser. Usable outdoors on a phone: large touch targets, high
contrast, functional before pretty, but both. **Every existing capability is
rearranged, none is dropped** -- tap-toggle, speaker buttons, badges,
renaming, the bilingual transcript, auto-scroll, connection and queue state.

### 19.4 Pipeline speed (unbuilt, measured target)

§18.4 stands: 85 % of the 3.95 s is the first clause's synthesis. Build, in
order: a shorter FIRST synthesis unit (synthesize and stream the opening
clause immediately, the rest follows), MT streamed at sentence boundaries, ASR
partials into the transcript (a reading-mode win, not an audio-latency one).
Then pull the latency assertions up in the gate (`--max-first-audio-s` exists
and is off by default until there is a new floor worth defending).

### 19.5 Quality ladder, upwards (unbuilt)

User: "couldn't it be somewhat better quality? if there were enough bandwidth,
with a fallback?" -- he is on LAN wifi with bandwidth to spare, and the
pipeline runs 16 kHz everywhere. This is the upward extension of the existing
bad-network ladder (addendum 6 asked for adaptive Opus downwards).

**The premise is verified, not assumed:** the talker synthesizes at 24 kHz
(`inprocess_tts.py:69`, `config.py:69`) and `PIPELINE_SAMPLE_RATE` is 16000
(`audio.py:62`), so the wire downsamples for no reason on a LAN.

1. **Output side, the largest audible win.** Carry native 24 kHz when
   bandwidth allows (high-bitrate Opus, or PCM on a LAN), with the client
   playing it at the correctly announced rate. EXTEND the rate chain, do not
   rebuild it -- it is freshly fixed and under test.
2. **Capture side.** Capture device-native (typically 48 kHz) and carry it
   when bandwidth allows. The chief beneficiaries are the CLONE REFERENCES:
   voices cloned from 16 kHz references sound dull, so the reference buffers
   should hold the highest available rate (check what the TTS reference path
   natively digests). ASR resamples to 16 kHz internally anyway. This couples
   directly to the reference rebuild after the capture fixes -- build the NEW
   references at high quality immediately.
3. **Negotiation.** A capability handshake measures or estimates bandwidth
   (a LAN identifies itself by RTT and throughput), picks a rung, and
   renegotiates on degradation over the existing reconnect path. The chosen
   rung is VISIBLE (a quality badge, which belongs in the new §19.3 surface)
   and every change is named rather than silent. Sticky per session, with a
   manual downgrade.
4. **Proofs.** Per rung, the rate/duration proof in the gate (an N-second clip
   stays N seconds); for the quality claim, an A/B artefact -- the same
   utterance rendered at 16 kHz and 24 kHz, written to the results directory
   so the user can listen; and the bandwidth fallback tested hermetically
   through an injected throttle (check the reach of the addendum-6f network
   chaos harness first).

### 19.3b A reset button (unbuilt, user order 2026-08-03)

User: "it generally needs a reset button." A reset in the surface that
starts the conversation genuinely fresh, and is effective on the SERVER, not
only in the display:

1. the transcript is cleared (the existing clear semantics);
2. every speaker profile goes -- voice references, centroids, names and pool
   assignments;
3. the queue and any pending turns are dropped;
4. a fresh session id, which deliberately ends the sticky resume of the old
   session.

Behind a confirmation dialog that NAMES what is lost: a mistap must not
delete a conversation. Placed well away from the speaking controls in the new
layout -- no fat-finger risk beside the big button.

It is worth double right now: this is the user's own way to get rid of
contaminated old profiles without anybody intervening server-side. Which is
also the honest reading of what a "reset" is for -- the state that accumulates
here is exactly the state that goes wrong.

Gate arm: after a reset the health session is fresh, the speaker list is
empty, and the first turn mints a speaker again.

### 19.7 A temporal continuity prior on speaker assignment (unbuilt, user order)

User: "generally somebody speaks continuously and nobody else chatters in
-- the assumption should hold first that the same person is speaking, above
all when the words or sentences are close together." Recorded after §17.8.5's continuity guard had
shipped and STILL produced a new speaker on turn 3 of a clean session.

The assignment combines embedding evidence with a prior taken from the gap to
the previous turn. A short gap -- order of a few seconds, the threshold chosen
from his real session data and NOT guessed -- is strong evidence for the same
speaker as before; the prior decays as the gap grows. Only clearly
contradicting voice evidence (well past the minting bound AND nearer to a
different existing profile) or an explicit user action (speaker button, "+")
overrides it.

This composes with the existing guard rather than replacing it: the guard
handles the uncertainty band, the prior additionally moves the boundary itself
when temporal coherence argues for it. Conversational reality is that speaker
changes happen at pauses, not mid-flow.

The edge cases, handled honestly:

* a direct attribution by speaker button is absolute -- the prior is
  irrelevant there;
* after a long pause the prior is neutral;
* in a genuine dialogue exchange the prior must not drown out a REAL change.
  The turn's language identification is a strong counter-signal here and it is
  already available for free: a turn arriving in the OTHER language of the
  pair makes a change more likely than continuity, and it is folded in.

Made provable: the prior component is persisted per decision alongside the
cosine and the final score, and the falsifier drives synthetic sequences --
short gaps in one voice must NEVER mint, and a language-change turn must not
be overridden by the prior.

### 19.6 Cheap wins from the competition, to fold in where they are cheap

Replay button per line and tap-a-line-to-replay (this one also buys back what
§17.8.4's "never speak history" gives up), automatic language badges per
bubble, a visible recording level, and copy/share of the transcript. Judge
each on effort against payoff with the deadline in view; the core (§19.1-19.4)
comes first.

### 19.8 Denoising the capture, before anything analyses it (user order 2026-08-03)

User: *"my own voice still hisses badly ... run very good/strong denoising and
background-noise filtering, ideally some model, over the audio input before
the samples are analysed."* Recorded together with a positive confirmation on
the same build -- *"now it updates properly and it comes through"* -- so this
is a quality complaint on a working pipeline, not another delivery defect.

**(a) The free check, done first and answered.** The browser constraints are
already requested: `index.html` asks for `echoCancellation`,
`noiseSuppression` and `autoGainControl`, all `true`. So "they are switched
off" is refuted at the source and is not the explanation.

But that is what was REQUESTED, and a plain `true` in a constraint dictionary
is `ideal`, not `exact` -- a browser may ignore any of them and report
nothing. Nothing in the system ever checked what the track actually got. The
capture diagnostics now report `track_echo_cancellation`,
`track_noise_suppression` and `track_auto_gain` from
`MediaStreamTrack.getSettings()`, which the server already logs verbatim
beside the playback half. **The next diagnostics frame from his phone settles
whether the browser is denoising at all**, and that answer decides how much
(b) has to carry. This is the parameter-reach rule applied to a constraint we
do not own: requesting is not binding.

**(b) A real neural denoiser server-side**, ahead of the reference buffer and
the embedder. Candidates: DeepFilterNet3 and the RNNoise class -- streaming
capable, low latency. Licence to be checked before installation, installed
into the translator venv, integrated as an in-process module (one-runtime
law: no sidecar).

**(c) Measure, do not believe.** Denoising can HARM ASR -- this is a known
effect, not a hedge -- so the arms are separated and each is measured on his
real turns: WER with/without, speaker similarity with/without, and a listening
artefact check on the clone with/without, because voice identity must not be
ironed flat by the very step meant to clean it. Expected outcome is a
DIFFERENTIATED switch rather than one global flag: references always denoised
(they are the direct cause of clone quality), ASR only if WER does not suffer.
The raw signal stays in the journal in parallel, so nothing is irreversible.
The denoiser's own latency is quantified and has to fit the real-time chain --
it is added to the §18.4 decomposition like any other stage.

This couples to §19.5's capture-side rung: a 48 kHz reference that is also
denoised is the target state, and both changes land in the same consumer.

### 19.9 A faster MT model, and the NVFP4 lane's first real load (user order)

User: *"qwen3.6-35b-a3b for example?"*, and then: *"prefer NVFP4 -- that would
be a good test of that at the same time"*, and finally *"of course the NVFP4
MoE model with the phase-optimal cuts for prefill and decode, and with MTP."*

**The honest arithmetic first, because it bounds the prize.** Per §18.4, MT to
first token is 0.30 s of a 3.95 s first-audio figure. A faster LLM can
therefore remove at most ~0.3 s of START latency; the dominant term remains
the first clause's synthesis at 3.36 s (85 %), which is §19.4's job and not
this one. Anyone reading this expecting a latency fix should read §18.4 first.

**Where it could nevertheless pay, stated as hypotheses to be measured:**

1. **MT throughput on long sentences.** An A3B MoE activates ~3 B parameters,
   so the per-token cost falls sharply against a dense 27 B. Long turns are
   where the current chain visibly stutters.
2. **SM contention against the talker.** This is the more interesting one.
   The talker's measured RTF is 1.23 -- it is AT the capacity limit
   (§17.8.2), and it shares the 5090 with MT. Less LLM work on that card may
   help the term that actually dominates, indirectly. Whether it does is an
   empirical question about co-residency, not about the LLM's own speed.

**The evaluation is deliberately a full-stack one**, because the model is also
the vehicle for three separate questions and running them separately would
cost three boots:

* **NVFP4 lane proof.** Checkpoint availability on the box first (otherwise
  name HF candidates with download size and check disk). Then read the
  #332/#336/#323 line's reach AT THE CODE before assuming anything -- native
  fp4 on the 5090 (sm120) versus the dequant fallback for unpackable layers
  on the sm86 ranks. The PLACEMENT is a deliberate choice and must be named
  in the report: MT alone on the 5090 is a purely NATIVE fp4 test; spread
  across the 3080s additionally exercises the dequant lane. Both are
  valuable; run both if cheap, and say which arm produced which number.
* **Phase-optimal cuts.** Not a single compromise layout: have the planner
  solve prefill and decode layouts separately for this model x quant (#485
  phase matrix / `uneven_perf`), with MoE as its own family axis. Verify the
  machinery's reach at the code, per the mechanism-reach law.
* **MTP/spec.** The model carries an MTP head. Watch the #318/#387 family:
  a draft that wrongly inherits the target's quantisation method, or an
  acceptance rate that collapses on a quantised target. Acceptance length is
  measured from `meta_info`, NEVER from `spec_ema_accept_len` -- that field
  is not the acceptance length.

**A tight fit is NOT a rejection reason** (user correction, 2026-08-03, aimed
at exactly the wrong phrasing "the VRAM arithmetic beside talker/ASR is
mandatory", which read as a go/no-go). When space is short, the fork spills
and offloads -- that is what the machinery is for. So the VRAM arithmetic
stays, but as a PLANNING INPUT (how much expert residency stays on the card,
what goes to RAM), never as a gate. Reporting "does not fit beside the talker"
as a stopper is a defect in the report.

The machinery to price, with its reach verified AT THE CODE and cited
`file:line` -- never from a catalogue shorthand, per the repo's mechanism-reach
law, and the catalogue sections read must be named in the report:

* MoE expert offload to RAM, load-time aware (#77/#123 line; the NVFP4 half
  was brought up with the #323 round);
* the CUDA-graph-compatible offload route (#122/#462 breakable path), so
  offload does not cost the graph;
* the KV pressure ladder and the runtime VRAM budget dial (#330), if
  coexistence with the live translator service needs adjusting.

**And the option space is not one-dimensional** (second user correction, same
day): spill/offload applies to the DENSE 27 B too, not only to the MoE
candidate. The comparison is therefore NOT "27 B fully resident versus A3B
with expert offload" -- both candidates are priced with their best
offload-supported placement out of the memory matrix. Dense weight shards are
just as parkable/spillable (weight-shard asset class, KV session spill, graph
state offload #93/#102/#286, the GDN slot ladder #364, the #330 dial); MoE
merely adds one extra and unusually cheap axis in its experts. Give the best
offload-supported variant per candidate, not only the fully resident one.

**Side effect worth having:** the run then also exercises offload x NVFP4,
a crossing point that has never been booted. Report it as its own finding --
never-booted crossings are exactly the gaps an integration run exists to find.

**Deliverables:** the placement/offload arithmetic per candidate (planning
input, not a gate); a DE<->ES quality sample of 10 sentences side by side
against the current 27 B; first-token and throughput numbers; and the NVFP4
and offload-crossing findings reported SEPARATELY from the MT decision,
because they feed the capability catalogue whichever way the model choice
goes. **A model swap is a ship decision and waits for an explicit go.**

### 19.10 The placeholder voice must match the speaker (user order, field defect)

User: *"in the initial selection of the synthetic voices something has to
quickly detect whether a woman or a man or a child is speaking. right now it
sometimes takes a female voice for me when there is no profile yet. for my
wife it takes a male voice until her own voice clone exists."*

Both directions observed on real turns, which makes this the most
user-visible defect remaining: it is what EVERY first contact sounds like,
before any clone exists.

The requirement is a fast voice-class decision from the first turn's audio --
fractions of a second, at the start of the turn -- choosing among the
manually creatable man/woman/boy/girl profiles rather than inventing a new
taxonomy.

**Check what is already computed before fetching a model.** The speaker
embedding is produced for every turn anyway, and ECAPA/ResNet-class embeddings
carry sex information densely; an F0/pitch median over the segment is nearly
free and is the classical discriminator (with the well-known adult-female /
child overlap that a formant check or the embedding can break). The order of
investigation is therefore: what the existing embedding already separates,
then pitch, then a dedicated classifier -- and only if the cheap paths
measurably fail.

Test cases are drawn from the journal rather than synthesised: his turns that
picked a female voice, and his wife's that picked a male one. Both must flip
under the fix, and that is the falsifier.

This sits in the speaker complex AHEAD of threshold calibration: calibration
improves a number, this fixes something the user hears on contact.

### 19.11 The audio stack is a movable asset class, not a fixed cost (user order)

User, extending the placement calculus of §19.9: *"asr/talker naturally have
to be spilled too."* So ASR, the TTS talker, its codec/vocoder and the
diarization embedder are SHIFTABLE asset classes in every coexistence
calculation -- never a fixed overhead to be subtracted before the interesting
question starts. Any layout arithmetic that treats the 7.5 GiB audio budget as
immovable is answering an easier question than the one asked.

**(a) Price the vacate options too.** Alongside the MT candidates' placements,
cost "ASR/talker idle-vacate or park": the talker's weights can go to RAM
during pure LISTENING phases -- there is no active speaking turn then -- and be
restored at turn start, PROVIDED the restore latency lands under the TTS start
offset that already exists in the chain. That proviso is a MEASUREMENT, not an
assumption: quantify park and restore time for the talker weights and put the
number against the §18 latency chain before claiming the option is free.
The shape is promising precisely because §18.4 shows the talker is the
dominant term and idle between turns.

**(b) What is reachable TODAY, stated honestly.** As long as the audio stack
runs as its own tenant process (the state before #488), the htsglang ledger
cannot see it -- so the fork's spill machinery does not apply to it. Process
local park/load still works (unload the model to RAM, load it back); it simply
does not get the arbitration. Saying "spill the talker" today means that
cruder mechanism, and a report must not blur the two.

**(c) The #488 target architecture is confirmed by this order.** ASR, talker
and codec become LEDGER asset classes in the #286 register -- parkable, with a
residency ladder in the #305 sense. Recorded here as user-confirmed direction.
No immediate rebuild obligation; but from now on the calculations and the
design carry it.

#### 17.8.10 Handover, 2026-08-03 (fifth session)

**Live: server `cba698fb5b` (tenant PID 3776998, started 20:41:35, log
`tenant17_stopstream.log`), client `173d52e541`.** The client is read from
disk on every request, so a client-only merge goes live WITHOUT a restart --
that is why Cut 1 needed no second window and why the two halves can differ.
Always read both identities before diagnosing anything.

**Shipped this session**

* the three undeployed server posten of §19.12/§19.13/§19.10 (playback.stop,
  text decoupled from audio, placeholder voice class) went live in one
  restart, taken while `health` reported `sessions: 0`;
* the UI agent's Cut 1 (stop button + silence) merged onto that tip, client
  build `173d52e541`;
* two gate arms, `dcfb8ab57e` (below);
* the ASR capability reach fix, `974300a315` -- NOT YET DEPLOYED, it needs
  the next restart.

**The gate is 1 green / 1 red on the merged build, so the user must NOT be
asked to test yet.** Run A (`gate_cut1a.log`): 4 turns, reload arm, stop arm,
console clean, PASS. Run B (`gate_cut1b.log`), identical invocation: turn 3
produced a transcript line at 4.0 s and then NO translation and no audio
inside the 90 s budget -- the page sat on `translating`. What was excluded
before writing this, by probe rather than argument: the MT server was alive
throughout (PID 3602337, unchanged since 14:21, `/v1/models` answering); the
tenant log holds no error, traceback or refusal in that window, only the
gap itself (last synthesis 20:56:41, next diagnostics 20:58:43 with the push
counter frozen at 232); and no third process was on the cards --
`nvidia-smi --query-compute-apps` showed only the three 27B schedulers and
the tenant, so the 5090's rise from 24983 to 29379 MiB was the talker's own
allocation and not an intruder. **The stall is therefore unexplained and is
NOT attributed to the parallel INT8 work, which was the convenient answer.**
It is one occurrence in six runs of this build.

**Turn 4 then corrected the diagnosis, which is why the run was read to the
end instead of stopped at the first red.** It reported `line 85.1 s / audio
83.9 s / frames 8188` -- roughly thirty-five times a normal turn's audio,
arriving in one burst. So nothing was wedged: the pipeline was BACKED UP by
about eighty seconds and then flushed everything at once. "Stalled and never
recovered" would have been the wrong bug to hunt. This is the queueing family
of §17.8.2 (one talker, RTF 1.23, no scheduling policy makes two turns fit),
not a new fault -- what is unexplained is the SIZE of the backlog, since the
gate drives one conversation with 8 s gaps. Next step is a rerun with the MT
hop and the talker queue depth instrumented per turn, which is exactly what
the `--enable-metrics` item below is for; reasoning further without that
instrument is how the last four rounds were lost.

**Two gate arms, and what their can-fail proofs actually cover**

`--stop-during-playback N` fires `playback.stop` on the page's own socket one
poll after the first frame reaches playback. Verified live: ack
`aborted_turn_id 427d34bee557 / dropped_queued 0 / stop_epoch 1`, 0 frames
after, socket OPEN, and the NEXT turn completed in 4.7 s -- the talker was
freed rather than wedged, which is the assertion worth having.

`--stop-sabotage` is the can-fail proof and it came back HALF green, which is
recorded rather than rounded up: the ack half failed as required, the
quiescence half did not. With a batch talker and a single-clause turn all
audio has arrived before the arm looks, so "it went quiet" is true with or
without a stop. **That half is a decoration until the arm is driven with a
backlog** (a multi-clause turn, or a queued second turn). The UI agent
reports his own stop arm going red on the audio assertions; prefer his for
that half.

Text-vs-audio ordering is asserted on every turn, both sides stamped on the
page's clock. Only PARTIAL events are judged -- the shipped contract filters
on that flag, and the final non-partial event is the whole-translation record
journalled at turn end. Asserting over both fails every healthy turn by
~0.02 s, which is what the arm did until the contract was read instead of
assumed. Measured through the real client: the last streamed clause lands
**2.6-5.8 s before** the first audio frame. §19.13 works, seen from the page.

**Queue item 2 (roster) is answered and needs no server change.** A frame DOES
go out on automatic naming: `name_speaker` (session.py:598-628) journals
`SESSION_STATE` with `{speaker_named, label, lines_updated}` and re-emits every
relabelled transcript line, which is exactly why the transcript said
"Matthias" while the button row did not. What it does NOT carry is a roster,
and the client's handler is `applyState(event.state)` (index.html:2496-2498)
-- no emitter of `SESSION_STATE` anywhere sets a `state` key, so that handler
receives `undefined` on the arming path too. Embedding a full snapshot was
considered and REJECTED: journal events replay on resume, and a replayed
snapshot would overwrite a newer roster, while `{speaker_named, label}` is a
monotone fact that replays correctly. Cut 2's client-side fix (refresh
`state()` on divergence) is the right shape; its premise is verified --
`state()["speakers"]` carries `label` (speakers.py:283).

**Queue item 4 (languages): the machinery was complete and answering with the
wrong set.** `LanguageMatrix` already derives the honest intersection and
`GET /api/translator/languages` already ships per-language `as_source` /
`as_target` / `usable` / `reason` -- that IS the format for the UI agent, no
new frame needed. Measured live BEFORE the fix: `stages.asr ["de","es"]`,
eight languages carrying `"ASR cannot hear it"`, from a model that hears 102.
One attribute did two jobs: `_restrict` was both the deployment bound and the
per-turn decode whitelist that §17.8.6 pushes before every recognition, and
`supported_languages()` read it -- so the capability answer was whatever the
last conversation selected, leaking across sessions since one recognizer
serves them all. Split into `_deployment_languages` (capability) and
`_restrict` (decode). Falsifier `test_asr_capability_reach.py`, can-fail
proven in both adapters.

**The remaining half of "all languages selectable", NOT built.**
`RoutingTable.validate_against` (languages.py:367-370) calls
`matrix.require_pair` for EVERY direction between participants, from the
Session constructor (session.py:378). A participant whose language has no TTS
coverage therefore refuses the whole conversation instead of degrading that
direction to text-only. The client already carries
`PARTIAL_PARTICIPANTS_SUPPORTED` for it. This is the precondition for the
picker to be usable, and it is the top build item.

**Open, in order:** reproduce and root-cause the run-B stall with the MT hop
instrumented; deploy the ASR fix together with Cut 2 in one restart and gate
twice; the require_pair degradation; then §19.5 quality ladder,
`--enable-metrics` for `launch.py` (still absent from `boot_tenant.sh` --
every boot in this session violated that standing order and the script is
where to fix it), the man-02 listening check, and §19.9.

**One instrument note.** `tts_total_ms` reads 0.00 s on every turn and always
has -- it is in the OLD logs too, so it is not a regression of the FIFO
worker. It measures first-audio-to-last-audio, which for a batch talker is
genuinely nothing. It is a metric with no information at the served geometry;
either give it a meaning or drop it when `--enable-metrics` lands.

#### 17.8.11 The bundled deploy, 2026-08-03 (end of fifth session)

**Live: server `da46f2759b` (tenant PID 3799451, log `tenant18_bundle.log`),
client `9d39ac76b8`.** Restart taken only after the user's field test ended
(`sessions: 0`); it had been deferred once while he was mid-test, which was
the right call and is why this window is late in the session.

Four things shipped in the one restart, each verified by a STATE PROBE rather
than by the boot succeeding:

* `/metrics` answers, with all four live depth gauges present;
* the ASR capability fix: `stages.asr` went from 2 to **98**, and **usable
  went from 2 to 10** (de, en, es, fr, it, ja, ko, pt, ru, zh). The talker's
  coverage is now the binding constraint, which is what §19.5 always said it
  should be;
* Cut 2 (roster + picker), client `9d39ac76b8`;
* `boot_tenant.sh` carries `--enable-metrics` from now on.

**Gate: run 1 PASS** (4 turns, reload arm, stop arm with
`aborted_turn_id 37579feb87d3`, 0 frames after the stop, socket OPEN, turn 4
fine afterwards, console clean). **Run 2 is still owed** -- the standing rule
is twice, so the user must NOT be asked to test on this build yet.

**The metrics earned their keep within one run.** Read live mid-turn:
`translator_talker_busy 1.0` with `translator_session_queue_depth_max 0.0`.
That combination is the discriminator §17.8.10 lacked -- the pipeline was IN
synthesis, not queued behind it. One reading is not a diagnosis, but the
instrument that was missing now exists and answers.

**Standing decisions recorded so they are not re-litigated:**

* the §19.4 decomposition experiment (fixed overhead vs RTF slope over clause
  lengths, plus an MT-idle arm for the contention share) waits until AFTER
  the parallel serving switch to INT8 / NVFP4-A3B. That switch changes the
  5090's contention landscape fundamentally -- an A3B activates ~3 B against
  27 B dense -- so measuring now would characterise a landscape that will not
  exist. Run both arms (MT idle, MT loaded) in the same exclusive window if
  cheap;
* the overlap hypothesis is REFUTED and must not be rebuilt: the server never
  waits for playback, so synthesis of clause n+1 already overlaps playback of
  clause n. The gaps are arithmetic -- RTF 1.23 means synthesis costs more
  than the audio it produces, so a gap of roughly `fixed + 0.23 x D` opens
  after every clause and compounds;
* `non_streaming_mode=True` (`inprocess_tts.py:356`) is NOT the streaming
  lever it looks like. The library's own docstring
  (`qwen3_tts_model.py:513-515`) says the flag "only simulates streaming text
  INPUT... rather than enabling true streaming input or streaming
  generation". There is no true streaming generation through this call.
  Checked before anyone spent a window on it;
* `turn.speech` (`{turn_id, target, unit_index, state:
  queued|synthesizing|spoken}`) goes into the next server cut together with
  the `require_pair` degradation. Nothing today announces that a clause is
  being synthesized, which is exactly the interval the user experiences as
  the gap.

**The wire-vs-DOM gap is the THIRD instance of the §17.8 class and belongs in
the standing gate.** The ordering arm reads `turn.translation` off a
WebSocket listener, so its "text lands 2.6-7.1 s before audio" is frame
ARRIVAL, not visibility; and `line_s` counts transcript-line nodes created on
`turn.transcript`, not on the translation. So the gate has never asserted
that translated text is VISIBLE. The UI agent is building the DOM-visibility
arm; adopt it into `client_gate.py`'s default repertoire when it lands rather
than leaving it in his harness.

#### 17.8.12 The stop arm was the flake, 2026-08-03 (sixth session)

**Live: server `da46f2759b` (tenant PID 3799451, unchanged), client
`9f60fbb742`.** The client is read from disk per request and the tenant runs
on `PYTHONPATH=/spinning/wt-466-translator/python`, so merging a client-only
branch into this worktree IS the deploy -- verified rather than assumed by
reading `CLIENT_BUILD` back off the served page.

**Gate run 2 of the bundle failed, and the server was not at fault.** Turn 3
returned `aborted_turn_id None` -- "the arm tested nothing". The arm fired one
poll after the first audio frame reached playback, which reads as
mid-playback and is not: the talker is BATCH, so a unit's first frame reaches
the client only once that unit is fully synthesized, and for a single-unit
turn that instant is also when `_active_turn_id` is cleared
(`session.py:1073`). Whether the ack could still name a turn was a race
between a 250 ms poll and the server's own teardown. Run 1 won it, run 2 lost
it. Everything else in run 2 was green: four turns translated, reload arm
passed, console clean.

**The trigger is now the transcript line.** A line on the page means the
server is inside `_run_turn_locked` past recognition with MT and synthesis
still ahead of it -- a guarantee instead of a race, worth ~6-10 s of margin
against ~0. `--stop-trigger audio` keeps the old behaviour for the client-side
half (drop what is already buffered), which the UI agent's arm covers.

**This also turned the quiescence half from decoration into evidence**, which
§17.8.10 flagged and could not fix at the old trigger. Fired before any audio
exists, a working stop means the turn is never spoken AT ALL. The window is
watched to its end there rather than breaking early on silence, because
silence is the state the arm starts in. Measured, twice:
`frames after stop 0, quiet 20.2 s` with the ack naming the turn, against a
sabotage control that saw **284 frames** in the same window. The control is
what proves the window is long enough to have seen the audio a stop prevents.

**The §17.8.11 discriminator was used as intended and answered.** The arm now
reads `translator_talker_busy` and `translator_session_queue_depth_max` at the
instant the stop is written; both gate runs and the sabotage run recorded
`talker_busy 1.0 / queue_depth 0.0`, i.e. the server was IN synthesis, not
queued behind it. Printed and never asserted -- both gauges are process-wide,
not per session.

**The DOM-visibility arm is adopted into `client_gate.py`** rather than left
in one agent's harness, which closes the third instance of the §17.8 class.
It polls the rendered translation row (non-empty, non-zero box) beside the
wire events on one clock. Measured on the merged client: text readable
**4.0-6.2 s** after the tap against first audio at **7.3-12.8 s**. The defect
it exists for -- `case turn.translation` opening with `if (event.partial)
break;`, so every streamed clause was discarded and the screen held a
placeholder until speaking time -- passed every wire-level assertion this gate
had.

**Gate: 2x PASS on client `9f60fbb742`** (`gate_int8_a.log`,
`gate_int8_b.log`), each 4 turns with the reload arm, the stop arm naming its
aborted turn, 0 frames after the stop, turn 4 healthy afterwards, console
clean. **The user may be asked to test this build.**

**The MT backend switched to INT8 under us and the smoke is clean.** Both
gate runs are end-to-end turns against the new server: Spanish output correct,
no `</think>` artefacts, `mt_first_token` med 0.24 s / `mt_total` med 0.37 s,
in line with the FP8 numbers. The tenant was never at risk of the leak the
serving agent warned about -- it sends `chat_template_kwargs
{"enable_thinking": False}` by default (`launch.py:327-328`), and a direct
probe of the new backend returned `reasoning_content: None`. A defensive
`</think>` strip in `mt.py` remains cheap future-proofing, not a fix for
anything currently broken.

**`turn.speech` is BUILT AND COMMITTED BUT NOT DEPLOYED** -- it needs a
restart, and the plan is to take that restart together with the `require_pair`
degradation so the picker becomes usable in the same cut. States
`queued -> synthesizing -> spoken` per unit, emitted at the transition;
`synthesizing` fires BEFORE the synthesize call because naming the interval
before any audio exists is the whole point. `spoken` means handed to the wire,
not heard -- playback position is the client's audio clock, which draws the
"about N s left" strip. Units that will never be spoken are never announced
(reading mode would otherwise leave a spinner nothing can clear). Five cases,
can-fail proven by two injections; full translator suite 469 passed.

**Open, in order:** the `require_pair` degradation (still the top build item,
`languages.py:367-370` from `session.py:378`, client constant
`PARTIAL_PARTICIPANTS_SUPPORTED` waiting on it), then deploy it with
`turn.speech` in one restart and gate twice; then §19.5 quality ladder, the
man-02 listening check, the `getSettings()` telemetry of the next phone
connect (§19.8(b)), and §19.9.

**Not re-litigated, per §17.8.11:** the §19.4 decomposition experiment is now
a BASELINE arm inside the #488 talker-lane agent's window (he measures
in-process RTF 1.23 against the native lane), not a window of its own. Clause
coalescence after the first unit stays here and applies to the lane too. The
tenant's switch from `inprocess_tts` to the lane API will be this worktree's
cut when #488 delivers the interface. Capacity note from the serving agent:
the new server reserves 7500 MiB on the 5090 for this tenant, whose declared
budget is asr 3000 + tts 4000 + diarization 500 -- if that budget grows, the
reserve must grow with it.

#### 17.8.13 Cut A in flight, 2026-08-03 (sixth session, end)

**Live: server `da46f2759b` (tenant PID 3799451, UNCHANGED -- no restart
taken this session), client `8498385d90`.** The client is served from this
worktree per request, so the resampler fix went live on merge. **It is NOT
gated**: the one gate run against it went red for MT reasons (below), so the
user must not be asked to test this build.

**Shipped and gated: nothing new. Shipped and ungated: the client
resampler.** Everything else in Cut A is committed and waits on one tenant
restart: `turn.speech`, the MT retry, `--mt-timeout-s`, `--mt-lane`.

**The click is per-frame resampling, proven with a control.** See commit
`94bd13cc82`. The client built one `AudioBufferSource` per 20 ms frame at
the WIRE rate while the context runs at 44.1/48 kHz, so every frame was
resampled as an island. Measured against the same signal rendered as one
buffer: residual peak **0.490 against a 0.5 signal**, with **exactly zero**
energy between frame edges; at 16k->16k the residual is 0.0, which is what
proves the mechanism is the resampler and not the scheduling.

**The fix caught two of its own defects by being verified framed-against-
whole rather than by inspection.** An accumulated read position drifts
differently across one long call than across many short ones -- at 48 kHz
the ratio is 1/3 and the rounding is systematic, so the fix was broken at
exactly the rate a phone uses while clean at the 44.1 kHz it was first run
at. And an earlier special-case for a zero fraction never fired, because 1/3
is not exact in binary; the identical failing number said so. Deriving the
position from the global output index makes framed and whole agree by
construction: bit-identical at 16/22.05/32/44.1/48 kHz.

**The gate red is MT, not the resampler, and the shape says so.** Run
`gate_resample_a.log`: turns 1 and 4 produced no audio AND no visible text,
and text never touches the resampler. Turn 2 DID produce audio through it
(236 frames) with text visible at 20.2 s -- against a 4.8 s median on the
previous build. That matches the independently measured 19.7 s MT
time-to-first-token behind a 46k-token co-tenant prefill. A probe taken
after the run: 25.0 s and 12.0 s for a FOUR-token request. Gating in that
state measures the neighbour, not the build.

**Next, in order:** wait for the 30030 restart carrying `--enable-fast-lane`,
then re-gate the client twice; then the `require_pair` degradation and
auto-scroll, which are the two Cut A items still unbuilt; then take ONE
tenant restart for the whole server side (turn.speech, retry, timeout, lane)
and gate twice again. The onset probe is the standing regression arm for the
seams, and `probe_resample_seam.py` for the output.

**Do not re-litigate:** the three original click candidates are each
falsified with numbers (§17.8.12) -- the talker's onset rises over 11.6 ms,
seams are not the step outlier, and 1 of 268 schedules started in the past
and it was the benign initialization. The talker module is not implicated
and remains the #488 agent's.

#### 17.8.14 Auto-scroll fixed, roster management built, 2026-08-03 (seventh session)

**Live: server `da46f2759b` (tenant PID 3857538, UNCHANGED -- no restart taken
this session), client `01ed65f1e2`.** The page is served from this worktree
per request, so both client cuts went live on merge; verified by reading
`CLIENT_BUILD` back off the tenant rather than assuming it.

**The MT neighbour that reddened §17.8.13 is gone, and the number says so.**
Same tiny request, same instant: **26.5 s on the default lane, 0.19 s with
`lane:fast`** -- the tenant sends `lane:fast` (`launch.py:351-352`), so it is
on the fast side. Gating now measures the build, which §17.8.13 explicitly
said it could not. `mt_first_token` across the runs: **0.13-0.37 s**, against
19.7 s in the state that forced the previous session to stop gating.

**AUTO-SCROLL. The mechanism the previous session suspected is confirmed, and
it is worse than a missed turn -- it is a LATCH.** The old shape sampled
`atBottom()` at append time and scrolled once; a bubble then keeps growing
(`turn.queued` writes the waiting notice, every `turn.translation` partial
adds a clause, the final event replaces the accumulation) and none of those
paths scrolls. Measured per mutation on the pre-fix client:

```
  turn 1 clause 1  visible True,  overflow below   -38px, at bottom True
  turn 1 clause 2  visible False, overflow below    +5px, at bottom True
  turn 1 clause 3  visible False, overflow below   +59px, at bottom False
  turn 4 final     visible False, overflow below +1045px, at bottom False
```

The FIRST growth pushes the box off the bottom; `atBottom()` reads false from
then on and the follow never fires again for the rest of the conversation.
The gate's +95px was one sample per turn of a number that keeps climbing.

Re-measuring at growth time cannot fix it -- the growth is what moved the
content past the fold. The fix is a `following` flag written only by the
scroll event (the one signal that is actually the reader's), plus a
MutationObserver and a ResizeObserver over the stream that re-pin on any size
change. Every future event handler is covered without knowing about it, which
is how this got in one handler at a time. The reader who scrolled up keeps the
60 px exception and now gets a small "new message" pill.

**ONE OF THE TWO REDS WAS THE ARM.** With the client provably pinned
(`overflow below -4px, at bottom True`) the predicate still said False: it
demanded the whole newest line fit inside the box, and one long turn makes a
bubble TALLER than the box on a phone, where that is unsatisfiable at any
scroll position. The assertion is the bottom edge; the top edge is only
required when the line is short enough to have one. Corrected in
`probe_autoscroll.py` and in `client_gate.py`, which carried the same
predicate.

**New arm: `scripts/translator/probe_autoscroll.py`.** Executes the shipped
page in Chromium, drives the real `onEvent` with the real event order, and
samples the geometry after EVERY mutation -- seconds, no server, no GPU,
against the gate's ~40 s per turn and one sample per turn. Can-fail proven
against `8011c6fb05` (21 of 24 samples red) with the corrected predicate in
both runs. It also refuses to pass a run in which the transcript never
overflowed, because that run could not have seen the defect.

**Gate: 2x PASS on client `e2650d75df`** (`gate_scroll_a.log`,
`gate_scroll_b.log`), 4 turns each with the reload arm and the stop arm, 0
frames after the stop, console clean. The scroll arm is green on every sample
including the 3-line shape that was `+95px` red. Note for the next reader: run
B never reached 3 lines, because the reload after turn 2 resets the DOM -- the
decisive live sample is run A turn 2, and the probe is what covers the deeper
overflow.

**ROSTER MANAGEMENT (user order).** Delete was complete server-side
(`delete_speaker`, `speakers.remove` keeping the transcript) and unreachable:
a 600 ms long press into a sheet. It is now an "x" on the entry, deliberately
NOT a long press -- that gesture is the merge drag, and one gesture must not
mean two irreversible things.

Merge did not exist. `SpeakerRegistry.merge` unions the clusters
(observation-weighted centroid, concatenated reference buffers re-evicted
through the normal budget, `enrolled` OR-ed); `Session.merge_speakers` moves
the transcript attribution retroactively, the armed button, name suggestions
and the typed-name protection, and gives the source's preset voice back. It
reads both profiles before mutating, so an unknown id leaves nothing
half-done. `speaker.merge` on the socket, `POST .../speakers/{target}/merge`
over REST.

**THE FIRST MERGE ARM DID NOT DISCRIMINATE.** With `merge` replaced by a plain
`remove`, "the follow-up segment lands on the merged speaker" still passed --
deleting the other cluster also leaves one speaker standing. What only a real
union can do is inherit the source's identity, so the arm is now two clusters
at cosine 0.30, below the 0.637 bar: the source's voice does not reach the
target before the merge (**0.300**) and does after (**0.806**). Red under
sabotage, green whole. The pipeline-level arm was also merging in the
direction that wins on its own and now merges the other way.

**The drag hint was moving the drop targets.** As a flow row it made the
footer taller mid-gesture, so the targets slid out from under a finger already
on its way to one -- caught by `probe_roster.py`, whose coordinates are taken
before the lift exactly as a user's aim is. It is absolutely positioned now
and takes no layout.

**CAPABILITY GATE, and it is the general answer to this worktree's deploy
asymmetry.** The page is live on merge; its server half needs a restart. In
that window a merge drag would look like the app ignoring the user. The
gesture is bound only when the connected server announces
`supports.speaker_merge` in its state frame (`session.state()`); an older
server omits the key and the long press keeps its old meaning. **No second
client deploy is needed** -- the restart arms it. Delete needs no server
change and is live now.

**Arms and results:** `probe_roster.py` PASS (delete incl. its cancel path and
that it does not reach the speak button underneath, the full drag with drop
zones and highlighted target, a drop on nobody falling back to the sheet, the
capability gate); 11 new hermetic server tests; full translator suite **487
passed**. Screenshots recorded for the drag mid-gesture and for the unread
pill.

**Gate on the roster build: 2x PASS on client `01ed65f1e2`**
(`gate_roster_a.log`, `gate_roster_b.log`), same shape -- 4 turns each with
the reload arm, the stop arm naming its aborted turn (`frames after stop 0,
quiet 20.2 s`), console clean, the scroll arm green on every sample. MT
medians across the run: `mt_first_token` 0.14 s, `first_audio` 5.99 s. Four
gate runs were taken this session in total, all green, across the two client
cuts. **The user may be asked to test `01ed65f1e2`.**

**THE RESTART BUNDLE has grown and is now the top item.** One tenant restart
carries: `turn.speech`, the MT retry, `--mt-timeout-s`, `--mt-lane` (all from
Cut A, still undeployed), PLUS `speaker.merge` + its REST route + the
`supports` capability key. Still to build before taking it: the `require_pair`
degradation (`languages.py:362-366` -> `matrix.require_pair`, called from
`session.py:411`; client constant `PARTIAL_PARTICIPANTS_SUPPORTED` waiting on
it) and the boot warmup synthesis (a dummy sentence through the talker before
"status ok", which removes the ~15 s cold start of turn 1). The `turn.unrouted`
event and its client case already exist for the manual-routing path and are the
shape the degradation should reuse -- a participant language whose direction
this deployment cannot serve should open the session and tag the turn, not
refuse the session.

**Then:** §19.5 Opus 48k uplink (the resampler is rate-agnostic and goes to
pass-through), the man-02 listening check, the `getSettings()` telemetry of the
next phone connect (§19.8(b)), and §19.9.

**Do not re-litigate:** the auto-scroll mechanism is measured, not suspected
-- do not "simplify" it back to a per-append `atBottom()` sample. The gate's
visibility predicate is deliberately asymmetric about the top edge and the
reason is written at both copies.

#### 17.8.15 The restart bundle is deployed, and the warmup is unproven (seventh session, end)

**Live: tenant PID 3893121, client `e777cd20dd`.** One restart carried the
whole bundle: `turn.speech`, the MT retry, `--mt-timeout-s`, `--mt-lane` (all
Cut A, undeployed since §17.8.12), plus `speaker.merge` + its REST route + the
`supports` key, plus the `require_pair` degradation and the boot warmup. Read
back off the live socket: `supports {'speaker_merge': True,
'partial_participants': True}`.

**Gate: 2x PASS on `e777cd20dd`** (`gate_bundle_a.log`, `gate_bundle_b.log`),
4 turns each with the reload and stop arms, `frames after stop 0, quiet
20.2 s`, console clean, the auto-scroll arm green on every sample. The merge
capability arms only now, so it was exercised against the LIVE server directly:
target survives with its label, source gone, and the three refusals answer
404 / 400 / 400.

**REQUIRE_PAIR DEGRADATION.** The constructor's
`conversation.validate_against(matrix)` raised on the first unservable
direction, so one participant language without a TTS voice refused the whole
conversation and took every other chosen language with it. The turn path
already degraded per target; only the constructor was all-or-nothing. It now
opens on the servable directions and tags the rest with `turn.unrouted` (an
error toast per utterance would make a conversation working in one direction
look broken in both). No servable direction at all is still refused.
`PARTIAL_PARTICIPANTS_SUPPORTED` is gone -- the picker asks the server, same
capability mechanism as the merge gesture.

A PIN HAD TO CHANGE: `test_an_unroutable_pair_is_refused_not_guessed` asserted
the constructor raises. Rewritten rather than deleted, around what did NOT
change and what its name is about -- nothing is guessed: the unservable
direction produces no translation and no substituted target (`mt.calls == []`),
it produces a tagged turn naming the stage.

**THE WARMUP IS UNPROVEN AND THE CONTROL SAYS SO.** It was built to remove the
~15 s first-turn cold start. Same build, two boots differing only in the flag,
gate turns:

```
  --no-warmup   turn 1 tts_first_audio  7.38 s   turn 2 2.36 s
  --warmup      turn 1 tts_first_audio 12.75 s (tts_wait 5.83 s)
  --warmup      turn 1 tts_first_audio  5.71 s   (final boot)
```

Two things fall out. **The 15 s cold start did not reproduce** -- the
first-turn penalty without any warmup is about 5 s (7.38 against a 2.36 warm
turn). And **the warmup shows no measurable benefit**: its two first-turn
samples, 12.75 s and 5.71 s, straddle the control. The coordinator's
acceptance criterion (first turn under 3 s of talker share) is **RED in both
configurations**.

Single samples, no repetition, so none of these is an estimate; what they
jointly refuse is the claim that the warmup removes the cold start. The
mechanism is left on -- it costs ~4 s of boot and this rig's "cold" boot still
reuses a hot driver and page cache, so it is not the cold start a fresh
machine has -- but the log line now states only what it did (`talker warmup:
6 chunks in 4.35s`), the flag help says UNPROVEN with the numbers, and the
docstring carries the control. **The next session either produces a repeated
A-vs-A or turns the default off.** Do not restore the "turn 1 no longer pays
the cold start" wording; it is the exact success-claim-without-evidence class
this repo has a rule about, and it survived one commit before the control
caught it.

**Also caught here:** the first boot with the warmup printed `translator
ready` five seconds BEFORE `talker warm`. The port ordering was always right
(uvicorn starts after the synthesis, so no turn can reach the cold path), but
the log announced readiness while the talker was still cold. Fixed by moving
the call above the readiness line.

**Harness note for the next session:** gate runs launched into the background
from a tool call are reaped and exit after two log lines with no error. Run
the gate in the FOREGROUND with a bounded `timeout`; four runs were lost to
this before it was diagnosed. The same session lost time to `until ! pgrep -f
"<pattern>"` wait loops whose own command line matches `<pattern>` -- they
never exit. Wait on a PID.

**Next:** §19.5 Opus 48k uplink, the man-02 listening check, the
`getSettings()` telemetry of the next phone connect (§19.8(b)), §19.9 -- all
behind the DSV4F focus window, which takes serving and this tenant down for a
while; its agent reboots the tenant at the end of that window.

#### 17.8.16 Language before embedding, and one speaker at a time (user directive, eighth session)

**The order, verbatim (2026-08-04):** "der größte hebel zu erkennen ob jemand
anderes spricht ist die ausgangssprache erstmal. weil es kann auch jemand
spanisch reinquatschen, wenn ich deutsch gerade spreche, dann muss
selbstverständlich der spanisch erkannte teil, die spanische stimme, ignoriert
werden. es soll nur ein sprecher gleichzeitig 'als audio rauskommen'."

This reorders the whole identity stack. Until now the embedding decided who
spoke and the language was a consequence; from here the language decides
whether this is even the same speaker, and the embedding is what separates two
people who share one.

**WHAT THE CODE ALREADY GUARANTEES, AND WHAT IT DOES NOT.** The output half of
the order is already true and needs no work: `run_turn_multi` holds
`self._turn_lock` across recognition, MT *and* synthesis (`session.py:1219-1233`,
with `_translate_and_speak` awaited inside `_run_turn_locked`), and `drain()`
pops one segment at a time (`session.py:1191-1197`). Two speakers can therefore
never be synthesized concurrently today, and any claim to have "built" that
would be a claim on existing behaviour. What is NOT prevented is the other half:
an interjection is QUEUED (`enqueue`, `session.py:1136-1150`) and spoken once
the running turn ends. "Ignoriert werden" forbids exactly that replay. Rule 2 is
therefore a DISCARD at the queue boundary, not a mutex.

**RULE 1 MUST RUN BETWEEN STEP 1 AND STEP 2, AND TODAY NOTHING RUNS THERE.**
Recognition is step 1 (`session.py:1336-1357`), identification step 2
(`session.py:1361`), source resolution step 3 (`session.py:1383`). Enrollment
happens inside step 2 -- `speakers.assign(...)` takes the audio and the text and
moves the centroid (`session.py:1631-1637`). A language filter placed at step 3,
where `_resolve_source` sits, would refuse the translation of a segment whose
voice it had already folded into the pinned speaker's cluster. The filter goes
BEFORE `_identify` or it protects only half of what it exists to protect.

**THE FLOOR.** One concept serves both rules, and naming it once keeps them from
being two mechanisms that disagree. The *floor language* is the language that
currently holds the conversation: the sticky pin's language while a pin is held,
and the running turn's resolved source while a turn is in flight. A segment
resolved to a *different participant language* while the floor is held is by
construction a different speaker -- no embedding is consulted, because the order
says language comes first.

**THE TRADE-OFF I DECIDED, AND IT NEEDS TO BE SEEN.** A floor that is held for
as long as the pin is held would silence the app's own purpose: a DE speaker and
an ES speaker alternating into one phone means every second turn is "the other
language under a held pin". The order is about `reinquatschen` -- talking OVER
someone -- so the discard is scoped to OVERLAP, which is a predicate the code can
actually evaluate: `self._active_turn_id is not None` at `enqueue()` time is
exactly "this segment was captured while another turn was being processed". It
is stamped on the queued segment there (the only place the temporal relation is
known) and acted on after recognition (the earliest place the language is
known).

  - segment OVERLAPS a running turn, other participant language
      -> discarded. No TTS, no enrollment, no queueing. Transcript line
         written and tagged, because a swallowed utterance is
         indistinguishable from a broken microphone.
  - segment does NOT overlap, other participant language, pin held
      -> translated normally, but NOT enrolled into the pinned cluster, and
         the UI says the pin is still on someone else. This is the ordinary
         hand-over and must keep working.

If the user wants the stricter reading -- a held pin silences the other language
outright, overlap or not -- it is one predicate, but it should be his call and
not a side effect of mine.

**WHY THE NAIVE DIRECTION GUARD (item 3d) IS UNREACHABLE, CONFIRMED IN CODE.**
`Conversation.targets_for` returns `participants - {src}` (`languages.py:360`)
and `require_pair` refuses `src == tgt` before anything else
(`languages.py:261-265`). "Detected language equals chosen target" can therefore
never occur downstream: the target set is built by excluding the source. The
defect the user sees is a WRONG `source`, and the only place to fix it is source
resolution -- which is what rules 1 and 2 are. The direction bolt stays as an
assertion that the invariant holds, not as a repair.

**LID HARDENING, AND WHAT A CONFIDENCE GATE CANNOT DO.** `_resolve_source`
(`session.py:1720-1735`) trusts the detected language at
`language_confidence >= 0.5` and otherwise believes `profile.last_language` --
which is how the TTS-echo cluster (speaker-4, `language es`, conf 0.026) routes
German to German. Under the directive the order becomes: confident detection
first, then the PIN's language, and `profile.last_language` only for a profile
the user actually named or enrolled -- never a freshly minted automatic cluster,
which is precisely the phantom this failure is made of.

That fixes the echo. It does NOT necessarily fix the child. If the LID reports
`es` for his German with HIGH confidence, no confidence gate can reach it and
only a prior can. Text-LID over the decoded text was considered and is NOT the
answer: Whisper decodes in the language it has already chosen, so a wrong
decision produces wrong-language text as well -- line 2 of the live transcript,
`"Und bingst du hási."`, is what a decode fighting its own language decision
looks like. Whether a pin prior suffices or a session language lock is required
is decided by the §2 measurement (his turns, with LID language and confidence
per turn), not here.

**THE HONEST LIMIT, STATED BECAUSE IT WILL BE FORGOTTEN.** Language separates
speakers only ACROSS the pair. Two German speakers are invisible to every
mechanism in this section, and the measured cosine threshold remains the only
thing that separates them. That is why the threshold work stays in the bundle --
demoted to the second line of defence, not dropped.

**GATE ARMS, both with the control that proves they discriminate:**

  (a) barge-in: a DE turn in flight, an ES segment injected while
      `_active_turn_id` is set. Assert exactly one synthesis for the window,
      the DE turn's audio uninterrupted, the ES segment tagged and never
      spoken, and the pinned cluster's centroid unmoved. Control: the same
      injection with the floor filter disabled must produce the second audio.
  (b) the child: a DE segment at low LID confidence under a DE pin stays DE.
      Control: the same segment with the pin released takes the old path and
      routes wrongly, which is what proves the pin is what decided it.

#### 17.8.17 The LLM arbiter, and why only half of it belongs in the restart bundle

**The idea, verbatim (user, 2026-08-04):** "das qwen3.6 modell sollte doch auch
mit gutem briefing aus den gemessenen daten und dem übergebenen transkript von
bisher viel besser entscheiden können wer gerade spricht?"

**Architecture, as ordered:** three stages, not a replacement. Rules first (pin,
the §17.8.16 language filter, an unambiguous cosine) decide with no latency at
all. Only inside a DOUBT BAND -- language does not separate, LID and embedding
disagree, or the child class -- does a structured evidence package go to Qwen
over the fast lane: LID language and confidence, cosine to EVERY known cluster,
pin state, overlap flag, and the last N transcript lines WITH their speaker
labels. Fixed answer schema (`speaker_id` | `NEW` | `DISCARD` plus one sentence
of reasoning), hard token ceiling, timeout falling back to the rule decision.
Shadow mode first: the arbiter judges, the judgement is logged, the rules still
decide, and authority is granted only against a measured hit rate.

**MY CALL ON TIMING: the bundle carries the EVIDENCE RECORD, the arbiter is the
cut immediately after. The split is a dependency, not a preference.**

The doubt band cannot be defined honestly today. Its edges are exactly the
numbers §2 is being measured for -- the within-speaker cosine distribution on
the NEW signal path, after the language filter and the sticky pin have changed
which segments even reach the comparison. An arbiter shipped against the old
0.637 bar would be invoked on the wrong set of turns, and its hit rate would
then be measured against a band nobody can defend. That is the §493 class:
a mechanism whose threshold never binds where it was supposed to.

What DOES belong in the bundle, and would be built for §2 regardless, is the
evidence record itself: the per-decision log extended with LID language and
confidence, cosine against every cluster (not just the nearest), pin state,
overlap flag, and the transcript window. It is the measurement instrument for
the threshold AND the exact payload the arbiter will later be handed, so
building it once serves both. Shipping it in the bundle means the arbiter cut
starts with real data instead of a fresh instrument.

**Three integration constraints, recorded now because each is a silent defect
later:**

  1. THE ARBITER MUST NOT TOUCH THE MT HISTORY. `mt.remember` is called per
     translated turn (`session.py:1499-1501`) and the backend carries a rolling
     six-turn context (`mt.py:200-217, 235-236`). An arbiter prompt issued
     through the translating path would inject speaker-attribution reasoning
     into the conversation context and degrade every following translation.
     It needs a history-free call, and `translate()` is not it.
  2. IT SITS IN THE CRITICAL PATH, because attribution precedes translation.
     Bounded only: the fast lane answers `mt_first_token` in 0.13-0.37 s
     measured, so a capped arbiter call is affordable *inside the doubt band*
     and nowhere else. A timeout is an instrument failure with its own state,
     never a blocked turn.
  3. ONE RUNTIME. The arbiter is a call into the same INT8 serving lane the
     tenant already uses (`launch.py:351-352`, `lane:fast`) -- no second
     engine, no sidecar.

**What shadow mode is measured against.** Later truth already exists in the
code and needs no new capture: `resolve_line` (a user re-attributing a line),
`merge_speakers`, and a name typed onto a cluster are all explicit human
corrections, and a pin change is a weaker one. The hit rate is the arbiter's
judgement compared against the correction that came after it -- which is why
the judgement has to be logged at decision time with the evidence it saw, not
reconstructed.

**The honest risk, named before it is built.** This is the ANALYSE_532 class
"bound decision with delivered material", which is where this model is strong.
It is also the class that produces PLAUSIBLE WRONG answers: an arbiter handed
four clusters and a transcript will always name one, and it will sound
reasonable doing it. Shadow mode is not caution theatre here -- it is the only
way to tell those two apart before the thing has authority over what the user
hears.
