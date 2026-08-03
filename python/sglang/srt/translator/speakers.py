# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Who said it, and what their voice sounds like.

Two jobs, deliberately kept in one place because they share a decision:

1. **Identity.** Assign each closed segment to a speaker, by cosine-matching
   its embedding against the speakers seen so far and starting a new one when
   nothing matches. This is online clustering over *finished utterances*, not
   frame-level diarization -- in a half-duplex, turn-taking conversation the
   segmenter has already done the hard part (it knows where the turn ended),
   so the remaining question is only "which of the known voices is this",
   which one embedding per segment answers. Frame-level diarization earns its
   complexity when speakers overlap; under half-duplex they do not, and the
   cheaper thing is also the more robust thing.

2. **Reference audio.** Keep, per speaker, the best few seconds of their own
   voice for the cloning TTS to condition on. This buffer is the direct cause
   of output quality, so its selection policy is explicit rather than
   "whatever was last": segments are scored and the best ones retained up to
   a target duration.

The enrollment path is the same machinery with a pinned label: the user's own
voice can be registered from provided samples before the first turn, which
starts them at full reference quality rather than cold.

An important asymmetry the reference policy encodes: the reference is what
the speaker sounds like, and a *longer, cleaner* reference beats a *more
recent* one for identity. Recency only matters for tracking a changing
channel (the phone moved, the room changed), so recency is a tiebreak and not
the primary key.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sglang.srt.translator.backends import AudioChunk, SpeakerEmbedding

__all__ = [
    "SpeakerProfile",
    "SpeakerRegistryConfig",
    "SpeakerRegistry",
    "ReferenceTooShort",
]


class ReferenceTooShort(RuntimeError):
    """A speaker has no usable reference audio yet."""

    def __init__(self, speaker_id: str, have_s: float, need_s: float) -> None:
        super().__init__(
            f"speaker {speaker_id} has {have_s:.2f}s of reference audio, "
            f"needs {need_s:.2f}s before it can be cloned"
        )
        self.speaker_id = speaker_id
        self.have_s = have_s
        self.need_s = need_s


@dataclasses.dataclass(eq=False)
class _ReferenceItem:
    """One retained slice of a speaker's own voice.

    ``eq=False`` on purpose: the item holds an ``AudioChunk`` whose ndarray
    makes a generated ``__eq__`` raise "truth value of an array is ambiguous"
    the moment anything compares two items. Identity is the only equality that
    means anything here anyway -- two slices with identical samples are still
    two distinct retained slices.
    """

    audio: AudioChunk
    text: str
    language: str
    score: float
    added_at: float
    #: Enrollment slices occupy their own budget and are never evicted.
    enrolled: bool = False


@dataclasses.dataclass
class SpeakerProfile:
    """Everything the pipeline knows about one voice in the conversation."""

    speaker_id: str
    #: Running centroid of this speaker's embeddings, unit-norm. ``None`` for
    #: a speaker the user DECLARED with the "+" button but who has not been
    #: heard yet (§17.5); their first audio seeds it.
    centroid: Optional[SpeakerEmbedding]
    #: How many segments have been folded into the centroid.
    observations: int = 1
    #: Human label, set by enrollment ("matthias") or left None.
    label: Optional[str] = None
    #: True when the profile came from explicit enrollment. Enrolled profiles
    #: never have their reference buffer evicted below the enrolled material:
    #: the user's curated samples are better than anything a noisy street
    #: segment will produce, and letting field audio displace them is a
    #: quality regression that would be invisible until playback.
    enrolled: bool = False
    #: Last language this speaker was confidently recognized in. Used as the
    #: ASR hint and as the fallback when a later segment's language ID is
    #: uncertain -- people do not usually switch language mid-conversation.
    last_language: Optional[str] = None
    last_seen: float = 0.0
    references: List[_ReferenceItem] = dataclasses.field(default_factory=list)

    def reference_seconds(self) -> float:
        return sum(item.audio.duration_s for item in self.references)

    def reference_audio(self, sample_rate: Optional[int] = None) -> AudioChunk:
        """Concatenate the retained references into one conditioning clip.

        Ordered best-first so that a backend which truncates its reference
        (several do, at 10-15 s) truncates the *worst* material.

        A slice recorded at a different rate is RESAMPLED, never skipped. It
        used to be skipped, and the consequence was invisible until a real
        backend ran: the registry stores at the 16 kHz pipeline rate while
        Qwen3-TTS asks for 24 kHz, so every slice was dropped and this
        returned an EMPTY clip -- silently, because the caller's
        ``reference_seconds()`` guard had already passed on the stored
        durations. The turn then died in the synthesizer with "reference is
        0.00s, need >= 3.0s" for a speaker holding 6.2 s of admitted audio.
        The desk suite could not see it: its fake TTS runs at the pipeline
        rate, so the rates always matched and the filter never dropped
        anything.
        """
        from sglang.srt.translator.audio import resample

        if not self.references:
            raise ReferenceTooShort(self.speaker_id, 0.0, 0.0)
        ordered = sorted(self.references, key=lambda i: (-i.score, -i.added_at))
        rate = sample_rate or ordered[0].audio.sample_rate
        merged = AudioChunk(np.zeros(0, dtype=np.float32), rate)
        for item in ordered:
            audio = item.audio
            if audio.sample_rate != rate:
                audio = resample(audio, rate)
            merged = merged.concat(audio)
        if merged.duration_s <= 0.0:
            # Unreachable by construction (`references` is non-empty and every
            # slice now survives), which is exactly why it is asserted: this
            # method returning an empty clip is not an error anywhere
            # downstream, it is a turn that dies three layers away.
            raise ReferenceTooShort(self.speaker_id, 0.0, 0.0)
        return merged

    def reference_text(self) -> str:
        ordered = sorted(self.references, key=lambda i: (-i.score, -i.added_at))
        return " ".join(item.text for item in ordered if item.text).strip()


@dataclasses.dataclass(frozen=True)
class SpeakerRegistryConfig:
    """Clustering thresholds and reference-buffer policy.

    ``match_threshold`` is the cosine above which a segment is considered the
    same speaker. 0.70 is the conventional operating point for ECAPA-class
    embeddings on short utterances; it trades a rare voice split (the same
    person gets two ids, costing only a duplicated reference buffer) against a
    voice merge (two people share one buffer, which corrupts BOTH voices and
    is audible immediately). The asymmetry is why the default sits high: a
    split degrades gracefully, a merge does not.

    ``max_speakers`` bounds a public-space conversation where every passer-by
    would otherwise mint a profile.
    """

    #: Measured on THIS embedder rather than taken from the literature.
    #: `probe_speaker_change.py --pool` over the 17-voice pool at the shipped
    #: 2.5 s window gives within-speaker p05 0.637 (min 0.624) against
    #: between-speaker p95 0.583. The conventional ECAPA operating point of
    #: 0.70 sits ABOVE the floor of the same-speaker population, so it splits
    #: one person into a new profile on every few utterances -- which is
    #: exactly what the first working phone conversation did: one speaker,
    #: a different identity every turn. A match bar has to sit at or below
    #: the same-speaker floor to be a match bar at all.
    match_threshold: float = 0.637
    #: Below this cosine a segment is definitely a new speaker. Between the
    #: two thresholds the segment is assigned to the nearest speaker but is
    #: NOT admitted to their reference buffer -- an ambiguous segment is fine
    #: to translate and dangerous to clone from.
    reference_threshold: float = 0.80
    max_speakers: int = 8
    #: Fixed enrollment prompt: curated audio, trimmed to this length, never
    #: evicted. Zero when the speaker was never enrolled.
    enrolled_prompt_s: float = 6.0
    #: Rolling prompt: the best of the speaker's recent field audio, capped
    #: here. Two slots rather than one pool, following the session-level
    #: speaker-prompt manager described in X-Translator (arXiv 2607.17544):
    #: a fixed enrollment prompt anchors identity, a rolling recent prompt
    #: tracks the current channel (the phone moved, the room changed).
    #: Reimplemented from the paper's description, not from their code.
    rolling_prompt_s: float = 6.0
    #: Half-life of the recency bonus in the retention score, in seconds of
    #: conversation time. The two sources disagree here and the disagreement
    #: is real: speaker-verification practice says keep the best-K segments,
    #: the prompt-manager design says keep the most RECENT window. Both are
    #: right about different failure modes -- a great old slice beats a poor
    #: new one, but a slice from a different room is worse than a mediocre
    #: one from this one. Scoring quality with an exponential recency decay is
    #: the synthesis, and this constant is where the trade-off is tuned.
    recency_half_life_s: float = 180.0
    #: Never keep a slice shorter than this. Splices are audible: a zero-shot
    #: cloner reproduces the prosodic discontinuity at a concatenation seam,
    #: so one contiguous 6-12 s slice beats six 2 s ones at the same total
    #: duration. The scoring below therefore rewards duration super-linearly
    #: through the eviction order, and this floor keeps the shortest material
    #: out entirely.
    min_slice_s: float = 2.0
    #: Never keep a slice longer than this; one long slice would monopolise
    #: the buffer and remove the diversity that makes cloning robust.
    max_slice_s: float = 8.0
    #: Segments quieter than this RMS are not admitted as reference material.
    min_reference_rms: float = 0.01
    #: Embedding centroid update rate. Low, because a speaker's identity does
    #: not drift; this only absorbs channel variation.
    centroid_alpha: float = 0.15
    #: Below this cosine a segment is confidently somebody NEW, and the line
    #: carries no uncertainty badge. Measured, not chosen: it is the
    #: between-speaker p95 of `probe_speaker_change.py --pool` over the
    #: 17-voice pool at the shipped 2.5 s window (design §17.4). Between it
    #: and ``match_threshold`` the assignment is made as usual and the LINE is
    #: marked uncertain, because that is the range where the embedder cannot
    #: separate "same person" from "different person".
    uncertain_floor: float = 0.583
    #: Within-speaker p05 from the same sweep. Used for auto-resolution: a
    #: stored embedding that later scores above this against an updated
    #: centroid is no longer ambiguous.
    within_speaker_floor: float = 0.637


class SpeakerRegistry:
    """Online speaker assignment plus per-speaker reference buffers."""

    def __init__(
        self,
        config: Optional[SpeakerRegistryConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or SpeakerRegistryConfig()
        self._clock = clock
        self._profiles: Dict[str, SpeakerProfile] = {}
        self._next_id = 1

    # -- read side ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._profiles)

    def profiles(self) -> Tuple[SpeakerProfile, ...]:
        return tuple(self._profiles.values())

    def get(self, speaker_id: str) -> SpeakerProfile:
        try:
            return self._profiles[speaker_id]
        except KeyError:
            raise KeyError(f"unknown speaker {speaker_id!r}") from None

    def to_json(self) -> List[Dict[str, object]]:
        return [
            {
                "speaker_id": p.speaker_id,
                "label": p.label,
                "enrolled": p.enrolled,
                "observations": p.observations,
                "last_language": p.last_language,
                "reference_seconds": round(p.reference_seconds(), 2),
                "clonable": p.reference_seconds() > 0.0,
            }
            for p in sorted(self._profiles.values(), key=lambda p: p.speaker_id)
        ]

    # -- write side ---------------------------------------------------------

    def enroll(
        self,
        label: str,
        embedding: SpeakerEmbedding,
        audio: AudioChunk,
        text: str = "",
        language: Optional[str] = None,
        speaker_id: Optional[str] = None,
    ) -> SpeakerProfile:
        """Register a known voice from curated samples before the first turn."""
        sid = speaker_id or f"enrolled:{label}"
        profile = SpeakerProfile(
            speaker_id=sid,
            centroid=embedding,
            label=label,
            enrolled=True,
            last_language=language,
            last_seen=self._clock(),
        )
        self._profiles[sid] = profile
        # The enrollment prompt is trimmed to a fixed length rather than kept
        # whole. Cloners truncate their reference anyway (10-15 s typically),
        # so handing them 60 s of enrollment audio does not improve identity;
        # it just decides for them which part to discard. Taking the middle
        # skips the microphone-handling noise at both ends.
        keep = self.config.enrolled_prompt_s
        if keep > 0 and audio.duration_s > keep:
            n = int(keep * audio.sample_rate)
            start = (len(audio.samples) - n) // 2
            audio = AudioChunk(audio.samples[start : start + n], audio.sample_rate)
        profile.references.append(
            _ReferenceItem(
                audio=audio,
                text=text,
                language=language or "",
                score=float("inf"),
                added_at=self._clock(),
                enrolled=True,
            )
        )
        return profile

    def rank(
        self, embedding: SpeakerEmbedding, limit: int = 8
    ) -> List[Tuple[str, float]]:
        """Every known speaker by similarity, best first.

        The same cosine the assignment decides on -- the candidate list on an
        uncertain line must not be a second, differently-derived opinion, or
        the user would be choosing between numbers the machine did not use.
        """
        scored = [
            (sid, profile.centroid.similarity(embedding))
            for sid, profile in self._profiles.items()
            if profile.centroid is not None
        ]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:limit]

    def uncertainty(
        self, ranked: Sequence[Tuple[str, float]], assigned_id: str
    ) -> Tuple[bool, List[Dict[str, object]]]:
        """Is this attribution ambiguous, and what are the alternatives (§17.4)?

        ``assigned_id`` is what the assignment actually chose, so a freshly
        minted speaker appears in the list as itself.

        The new-speaker option is ranked at ``uncertain_floor`` rather than
        appended after the known ones. That gives it a defensible meaning: a
        known speaker outranks "somebody new" exactly when their similarity
        beats the between-speaker p95 -- the point above which the population
        of DIFFERENT people has essentially run out. Without it, the children
        case (two similar young voices where the answer is often "neither of
        the two you know") could never surface the right answer.
        """
        best_sim = ranked[0][1] if ranked else -1.0
        known = {sid for sid, _ in ranked}
        uncertain = (
            self.config.uncertain_floor <= best_sim < self.config.match_threshold
        )
        entries: List[Dict[str, object]] = [
            {
                "speaker_id": sid,
                "label": self._profiles[sid].label or sid,
                "similarity": round(sim, 3),
                "new": False,
            }
            for sid, sim in ranked
        ]
        entries.append(
            {
                "speaker_id": assigned_id,
                "label": (
                    self._profiles[assigned_id].label
                    if assigned_id in self._profiles
                    else assigned_id
                )
                or assigned_id,
                "similarity": round(self.config.uncertain_floor, 3),
                "new": True,
            }
            if assigned_id not in known
            else {
                "speaker_id": f"speaker-{self._next_id}",
                "label": f"speaker-{self._next_id}",
                "similarity": round(self.config.uncertain_floor, 3),
                "new": True,
            }
        )
        entries.sort(key=lambda entry: -float(entry["similarity"]))
        return uncertain, entries[:3]

    def fold_confirmed(
        self, speaker_id: str, embedding: SpeakerEmbedding
    ) -> SpeakerProfile:
        """Absorb an embedding a human confirmed (§17.4).

        Only ever called after a confirmation. An unconfirmed ambiguous slice
        that moved a centroid would corrupt that identity permanently, and
        nothing downstream could tell it had happened.
        """
        profile = self.get(speaker_id)
        if profile.centroid is None:
            profile.centroid = embedding
            profile.observations = 1
        else:
            self._fold(profile, embedding)
        profile.last_seen = self._clock()
        return profile

    def create_speaker(self, label: Optional[str] = None) -> SpeakerProfile:
        """Mint an empty profile for the client's "+" button (§17.5).

        The centroid is left unset: this speaker has been DECLARED, not heard.
        Their first audio seeds it in :meth:`assign_manual` rather than being
        matched against the existing profiles, which is the whole point of the
        button — the user already knows this is somebody new, and a match
        against a similar-sounding participant would overrule them.
        """
        sid = f"speaker-{self._next_id}"
        self._next_id += 1
        profile = SpeakerProfile(
            speaker_id=sid,
            centroid=None,
            observations=0,
            label=label,
            last_seen=self._clock(),
        )
        self._profiles[sid] = profile
        return profile

    def remove(self, speaker_id: str) -> SpeakerProfile:
        """Forget a speaker completely (user order, 2026-08-03).

        Everything this profile holds goes: the centroid, the reference
        buffer, the label. That is the point -- a speaker created by mistake,
        or one who left the room, otherwise keeps a slot under
        ``max_speakers`` and a preset voice for the rest of the session, and
        keeps attracting utterances that belong to somebody else.

        The id is NOT recycled. Transcript lines already attributed to it stay
        as they are: history is what was said, and silently reattributing old
        lines to somebody else would be a worse answer than a dangling name.
        """
        profile = self._profiles.pop(speaker_id, None)
        if profile is None:
            raise KeyError(f"unknown speaker {speaker_id!r}")
        return profile

    def assign_manual(
        self,
        speaker_id: str,
        audio: AudioChunk,
        embedding: Optional[SpeakerEmbedding] = None,
        text: str = "",
        language: Optional[str] = None,
    ) -> Tuple[SpeakerProfile, bool]:
        """Attribute a segment to a speaker the user named (§17.5).

        Ground truth: no comparison happens and no threshold can overrule it.
        The identity thresholds exist to keep MISIDENTIFIED audio out of a
        reference buffer, and identification is exactly what was skipped — so
        ``reference_threshold`` does not apply here.

        The audio QUALITY criteria still do. A human can vouch for who spoke;
        they cannot vouch for whether the microphone clipped or the segment is
        four tenths of a second long, and a splice-length slice degrades the
        clone for everything that speaker says afterwards.
        """
        profile = self.get(speaker_id)
        now = self._clock()
        profile.last_seen = now
        if embedding is not None:
            if profile.centroid is None:
                # First audio for a declared speaker seeds the centroid, so
                # later automatic turns can find them without another tap.
                profile.centroid = embedding
                profile.observations = 1
            else:
                self._fold(profile, embedding)
        if language:
            profile.last_language = language
        admitted = self._maybe_admit_reference(
            profile, audio, text, language, similarity=1.0, enforce_identity=False
        )
        return profile, admitted

    def assign(
        self,
        embedding: SpeakerEmbedding,
        audio: AudioChunk,
        text: str = "",
        language: Optional[str] = None,
        language_confidence: float = 1.0,
    ) -> Tuple[SpeakerProfile, float, bool]:
        """Match or create a speaker, and maybe admit the audio as reference.

        Returns ``(profile, similarity, admitted_as_reference)``. The
        similarity is reported so the session can put it on the turn event --
        a diarization decision the user can see is one they can correct.
        """
        best_id, best_sim = self._nearest(embedding)
        now = self._clock()
        #: True when this turn was attributed by the continuity guard rather
        #: than by a confident match. Such a turn must never reach the clone
        #: reference: the identity gate in `_maybe_admit_reference` is
        #: deliberately waived for a profile's FIRST reference (the enrollment
        #: anchor), and a guarded assignment does not fold, so `observations`
        #: never grows past 1 and every borderline turn would slip through
        #: that exemption and become the voice.
        guessed = False

        if best_id is not None and best_sim >= self.config.match_threshold:
            profile = self._profiles[best_id]
            self._fold(profile, embedding)
        elif best_id is not None and best_sim >= self.config.uncertain_floor:
            # CONTINUITY GUARD. In the uncertainty band this used to fall
            # through and MINT A NEW SPEAKER, and that is the defect a real
            # conversation reported as "lots of different voices came out of
            # what I said -- it cloned a new voice every time". The cascade is
            # short and brutal: a borderline cosine becomes a new speaker, a
            # new speaker has no reference buffer, so it gets a fresh preset
            # or a fresh clone, and one person changes voice mid-conversation.
            #
            # The asymmetry is deliberate. A wrong continuity is a badge the
            # user can tap to correct, and the line already carries that badge
            # because `uncertainty()` reads the same band. A wrong new speaker
            # is a different voice in the middle of a conversation, which is
            # not correctable after the fact -- it has already been heard.
            #
            # The centroid is deliberately NOT folded here. Continuity is a
            # guess, and a guess must not rewrite the identity it guessed at:
            # if this turn was in fact a second person, folding would drag the
            # first person's centroid toward them and every later decision
            # would inherit the error. That invariant predates this guard
            # ("an unconfirmed uncertain line never moves a centroid") and it
            # survives it -- the guard changes WHO the line is attributed to,
            # not what the system believes afterwards. Confirmation, manual or
            # by a later confident turn, is what moves centroids.
            #
            # Reference admission is likewise untouched: it is gated at
            # `reference_threshold` (0.80), far above this band, so a
            # possibly-wrong turn can never reach the clone prompt.
            profile = self._profiles[best_id]
            guessed = True
        elif len(self._profiles) >= self.config.max_speakers:
            # At capacity: rather than evicting a profile (which would drop a
            # reference buffer someone is still using), the segment joins the
            # nearest speaker and is explicitly barred from their reference
            # buffer below. Translation still works; only cloning fidelity is
            # capped, which is the right thing to sacrifice under crowding.
            if best_id is None:
                raise RuntimeError("speaker registry at capacity with no profiles")
            profile = self._profiles[best_id]
            best_sim = min(best_sim, self.config.reference_threshold - 1e-6)
        else:
            sid = f"speaker-{self._next_id}"
            self._next_id += 1
            profile = SpeakerProfile(
                speaker_id=sid, centroid=embedding, last_seen=now
            )
            self._profiles[sid] = profile
            best_sim = 1.0

        profile.last_seen = now
        if language and language_confidence >= 0.5:
            profile.last_language = language

        admitted = (
            False
            if guessed
            else self._maybe_admit_reference(
                profile, audio, text, language, best_sim
            )
        )
        return profile, best_sim, admitted

    # -- internals ----------------------------------------------------------

    def _nearest(
        self, embedding: SpeakerEmbedding
    ) -> Tuple[Optional[str], float]:
        best_id: Optional[str] = None
        best_sim = -1.0
        for sid, profile in self._profiles.items():
            if profile.centroid is None:
                # Declared but never heard: there is nothing to compare
                # against, and treating a missing centroid as a distance of
                # zero would make every declared speaker the nearest match.
                continue
            sim = profile.centroid.similarity(embedding)
            if sim > best_sim:
                best_id, best_sim = sid, sim
        return best_id, best_sim

    def _fold(self, profile: SpeakerProfile, embedding: SpeakerEmbedding) -> None:
        alpha = self.config.centroid_alpha
        blended = (1.0 - alpha) * profile.centroid.vector + alpha * embedding.vector
        profile.centroid = SpeakerEmbedding(blended)
        profile.observations += 1

    def _maybe_admit_reference(
        self,
        profile: SpeakerProfile,
        audio: AudioChunk,
        text: str,
        language: Optional[str],
        similarity: float,
        enforce_identity: bool = True,
    ) -> bool:
        cfg = self.config
        if (
            enforce_identity
            and similarity < cfg.reference_threshold
            and profile.observations > 1
        ):
            return False
        if audio.duration_s < cfg.min_slice_s:
            return False
        rms = audio.rms()
        if rms < cfg.min_reference_rms:
            return False

        slice_audio = audio
        if audio.duration_s > cfg.max_slice_s:
            # Keep the middle: onsets and trailing silence are the two parts
            # least representative of a steady voice.
            n = int(cfg.max_slice_s * audio.sample_rate)
            start = (len(audio.samples) - n) // 2
            slice_audio = AudioChunk(
                audio.samples[start : start + n], audio.sample_rate
            )

        # Quality score: long and loud, with confident identity as a
        # multiplier. Recency is applied at eviction time, not here, because
        # a slice's quality does not change but its relevance does.
        score = slice_audio.duration_s * min(rms / max(cfg.min_reference_rms, 1e-6), 4.0)
        score *= max(similarity, 0.0)

        now = self._clock()
        item = _ReferenceItem(
            audio=slice_audio,
            text=text,
            language=language or "",
            score=score,
            added_at=now,
        )
        profile.references.append(item)
        self._evict(profile, now)
        return any(kept is item for kept in profile.references)

    def _retention_score(self, item: _ReferenceItem, now: float) -> float:
        """Quality discounted by age, so the buffer rolls without thrashing."""
        half_life = max(self.config.recency_half_life_s, 1e-6)
        age = max(now - item.added_at, 0.0)
        return item.score * (0.5 ** (age / half_life))

    def _evict(self, profile: SpeakerProfile, now: float) -> None:
        """Trim the ROLLING slot. The enrollment slot is not touched.

        Two budgets, evicted independently: enrolled slices are the anchor and
        must survive any amount of field audio, while field slices compete
        against each other for ``rolling_prompt_s`` seconds.
        """
        cfg = self.config
        enrolled = [i for i in profile.references if i.enrolled]
        rolling = [i for i in profile.references if not i.enrolled]
        rolling.sort(key=lambda i: (-self._retention_score(i, now), -i.added_at))
        kept: List[_ReferenceItem] = []
        total = 0.0
        for item in rolling:
            if kept and total >= cfg.rolling_prompt_s:
                break
            kept.append(item)
            total += item.audio.duration_s
        profile.references = enrolled + kept

    def reference_for(
        self, speaker_id: str, min_seconds: float, sample_rate: Optional[int] = None
    ) -> Tuple[AudioChunk, str]:
        """Reference clip + its transcript, or refuse with the shortfall named.

        Refusing is a real outcome, not an error path to be swallowed: a
        speaker's first turn frequently arrives before they have accumulated a
        clonable reference. The caller (``session.py``) turns this into a
        documented fallback voice rather than a failed turn, because hearing
        the translation in a stranger's voice beats hearing nothing.
        """
        profile = self.get(speaker_id)
        have = profile.reference_seconds()
        if have < min_seconds:
            raise ReferenceTooShort(speaker_id, have, min_seconds)
        return profile.reference_audio(sample_rate), profile.reference_text()


def split_points_by_dispersion(
    window_embeddings: Sequence[SpeakerEmbedding],
    threshold: float = 0.62,
) -> Tuple[int, ...]:
    """Window indices where the voice changes inside one VAD segment.

    The one failure mode a per-segment (rather than frame-level) diarizer
    genuinely has: two people speaking back-to-back with no pause between
    them land in a single segment, get one embedding, and one of them
    contributes their voice to the other's reference buffer. That is the
    expensive kind of mistake -- a poisoned reference buffer is audible in
    every later turn by that speaker.

    The cheap guard is to embed ~1.5 s windows across the segment and look for
    an adjacent pair whose cosine similarity falls below ``threshold``. It
    costs one extra forward pass per window on a ~25 M-parameter embedder and
    catches the back-to-back case without any of the machinery a streaming
    diarizer would bring. It does NOT handle overlapped speech; under
    half-duplex turn-taking, overlapped speech produces unusable ASR anyway.

    Returns the indices ``i`` such that a boundary falls between window
    ``i-1`` and window ``i``. Empty means one speaker throughout.
    """
    if len(window_embeddings) < 2:
        return ()
    points: List[int] = []
    for i in range(1, len(window_embeddings)):
        if window_embeddings[i - 1].similarity(window_embeddings[i]) < threshold:
            points.append(i)
    return tuple(points)


def rolling_reference_from_segments(
    segments: Sequence[Tuple[AudioChunk, float]], target_s: float
) -> AudioChunk:
    """Pick the highest-scoring segments up to ``target_s`` and concatenate.

    Standalone helper for the enrollment tool, which has a pile of samples and
    no registry yet. Same policy as the registry so enrollment and field
    accumulation cannot drift apart.
    """
    if not segments:
        raise ValueError("no segments to build a reference from")
    rate = segments[0][0].sample_rate
    ordered = sorted(segments, key=lambda pair: -pair[1])
    merged = AudioChunk(np.zeros(0, dtype=np.float32), rate)
    for audio, _score in ordered:
        if merged.duration_s >= target_s:
            break
        merged = merged.concat(audio)
    return merged
