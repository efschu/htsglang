# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The 18 preset voices, as data — descriptors now, clips in the GPU window.

**The chicken-and-egg this file resolves.** The serving checkpoint,
Qwen3-TTS-12Hz-0.6B-Base, is a *cloning* model: it needs a reference clip to
produce a voice, so it cannot invent one. Eighteen novel artificial voices
therefore cannot come from the serving model itself. Two ways out, and only one
is clean:

* record eighteen real people — not happening, and it drags consent and
  likeness questions into a holiday project;
* render them once, offline, with **Qwen3-TTS-12Hz-1.7B-VoiceDesign**
  (Apache-2.0), which synthesizes a voice from a natural-language description.
  The rendered clips then become ordinary reference audio for the Base model at
  serving time.

So this module holds the *descriptions and seeds*, the render is a one-time GPU
step, and the pool loader afterwards sees nothing unusual — just wav files in
`<class>/<voice_id>.<language>.wav`. Nothing about the runtime path knows the
clips were synthetic.

**Sizing** (user decision 2026-08-03): 6 man / 6 woman / 3 boy / 3 girl = 18.
Realistic worst case is 6-8 participants, usually 2-4; eight can plausibly skew
hard to one adult class, children rarely exceed three.

**Distinctness is the design goal, not variety for its own sake.** In preset
mode the voice is the listener's ONLY cue for who is speaking, so within a
class the descriptions are pushed apart on the axes a listener actually
discriminates on -- pitch height, speech rate, brightness/weight, and age --
rather than on flavour adjectives that a synthesizer renders identically.

The rendered clips are Apache-2.0 model output with no real person's likeness
involved, so they carry no consent or backup obligation.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Sequence, Tuple

from sglang.srt.translator.voices import VoiceClass

__all__ = [
    "PresetDescriptor",
    "PRESET_DESCRIPTORS",
    "RENDER_SENTENCES",
    "VOICE_DESIGN_MODEL",
    "descriptors_for_class",
    "render_plan",
]

#: The model that renders the descriptors into clips. Not the serving model.
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
VOICE_DESIGN_REVISION = "5ecdb67327fd"

#: What each preset says in its clip, per language.
#:
#: Chosen to be phonetically broad rather than meaningful: a reference clip is
#: read for timbre, and a sentence that misses whole phoneme classes gives the
#: cloner less to work with. Each is ~8-10 s read at a normal pace, which sits
#: in the 6-12 s contiguous window the reference policy prefers.
#:
#: Content is deliberately banal. These clips are heard by strangers in a
#: conversation only if something goes wrong, and a preset that says something
#: memorable would be a liability.
RENDER_SENTENCES: Dict[str, str] = {
    "de": (
        "Guten Tag, ich zähle einmal langsam von eins bis zehn und wünsche "
        "einen ruhigen, freundlichen Nachmittag am Hafen."
    ),
    "es": (
        "Buenos días, cuento despacio del uno al diez y les deseo una tarde "
        "tranquila y agradable junto al puerto."
    ),
    "en": (
        "Good afternoon, I will count slowly from one to ten and wish you a "
        "calm and pleasant evening by the harbour."
    ),
    "fr": (
        "Bonjour, je compte lentement de un à dix et vous souhaite une "
        "après-midi calme et agréable près du port."
    ),
    "it": (
        "Buongiorno, conto lentamente da uno a dieci e vi auguro un "
        "pomeriggio tranquillo e piacevole vicino al porto."
    ),
    "pt": (
        "Bom dia, conto devagar de um a dez e desejo-lhes uma tarde "
        "tranquila e agradável junto ao porto."
    ),
}


@dataclasses.dataclass(frozen=True)
class PresetDescriptor:
    """One preset voice, before it has been rendered.

    ``description`` is the VoiceDesign prompt. ``seed`` pins the render so a
    re-render after a lost pool produces the SAME voice -- otherwise a speaker
    the user has learned to recognise would silently change identity between
    sessions.
    """

    voice_id: str
    voice_class: VoiceClass
    label: str
    description: str
    seed: int

    def directory_name(self) -> str:
        """Pool subdirectory: the class, exactly as the loader expects."""
        return self.voice_class.value

    def filename(self, language: str) -> str:
        return f"{self.voice_id}.{language}.wav"


# Six adult men, spread over pitch, rate, weight and age.
_MEN: Tuple[PresetDescriptor, ...] = (
    PresetDescriptor(
        "man-01", VoiceClass.MAN, "calm baritone",
        "A calm adult man with a low, warm baritone voice, speaking at a "
        "relaxed unhurried pace, clear articulation, neutral and friendly.",
        466001),
    PresetDescriptor(
        "man-02", VoiceClass.MAN, "bright tenor",
        "An adult man with a noticeably high, bright, light tenor voice, "
        "speaking briskly and energetically with crisp consonants.",
        466002),
    PresetDescriptor(
        "man-03", VoiceClass.MAN, "deep slow",
        "An older adult man with a very deep, heavy, gravelly voice, "
        "speaking slowly and deliberately with long pauses.",
        466003),
    PresetDescriptor(
        "man-04", VoiceClass.MAN, "young quick",
        "A young adult man in his early twenties with a mid-pitched, thin, "
        "slightly nasal voice, speaking quickly and casually.",
        466004),
    PresetDescriptor(
        "man-05", VoiceClass.MAN, "measured mid",
        "A middle-aged adult man with a mid-pitched, resonant, rounded voice, "
        "speaking at a measured even pace, formal and precise.",
        466005),
    PresetDescriptor(
        "man-06", VoiceClass.MAN, "soft breathy",
        "An adult man with a soft, breathy, quiet voice, mid to low pitch, "
        "speaking gently and slowly as if in a small room.",
        466006),
)

# Six adult women, spread on the same axes.
_WOMEN: Tuple[PresetDescriptor, ...] = (
    PresetDescriptor(
        "woman-01", VoiceClass.WOMAN, "warm alto",
        "A calm adult woman with a low, warm alto voice, speaking at a "
        "relaxed unhurried pace, clear articulation, neutral and friendly.",
        466011),
    PresetDescriptor(
        "woman-02", VoiceClass.WOMAN, "bright soprano",
        "An adult woman with a high, bright, light voice, speaking briskly "
        "and energetically with crisp consonants.",
        466012),
    PresetDescriptor(
        "woman-03", VoiceClass.WOMAN, "older slow",
        "An older adult woman with a lower, slightly husky voice, speaking "
        "slowly and deliberately with long pauses.",
        466013),
    PresetDescriptor(
        "woman-04", VoiceClass.WOMAN, "young quick",
        "A young adult woman in her early twenties with a mid-high, thin, "
        "slightly nasal voice, speaking quickly and casually.",
        466114),
    PresetDescriptor(
        "woman-05", VoiceClass.WOMAN, "measured mid",
        "A middle-aged adult woman with a mid-pitched, resonant, rounded "
        "voice, speaking at a measured even pace, formal and precise.",
        466015),
    PresetDescriptor(
        "woman-06", VoiceClass.WOMAN, "soft breathy",
        "An adult woman with a soft, breathy, quiet voice, mid pitch, "
        "speaking gently and slowly as if in a small room.",
        466016),
)

# Three boys and three girls. The classifier CANNOT separate boy from girl
# (prepubescent voices are not discriminable on F0), so both sets serve any
# speaker classified CHILD -- these are six child voices that happen to be
# labelled, and the labels only matter for a manual override.
_BOYS: Tuple[PresetDescriptor, ...] = (
    PresetDescriptor(
        "boy-01", VoiceClass.BOY, "young boy bright",
        "A young boy around eight years old with a high, bright, clear voice, "
        "speaking cheerfully at a moderate pace.",
        466021),
    PresetDescriptor(
        "boy-02", VoiceClass.BOY, "older boy",
        "A boy around twelve years old with a mid-high voice just beginning "
        "to deepen, speaking calmly and a little shyly.",
        466022),
    PresetDescriptor(
        "boy-03", VoiceClass.BOY, "small boy quick",
        "A small boy around five years old with a very high, light, piping "
        "voice, speaking quickly and excitedly.",
        466023),
)

_GIRLS: Tuple[PresetDescriptor, ...] = (
    PresetDescriptor(
        "girl-01", VoiceClass.GIRL, "young girl bright",
        "A young girl around eight years old with a high, bright, clear "
        "voice, speaking cheerfully at a moderate pace.",
        466031),
    PresetDescriptor(
        "girl-02", VoiceClass.GIRL, "older girl",
        "A girl around twelve years old with a mid-high, steady voice, "
        "speaking calmly and confidently.",
        466032),
)

#: All 18, in a stable order. The pool's allocation sorts by voice_id, so this
#: order also decides who gets assigned first -- the "01" voices are the most
#: neutral of each class on purpose, because the first speaker in a
#: conversation is usually the user.


#: RETIRED 2026-08-03 after three attempts, and the reason is a MODEL limit
#: rather than a bad seed or a weak description.
#:
#:   original description          girl-01 / girl-03 = 0.738
#:   new seed (466033 -> 466133)                     = 0.777
#:   rewritten description, moved on timbre,
#:   dynamics and pace simultaneously                = 0.769
#:
#: All above the registry's 0.70 same-speaker line. A seed samples WITHIN a
#: description and cannot escape it, and the description was already moved on
#: three independent axes -- so the remaining explanation is that VoiceDesign
#: has little separable range left in the small-girl corner of its voice
#: space. Pruning is the right response to a model limit; it was the WRONG
#: response earlier, when the cause was the derive step's compression and the
#: fix was to remove that instead (see DESIGN_466 SS15b).
#:
#: Cost is small: DESIGN_466 SS4.3 notes boy/girl are not distinguishable from
#: F0 before puberty, so both classes match a CHILD classification. Two girls
#: plus three boys still gives five distinct child voices against a stated
#: worst case of three children.
_RETIRED: Tuple[PresetDescriptor, ...] = (
    PresetDescriptor(
        # Rewritten 2026-08-03, and the DESCRIPTION was the lever, not the
        # seed. The original ("five years old, very high, light, piping,
        # quickly and excitedly") differed from girl-01 ("eight years old,
        # high, bright, clear, cheerfully") only in degree: both are bright
        # high-pitched cheerful young girls, and the model rendered them as
        # one voice (0.738, then 0.777 after a re-seed -- a new seed samples
        # WITHIN a description and cannot escape it). This one is moved on
        # three independent axes at once: timbre (husky/breathy vs bright and
        # clear), dynamics (soft vs cheerful) and pace (hesitant with pauses
        # vs moderate).
        "girl-03", VoiceClass.GIRL, "small girl soft",
        "A small girl around six years old with a slightly husky, breathy "
        "voice and a narrow pitch range, speaking softly and hesitantly with "
        "frequent pauses.",
        466133),
)


PRESET_DESCRIPTORS: Tuple[PresetDescriptor, ...] = _MEN + _WOMEN + _BOYS + _GIRLS


def descriptors_for_class(voice_class: VoiceClass) -> Tuple[PresetDescriptor, ...]:
    return tuple(d for d in PRESET_DESCRIPTORS if d.voice_class is voice_class)


def render_plan(
    languages: Sequence[str], pool_root: str
) -> Tuple[Dict[str, object], ...]:
    """Every clip that must be rendered, as plain data.

    Consumed by the GPU-window render script. Returned rather than executed
    because rendering needs a card and this module must stay importable on the
    desk -- the same split every other backend here uses.

    A language with no sentence in :data:`RENDER_SENTENCES` is refused rather
    than rendered from another language's text, which would bake that
    language's accent into every preset of the pool.
    """
    missing = [c for c in languages if c not in RENDER_SENTENCES]
    if missing:
        raise ValueError(
            f"no render sentence for {missing}; add one to RENDER_SENTENCES "
            "rather than reusing another language's text, which would give "
            "every preset that language's accent"
        )
    plan = []
    for descriptor in PRESET_DESCRIPTORS:
        for language in languages:
            plan.append(
                {
                    "voice_id": descriptor.voice_id,
                    "voice_class": descriptor.voice_class.value,
                    "language": language,
                    "description": descriptor.description,
                    "text": RENDER_SENTENCES[language],
                    "seed": descriptor.seed,
                    "path": (
                        f"{pool_root}/{descriptor.directory_name()}/"
                        f"{descriptor.filename(language)}"
                    ),
                }
            )
    return tuple(plan)
